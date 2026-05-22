# src/data/processor.py
"""
ProductionDataProcessor — sklearn Pipeline backed processor.

KEY CHANGE from previous version:
  - No manual config required
  - AutoConfigBuilder introspects DataFrame automatically
  - fit() always called on REFERENCE data
  - transform() called on both reference AND production
  - Returns ProcessingResult with both processed DataFrames
"""
from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from src.data.column_config import AutoConfigBuilder
from src.data.pipeline_builder import ProcessingPipelineBuilder
from src.data.processing_config import (
    EncodingStrategy,
    ImputationStrategy,
    OutlierStrategy,
    ProcessingConfig,
    ScalingStrategy,
)
from src.data.processing_report import ProcessingReport
from src.observability.event_bus import EventType, event_bus, make_event

logger = logging.getLogger("ml_platform.data.processor")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@dataclass
class ProcessingResult:
    """
    Result of processing both reference and production datasets.
    Both are now in the SAME feature space → drift comparison valid.
    """
    reference_processed: pd.DataFrame
    production_processed: pd.DataFrame

    # Reports
    reference_report: ProcessingReport
    production_report: ProcessingReport

    # Metadata
    feature_names: list[str] = field(default_factory=list)
    processor_path: str = ""

    # Summary stats for orchestrator
    reference_rows_in: int = 0
    reference_rows_out: int = 0
    production_rows_in: int = 0
    production_rows_out: int = 0
    n_features_out: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": {
                "rows_in":  self.reference_rows_in,
                "rows_out": self.reference_rows_out,
                "report":   self.reference_report.to_dict(),
            },
            "production": {
                "rows_in":  self.production_rows_in,
                "rows_out": self.production_rows_out,
                "report":   self.production_report.to_dict(),
            },
            "n_features_out": self.n_features_out,
            "feature_names":  self.feature_names,
            "processor_path": self.processor_path,
            "warnings":       self.warnings,
        }


class ProductionDataProcessor:
    """
    Production-grade data processor.

    AUTOMATIC operation — no manual config needed.
    AutoConfigBuilder inspects reference DataFrame and selects strategies.

    CYCLE FLOW:
    ───────────
    Cycle 1 (initial):
      reference  = historical training data
      production = this month's real user data

      processor.fit(reference)              ← fits on reference
      ref_proc  = processor.transform(ref)  ← reference in fitted space
      prod_proc = processor.transform(prod) ← production in SAME fitted space
      → drift(ref_proc, prod_proc)          ← valid comparison
      → retrain on prod_proc
      → save production as new reference

    Cycle 2 (next month):
      reference  = last month's production (saved from cycle 1)
      production = this month's real user data

      processor.fit(reference)              ← fits on NEW reference
      ref_proc  = processor.transform(ref)
      prod_proc = processor.transform(prod)
      → drift(ref_proc, prod_proc)
      ...

    WHY fit on reference (not production)?
      Reference = what the current champion model "knows".
      We want both datasets in REFERENCE feature space.
      Drift = "has production moved away from reference space?"
      If we fit on production → we'd be asking if reference drifted
      from production, which is the wrong question.
    """

    def __init__(
        self,
        model_id: str = "",
        pipeline_run_id: str = "",
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        onehot_threshold: int = 10,
        skewness_threshold: float = 2.0,
        max_null_rate_to_drop: float = 0.70,
    ) -> None:
        self.model_id             = model_id
        self.pipeline_run_id      = pipeline_run_id
        self.target_column        = target_column
        self.id_columns           = id_columns or []
        self.onehot_threshold     = onehot_threshold
        self.skewness_threshold   = skewness_threshold
        self.max_null_rate_to_drop = max_null_rate_to_drop

        self._is_fitted           = False
        self._pipeline: Optional[Pipeline] = None
        self._config: Optional[ProcessingConfig] = None
        self._feature_names_out: list[str] = []

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def fit_transform_both(
        self,
        reference_df: pd.DataFrame,
        production_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Fit on reference, transform BOTH reference and production.

        This is the ONLY method the pipeline calls.
        Returns ProcessingResult with both processed DataFrames.

        Parameters
        ----------
        reference_df:
            The data the current champion was trained on.
            Processor FITS on this → learns the reference distribution.

        production_df:
            Real user data collected since last training.
            Processor TRANSFORMS this → puts it in reference space.

        save_path:
            Where to save the fitted processor.
        """
        self._validate_input(reference_df, "reference")
        self._validate_input(production_df, "production")

        logger.info(
            f"Processing both datasets: "
            f"reference={len(reference_df)} rows, "
            f"production={len(production_df)} rows"
        )

        # ── Step 1: Auto-build config from reference data ──────────────────
        # Why reference? Config should reflect what model was trained on.
        # OHE vocabulary, frequency maps, scaling bounds — all from reference.
        auto_builder  = AutoConfigBuilder(
            target_column=self.target_column,
            id_columns=self.id_columns,
            max_null_rate_to_drop=self.max_null_rate_to_drop,
            onehot_threshold=self.onehot_threshold,
            skewness_threshold=self.skewness_threshold,
        )
        self._config = auto_builder.build(reference_df)

        # ── Step 2: Build sklearn Pipeline from auto-config ────────────────
        builder        = ProcessingPipelineBuilder(self._config)
        ref_features, ref_target = self._separate_target(reference_df)
        self._pipeline = builder.build(ref_features)

        # ── Step 3: Fit on reference data ──────────────────────────────────
        y_ref = ref_target.values if ref_target is not None else None

        self._pipeline.fit(ref_features, y=y_ref)
        self._is_fitted = True
        self._feature_names_out = self._get_feature_names()

        logger.info(
            f"Pipeline fitted on reference data. "
            f"Features out: {len(self._feature_names_out)}"
        )

        # ── Step 4: Transform reference ────────────────────────────────────
        ref_report    = self._make_report(reference_df, "reference")
        ref_processed = self._apply_transform(
            reference_df, ref_report, dataset_label="reference"
        )

        # ── Step 5: Transform production ───────────────────────────────────
        # Uses EXACTLY same fitted parameters as reference transform
        # No production statistics contaminate the processor
        prod_report    = self._make_report(production_df, "production")
        prod_processed = self._apply_transform(
            production_df, prod_report, dataset_label="production"
        )

        # ── Step 6: Save fitted processor ──────────────────────────────────
        actual_save_path = save_path or (
            f"artifacts/processors/{self.model_id}/processor.joblib"
        )
        self.save(actual_save_path)

        # ── Step 7: Build result ───────────────────────────────────────────
        all_warnings = ref_report.warnings + prod_report.warnings

        result = ProcessingResult(
            reference_processed=ref_processed,
            production_processed=prod_processed,
            reference_report=ref_report,
            production_report=prod_report,
            feature_names=list(self._feature_names_out),
            processor_path=actual_save_path,
            reference_rows_in=len(reference_df),
            reference_rows_out=len(ref_processed),
            production_rows_in=len(production_df),
            production_rows_out=len(prod_processed),
            n_features_out=len(self._feature_names_out),
            warnings=all_warnings,
        )

        self._emit_processing_event(result)
        return result

    def transform_single(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ProcessingReport]:
        """
        Transform a single DataFrame using fitted processor.
        Used at inference time (single batch or request).
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError(
                "Processor not fitted. "
                "Call fit_transform_both() or load a saved processor."
            )
        self._validate_input(df, "inference")
        report      = self._make_report(df, "inference")
        processed   = self._apply_transform(df, report, "inference")
        return processed, report

    def save(self, path: str) -> str:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Processor saved: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "ProductionDataProcessor":
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected ProductionDataProcessor, got {type(obj)}"
            )
        logger.info(f"Processor loaded: {path}")
        return obj

    def get_feature_names_out(self) -> list[str]:
        return list(self._feature_names_out)

    # ─────────────────────────────────────────────────────────────────────
    # Internal transform
    # ─────────────────────────────────────────────────────────────────────

    def _apply_transform(
        self,
        df: pd.DataFrame,
        report: ProcessingReport,
        dataset_label: str,
    ) -> pd.DataFrame:
        """
        Apply fitted Pipeline.transform() to a DataFrame.
        Reconstructs DataFrame with proper column names.
        """
        df_features, target_series = self._separate_target(df)

        try:
            if self._pipeline is None:
                raise RuntimeError("Pipeline not built.")
            transformed_array = self._pipeline.transform(df_features)
        except Exception as exc:
            logger.error(
                f"Pipeline transform failed on {dataset_label}: {exc}",
                exc_info=True,
            )
            raise

        result_df = self._array_to_dataframe(
            transformed_array, df_features.index
        )

        # Reattach target (untouched)
        if target_series is not None:
            result_df[self.target_column] = (
                target_series.reindex(result_df.index).values
            )

        result_df = self._final_cleanup(result_df)

        # Enrich report
        report.output_rows = len(result_df)
        report.output_cols = len(result_df.columns)
        report.completed_at = datetime.now(timezone.utc)
        self._enrich_report(report, df, result_df)

        logger.info(
            f"Transform [{dataset_label}]: "
            f"{len(df)}→{len(result_df)} rows, "
            f"{len(df.columns)}→{len(result_df.columns)} cols, "
            f"warnings={len(report.warnings)}"
        )

        return result_df

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _separate_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.target_column and self.target_column in df.columns:
            return (
                df.drop(columns=[self.target_column]),
                df[self.target_column].copy(),
            )
        return df.copy(), None

    def _get_feature_names(self) -> list[str]:
        """Extract and clean feature names from ColumnTransformer."""
        if self._pipeline is None:
            return []
        try:
            for _, step in self._pipeline.steps:
                if hasattr(step, "transformers"):
                    raw = step.get_feature_names_out()
                    return [
                        n.split("__", 1)[1] if "__" in n else n
                        for n in raw
                    ]
        except Exception as exc:
            logger.warning(f"Could not get feature names: {exc}")
        return []

    def _array_to_dataframe(
        self,
        array: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        n_cols = array.shape[1] if array.ndim > 1 else 1
        cols   = (
            self._feature_names_out
            if len(self._feature_names_out) == n_cols
            else [f"feature_{i}" for i in range(n_cols)]
        )
        actual_index = index[:len(array)]
        return pd.DataFrame(array, columns=cols, index=actual_index)

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        target = self.target_column

        for col in df.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                n_null = int(df[col].isnull().sum())
                if n_null > 0:
                    df[col] = df[col].fillna(0.0)

        for col in df.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float32)

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        return df

    def _make_report(
        self, df: pd.DataFrame, label: str
    ) -> ProcessingReport:
        return ProcessingReport(
            run_id=f"{str(uuid.uuid4())[:8]}_{label}",
            model_id=self.model_id,
            pipeline_run_id=self.pipeline_run_id,
            started_at=datetime.now(timezone.utc),
            input_rows=len(df),
            input_cols=len(df.columns),
        )

    def _enrich_report(
        self,
        report: ProcessingReport,
        original: pd.DataFrame,
        result: pd.DataFrame,
    ) -> None:
        orig_cols   = set(original.columns)
        result_cols = set(result.columns)
        target_set  = {self.target_column} if self.target_column else set()

        report.dropped_columns.extend(
            sorted(orig_cols - result_cols - target_set)
        )
        report.added_columns.extend(
            sorted(result_cols - orig_cols - target_set)
        )

        # Collect pipeline step warnings
        if self._pipeline:
            for name, step in self._pipeline.steps:
                if hasattr(step, "cols_to_drop_"):
                    for col in step.cols_to_drop_:
                        report.warnings.append(
                            f"[{name}] dropped '{col}'"
                        )
                if hasattr(step, "leakage_warnings_"):
                    report.warnings.extend(step.leakage_warnings_)

    def _emit_processing_event(self, result: ProcessingResult) -> None:
        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.PIPELINE_STAGE_COMPLETED,
            step_name="data_processing",
            title=(
                f"⚙️ Processing Complete | "
                f"ref: {result.reference_rows_in}→{result.reference_rows_out} | "
                f"prod: {result.production_rows_in}→{result.production_rows_out}"
            ),
            message=(
                f"Features out: {result.n_features_out} | "
                f"Warnings: {len(result.warnings)}"
            ),
            model_id=self.model_id,
            status="success" if not result.warnings else "warning",
            data=result.to_dict(),
        ))

    @staticmethod
    def _validate_input(df: pd.DataFrame, label: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{label}: expected DataFrame, got {type(df)}"
            )
        if df.empty:
            raise ValueError(f"{label}: cannot process empty DataFrame.")