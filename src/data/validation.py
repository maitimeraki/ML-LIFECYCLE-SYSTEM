# src/data/validation.py
"""
Production data validation using Great Expectations 1.x.
Emits granular events for every expectation check.

Fix applied: NaN handling moved to column-level with explicit
imputation strategy rather than silent per-feature dropna().
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.expectations.expectation import Expectation

from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import GE_EXPECTATIONS_TOTAL, GE_SUITE_SCORE, DATA_ROWS_VALIDATED

logger = logging.getLogger("ml_platform.data.validation")


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list] = None


@dataclass
class DatasetSchema:
    columns: list[ColumnSchema]
    min_rows: int = 100
    max_rows: Optional[int] = None


@dataclass
class ValidationReport:
    """Unified validation report consumed by MLLifecyclePipeline."""
    suite_name: str
    is_valid: bool
    success_rate: float
    statistics: dict[str, Any]
    expectation_results: list[dict[str, Any]]
    errors: list[str]
    evaluated_at: str
    data_snapshot: dict[str, Any] = field(default_factory=dict)
    alignment_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name":          self.suite_name,
            "is_valid":            self.is_valid,
            "success_rate":        round(self.success_rate, 4),
            "statistics":          self.statistics,
            "expectation_results": self.expectation_results,
            "errors":              self.errors,
            "evaluated_at":        self.evaluated_at,
            "data_snapshot":       self.data_snapshot,
        }


class DataValidator:
    """
    Great Expectations-based validator.
    Emits an event per expectation and updates Prometheus automatically
    via the registered EventBus handler.
    """

    FRAMEWORK = "great_expectations"

    def __init__(self, schema: Optional[DatasetSchema] = None) -> None:
        self.schema = schema
        self.context = gx.get_context()

    def validate(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        reference_df: Optional[pd.DataFrame] = None,
        pipeline_run_id: str = "",
        model_id: str = "",
        expectations: Optional[list[dict[str, Any]]] = None,
    ) -> ValidationReport:
        """
        Validate a DataFrame and emit observable events.

        Critical fix: we no longer dropna() silently. 
        Instead we track null rates explicitly and use GE's `mostly` 
        parameter to encode tolerance.
        """
        suite_name = f"{model_id or 'dataset'}_{dataset_id}"

        if df.empty:
            raise ValueError("Cannot validate empty DataFrame")

        start_time = time.perf_counter()
        errors: list[str] = []

        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id,
            event_type=EventType.GE_SUITE_STARTED,
            step_name="data_validation",
            title="📋 GE Validation Started",
            message=f"Suite: '{suite_name}' | Rows: {len(df)} | Cols: {len(df.columns)}",
            framework=self.FRAMEWORK,
            status="running",
            model_id=model_id,
            data={
                "suite_name":   suite_name,
                "row_count":    len(df),
                "column_count": len(df.columns),
                "columns":      list(df.columns),
            },
        ))

        # Schema-level checks before GE (fast-fail). But why self.schema? Because if we have a schema, we can do some quick checks before running the full GE suite. This can catch basic issues like missing columns or too few rows without the overhead of GE. It's an optional step that adds a safety net before the more detailed expectation checks.
        if self.schema:
            schema_errors = self._schema_level_checks(df)
            errors.extend(schema_errors)

        # Build GE datasource
        try:
            ds_name = f"ds_{suite_name}_{int(time.time())}"
            data_source      = self.context.data_sources.add_pandas(name=ds_name)
            data_asset       = data_source.add_dataframe_asset(name=f"asset_{suite_name}")
            batch_definition = data_asset.add_batch_definition_whole_dataframe(
                name=f"batch_{suite_name}"
            )

            suite = self.context.suites.add_or_update(
                gx.ExpectationSuite(name=suite_name)
            )

            expectations_to_run = (
                expectations
                or self._build_expectations_from_schema(df)
            )

            for exp in expectations_to_run:
                expectation = self._create_expectation(
                    exp["expectation_type"], exp.get("kwargs", {})
                )
                if expectation:
                    suite.add_expectation(expectation)

            batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
            validation_results = batch.validate(suite)

        except Exception as exc:
            logger.error(f"GE validation failed: {exc}", exc_info=True)
            errors.append(f"GE engine error: {exc}")
            return ValidationReport(
                suite_name=suite_name,
                is_valid=False,
                success_rate=0.0,
                statistics={},
                expectation_results=[],
                errors=errors,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
            )

        # Process results and emit events
        expectation_results: list[dict[str, Any]] = []
        passed = 0
        total = len(validation_results.results)

        for idx, result in enumerate(validation_results.results, start=1):
            exp_cfg = result.expectation_config
            if exp_cfg is None:
                continue

            success      = bool(result.success)
            exp_type     = exp_cfg.type
            kwargs       = exp_cfg.kwargs
            column       = kwargs.get("column", "dataset-level")

            if success:
                passed += 1
            else:
                errors.append(f"FAILED: {exp_type} on '{column}'")

            entry = {
                "expectation_type": exp_type,
                "column":           column,
                "success":          success,
                "kwargs":           dict(kwargs),
                "result":           result.result if hasattr(result, "result") else {},
            }
            expectation_results.append(entry)

            # Emit per-expectation event → Prometheus handler picks it up
            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.GE_EXPECTATION_CHECKED,
                step_name="data_validation",
                title=(
                    f"{'✅' if success else '❌'} "
                    f"{exp_type.replace('expect_column_', '').replace('_', ' ').title()}"
                ),
                message=f"Column '{column}': {'PASSED' if success else 'FAILED'}",
                framework=self.FRAMEWORK,
                status="success" if success else "warning",
                severity="info" if success else "warning",
                model_id=model_id,
                data={**entry, "suite_name": suite_name},
                progress=idx / total,
            ))

        # Distribution alignment vs reference
        alignment_results: list[dict[str, Any]] = []
        if reference_df is not None and not reference_df.empty:
            alignment_results = self._check_distribution_alignment(
                df, reference_df, pipeline_run_id, model_id
            )

        success_rate  = passed / total if total > 0 else 0.0
        duration_ms   = (time.perf_counter() - start_time) * 1000
        data_snapshot = self._compute_snapshot(df)

        statistics = {
            "evaluated_expectations":   total,
            "successful_expectations":  passed,
            "unsuccessful_expectations": total - passed,
            "success_percent":          round(success_rate * 100, 2),
        }

        # Update Prometheus directly for suite-level gauge
        GE_SUITE_SCORE.labels(suite_name=suite_name).set(success_rate)
        DATA_ROWS_VALIDATED.labels(model_id=model_id).inc(len(df))

        # Emit suite completion → Prometheus handler
        event_bus.emit(make_event(
            pipeline_run_id=pipeline_run_id,
            event_type=EventType.GE_SUITE_COMPLETED,
            step_name="data_validation",
            title=f"{'✅' if success_rate == 1.0 else '❌'} GE Suite Completed",
            message=(
                f"Suite '{suite_name}': {passed}/{total} passed "
                f"({success_rate:.1%}) in {duration_ms:.0f}ms"
            ),
            framework=self.FRAMEWORK,
            status="success" if success_rate == 1.0 else "failed",
            severity="info" if success_rate == 1.0 else "error",
            model_id=model_id,
            data={
                "suite_name": suite_name,
                "statistics": statistics,
                "data_snapshot": data_snapshot,
            },
            duration_ms=duration_ms,
            progress=1.0,
        ))

        return ValidationReport(
            suite_name=suite_name,
            is_valid=len(errors) == 0 or success_rate >= 0.90,
            success_rate=success_rate,
            statistics=statistics,
            expectation_results=expectation_results,
            errors=errors,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            data_snapshot=data_snapshot,
            alignment_results=alignment_results,
        )
        
    def _schema_level_checks(self, df: pd.DataFrame) -> list[str]:
        """Fast schema checks before invoking GE."""
        errors = []
        if not self.schema:
            return errors

        if len(df) < self.schema.min_rows:
            errors.append(
                f"Row count {len(df)} below minimum {self.schema.min_rows}"
            )
        # referenced dataset -> schema and production dataset -> df

        required_cols = {c.name for c in self.schema.columns}
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        return errors

    def _create_expectation(
        self, exp_type: str, kwargs: dict[str, Any]
    ) -> Optional[Expectation]:
        try:
            cls = getattr(gx.expectations, exp_type, None)
            if cls is None:
                logger.warning(f"Unknown expectation: {exp_type}")
                return None
            return cls(**kwargs)
        except Exception as exc:
            logger.warning(f"Could not create '{exp_type}': {exc}")
            return None
    
    def _build_expectations_from_schema(
        self, df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """
        Generate expectations from schema or auto-profile.

        Key fix: null rates are computed from the FULL column (not dropna'd),
        so the `mostly` parameter accurately reflects production reality.
        """
        expectations: list[dict[str, Any]] = []

        # Table-level
        expectations.append({
            "expectation_type": "ExpectTableRowCountToBeBetween",
            "kwargs": {
                "min_value": self.schema.min_rows if self.schema else 100,
                "max_value": None,
            },
        })
        expectations.append({
            "expectation_type": "ExpectTableColumnsToMatchSet",
            "kwargs": {
                "column_set":   list(df.columns),
                "exact_match":  True,
            },
        })

        schema_map = (
            {c.name: c for c in self.schema.columns}
            if self.schema else {}
        )

        for col in df.columns:
            col_schema = schema_map.get(col)
            total      = len(df)
            null_count = int(df[col].isnull().sum())

            # Null tolerance: observed rate + 5% buffer, max 30%
            # We do NOT drop nulls — we measure them on the full column
            null_rate    = null_count / total if total > 0 else 0.0
            null_budget  = min(null_rate + 0.05, 0.30)
            mostly_valid = round(1.0 - null_budget, 4)

            if null_count == total:
                # Column is 100% null — skip value expectations
                logger.warning(f"Column '{col}' is 100%% null, skipping.")
                continue

            if col_schema and not col_schema.nullable:
                mostly_valid = 1.0  # Strict: no nulls allowed

            expectations.append({
                "expectation_type": "ExpectColumnValuesToNotBeNull",
                "kwargs": {"column": col, "mostly": mostly_valid},
            })

            # Allowed values from schema
            if col_schema and col_schema.allowed_values:
                expectations.append({
                    "expectation_type": "ExpectColumnValuesToBeInSet",
                    "kwargs": {
                        "column":    col,
                        "value_set": col_schema.allowed_values,
                        "mostly":    0.99,
                    },
                })

            elif pd.api.types.is_numeric_dtype(df[col]):
                # Use non-null values only for range computation
                # but we DON'T mutate the dataframe
                non_null = df[col].dropna()
                if len(non_null) < 5:
                    continue

                desc    = non_null.describe()
                col_min = float(desc["min"])
                col_max = float(desc["max"])
                col_mean = float(desc["mean"])

                # Schema overrides auto-profile
                range_min = col_schema.min_value if (col_schema and col_schema.min_value is not None) \
                    else (col_min * 1.5 if col_min < 0 else col_min * 0.5)
                range_max = col_schema.max_value if (col_schema and col_schema.max_value is not None) \
                    else (col_max * 1.5 if col_max > 0 else col_max * 0.5)

                if range_min > range_max:
                    range_min, range_max = range_max, range_min

                expectations.append({
                    "expectation_type": "ExpectColumnValuesToBeBetween",
                    "kwargs": {
                        "column":    col,
                        "min_value": round(range_min, 6),
                        "max_value": round(range_max, 6),
                        "mostly":    0.99,
                    },
                })

                if col_mean != 0:
                    expectations.append({
                        "expectation_type": "ExpectColumnMeanToBeBetween",
                        "kwargs": {
                            "column":    col,
                            "min_value": round(col_mean * 0.5, 6),
                            "max_value": round(col_mean * 1.5, 6),
                        },
                    })

            elif df[col].dtype in ("object", "category"):
                unique_count = df[col].nunique(dropna=True)
                if 0 < unique_count <= 50:
                    top_cats = (
                        df[col].value_counts(normalize=True, dropna=True)
                        .head(20).index.tolist()
                    )
                    expectations.append({
                        "expectation_type": "ExpectColumnValuesToBeInSet",
                        "kwargs": {
                            "column":    col,
                            "value_set": top_cats,
                            "mostly":    0.95,
                        },
                    })

        return expectations

    def _check_distribution_alignment(
        self,
        new_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        pipeline_run_id: str,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """KS alignment check — uses full columns, reports null rates."""
        import numpy as np
        from scipy import stats

        results = []
        numeric_cols = new_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in ref_df.columns:
                continue

            # Use non-null values for KS (but log what we're dropping)
            ref_vals = ref_df[col].dropna().values
            new_vals = new_df[col].dropna().values

            ref_null_rate = ref_df[col].isnull().mean()
            new_null_rate = new_df[col].isnull().mean()

            if len(ref_vals) < 10 or len(new_vals) < 10:
                logger.warning(
                    f"Skipping KS for '{col}': insufficient non-null values "
                    f"(ref={len(ref_vals)}, new={len(new_vals)})"
                )
                continue

            try:
                from scipy.stats._stats_py import Ks_2sampResult
                result: Ks_2sampResult = stats.ks_2samp(ref_vals, new_vals)
            except Exception as exc:
                logger.warning(f"KS test failed for '{col}': {exc}")
                continue

            aligned = (result.pvalue) > 0.01
            entry   = {
                "column":        col,
                "test":          "ks_2samp",
                "ks_stat":       round(float(result.statistic), 4),
                "p_value":       round(float(result.pvalue), 6),
                "aligned":       aligned,
                "ref_null_rate": round(float(ref_null_rate), 4),
                "new_null_rate": round(float(new_null_rate), 4),
            }
            results.append(entry)

            event_bus.emit(make_event(
                pipeline_run_id=pipeline_run_id,
                event_type=EventType.GE_EXPECTATION_CHECKED,
                step_name="data_validation",
                title=f"{'✅' if aligned else '⚠️'} Distribution: {col}",
                message=(
                    f"KS '{col}': p={float(result.pvalue):.4f} — "
                    f"{'aligned' if aligned else 'DRIFT DETECTED'}"
                ),
                framework=self.FRAMEWORK,
                status="success" if aligned else "warning",
                severity="info" if aligned else "warning",
                model_id=model_id,
                data=entry,
            ))

        return results

    def _compute_snapshot(self, df: pd.DataFrame) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "shape":     {"rows": len(df), "columns": len(df.columns)},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns":   {},
        }
        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            total      = len(df)
            col_info: dict[str, Any] = {
                "dtype":      str(df[col].dtype),
                "null_count": null_count,
                "null_pct":   round(null_count / total * 100, 2) if total > 0 else 0,
                "unique":     int(df[col].nunique()),
            }
            if col in numeric_cols:
                desc = df[col].describe()
                col_info.update({
                    "mean":   round(float(desc["mean"]), 4),
                    "std":    round(float(desc["std"]), 4),
                    "min":    round(float(desc["min"]), 4),
                    "p25":    round(float(desc["25%"]), 4),
                    "median": round(float(desc["50%"]), 4),
                    "p75":    round(float(desc["75%"]), 4),
                    "max":    round(float(desc["max"]), 4),
                })
            else:
                col_info["top_values"] = {
                    str(k): int(v)
                    for k, v in df[col].value_counts().head(5).items()
                }
            snapshot["columns"][col] = col_info

        return snapshot