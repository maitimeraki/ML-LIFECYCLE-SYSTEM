# src/data/pipeline_builder.py
"""
Builds the sklearn Pipeline + ColumnTransformer.

ColumnTransformer is the core:
  It routes each column to the correct transformer
  based on dtype and config.

Structure:
  Pipeline([
    ("structural", StructuralCleaner()),
    ("type_fix",   TypeFixer()),
    ("outlier",    OutlierHandler()),
    ("column_transform", ColumnTransformer([
        ("numeric",     numeric_pipeline,     numeric_cols),
        ("categorical", categorical_pipeline, cat_cols),
        ("ohe",         ohe_pipeline,         ohe_cols),
    ])),
    ("leakage",    LeakageGuard()),
  ])
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
    FunctionTransformer,
)

try:
    from sklearn.preprocessing import TargetEncoder
    _HAS_TARGET_ENCODER = True
except ImportError:
    _HAS_TARGET_ENCODER = False

from src.data.processing_config import (
    EncodingStrategy,
    ImputationStrategy,
    OutlierStrategy,
    ProcessingConfig,
    ScalingStrategy,
)
from src.data.transformers import (
    FrequencyEncoder,
    LeakageGuard,
    OutlierHandler,
    StructuralCleaner,
    TypeFixer,
)

logger = logging.getLogger("ml_platform.data.pipeline_builder")


class ProcessingPipelineBuilder:
    """
    Builds a complete sklearn Pipeline from ProcessingConfig.

    Usage:
        builder  = ProcessingPipelineBuilder(config)
        pipeline = builder.build(df)   # inspects df to determine column types
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config

    def build(self, df: pd.DataFrame) -> Pipeline:
        """
        Inspect df column types and build the full Pipeline.

        Steps:
        1. StructuralCleaner    — drops bad columns/rows
        2. TypeFixer            — fixes value-level issues
        3. OutlierHandler       — clips/transforms numeric outliers
        4. ColumnTransformer    — impute + scale + encode per column type
        5. LeakageGuard         — final safety check

        Returns a sklearn Pipeline ready for fit/transform.
        """
        target = self.config.target_column
        df_features = df.drop(
            columns=[target] if target and target in df.columns else []
        )

        # Categorize columns
        (
            numeric_impute_scale_cols,
            numeric_knn_cols,
            ordinal_cols,
            ohe_cols,
            freq_cols,
            target_enc_cols,
            passthrough_cols,
        ) = self._categorize_columns(df_features)

        logger.info(
            f"Pipeline builder column routing:\n"
            f"  numeric (median+robust):   {numeric_impute_scale_cols}\n"
            f"  numeric (knn):             {numeric_knn_cols}\n"
            f"  ordinal:                   {ordinal_cols}\n"
            f"  one-hot:                   {ohe_cols}\n"
            f"  frequency:                 {freq_cols}\n"
            f"  target-encode:             {target_enc_cols}\n"
            f"  passthrough:               {passthrough_cols}"
        )

        # Build sub-pipelines
        transformers = []

        # ── Numeric: standard impute + scale ──────────────────────────────
        if numeric_impute_scale_cols:
            transformers.append((
                "numeric",
                self._build_numeric_pipeline(numeric_impute_scale_cols),
                numeric_impute_scale_cols,
            ))

        # ── Numeric: KNN impute + scale ────────────────────────────────────
        if numeric_knn_cols:
            transformers.append((
                "numeric_knn",
                self._build_knn_pipeline(numeric_knn_cols),
                numeric_knn_cols,
            ))

        # ── Ordinal encoding ───────────────────────────────────────────────
        if ordinal_cols:
            transformers.append((
                "ordinal",
                self._build_ordinal_pipeline(ordinal_cols),
                ordinal_cols,
            ))

        # ── One-hot encoding ───────────────────────────────────────────────
        if ohe_cols:
            transformers.append((
                "onehot",
                self._build_ohe_pipeline(ohe_cols),
                ohe_cols,
            ))

        # ── Frequency encoding ─────────────────────────────────────────────
        if freq_cols:
            transformers.append((
                "frequency",
                self._build_frequency_pipeline(freq_cols),
                freq_cols,
            ))

        # ── Target encoding ────────────────────────────────────────────────
        if target_enc_cols and _HAS_TARGET_ENCODER:
            transformers.append((
                "target_encode",
                self._build_target_encode_pipeline(target_enc_cols),
                target_enc_cols,
            ))
        elif target_enc_cols:
            # Fallback if TargetEncoder not available
            logger.warning(
                "TargetEncoder requires sklearn>=1.3. "
                "Falling back to FrequencyEncoder for: "
                f"{target_enc_cols}"
            )
            transformers.append((
                "target_fallback",
                self._build_frequency_pipeline(target_enc_cols),
                target_enc_cols,
            ))

        # ── Passthrough ────────────────────────────────────────────────────
        if passthrough_cols:
            transformers.append((
                "passthrough",
                "passthrough",
                passthrough_cols,
            ))

        # Build ColumnTransformer
        # remainder="drop" → any column not explicitly listed is dropped
        # This is safe: we've explicitly categorized all known columns
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,   # e.g. "numeric__feature_1"
            n_jobs=-1,
        )

        # Build OutlierHandler column strategies
        outlier_col_strategies = {}
        for col in df_features.columns:
            cfg      = self.config.get_column_config(col, df_features[col])
            strategy = cfg.outlier_strategy.value
            if strategy != "none" and pd.api.types.is_numeric_dtype(
                df_features[col]
            ):
                outlier_col_strategies[col] = (
                    strategy if strategy != "iqr_remove" else "iqr_clip"
                )

        # Full Pipeline
        pipeline = Pipeline(steps=[
            (
                "structural",
                StructuralCleaner(
                    max_null_rate=self.config.max_null_rate_to_drop_column,
                    min_variance=self.config.min_variance_threshold,
                    duplicate_subset=self.config.duplicate_subset,
                ),
            ),
            (
                "type_fix",
                TypeFixer(column_configs=self.config.columns),
            ),
            (
                "outlier",
                OutlierHandler(
                    strategy="clip",
                    lower_pct=1.0,
                    upper_pct=99.0,
                    column_strategies=outlier_col_strategies,
                ),
            ),
            (
                "column_transform",
                column_transformer,
            ),
            (
                "leakage",
                LeakageGuard(
                    target_column=target,
                    correlation_threshold=0.99,
                ),
            ),
        ])

        return pipeline

    # ─────────────────────────────────────────────────────────────────────
    # Sub-pipeline builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_numeric_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        Numeric pipeline:
          SimpleImputer(median) → RobustScaler

        RobustScaler is default for production:
          - Uses median + IQR (not mean + std)
          - Resistant to outliers that survived clipping
          - Appropriate for financial, age, count data
        """
        # Determine scaling strategy from config
        # Use first column's config as representative
        if columns:
            first_cfg  = self.config.get_column_config(columns[0])
            sk_strategy = first_cfg.scaling_strategy
        else:
            sk_strategy = ScalingStrategy.ROBUST

        scaler = self._get_scaler(sk_strategy)

        # Determine imputation strategy
        if columns:
            imp_strategy = first_cfg.imputation_strategy
            sk_imp, fill = self._get_imputer_params(imp_strategy, first_cfg)
        else:
            sk_imp, fill = "median", None

        steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy=sk_imp,
                    fill_value=fill,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", scaler),
        ]

        return Pipeline(steps=steps)

    def _build_knn_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        KNN imputation pipeline:
          KNNImputer → RobustScaler

        KNNImputer.fit() stores training data internally.
        KNNImputer.transform() finds neighbors in TRAINING space.
        Zero leakage at transform time.
        """
        return Pipeline(steps=[
            (
                "knn_imputer",
                KNNImputer(
                    n_neighbors=5,
                    weights="distance",
                    metric="nan_euclidean",
                ),
            ),
            ("scaler", RobustScaler()),
        ])

    def _build_ordinal_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        Ordinal encoding pipeline:
          SimpleImputer(most_frequent) → OrdinalEncoder

        OrdinalEncoder:
          fit  → learns category-to-integer mapping from training data
          transform → applies SAME mapping
          handle_unknown="use_encoded_value" → unseen → -1 (not crash)
        """
        # Build per-column categories if specified
        all_categories = []
        for col in columns:
            cfg = self.config.get_column_config(col)
            if cfg.ordinal_categories:
                all_categories.append(cfg.ordinal_categories)
            else:
                all_categories.append("auto")

        return Pipeline(steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=all_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
            ),
        ])

    def _build_ohe_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        One-hot encoding pipeline:
          SimpleImputer(most_frequent) → OneHotEncoder

        OneHotEncoder:
          fit  → learns vocabulary from training data
          transform → same columns always, regardless of new data
          handle_unknown="ignore" → unseen categories → all zeros
          sparse_output=False → return dense array
        """
        return Pipeline(steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                    drop=None,
                ),
            ),
        ])

    def _build_frequency_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        Frequency encoding pipeline:
          SimpleImputer(most_frequent) → FrequencyEncoder

        FrequencyEncoder:
          fit  → computes value_counts from training data
          transform → applies SAME frequency map
          Unknown → 0.0
        """
        return Pipeline(steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                FrequencyEncoder(
                    columns=None,    # apply to all passed columns
                    handle_unknown="zero",
                ),
            ),
        ])

    def _build_target_encode_pipeline(
        self, columns: list[str]
    ) -> Pipeline:
        """
        Target encoding pipeline (sklearn 1.3+):
          SimpleImputer → TargetEncoder

        TargetEncoder:
          fit  → learns category→target_mean mapping WITH cross-validation
                 cv=5 prevents target leakage within training set
          transform → applies SAME mapping
          Unknown → global mean (from training data)
        """
        return Pipeline(steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                TargetEncoder(
                    target_type="auto",
                    smooth="auto",
                    cv=5,
                ),
            ),
        ])

    # ─────────────────────────────────────────────────────────────────────
    # Column categorization
    # ─────────────────────────────────────────────────────────────────────

    def _categorize_columns(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        list[str], list[str], list[str],
        list[str], list[str], list[str], list[str]
    ]:
        """
        Route each column to the appropriate sub-pipeline.

        Returns:
            numeric_impute_scale_cols  → SimpleImputer + RobustScaler
            numeric_knn_cols           → KNNImputer + RobustScaler
            ordinal_cols               → OrdinalEncoder
            ohe_cols                   → OneHotEncoder
            freq_cols                  → FrequencyEncoder
            target_enc_cols            → TargetEncoder
            passthrough_cols           → no transformation
        """
        numeric_impute_scale: list[str] = []
        numeric_knn:          list[str] = []
        ordinal:              list[str] = []
        ohe:                  list[str] = []
        freq:                 list[str] = []
        target_enc:           list[str] = []
        passthrough:          list[str] = []

        for col in df.columns:
            cfg = self.config.get_column_config(col, df[col])

            # Skip id columns
            if cfg.is_id_column or col in self.config.id_columns:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_cat     = df[col].dtype in ("object", "category")
            n_unique   = df[col].nunique(dropna=True)

            if is_numeric:
                if cfg.imputation_strategy == ImputationStrategy.KNN:
                    numeric_knn.append(col)
                else:
                    numeric_impute_scale.append(col)

            elif is_cat:
                enc_strategy = cfg.encoding_strategy

                if enc_strategy == EncodingStrategy.ORDINAL:
                    ordinal.append(col)

                elif enc_strategy == EncodingStrategy.ONEHOT:
                    if n_unique <= cfg.onehot_max_cardinality:
                        ohe.append(col)
                    else:
                        # High cardinality → frequency
                        logger.warning(
                            f"'{col}': {n_unique} unique > "
                            f"{cfg.onehot_max_cardinality} max for OHE. "
                            f"→ FrequencyEncoder"
                        )
                        freq.append(col)

                elif enc_strategy == EncodingStrategy.TARGET:
                    target_enc.append(col)

                elif enc_strategy == EncodingStrategy.FREQUENCY:
                    freq.append(col)

                elif enc_strategy == EncodingStrategy.NONE:
                    passthrough.append(col)

                else:
                    # Default for categorical: frequency
                    freq.append(col)

        return (
            numeric_impute_scale,
            numeric_knn,
            ordinal,
            ohe,
            freq,
            target_enc,
            passthrough,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_scaler(strategy: ScalingStrategy) -> Any:
        if strategy == ScalingStrategy.STANDARD:
            return StandardScaler()
        elif strategy == ScalingStrategy.MINMAX:
            return MinMaxScaler(feature_range=(0, 1))
        elif strategy == ScalingStrategy.ROBUST:
            return RobustScaler(quantile_range=(25.0, 75.0))
        elif strategy == ScalingStrategy.LOG:
            return FunctionTransformer(
                func=np.log1p,
                inverse_func=np.expm1,
                validate=False,
            )
        else:
            return "passthrough"

    @staticmethod
    def _get_imputer_params(
        strategy: ImputationStrategy,
        cfg: Any,
    ) -> tuple[str, Any]:
        mapping = {
            ImputationStrategy.MEAN:     ("mean",           None),
            ImputationStrategy.MEDIAN:   ("median",         None),
            ImputationStrategy.MODE:     ("most_frequent",  None),
            ImputationStrategy.CONSTANT: ("constant",       cfg.imputation_constant or 0),
        }
        return mapping.get(strategy, ("median", None))