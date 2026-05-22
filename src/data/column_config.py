# src/data/auto_config.py
"""
Automatic ProcessingConfig builder.
Introspects a DataFrame and generates config automatically.
No manual column definitions required.

Strategy selection rules:
─────────────────────────
Numeric columns:
  null_rate > 20%     → KNN imputation
  null_rate <= 20%    → MEDIAN imputation
  right-skewed (>2)   → LOG outlier + LOG scaling
  normal-ish          → CLIP outlier + ROBUST scaling

Categorical columns:
  n_unique <= 10      → ONE-HOT encoding
  n_unique <= 50      → FREQUENCY encoding
  n_unique > 50       → FREQUENCY encoding (OHE too wide)

Boolean-like columns:
  only 2 unique vals  → ORDINAL encoding

Target column:
  always excluded from all transformations
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.data.processing_config import (
    ColumnProcessingConfig,
    EncodingStrategy,
    ImputationStrategy,
    OutlierStrategy,
    ProcessingConfig,
    ScalingStrategy,
)

logger = logging.getLogger("ml_platform.data.auto_config")


class AutoConfigBuilder:
    """
    Builds ProcessingConfig automatically from DataFrame inspection.

    Parameters
    ----------
    target_column : str | None
        Target column — excluded from all transforms.
    id_columns : list[str]
        ID columns to drop.
    max_null_rate_to_drop : float
        Drop column if null rate exceeds this.
    onehot_threshold : int
        Use OHE if n_unique <= this, else FREQUENCY.
    skewness_threshold : float
        Use LOG transform if |skewness| > this.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        id_columns: Optional[list[str]] = None,
        max_null_rate_to_drop: float = 0.70,
        onehot_threshold: int = 10,
        skewness_threshold: float = 2.0,
    ) -> None:
        self.target_column        = target_column
        self.id_columns           = id_columns or []
        self.max_null_rate_to_drop = max_null_rate_to_drop
        self.onehot_threshold     = onehot_threshold
        self.skewness_threshold   = skewness_threshold

    def build(self, df: pd.DataFrame) -> ProcessingConfig:
        """
        Inspect df and return a fully populated ProcessingConfig.
        Called automatically — user does not write any config.
        """
        column_configs: dict[str, ColumnProcessingConfig] = {}

        for col in df.columns:
            if col == self.target_column:
                continue
            if col in self.id_columns:
                continue

            col_cfg = self._build_column_config(col, df[col])
            column_configs[col] = col_cfg

            logger.debug(
                f"AutoConfig '{col}': "
                f"dtype={df[col].dtype}, "
                f"impute={col_cfg.imputation_strategy.value}, "
                f"outlier={col_cfg.outlier_strategy.value}, "
                f"scale={col_cfg.scaling_strategy.value}, "
                f"encode={col_cfg.encoding_strategy.value}"
            )

        config = ProcessingConfig(
            target_column=self.target_column,
            id_columns=self.id_columns,
            max_null_rate_to_drop_column=self.max_null_rate_to_drop,
            columns=column_configs,
        )

        logger.info(
            f"AutoConfig built: {len(column_configs)} columns configured"
        )
        return config

    def _build_column_config(
        self,
        col: str,
        series: pd.Series,
    ) -> ColumnProcessingConfig:
        """Determine best strategy for one column."""
        null_rate  = float(series.isnull().mean())
        n_unique   = int(series.nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_cat     = series.dtype in ("object", "category")
        n_total    = len(series)

        cfg = ColumnProcessingConfig(name=col)

        # ── ID-like detection ──────────────────────────────────────────────
        if is_cat and n_unique == n_total and n_total > 100:
            cfg.is_id_column = True
            return cfg

        # ── Numeric column strategies ──────────────────────────────────────
        if is_numeric:
            cfg.expected_dtype = "float64"
            cfg.encoding_strategy = EncodingStrategy.NONE

            # Imputation
            if null_rate > 0.20:
                cfg.imputation_strategy = ImputationStrategy.KNN
            else:
                cfg.imputation_strategy = ImputationStrategy.MEDIAN

            # Outlier + Scaling based on distribution shape
            non_null = series.dropna()
            if len(non_null) >= 8:
                skewness = float(scipy_stats.skew(non_null.values))

                if abs(skewness) > self.skewness_threshold and non_null.min() >= 0:
                    # Right-skewed positive data (income, counts, prices)
                    cfg.outlier_strategy = OutlierStrategy.LOG_TRANSFORM
                    cfg.scaling_strategy = ScalingStrategy.STANDARD
                else:
                    cfg.outlier_strategy = OutlierStrategy.CLIP
                    cfg.scaling_strategy = ScalingStrategy.ROBUST
            else:
                cfg.outlier_strategy = OutlierStrategy.CLIP
                cfg.scaling_strategy = ScalingStrategy.ROBUST

            # Min value inference
            non_null_min = float(non_null.min()) if len(non_null) > 0 else 0.0
            if non_null_min >= 0:
                cfg.min_value = 0.0

        # ── Categorical column strategies ──────────────────────────────────
        elif is_cat:
            cfg.expected_dtype    = "object"
            cfg.outlier_strategy  = OutlierStrategy.NONE
            cfg.scaling_strategy  = ScalingStrategy.NONE

            # Imputation
            cfg.imputation_strategy = ImputationStrategy.MODE

            # Boolean-like (only 2 unique values)
            if n_unique <= 2:
                cfg.encoding_strategy = EncodingStrategy.ORDINAL

            # Low cardinality → OHE
            elif n_unique <= self.onehot_threshold:
                cfg.encoding_strategy = EncodingStrategy.ONEHOT
                cfg.onehot_max_cardinality = self.onehot_threshold

            # Medium/high cardinality → FREQUENCY
            else:
                cfg.encoding_strategy = EncodingStrategy.FREQUENCY

        # ── Boolean columns ────────────────────────────────────────────────
        elif series.dtype == bool:
            cfg.expected_dtype    = "int64"
            cfg.outlier_strategy  = OutlierStrategy.NONE
            cfg.scaling_strategy  = ScalingStrategy.NONE
            cfg.encoding_strategy = EncodingStrategy.NONE
            cfg.imputation_strategy = ImputationStrategy.CONSTANT
            cfg.imputation_constant = 0

        return cfg