# src/data/processing_config.py
"""
Per-column and global processing configuration.
No changes needed here — config is just data classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ImputationStrategy(str, Enum):
    MEAN         = "mean"
    MEDIAN       = "median"
    MODE         = "mode"
    CONSTANT     = "constant"
    KNN          = "knn"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    DROP_ROW     = "drop_row"
    NONE         = "none"


class OutlierStrategy(str, Enum):
    CLIP         = "clip"
    ZSCORE_CLIP  = "zscore_clip"
    IQR_REMOVE   = "iqr_remove"
    LOG_TRANSFORM = "log_transform"
    NONE         = "none"


class ScalingStrategy(str, Enum):
    STANDARD = "standard"    # StandardScaler
    MINMAX   = "minmax"      # MinMaxScaler
    ROBUST   = "robust"      # RobustScaler  ← recommended for production
    LOG      = "log"         # FunctionTransformer(log1p)
    NONE     = "none"


class EncodingStrategy(str, Enum):
    ORDINAL   = "ordinal"    # OrdinalEncoder
    ONEHOT    = "onehot"     # OneHotEncoder
    FREQUENCY = "frequency"  # Custom (no sklearn equivalent)
    TARGET    = "target"     # TargetEncoder (sklearn 1.3+)
    NONE      = "none"


@dataclass
class ColumnProcessingConfig:
    name: str
    expected_dtype: str = "float64"
    nullable: bool = True

    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    imputation_constant: Any = None

    outlier_strategy: OutlierStrategy = OutlierStrategy.CLIP
    outlier_lower_pct: float = 1.0
    outlier_upper_pct: float = 99.0
    outlier_zscore_threshold: float = 3.5

    scaling_strategy: ScalingStrategy = ScalingStrategy.ROBUST

    encoding_strategy: EncodingStrategy = EncodingStrategy.NONE
    onehot_max_cardinality: int = 20
    ordinal_categories: Optional[list] = None

    is_target: bool = False
    is_id_column: bool = False
    is_date_column: bool = False
    date_format: Optional[str] = None
    allowed_values: Optional[list] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    rename_to: Optional[str] = None


@dataclass
class ProcessingConfig:
    target_column: Optional[str] = None
    id_columns: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)

    default_numeric_imputation: ImputationStrategy = ImputationStrategy.MEDIAN
    default_categorical_imputation: ImputationStrategy = ImputationStrategy.MODE
    default_outlier_strategy: OutlierStrategy = OutlierStrategy.CLIP
    default_scaling: ScalingStrategy = ScalingStrategy.ROBUST
    default_encoding: EncodingStrategy = EncodingStrategy.FREQUENCY

    max_null_rate_to_drop_column: float = 0.70
    max_cardinality_for_onehot: int = 20
    min_variance_threshold: float = 1e-8
    duplicate_subset: Optional[list[str]] = None

    columns: dict[str, ColumnProcessingConfig] = field(default_factory=dict)

    def get_column_config(
        self,
        col_name: str,
        df_col: Any = None,
    ) -> ColumnProcessingConfig:
        if col_name in self.columns:
            return self.columns[col_name]

        cfg = ColumnProcessingConfig(name=col_name)

        if df_col is not None:
            import pandas as pd
            if pd.api.types.is_numeric_dtype(df_col):
                cfg.imputation_strategy = self.default_numeric_imputation
                cfg.outlier_strategy    = self.default_outlier_strategy
                cfg.scaling_strategy    = self.default_scaling
                cfg.encoding_strategy   = EncodingStrategy.NONE
            else:
                cfg.imputation_strategy = self.default_categorical_imputation
                cfg.outlier_strategy    = OutlierStrategy.NONE
                cfg.scaling_strategy    = ScalingStrategy.NONE
                cfg.encoding_strategy   = self.default_encoding

        return cfg