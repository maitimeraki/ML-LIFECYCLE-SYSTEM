# src/data/transformers.py
"""
Custom sklearn-compatible transformers.
Each class:
  - Inherits BaseEstimator (get_params/set_params, repr)
  - Inherits TransformerMixin (fit_transform = fit + transform)
  - Implements fit(X, y=None) → self
  - Implements transform(X) → X
  - All statistics computed in fit(), stored as attributes ending in _
  - transform() uses ONLY fitted attributes — zero leakage
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger("ml_platform.data.transformers")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 1: Structural Cleaner
# ─────────────────────────────────────────────────────────────────────────────

class StructuralCleaner(BaseEstimator, TransformerMixin):
    """
    Removes structurally problematic columns and rows.

    fit():
      - Identifies high-null columns (> max_null_rate)
      - Identifies near-zero variance columns
      - Saves list as cols_to_drop_ (never recomputed in transform)

    transform():
      - Drops the SAME columns identified during fit
      - Removes duplicate rows
      - Does NOT reidentify columns from new data

    Parameters
    ----------
    max_null_rate : float
        Drop column if null fraction exceeds this. Default 0.70.
    min_variance : float
        Drop column if variance below this. Default 1e-8.
    duplicate_subset : list[str] | None
        Columns to consider for duplicate detection.
    """

    def __init__(
        self,
        max_null_rate: float = 0.70,
        min_variance: float = 1e-8
    ) -> None:
        self.max_null_rate    = max_null_rate
        self.min_variance     = min_variance

    def fit(self, X: pd.DataFrame, y: Any = None) -> "StructuralCleaner":
        """
        Identify columns to drop from TRAINING data.
        Stores result in self.cols_to_drop_ (sklearn convention: trailing _)
        """
        self._validate_dataframe(X)
        self.cols_to_drop_: list[str] = []
        self.input_features_: list[str] = list(X.columns)

        for col in X.columns:
            # High null rate
            null_rate = float(X[col].isnull().mean())
            if null_rate > self.max_null_rate:
                self.cols_to_drop_.append(col)
                logger.debug(
                    f"StructuralCleaner: '{col}' null_rate={null_rate:.2%} "
                    f"> {self.max_null_rate:.2%} → will drop"
                )
                continue

            # Near-zero variance (numeric only)
            if pd.api.types.is_numeric_dtype(X[col]):
                non_null = X[col].dropna().values
                if len(non_null) > 0:
                    variance = float(np.var(np.asarray(non_null)))
                    if variance < self.min_variance:
                        self.cols_to_drop_.append(col)
                        logger.debug(
                            f"StructuralCleaner: '{col}' "
                            f"variance={variance:.2e} → will drop"
                        )

        logger.info(
            f"StructuralCleaner fitted: "
            f"{len(self.cols_to_drop_)} columns marked for drop: "
            f"{self.cols_to_drop_}"
        )
        return self

    def transform(self, X):
        """Drop only columns identified during fit. NO row operations."""
        check_is_fitted(self, ["cols_to_drop_", "input_features_"])
        self._validate_dataframe(X)
        X = X.copy()

        # Drop columns identified at fit time
        existing_drop = [c for c in self.cols_to_drop_ if c in X.columns]
        if existing_drop:
            X = X.drop(columns=existing_drop)

        # Schema drift handling
        known_cols = set(self.input_features_) - set(self.cols_to_drop_)
        unknown = set(X.columns) - known_cols
        if unknown:
            X = X.drop(columns=list(unknown))

        for col in known_cols:
            if col not in X.columns:
                X[col] = np.nan

        return X  # No y handling needed, no row drops

    @staticmethod
    def _validate_dataframe(X: Any) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"StructuralCleaner requires pd.DataFrame, got {type(X)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 2: Type Fixer
# ─────────────────────────────────────────────────────────────────────────────

class TypeFixer(BaseEstimator, TransformerMixin):
    """
    Fixes semantically wrong values before sklearn imputers run.

    This transformer is STATELESS (no fit-time statistics needed):
      - Null-string replacement uses a fixed vocabulary
      - Boolean normalization uses a fixed vocabulary
      - Inf replacement is deterministic
      - min/max constraints come from config, not data

    Therefore fit() just stores column configs and returns self.
    transform() applies deterministic, config-driven corrections.

    Parameters
    ----------
    column_configs : dict
        Maps column name → ColumnProcessingConfig.
        Provides min_value, max_value, allowed_values, is_date_column etc.
    """

    NULL_STRINGS = frozenset({
        "", "na", "n/a", "nan", "null", "none", "nil",
        "?", "-", "--", ".", "unknown", "missing",
        "#n/a", "#null!", "#ref!", "#value!",
    })

    BOOL_TRUE  = frozenset({"true", "yes", "1", "t", "y", "on"})
    BOOL_FALSE = frozenset({"false", "no", "0", "f", "n", "off"})

    def __init__(
        self,
        column_configs: Optional[dict] = None,
    ) -> None:
        self.column_configs = column_configs or {}

    def fit(self, X: pd.DataFrame, y: Any) -> "TypeFixer":
        """
        Stateless — just validate and store column list.
        fit() returns self to satisfy sklearn contract.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"TypeFixer requires pd.DataFrame, got {type(X)}")
        self.feature_names_in_ = list(X.columns)
        logger.info(f"TypeFixer fitted: {len(self.feature_names_in_)} columns")
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply deterministic type corrections. No data statistics used."""
        check_is_fitted(self, ["feature_names_in_"])

        X = X.copy()

        for col in X.columns:
            cfg            = self.column_configs.get(col)
            original_dtype = str(X[col].dtype)

            # ── Null-like strings → NaN ────────────────────────────────────
            if X[col].dtype == object:
                null_mask = (
                    X[col].astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(self.NULL_STRINGS)
                )
                X.loc[null_mask, col] = np.nan

            # ── Boolean normalization ──────────────────────────────────────
            if (
                cfg
                and cfg.expected_dtype in ("bool", "boolean", "int64")
                and X[col].dtype == object
            ):
                s = X[col].astype(str).str.strip().str.lower()
                X[col] = np.where(
                    s.isin(self.BOOL_TRUE),  1,
                    np.where(
                        s.isin(self.BOOL_FALSE), 0,
                        np.nan,
                    ),
                )

            # ── Date → Unix timestamp ──────────────────────────────────────
            if cfg and cfg.is_date_column:
                try:
                    parsed  = pd.to_datetime(
                        X[col], format=cfg.date_format, errors="coerce"
                    )
                    X[col]  = parsed.astype("int64") // 10**9
                    X[col]  = X[col].replace(
                        -9223372036854775808 // 10**9, np.nan
                    )
                except Exception as exc:
                    logger.warning(f"TypeFixer date parse '{col}': {exc}")

            # ── Numeric coercion ───────────────────────────────────────────
            if (
                cfg
                and cfg.expected_dtype in ("float64", "float32", "int64")
                and X[col].dtype == object
            ):
                cleaned = (
                    X[col].astype(str)
                    .str.replace(r"[^\d.\-+eE]", "", regex=True)
                    .str.strip()
                )
                X[col] = pd.to_numeric(cleaned, errors="coerce")

            # ── Inf / -Inf → NaN ───────────────────────────────────────────
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)

            # ── Hard min/max constraints ───────────────────────────────────
            if pd.api.types.is_numeric_dtype(X[col]) and cfg:
                if cfg.min_value is not None:
                    bad = X[col] < cfg.min_value
                    if bad.any():
                        logger.debug(
                            f"TypeFixer '{col}': "
                            f"{int(bad.sum())} values < {cfg.min_value} → NaN"
                        )
                        X.loc[bad, col] = np.nan

                if cfg.max_value is not None:
                    bad = X[col] > cfg.max_value
                    if bad.any():
                        logger.debug(
                            f"TypeFixer '{col}': "
                            f"{int(bad.sum())} values > {cfg.max_value} → NaN"
                        )
                        X.loc[bad, col] = np.nan

            # ── Allowed values ─────────────────────────────────────────────
            if (
                cfg
                and cfg.allowed_values
                and X[col].dtype == object
            ):
                allowed  = {str(v) for v in cfg.allowed_values}
                bad_mask = ~X[col].astype(str).isin(allowed) & X[col].notna()
                if bad_mask.any():
                    logger.debug(
                        f"TypeFixer '{col}': "
                        f"{int(bad_mask.sum())} values not in allowed set → NaN"
                    )
                    X.loc[bad_mask, col] = np.nan
        logger.info(f"TypeFixer: transformed DataFrame with shape {X.shape} and {y.shape if y is not None else 'no target'}")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 3: Outlier Handler
# ─────────────────────────────────────────────────────────────────────────────

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Fits outlier bounds from training data and applies them consistently.

    fit():
      Computes bounds per column from training data.
      Stores in self.bounds_ dict — never recomputed.

    transform():
      Applies ONLY self.bounds_ — no new statistics computed.

    Parameters
    ----------
    strategy : str
        "clip"        → Winsorize to [lower_pct, upper_pct]
        "zscore_clip" → Clip beyond N standard deviations
        "iqr_clip"    → Clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        "log"         → log1p transform (shift if needed)
    lower_pct : float
        Lower percentile for clip strategy. Default 1.0.
    upper_pct : float
        Upper percentile for clip strategy. Default 99.0.
    zscore_threshold : float
        Number of std devs for zscore_clip. Default 3.5.
    columns : list[str] | None
        Columns to apply outlier treatment.
        If None: all numeric columns.
    column_strategies : dict[str, str] | None
        Per-column strategy override.
        e.g. {"income": "log", "age": "clip"}
    """

    def __init__(
        self,
        strategy: str = "clip",
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
        zscore_threshold: float = 3.5,
        columns: Optional[list[str]] = None,
        column_strategies: Optional[dict[str, str]] = None,
    ) -> None:
        self.strategy          = strategy
        self.lower_pct         = lower_pct
        self.upper_pct         = upper_pct
        self.zscore_threshold  = zscore_threshold
        self.columns           = columns
        self.column_strategies = column_strategies or {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OutlierHandler":
        """
        Compute bounds from TRAINING data.
        All bounds saved in self.bounds_ — frozen after fit.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"OutlierHandler requires DataFrame, got {type(X)}")

        numeric_cols = (
            self.columns
            or X.select_dtypes(include=[np.number]).columns.tolist()
        )

        # self.bounds_ stores fitted parameters per column
        # Key convention: trailing underscore = fitted attribute (sklearn)
        self.bounds_: dict[str, dict[str, Any]] = {}
        self.log_shifts_: dict[str, float] = {}
        self.feature_names_in_: list[str]  = list(X.columns)

        for col in numeric_cols:
            if col not in X.columns:
                continue

            non_null = X[col].dropna().values
            if len(non_null) < 10:
                continue

            col_strategy = self.column_strategies.get(col, self.strategy)

            if col_strategy == "clip":
                self.bounds_[col] = {
                    "strategy": "clip",
                    "lower":    float(np.percentile(np.asarray(non_null), self.lower_pct)),
                    "upper":    float(np.percentile(np.asarray(non_null), self.upper_pct)),
                }

            elif col_strategy == "zscore_clip":
                mean = float(np.mean(np.asarray(non_null)))
                std  = float(np.std(np.asarray(non_null)))
                std  = std if std > 1e-10 else 1.0
                self.bounds_[col] = {
                    "strategy": "zscore_clip",
                    "lower":    mean - self.zscore_threshold * std,
                    "upper":    mean + self.zscore_threshold * std,
                    "mean":     mean,
                    "std":      std,
                }

            elif col_strategy == "iqr_clip":
                q1  = float(np.percentile(np.asarray(non_null), 25))
                q3  = float(np.percentile(np.asarray(non_null), 75))
                iqr = q3 - q1
                self.bounds_[col] = {
                    "strategy": "iqr_clip",
                    "lower":    q1 - 1.5 * iqr,
                    "upper":    q3 + 1.5 * iqr,
                    "q1":       q1,
                    "q3":       q3,
                    "iqr":      iqr,
                }

            elif col_strategy == "log":
                shift = 0.0
                if float(np.min(np.asarray(non_null))) <= 0:
                    shift = abs(float(np.min((np.asarray(non_null))))) + 1.0
                self.log_shifts_[col] = shift
                self.bounds_[col] = {
                    "strategy": "log",
                    "shift":    shift,
                }

        logger.info(
            f"OutlierHandler fitted: "
            f"{len(self.bounds_)} columns with bounds"
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Apply ONLY self.bounds_ — no statistics from new data.
        """
        check_is_fitted(self, ["bounds_", "feature_names_in_"])

        X = X.copy()

        for col, bounds in self.bounds_.items():
            if col not in X.columns:
                continue

            strategy = bounds["strategy"]
            values   = X[col].values.copy().astype(float)
            not_null = ~np.isnan(values)

            if not not_null.any():
                continue

            if strategy in ("clip", "zscore_clip", "iqr_clip"):
                lb = bounds["lower"]
                ub = bounds["upper"]
                values[not_null] = np.clip(values[not_null], lb, ub)

            elif strategy == "log":
                shift = bounds["shift"]
                values[not_null] = np.log1p(values[not_null] + shift)

            X[col] = values
        logger.info(f"OutlierHandler: transformed DataFrame with shape {X.shape}")
        return X

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_"])
        return np.array(self.feature_names_in_)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 4: Frequency Encoder
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns with their training-set frequency.

    fit():
      Computes value_counts(normalize=True) from TRAINING data.
      Stores in self.frequency_maps_ — never recomputed.

    transform():
      Maps categories using self.frequency_maps_.
      Unknown categories → 0.0 (not crash, not data-dependent value).

    Parameters
    ----------
    columns : list[str] | None
        Columns to encode. If None: auto-detect object/category columns.
    handle_unknown : str
        "zero"   → unknown → 0.0  (default, safe)
        "mean"   → unknown → mean frequency of training categories
    """

    def __init__(
        self,
        columns: Optional[list[str]] = None,
        handle_unknown: str = "zero",
    ) -> None:
        self.columns        = columns
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        """
        Compute frequency maps from training data.
        All maps saved in self.frequency_maps_ — frozen after fit.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"FrequencyEncoder requires DataFrame, got {type(X)}")

        cols = (
            self.columns
            or X.select_dtypes(include=["object", "category"]).columns.tolist()
        )

        self.frequency_maps_: dict[str, dict] = {}
        self.fallback_values_: dict[str, float] = {}
        self.feature_names_in_: list[str] = list(X.columns)
        self.encoded_columns_: list[str]  = []

        for col in cols:
            if col not in X.columns:
                continue

            freq_map = (
                X[col]
                .value_counts(normalize=True, dropna=True)
                .to_dict()
            )
            self.frequency_maps_[col] = freq_map
            self.encoded_columns_.append(col)

            # Fallback for unknown categories
            if self.handle_unknown == "mean" and freq_map:
                self.fallback_values_[col] = float(
                    np.mean(list(freq_map.values()))
                )
            else:
                self.fallback_values_[col] = 0.0

            logger.debug(
                f"FrequencyEncoder '{col}': "
                f"{len(freq_map)} categories fitted"
            )

        logger.info(
            f"FrequencyEncoder fitted: {len(self.encoded_columns_)} columns"
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Apply ONLY self.frequency_maps_ — no statistics from new data.
        """
        check_is_fitted(self, ["frequency_maps_", "feature_names_in_"])

        X = X.copy()

        for col, freq_map in self.frequency_maps_.items():
            if col not in X.columns:
                continue

            fallback = self.fallback_values_.get(col, 0.0)

            # Map using training frequencies
            # Unknown categories → fallback (never crashes)
            X[col] = (
                X[col]
                .astype(str)
                .map(freq_map)
                .fillna(fallback)
                .astype(np.float32)
            )
        logger.info(f"FrequencyEncoder: transformed DataFrame with shape {X.shape}")
        return X

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_"])
        return np.array(self.feature_names_in_)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 5: Leakage Guard
# ─────────────────────────────────────────────────────────────────────────────

class LeakageGuard(BaseEstimator, TransformerMixin):
    """
    Post-processing safety check.

    fit():
      Identifies columns that became constant after all transformations.
      Detects high correlation with target (leakage risk).
      Saves self.cols_to_drop_ — never recomputed.

    transform():
      Drops the SAME constant columns identified at fit.
      Drops remaining non-numeric columns (safety net).

    Parameters
    ----------
    target_column : str | None
        Target column name (excluded from checks, never dropped).
    correlation_threshold : float
        Warn if |corr(feature, target)| > threshold. Default 0.99.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        correlation_threshold: float = 0.99,
    ) -> None:
        self.target_column          = target_column
        self.correlation_threshold  = correlation_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LeakageGuard":
        """
        Identify constant columns and leakage risks from training data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"LeakageGuard requires DataFrame, got {type(X)}")

        self.cols_to_drop_: list[str]    = []
        self.leakage_warnings_: list[str] = []
        self.feature_names_in_: list[str] = list(X.columns)

        for col in X.columns:
            if col == self.target_column:
                continue

            # Constant column detection
            if pd.api.types.is_numeric_dtype(X[col]):
                n_unique = X[col].nunique(dropna=False)
                if n_unique <= 1:
                    self.cols_to_drop_.append(col)
                    logger.warning(
                        f"LeakageGuard: '{col}' is constant → will drop"
                    )
                    continue

        # Correlation with target
        if y is not None:
            y_series = (
                y if isinstance(y, pd.Series)
                else pd.Series(y, index=X.index)
            )
            num_cols = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c not in self.cols_to_drop_
                and c != self.target_column
            ]
            for col in num_cols:
                try:
                    corr = abs(float(X[col].corr(y_series)))
                    if corr > self.correlation_threshold:
                        warning = (
                            f"⚠️ LEAKAGE RISK: '{col}' "
                            f"corr={corr:.4f} with target — review this feature"
                        )
                        self.leakage_warnings_.append(warning)
                        logger.warning(warning)
                except Exception:
                    pass

        logger.info(
            f"LeakageGuard fitted: "
            f"{len(self.cols_to_drop_)} constant cols to drop, "
            f"{len(self.leakage_warnings_)} leakage warnings"
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Drop constant columns and non-numeric stragglers."""
        check_is_fitted(self, ["cols_to_drop_", "feature_names_in_"])

        X = X.copy()

        # Drop fitted constant columns
        existing = [c for c in self.cols_to_drop_ if c in X.columns]
        if existing:
            X = X.drop(columns=existing)

        # Safety net: drop any remaining non-numeric columns
        # (should have been encoded, but defensive programming)
        obj_cols = [
            c for c in X.select_dtypes(
                include=["object", "category"]
            ).columns
            if c != self.target_column
        ]
        if obj_cols:
            logger.warning(
                f"LeakageGuard: dropping non-numeric stragglers: {obj_cols}"
            )
            X = X.drop(columns=obj_cols)
        logger.info(f"LeakageGuard: transformed DataFrame with shape {X.shape}")
        return X

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_"])
        keep = [
            c for c in self.feature_names_in_
            if c not in self.cols_to_drop_
        ]
        return np.array(keep)