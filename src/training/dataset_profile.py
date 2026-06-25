"""
Dataset classification taxonomy and enriched profile.

Provides the DatasetCategory enum that classifies datasets into
9 distinct categories based on problem type, class balance ratio,
and dataset size. The enriched DatasetProfile carries this classification
through the pipeline so downstream components (strategy factory,
search space builder, trainer) can make informed decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_platform.training.dataset_profile")


class DatasetCategory(str, Enum):
    """Multi-tier dataset classification based on balance ratio and problem type.

    Balance ratio = minority_class_count / majority_class_count.
    Thresholds:
        EXTREME_IMBALANCE    < 0.01  (< 1% minority — e.g. credit card fraud)
        SEVERE_IMBALANCE     < 0.05  (1-5% minority — e.g. medical diagnosis)
        MODERATE_IMBALANCE   < 0.20  (5-20% minority — e.g. email spam)
        MILD_IMBALANCE       < 0.45  (20-45% minority — e.g. customer churn)
        BALANCED_BINARY      >= 0.45  binary with both classes well-represented
        MULTICLASS_BALANCED  >= 0.45  multiclass where every class >= 45% of majority
        MULTICLASS_IMBALANCED < 0.45 multiclass with at least one small class
        REGRESSION_SMALL     N/A     regression with < 5 000 samples
        REGRESSION_MEDIUM    N/A     regression with 5 000-50 000 samples
        REGRESSION_LARGE     N/A     regression with > 50 000 samples
    """

    # Classification categories (ordered by imbalance severity)
    EXTREME_IMBALANCE = "extreme_imbalance"
    SEVERE_IMBALANCE = "severe_imbalance"
    MODERATE_IMBALANCE = "moderate_imbalance"
    MILD_IMBALANCE = "mild_imbalance"
    BALANCED_BINARY = "balanced_binary"
    MULTICLASS_BALANCED = "multiclass_balanced"
    MULTICLASS_IMBALANCED = "multiclass_imbalanced"

    # Regression size buckets
    REGRESSION_SMALL = "regression_small"
    REGRESSION_MEDIUM = "regression_medium"
    REGRESSION_LARGE = "regression_large"

    @property
    def is_classification(self) -> bool:
        return self not in {
            DatasetCategory.REGRESSION_SMALL,
            DatasetCategory.REGRESSION_MEDIUM,
            DatasetCategory.REGRESSION_LARGE,
        }

    @property
    def is_regression(self) -> bool:
        return not self.is_classification

    @property
    def imbalance_severity(self) -> int:
        """Numeric severity: 0=balanced, 4=extreme. Regression = -1."""
        mapping = {
            DatasetCategory.EXTREME_IMBALANCE: 4,
            DatasetCategory.SEVERE_IMBALANCE: 3,
            DatasetCategory.MODERATE_IMBALANCE: 2,
            DatasetCategory.MILD_IMBALANCE: 1,
            DatasetCategory.BALANCED_BINARY: 0,
            DatasetCategory.MULTICLASS_BALANCED: 0,
            DatasetCategory.MULTICLASS_IMBALANCED: 2,
        }
        return mapping.get(self, -1)


class RegressionSize(str, Enum):
    """Size bucket for regression datasets."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# -- Threshold constants (single source of truth) -----------------------------

EXTREME_IMBALANCE_THRESHOLD = 0.01   # < 1%
SEVERE_IMBALANCE_THRESHOLD = 0.05    # < 5%
MODERATE_IMBALANCE_THRESHOLD = 0.20  # < 20%
MILD_IMBALANCE_THRESHOLD = 0.45      # < 45%

REGRESSION_SMALL_THRESHOLD = 5_000
REGRESSION_LARGE_THRESHOLD = 50_000


def classify_dataset(
    is_classification: bool,
    n_classes: Optional[int],
    class_balance_ratio: float,
    n_samples: int,
) -> DatasetCategory:
    """Classify a dataset into a DatasetCategory.

    Parameters
    ----------
    is_classification : bool
        Whether this is a classification problem.
    n_classes : int or None
        Number of target classes (None for regression).
    class_balance_ratio : float
        minority_count / majority_count (1.0 = perfectly balanced).
    n_samples : int
        Total number of samples.

    Returns
    -------
    DatasetCategory
        The determined category.
    """
    logger.debug(
        f"classify_dataset: is_classification={is_classification}, "
        f"n_classes={n_classes}, class_balance_ratio={class_balance_ratio:.6f}, "
        f"n_samples={n_samples}"
    )

    if not is_classification:
        if n_samples < REGRESSION_SMALL_THRESHOLD:
            category = DatasetCategory.REGRESSION_SMALL
        elif n_samples < REGRESSION_LARGE_THRESHOLD:
            category = DatasetCategory.REGRESSION_MEDIUM
        else:
            category = DatasetCategory.REGRESSION_LARGE
        logger.debug(f"classify_dataset: regression → {category.value}")
        return category

    # Classification
    if n_classes is None or n_classes < 2:
        logger.debug(f"classify_dataset: n_classes={n_classes} → balanced_binary")
        return DatasetCategory.BALANCED_BINARY

    if n_classes == 2:
        if class_balance_ratio < EXTREME_IMBALANCE_THRESHOLD:
            category = DatasetCategory.EXTREME_IMBALANCE
        elif class_balance_ratio < SEVERE_IMBALANCE_THRESHOLD:
            category = DatasetCategory.SEVERE_IMBALANCE
        elif class_balance_ratio < MODERATE_IMBALANCE_THRESHOLD:
            category = DatasetCategory.MODERATE_IMBALANCE
        elif class_balance_ratio < MILD_IMBALANCE_THRESHOLD:
            category = DatasetCategory.MILD_IMBALANCE
        else:
            category = DatasetCategory.BALANCED_BINARY
        logger.debug(
            f"classify_dataset: binary, ratio={class_balance_ratio:.6f} → {category.value}"
        )
        return category
    else:
        # Multiclass
        if class_balance_ratio < MILD_IMBALANCE_THRESHOLD:
            category = DatasetCategory.MULTICLASS_IMBALANCED
        else:
            category = DatasetCategory.MULTICLASS_BALANCED
        logger.debug(
            f"classify_dataset: multiclass, ratio={class_balance_ratio:.6f} → {category.value}"
        )
        return category


@dataclass
class DatasetProfile:
    """Enriched dataset profile with category classification.

    This extends the basic profile with the DatasetCategory and
    per-class statistics needed by the strategy factory.
    """
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    is_classification: bool
    class_balance_ratio: float
    has_imbalanced_classes: bool
    feature_types: dict[str, str]
    memory_mb: float
    recommended_families: list[str]
    recommended_scoring: str
    problem_subtype: str

    # -- New fields ---------------------------------------------------------
    category: DatasetCategory = DatasetCategory.BALANCED_BINARY
    """The determined dataset category."""

    minority_class_counts: dict[str, int] = field(default_factory=dict)
    """Per-class sample counts (key=class_label, value=count)."""

    scale_pos_weight: float = 1.0
    """For binary classification: neg/pos ratio. For multiclass: max per-class ratio."""

    per_class_scale_weights: dict[str, float] = field(default_factory=dict)
    """Per-class scale_pos_weight for multiclass (class_label -> majority/class_ratio)."""

    recommended_average: str = "binary"
    """Sklearn averaging strategy: 'binary', 'macro', 'weighted'."""

    is_multiclass: bool = False
    """True if n_classes > 2."""

    regression_size: Optional[RegressionSize] = None
    """Size bucket for regression problems."""

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> DatasetProfile:
        """Build a complete DatasetProfile from a DataFrame.

        This is the single entry point for dataset analysis.
        It computes all fields including the category.
        """
        n_samples = len(df)
        n_features = len(feature_columns)
        target = df[target_column]
        n_unique = target.nunique()

        # Determine problem type
        is_classification = (
            target.dtype == object
            or target.dtype.name == "category"
            or (np.issubdtype(target.dtype, np.integer) and n_unique <= 20)
            or (n_unique == 2)
        )

        n_classes: Optional[int] = None
        class_balance_ratio = 1.0
        has_imbalanced = False
        problem_subtype = "regression"
        recommended_scoring = "r2"
        spw = 1.0
        per_class_weights: dict[str, float] = {}
        recommended_avg = "binary"
        minority_counts: dict[str, int] = {}
        is_multiclass = False
        regression_size: Optional[RegressionSize] = None

        if is_classification:
            n_classes = n_unique
            value_counts = target.value_counts()
            minority_counts = {str(k): int(v) for k, v in value_counts.items()}

            if len(value_counts) >= 2:
                class_balance_ratio = float(
                    value_counts.min() / value_counts.max()
                )
                has_imbalanced = class_balance_ratio < MILD_IMBALANCE_THRESHOLD

            if n_classes == 2:
                problem_subtype = "binary"
                recommended_scoring = "f1"
                recommended_avg = "binary"
                # Do NOT assume labels are 0/1 — use sorted counts so that
                # minority = smallest count, majority = largest count,
                # regardless of the actual label values (e.g. 2/3, yes/no).
                sorted_counts = value_counts.sort_values()
                n_pos = int(sorted_counts.iloc[0])   # minority class count
                n_neg = int(sorted_counts.iloc[-1])  # majority class count
                if n_pos > 0:
                    spw = float(n_neg / n_pos)
            else:
                is_multiclass = True
                problem_subtype = "multiclass"
                recommended_scoring = "f1_macro"
                recommended_avg = "macro"
                max_count = int(value_counts.max())
                for cls_label, cls_count in value_counts.items():
                    per_class_weights[str(cls_label)] = float(
                        max_count / cls_count
                    )
                spw = float(max(per_class_weights.values())) if per_class_weights else 1.0
        else:
            if n_samples < REGRESSION_SMALL_THRESHOLD:
                regression_size = RegressionSize.SMALL
                problem_subtype = "regression_small"
            elif n_samples < REGRESSION_LARGE_THRESHOLD:
                regression_size = RegressionSize.MEDIUM
                problem_subtype = "regression_medium"
            else:
                regression_size = RegressionSize.LARGE
                problem_subtype = "regression_large"
            recommended_scoring = "neg_root_mean_squared_error"

        logger.debug(
            f"from_dataframe: computing category — "
            f"is_classification={is_classification}, n_classes={n_classes}, "
            f"class_balance_ratio={class_balance_ratio:.6f}, n_samples={n_samples}"
        )

        category = classify_dataset(
            is_classification=is_classification,
            n_classes=n_classes,
            class_balance_ratio=class_balance_ratio,
            n_samples=n_samples,
        )

        logger.info(
            f"from_dataframe: category={category.value}, "
            f"is_classification={is_classification}, n_classes={n_classes}, "
            f"class_balance_ratio={class_balance_ratio:.6f}, "
            f"scale_pos_weight={spw:.1f}, minority_counts={minority_counts}"
        )

        feature_types = {}
        for col in feature_columns:
            if col in df.columns:
                feature_types[col] = (
                    "categorical"
                    if df[col].dtype in (object, "category")
                    else "numeric"
                )

        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        recommended_families = _recommend_families_basic(
            n_samples, n_features, is_classification, has_imbalanced, memory_mb,
        )

        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            is_classification=is_classification,
            class_balance_ratio=class_balance_ratio,
            has_imbalanced_classes=has_imbalanced,
            feature_types=feature_types,
            memory_mb=memory_mb,
            recommended_families=recommended_families,
            recommended_scoring=recommended_scoring,
            problem_subtype=problem_subtype,
            category=category,
            minority_class_counts=minority_counts,
            scale_pos_weight=spw,
            per_class_scale_weights=per_class_weights,
            recommended_average=recommended_avg,
            is_multiclass=is_multiclass,
            regression_size=regression_size,
        )


def _recommend_families_basic(
    n_samples: int,
    n_features: int,
    is_classification: bool,
    has_imbalanced: bool,
    memory_mb: float,
) -> list[str]:
    """Basic family recommendation (StrategyFactory overrides per category)."""
    families: list[str] = []
    if is_classification:
        families.append("random_forest")
        if n_samples > 500:
            families.append("gradient_boosting")
        try:
            from xgboost import XGBClassifier  # noqa: F401
            if n_samples > 1000:
                families.append("xgboost")
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier  # noqa: F401
            families.append("lightgbm")
        except ImportError:
            pass
        if n_samples < 100_000:
            families.append("logistic_regression")
    else:
        families.append("random_forest")
        families.append("ridge_regression")
        if n_samples > 500:
            families.append("gradient_boosting")
        try:
            from xgboost import XGBRegressor  # noqa: F401
            if n_samples > 1000:
                families.append("xgboost")
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor  # noqa: F401
            families.append("lightgbm")
        except ImportError:
            pass
        if n_samples < 50_000:
            families.append("elasticnet_regression")
    return families
