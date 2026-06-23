"""
Strategy Factory — the decision engine.

Maps a DatasetProfile to a fully-configured ImbalanceStrategy.
This is where the entire strategy matrix (scoring, CV folds,
model family priority, algorithm-specific parameter overrides)
lives as code.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.training.dataset_profile import DatasetProfile, DatasetCategory
from src.training.imbalance_strategy import ImbalanceStrategy

logger = logging.getLogger("ml_platform.training.strategy_factory")

# -- Availability checks ------------------------------------------------------

try:
    from xgboost import XGBClassifier, XGBRegressor  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_strategy(
    profile: DatasetProfile,
    tuning_n_trials: int = 30,
) -> ImbalanceStrategy:
    """Build the appropriate ImbalanceStrategy for a dataset profile.

    Dispatches to category-specific builders that encode the full
    strategy matrix (scoring, CV folds, model priority, param overrides).
    """
    category = profile.category

    builders = {
        DatasetCategory.EXTREME_IMBALANCE: _build_extreme_imbalance,
        DatasetCategory.SEVERE_IMBALANCE: _build_severe_imbalance,
        DatasetCategory.MODERATE_IMBALANCE: _build_moderate_imbalance,
        DatasetCategory.MILD_IMBALANCE: _build_mild_imbalance,
        DatasetCategory.BALANCED_BINARY: _build_balanced_binary,
        DatasetCategory.MULTICLASS_BALANCED: _build_multiclass_balanced,
        DatasetCategory.MULTICLASS_IMBALANCED: _build_multiclass_imbalanced,
        DatasetCategory.REGRESSION_SMALL: _build_regression_small,
        DatasetCategory.REGRESSION_MEDIUM: _build_regression_medium,
        DatasetCategory.REGRESSION_LARGE: _build_regression_large,
    }

    builder = builders.get(category)
    if builder is None:
        logger.warning(
            f"Unknown category '{category}', falling back to balanced binary"
        )
        builder = _build_balanced_binary

    strategy = builder(profile, tuning_n_trials)

    logger.info(
        f"Strategy selected: category={strategy.category}, "
        f"scoring={strategy.scoring_metric}, cv_folds={strategy.cv_folds}, "
        f"families={strategy.model_families}, "
        f"class_weight={strategy.class_weight}, "
        f"spw={strategy.scale_pos_weight:.1f}"
    )
    return strategy


# -- Extreme Imbalance (< 1% minority) ---------------------------------------

def _build_extreme_imbalance(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    spw = profile.scale_pos_weight
    spw_low = max(1, int(spw * 0.8))
    spw_high = int(spw * 1.2)
    spw_values = sorted(set([spw_low, int(spw), spw_high]))
    spw_values = [v for v in spw_values if v >= 1]

    return ImbalanceStrategy(
        category=DatasetCategory.EXTREME_IMBALANCE.value,
        model_families=_filter_available(
            ["xgboost", "lightgbm", "random_forest", "gradient_boosting"]
        ),
        scoring_metric="pr_auc",
        cv_folds=3,
        class_weight="balanced_subsample",
        scale_pos_weight=spw,
        use_sample_weight=True,
        is_unbalanced_lgbm=True,
        search_space_overrides={
            "xgboost": {"scale_pos_weight": spw_values},
            "lightgbm": {"is_unbalanced": [True]},
            "random_forest": {"class_weight": ["balanced_subsample"]},
            "gradient_boosting": {"subsample": [0.6, 0.7, 0.8, 0.9]},
            "logistic_regression": {
                "class_weight": ["balanced"],
                "C": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
        },
        fit_extras={"early_stopping_rounds": 30},
    )


# -- Severe Imbalance (1-5% minority) ----------------------------------------

def _build_severe_imbalance(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    spw = profile.scale_pos_weight
    spw_low = max(1, int(spw * 0.9))
    spw_high = int(spw * 1.1)
    spw_values = sorted(set([spw_low, int(spw), spw_high]))
    spw_values = [v for v in spw_values if v >= 1]

    return ImbalanceStrategy(
        category=DatasetCategory.SEVERE_IMBALANCE.value,
        model_families=_filter_available(
            ["xgboost", "lightgbm", "random_forest", "gradient_boosting",
             "logistic_regression"]
        ),
        scoring_metric="f1",
        cv_folds=4,
        class_weight="balanced_subsample",
        scale_pos_weight=spw,
        use_sample_weight=True,
        is_unbalanced_lgbm=True,
        search_space_overrides={
            "xgboost": {"scale_pos_weight": spw_values},
            "lightgbm": {"is_unbalanced": [True]},
            "random_forest": {
                "class_weight": ["balanced_subsample", "balanced"],
            },
            "logistic_regression": {
                "class_weight": ["balanced"],
                "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        },
        fit_extras={"early_stopping_rounds": 20},
    )


# -- Moderate Imbalance (5-20% minority) -------------------------------------

def _build_moderate_imbalance(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.MODERATE_IMBALANCE.value,
        model_families=_filter_available(
            ["random_forest", "xgboost", "lightgbm", "gradient_boosting",
             "logistic_regression"]
        ),
        scoring_metric="f1_weighted",
        cv_folds=5,
        class_weight="balanced",
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "xgboost": {"scale_pos_weight": [1, 2, 5]},
            "lightgbm": {"is_unbalanced": [False, True]},
            "random_forest": {"class_weight": ["balanced", None]},
            "logistic_regression": {
                "class_weight": ["balanced", None],
                "C": [0.5, 1.0, 2.0, 5.0, 10.0],
            },
        },
        fit_extras={},
    )


# -- Mild Imbalance (20-45% minority) ----------------------------------------

def _build_mild_imbalance(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.MILD_IMBALANCE.value,
        model_families=_filter_available(
            ["logistic_regression", "random_forest", "xgboost",
             "gradient_boosting", "lightgbm"]
        ),
        scoring_metric="f1_weighted",
        cv_folds=5,
        class_weight="balanced",
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "random_forest": {"class_weight": ["balanced", None]},
            "logistic_regression": {
                "class_weight": [None, "balanced"],
                "C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            },
        },
        fit_extras={},
    )


# -- Balanced Binary ---------------------------------------------------------

def _build_balanced_binary(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.BALANCED_BINARY.value,
        model_families=_filter_available(
            ["logistic_regression", "random_forest", "xgboost",
             "gradient_boosting", "lightgbm"]
        ),
        scoring_metric="accuracy",
        cv_folds=5,
        class_weight=None,
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "logistic_regression": {"class_weight": [None]},
            "random_forest": {"class_weight": [None]},
        },
        fit_extras={},
    )


# -- Multiclass Balanced -----------------------------------------------------

def _build_multiclass_balanced(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.MULTICLASS_BALANCED.value,
        model_families=_filter_available(
            ["lightgbm", "xgboost", "random_forest", "gradient_boosting",
             "logistic_regression"]
        ),
        scoring_metric="f1_macro",
        cv_folds=5,
        class_weight=None,
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "logistic_regression": {"class_weight": [None, "balanced"]},
        },
        fit_extras={},
    )


# -- Multiclass Imbalanced ---------------------------------------------------

def _build_multiclass_imbalanced(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.MULTICLASS_IMBALANCED.value,
        model_families=_filter_available(
            ["lightgbm", "xgboost", "random_forest", "gradient_boosting"]
        ),
        scoring_metric="f1_macro",
        cv_folds=4,
        class_weight="balanced",
        scale_pos_weight=profile.scale_pos_weight,
        use_sample_weight=True,
        is_unbalanced_lgbm=True,
        search_space_overrides={
            "random_forest": {"class_weight": ["balanced"]},
        },
        fit_extras={"early_stopping_rounds": 20},
    )


# -- Regression Small (< 5K) -------------------------------------------------

def _build_regression_small(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.REGRESSION_SMALL.value,
        model_families=_filter_available(
            ["ridge_regression", "random_forest", "elasticnet_regression",
             "gradient_boosting"]
        ),
        scoring_metric="neg_root_mean_squared_error",
        cv_folds=5,
        class_weight=None,
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "random_forest": {
                "max_depth": [3, 5, 7, 10],
                "n_estimators": [50, 100, 200],
            },
            "gradient_boosting": {
                "max_depth": [3, 4, 5],
                "n_estimators": [50, 100, 200],
            },
        },
        fit_extras={},
    )


# -- Regression Medium (5K-50K) ----------------------------------------------

def _build_regression_medium(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.REGRESSION_MEDIUM.value,
        model_families=_filter_available(
            ["xgboost", "lightgbm", "random_forest", "gradient_boosting",
             "ridge_regression"]
        ),
        scoring_metric="neg_root_mean_squared_error",
        cv_folds=5,
        class_weight=None,
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "xgboost": {"max_depth": [4, 6, 8, 10]},
            "lightgbm": {"num_leaves": [31, 63, 127]},
        },
        fit_extras={},
    )


# -- Regression Large (> 50K) ------------------------------------------------

def _build_regression_large(
    profile: DatasetProfile, n_trials: int,
) -> ImbalanceStrategy:
    return ImbalanceStrategy(
        category=DatasetCategory.REGRESSION_LARGE.value,
        model_families=_filter_available(
            ["lightgbm", "xgboost", "random_forest", "gradient_boosting"]
        ),
        scoring_metric="neg_root_mean_squared_error",
        cv_folds=3,
        class_weight=None,
        scale_pos_weight=1.0,
        use_sample_weight=False,
        is_unbalanced_lgbm=False,
        search_space_overrides={
            "xgboost": {
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            },
            "lightgbm": {
                "subsample": [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            },
        },
        fit_extras={},
    )


# -- Helpers -----------------------------------------------------------------

def _filter_available(families: list[str]) -> list[str]:
    """Filter model families by availability."""
    result = []
    for f in families:
        if f == "xgboost" and not XGBOOST_AVAILABLE:
            continue
        if f == "lightgbm" and not LIGHTGBM_AVAILABLE:
            continue
        result.append(f)
    return result if result else ["random_forest"]
