# src/training/hyperparameter_tuner.py
"""
Production-grade automated hyperparameter tuning engine.

Designed for minimal human interaction: feed it a dataset, get back the
best model with optimized hyperparameters for precision, recall, F1, and
accuracy.

Architecture:
    AutoTrainer
      ├── Dataset analysis (problem type, class balance, size, feature types)
      ├── Model family selection (adaptive to dataset characteristics)
      ├── Multi-stage hyperparameter search:
      │   ├── Stage 1: Broad exploration (Optuna TPE or sklearn HalvingSearch)
      │   ├── Stage 2: Focused refinement (narrowed search space)
      │   └── Stage 3: Final training with best params via ModelTrainer
      └── Full observability (Prometheus, event bus, MLflow)

Usage:
    from src.training.hyperparameter_tuner import AutoTrainer, TuningConfig
    from src.training.trainer import TrainingConfig

    tuner = AutoTrainer(model_id="fraud_detector", pipeline_run_id="run_123")
    config = TrainingConfig(
        model_type="auto",
        hyperparameters={},
        feature_columns=features,
        target_column="label",
    )
    result = tuner.auto_train(df, config)
    print(result.tuning_report["best_params"])
"""
from __future__ import annotations

import json
import logging
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union,List,Dict

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
)
from sklearn.preprocessing import LabelEncoder

from src.common.exceptions import HyperparameterTuningError
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    TRAINING_RUNS_TOTAL,
    TRAINING_DURATION,
    MODEL_CV_SCORE,
    MODEL_VALIDATION_METRIC,
)
from src.training.trainer import ModelTrainer, TrainingConfig, TrainingResult, ModelProtocol
from src.training.dataset_profile import DatasetProfile, DatasetCategory
from src.training.strategy_factory import get_strategy
from src.training.imbalance_strategy import ImbalanceStrategy

logger = logging.getLogger("ml_platform.training.hyperparameter_tuner")

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports - graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.info("Optuna not installed. Using sklearn HalvingRandomSearchCV fallback.")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import sklearn
    # explicitly require this experimental feature
    from sklearn.experimental import enable_halving_search_cv # noqa
    # now you can import normally from model_selection
    from sklearn.model_selection import HalvingRandomSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics for tuning
# ─────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Gauge, Histogram

    HYPERPARAMETER_TRIALS_TOTAL = Counter(
        "ml_hyperparameter_trials_total",
        "Total hyperparameter optimization trials",
        ["model_id", "model_family", "status"],
    )
    TUNING_DURATION = Histogram(
        "ml_tuning_duration_seconds",
        "Hyperparameter tuning duration per model family",
        ["model_id", "model_family"],
        buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
    )
    BEST_TRIAL_SCORE = Gauge(
        "ml_best_trial_score",
        "Best CV score from hyperparameter tuning",
        ["model_id", "model_family"],
    )
    TUNING_STATUS = Gauge(
        "ml_tuning_status",
        "Tuning status (0=idle, 1=running, 2=completed, 3=failed)",
        ["model_id"],
    )
except ImportError:
    HYPERPARAMETER_TRIALS_TOTAL = None
    TUNING_DURATION = None
    BEST_TRIAL_SCORE = None
    TUNING_STATUS = None


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_number: int
    model_family: str
    hyperparameters: dict[str, Any]
    cv_mean: float
    cv_std: float
    primary_metric: str
    secondary_metrics: dict[str, float]
    fit_time_seconds: float
    is_pruned: bool = False


@dataclass
class FamilyTuningResult:
    """Complete tuning result for one model family."""
    model_family: str
    best_params: dict[str, Any]
    best_score: float
    best_std: float
    primary_metric: str
    all_trials: list[TrialResult]
    tuning_duration_seconds: float
    n_trials_completed: int
    n_trials_pruned: int
    status: str
    error_message: Optional[str] = None


@dataclass
class TuningConfig:
    """Configuration for the hyperparameter tuning process."""
    model_families: list[str] = field(default_factory=list)
    n_trials: int = 50
    timeout_seconds: int = 1800
    cv_folds: int = 5
    primary_metric: str = ""
    secondary_metrics: list[str] = field(default_factory=list)
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001
    search_strategy: str = "auto"
    stage2_refinement_ratio: float = 0.3
    n_jobs: int = -1
    random_state: int = 42
    use_multi_objective: bool = True
    ensemble_best_n: int = 0


@dataclass
class AutoTrainingResult:
    """Complete result of an automated training run."""
    training_result: TrainingResult
    dataset_profile: DatasetProfile
    tuning_config: TuningConfig
    family_results: dict[str, FamilyTuningResult]
    best_family: str
    best_params: dict[str, Any]
    best_score: float
    total_tuning_duration_seconds: float
    total_trials: int
    timestamp: datetime

    @property
    def tuning_report(self) -> dict[str, Any]:
        return {
            "dataset": {
                "samples": self.dataset_profile.n_samples,
                "features": self.dataset_profile.n_features,
                "problem_type": (
                    "classification"
                    if self.dataset_profile.is_classification
                    else "regression"
                ),
                "class_balance": round(self.dataset_profile.class_balance_ratio, 3),
                "is_imbalanced": self.dataset_profile.has_imbalanced_classes,
                "scale_pos_weight": round(self.dataset_profile.scale_pos_weight, 1),
                "recommended_families": self.dataset_profile.recommended_families,
            },
            "tuning_summary": {
                "total_trials": self.total_trials,
                "total_duration_seconds": round(self.total_tuning_duration_seconds, 1),
                "families_evaluated": list(self.family_results.keys()),
                "best_family": self.best_family,
                "best_score": round(self.best_score, 6),
                "best_params": self.best_params,
            },
            "family_details": {
                name: {
                    "best_score": round(r.best_score, 6),
                    "best_std": round(r.best_std, 6),
                    "n_trials": r.n_trials_completed,
                    "n_pruned": r.n_trials_pruned,
                    "duration_seconds": round(r.tuning_duration_seconds, 1),
                    "status": r.status,
                    "best_params": r.best_params,
                }
                for name, r in self.family_results.items()
            },
            "final_metrics": {
                k: round(v, 6) for k, v in self.training_result.metrics.items()
            },
            "cv_mean": round(self.training_result.cv_mean, 6),
            "cv_std": round(self.training_result.cv_std, 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Search space definitions
# ─────────────────────────────────────────────────────────────────────────────

class SearchSpace:
    """Dataset-adaptive hyperparameter search spaces."""

    @staticmethod
    def random_forest(is_classification, is_imbalanced, n_samples):
        class_weight_values = (
            ["balanced", "balanced_subsample", None] if is_imbalanced else [None, "balanced"]
        )
        sklearn_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, 30, None],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "class_weight": class_weight_values if is_classification else [None],
        }

        def optuna_space(trial):
            return {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, 30, None]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "class_weight": trial.suggest_categorical("class_weight", class_weight_values) if is_classification else None,
            }

        return sklearn_space, optuna_space

    @staticmethod
    def xgboost(is_classification, is_imbalanced, n_samples):
        # scale_pos_weight is critical for imbalanced classification
        # It controls the balance of positive/negative weights
        spw_values = [1, 5, 10, 50, 100] if is_imbalanced else [1]
        sklearn_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-4, 10),
            "min_child_weight": randint(1, 10),
            "gamma": loguniform(1e-4, 5),
            "scale_pos_weight": spw_values if is_imbalanced else [1],
        }

        def optuna_space(trial):
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            }
            if is_imbalanced:
                params["scale_pos_weight"] = trial.suggest_categorical(
                    "scale_pos_weight", spw_values
                )
            return params

        return sklearn_space, optuna_space

    @staticmethod
    def lightgbm(is_classification, is_imbalanced, n_samples):
        # LightGBM 4.6+ sklearn API does not support `is_unbalanced`.
        # Imbalance is handled via `scale_pos_weight` (set by the strategy
        # factory's search_space_overrides) or `class_weight` (native sklearn).
        sklearn_space = {
            "n_estimators": [100, 200, 300, 500],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 5, 10, 15, 20],
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-4, 10),
            "min_child_samples": randint(5, 50),
        }

        def optuna_space(trial):
            return {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15, 20]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            }

        return sklearn_space, optuna_space

    @staticmethod
    def logistic_regression(is_classification, is_imbalanced, n_samples):
        class_weight_values = ["balanced", None] if is_imbalanced else [None, "balanced"]
        sklearn_space = {
            "C": loguniform(1e-4, 100),
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["saga"],
            "max_iter": [500, 1000, 2000],
            "class_weight": class_weight_values,
            "l1_ratio": uniform(0, 1),
        }

        def optuna_space(trial):
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            return {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "penalty": penalty,
                "solver": "saga",
                "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000]),
                "class_weight": trial.suggest_categorical("class_weight", class_weight_values),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None,
            }

        return sklearn_space, optuna_space

    @staticmethod
    def gradient_boosting(is_classification, is_imbalanced, n_samples):
        # NOTE: GradientBoostingClassifier does NOT support class_weight.
        # For imbalanced data, we rely on:
        #   1. subsample < 1.0 (stochastic GB) for implicit regularization
        #   2. The trainer's sample_weight injection at fit time
        #   3. Stratified CV to preserve class distribution in folds
        sklearn_space = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        }

        def optuna_space(trial):
            return {
                "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, None]),
            }

        return sklearn_space, optuna_space

    @staticmethod
    def _safe_model_params(model_cls, params: Dict[Any,Any], random_state: int) -> Dict[str, Any]:
        """Filter params to only those accepted by the model class.
        Handles models like GradientBoosting that don't accept n_jobs."""
        import inspect
        sig = inspect.signature(model_cls.__init__)
        valid_params = set(sig.parameters.keys())
        safe = {"random_state": random_state}
        if params is None:
            return safe
        for k, v in params.items():
            if k in valid_params and v is not None:
                safe[k] = v
        return safe

    @staticmethod
    def ridge_regression(is_classification, is_imbalanced, n_samples):
        sklearn_space = {
            "alpha": loguniform(1e-4, 1000),
            "fit_intercept": [True, False],
        }

        def optuna_space(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 1000.0, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            }

        return sklearn_space, optuna_space

    @staticmethod
    def elasticnet_regression(is_classification, is_imbalanced, n_samples):
        sklearn_space = {
            "alpha": loguniform(1e-4, 100),
            "l1_ratio": uniform(0, 1),
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000],
        }

        def optuna_space(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "max_iter": trial.suggest_categorical("max_iter", [1000, 2000]),
            }

        return sklearn_space, optuna_space


# Model factory registry
MODEL_FACTORIES: dict[str, dict[str, Any]] = {
    "random_forest": {
        "classifier": RandomForestClassifier,
        "regressor": RandomForestRegressor,
        "search_space": SearchSpace.random_forest,
        "always_available": True,
    },
    "gradient_boosting": {
        "classifier": GradientBoostingClassifier,
        "regressor": GradientBoostingRegressor,
        "search_space": SearchSpace.gradient_boosting,
        "always_available": True,
    },
    "logistic_regression": {
        "classifier": LogisticRegression,
        "regressor": None,
        "search_space": SearchSpace.logistic_regression,
        "always_available": True,
    },
    "ridge_regression": {
        "classifier": None,
        "regressor": Ridge,
        "search_space": SearchSpace.ridge_regression,
        "always_available": True,
    },
    "elasticnet_regression": {
        "classifier": None,
        "regressor": ElasticNet,
        "search_space": SearchSpace.elasticnet_regression,
        "always_available": True,
    },
}

if XGBOOST_AVAILABLE:
    MODEL_FACTORIES["xgboost"] = {
        "classifier": XGBClassifier,
        "regressor": XGBRegressor,
        "search_space": SearchSpace.xgboost,
        "always_available": False,
    }

if LIGHTGBM_AVAILABLE:
    MODEL_FACTORIES["lightgbm"] = {
        "classifier": LGBMClassifier,
        "regressor": LGBMRegressor,
        "search_space": SearchSpace.lightgbm,
        "always_available": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AutoTrainer - main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AutoTrainer:
    """
    Production-grade automated training with hyperparameter optimization.

    Takes a dataset and produces the best model with minimal human interaction.
    Automatically detects problem type, selects model families, tunes
    hyperparameters, and trains the final model.

    Parameters
    ----------
    model_id : str
        Identifier for the model being trained.
    pipeline_run_id : str
        ID of the pipeline run (for event tracking).
    settings : optional
        Platform settings. Uses global settings if not provided.
    """

    def __init__(
        self,
        model_id: str = "",
        model_version : str ="",
        pipeline_run_id: str = "",
        settings: Optional[Any] = None,
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.pipeline_run_id = pipeline_run_id
        self.settings = settings

    def auto_train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        tuning_config: Optional[TuningConfig] = None,
    ) -> AutoTrainingResult:
        """
        End-to-end automated training.

        Parameters
        ----------
        df : pd.DataFrame
            Training data (features + target).
        config : TrainingConfig
            Base training configuration. Hyperparameters will be overridden.
        tuning_config : TuningConfig, optional
            Tuning-specific configuration. Uses defaults if not provided.

        Returns
        -------
        AutoTrainingResult
            Complete training result with tuning report.
        """
        tuning_config = tuning_config or TuningConfig()
        run_id = str(uuid.uuid4())
        total_start = time.time()

        logger.info(
            f"AutoTrain started: model={self.model_id}, "
            f"samples={len(df)}, features={len(config.feature_columns)}"
        )

        if TUNING_STATUS:
            TUNING_STATUS.labels(model_id=self.model_id).set(1)

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.TRAINING_STARTED,
            step_name="auto_tuning",
            title="Auto-Training Started",
            message=(
                f"Model: {self.model_id} | "
                f"Samples: {len(df)} | "
                f"Features: {len(config.feature_columns)}"
            ),
            model_id=self.model_id,
            data={
                "run_id": run_id,
                "n_samples": len(df),
                "n_features": len(config.feature_columns),
                "tuning_strategy": tuning_config.search_strategy,
            },
        ))

        try:
            # Step 1: Analyze dataset (already done above)
            # profile is already computed with category, scale_pos_weight, etc.
            profile = self._analyze_dataset(df, config)
            # Step 2: Get strategy from the strategy factory
            # The strategy encodes ALL decisions: families, scoring, CV folds,
            # class_weight, scale_pos_weight, search space overrides, fit extras.
            strategy = get_strategy(profile, tuning_config.n_trials)
            logger.info(
                f"Strategy selected: category={strategy.category}, "
                f"scoring_metric={strategy.scoring_metric}, "
                f"cv_folds={strategy.cv_folds}, "
                f"model_families={strategy.model_families}, "
                f"class_weight={strategy.class_weight}, "
                f"scale_pos_weight={strategy.scale_pos_weight:.1f}, "
                f"use_sample_weight={strategy.use_sample_weight}, "
                f"is_unbalanced_lgbm={strategy.is_unbalanced_lgbm}"
            )
            logger.debug(
                f"Strategy search_space_overrides={strategy.search_space_overrides}, "
                f"fit_extras={strategy.fit_extras}"
            )

            # Step 3: Determine model families from strategy
            # (strategy already filters by availability)
            families = tuning_config.model_families or strategy.model_families
            families = [f for f in families if f in MODEL_FACTORIES]
            if not families:
                raise HyperparameterTuningError(
                    "No suitable model families available for this dataset.",
                    details={
                        "requested": tuning_config.model_families,
                        "strategy_families": strategy.model_families,
                        "available": list(MODEL_FACTORIES.keys()),
                    },
                )

            # Step 4: Determine scoring from strategy
            primary_metric = tuning_config.primary_metric or strategy.scoring_metric

            # Step 5: Tune each model family
            family_results: dict[str, FamilyTuningResult] = {}
            total_trials = 0

            for family in families:
                logger.info(f"Tuning model family: {family}")
                try:
                    result = self._tune_model_family(
                        family=family, df=df, config=config,
                        tuning_config=tuning_config, profile=profile,
                        primary_metric=primary_metric,
                        strategy=strategy,
                    )
                    family_results[family] = result
                    total_trials += result.n_trials_completed
                    logger.info(
                        f"  {family}: best_score={result.best_score:.4f}, "
                        f"trials={result.n_trials_completed}, status={result.status}"
                    )
                except Exception as e:
                    logger.warning(f"  {family}: tuning failed - {e}. Skipping.")
                    family_results[family] = FamilyTuningResult(
                        model_family=family, best_params={}, best_score=0.0,
                        best_std=0.0, primary_metric=primary_metric,
                        all_trials=[], tuning_duration_seconds=0,
                        n_trials_completed=0, n_trials_pruned=0,
                        status="failed", error_message=str(e),
                    )

            # Step 6: Select best
            best_family, best_result = self._select_best(family_results)
            if best_result is None or not best_result.best_params:
                raise HyperparameterTuningError(
                    "All model families failed to produce valid hyperparameters.",
                    details={
                        "families_tried": list(family_results.keys()),
                        "errors": {
                            name: r.error_message for name, r in family_results.items()
                            if r.status == "failed"
                        },
                    },
                )

            logger.info(f"Best model: {best_family} (score={best_result.best_score:.4f})")

            # Step 7: Final training with best params
            # The strategy carries all imbalance handling decisions.
            final_config = TrainingConfig(
                model_type=best_family,
                hyperparameters=best_result.best_params,
                feature_columns=config.feature_columns,
                target_column=config.target_column,
                cv_folds=strategy.cv_folds,
                cv_scoring=primary_metric,
                random_state=config.random_state,
                test_size=config.test_size,
                stratify=config.stratify,
                tags={
                    **config.tags, "auto_tuned": "true",
                    "tuning_trials": str(total_trials),
                    "best_family": best_family,
                    "dataset_category": strategy.category,
                },
                is_imbalanced=strategy.needs_imbalance_handling,
                class_weight=strategy.class_weight,
                scale_pos_weight=strategy.scale_pos_weight,
                strategy=strategy,
            )

            model_factory = self._build_model_factory(best_family, profile.is_classification)
            trainer = ModelTrainer(
                model_factory=model_factory, preprocessing_fn=None,
                model_id=self.model_id, pipeline_run_id=self.pipeline_run_id,
            )
            training_result = trainer.train(
                df=df, config=final_config, model_id=self.model_id,
                model_version=self.model_version,
            )

            total_duration = time.time() - total_start
            auto_result = AutoTrainingResult(
                training_result=training_result, dataset_profile=profile,
                tuning_config=tuning_config, family_results=family_results,
                best_family=best_family, best_params=best_result.best_params,
                best_score=best_result.best_score,
                total_tuning_duration_seconds=total_duration,
                total_trials=total_trials,
                timestamp=datetime.now(timezone.utc),
            )

            if TUNING_STATUS:
                TUNING_STATUS.labels(model_id=self.model_id).set(2)

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.TRAINING_COMPLETED,
                step_name="auto_tuning",
                title="Auto-Training Complete",
                message=(
                    f"Best: {best_family} | Score: {best_result.best_score:.4f} | "
                    f"Trials: {total_trials} | Duration: {total_duration:.1f}s"
                ),
                model_id=self.model_id,
                data=auto_result.tuning_report,
            ))

            logger.info(
                f"AutoTrain complete: best={best_family}, "
                f"score={best_result.best_score:.4f}, trials={total_trials}, "
                f"duration={total_duration:.1f}s"
            )
            return auto_result

        except Exception as e:
            if TUNING_STATUS:
                TUNING_STATUS.labels(model_id=self.model_id).set(3)
            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.TRAINING_FAILED,
                step_name="auto_tuning",
                title="Auto-Training Failed",
                message=str(e), model_id=self.model_id,
                status="failed", severity="error",
                data={"error": str(e)},
            ))
            logger.error(f"AutoTrain failed: {e}", exc_info=True)
            raise

    # -- Dataset Analysis ---------------------------------------------------

    def _analyze_dataset(self, df: pd.DataFrame, config: TrainingConfig) -> DatasetProfile:
        """Analyze dataset using the enriched DatasetProfile.from_dataframe().

        This delegates to the dataset_profile module which classifies the
        dataset into a DatasetCategory and computes all derived fields.
        """
        profile = DatasetProfile.from_dataframe(
            df=df,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
        )

        # Log the classification results
        logger.info(
            f"Dataset profile: category={profile.category.value}, "
            f"samples={profile.n_samples}, features={profile.n_features}, "
            f"balance={profile.class_balance_ratio:.4f}, "
            f"scale_pos_weight={profile.scale_pos_weight:.1f}, "
            f"minority_class_counts={profile.minority_class_counts}, "
            f"recommended_families={profile.recommended_families}, "
            f"recommended_scoring={profile.recommended_scoring}"
        )
        logger.debug(
            f"Dataset profile details: is_classification={profile.is_classification}, "
            f"is_multiclass={profile.is_multiclass}, "
            f"has_imbalanced_classes={profile.has_imbalanced_classes}, "
            f"per_class_scale_weights={profile.per_class_scale_weights}, "
            f"memory_mb={profile.memory_mb:.2f}"
        )

        # Warn for extreme imbalance
        if profile.category == DatasetCategory.EXTREME_IMBALANCE:
            logger.warning(
                f"EXTREME imbalance detected: minority ratio="
                f"{profile.class_balance_ratio:.4f}, scale_pos_weight="
                f"{profile.scale_pos_weight:.0f}. Will use PR-AUC scoring, "
                f"3-fold CV, and aggressive class weighting."
            )
        elif profile.category == DatasetCategory.SEVERE_IMBALANCE:
            logger.warning(
                f"Severe imbalance: minority ratio="
                f"{profile.class_balance_ratio:.4f}, scale_pos_weight="
                f"{profile.scale_pos_weight:.0f}. Using F1 scoring and "
                f"4-fold CV."
            )

        return profile

    # -- Search Space Helpers ------------------------------------------------

    @staticmethod
    def _merge_overrides(
        base_space: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge strategy overrides into a base search space.

        Overrides replace or extend base space values.
        """
        merged = dict(base_space)
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                # Merge lists, removing duplicates while preserving order
                existing = set(str(v) for v in merged[key])
                merged[key] = merged[key] + [v for v in value if str(v) not in existing]
            else:
                merged[key] = value
        return merged

    # -- Model Family Tuning ------------------------------------------------

    def _tune_model_family(
        self,
        family: str,
        df: pd.DataFrame,
        config: TrainingConfig,
        tuning_config: TuningConfig,
        profile: DatasetProfile,
        primary_metric: str,
        strategy: ImbalanceStrategy,
    ) -> FamilyTuningResult:
        """Run hyperparameter search for a single model family.

        The strategy provides search_space_overrides that are merged into
        the base search space for this family.
        """
        family_start = time.time()
        factory_info = MODEL_FACTORIES[family]
        search_space_fn = factory_info["search_space"]
        sklearn_space, optuna_space = search_space_fn(
            is_classification=profile.is_classification,
            is_imbalanced=profile.has_imbalanced_classes,
            n_samples=profile.n_samples,
        )

        logger.debug(
            f"[{family}] Base search space keys: {list(sklearn_space.keys())}, "
            f"is_classification={profile.is_classification}, "
            f"is_imbalanced={profile.has_imbalanced_classes}, "
            f"n_samples={profile.n_samples}"
        )

        # Merge strategy overrides into the search space
        overrides = strategy.get_overrides(family)
        if overrides:
            sklearn_space = AutoTrainer._merge_overrides(sklearn_space, overrides)
            logger.info(
                f"[{family}] Strategy overrides applied: {list(overrides.keys())}"
            )
            logger.debug(
                f"[{family}] Override details: {overrides}"
            )
        else:
            logger.warning(f"[{family}] No strategy overrides applied")

        search_strategy = tuning_config.search_strategy
        if search_strategy == "auto":
            search_strategy = "optuna" if OPTUNA_AVAILABLE else "sklearn"

        logger.info(
            f"[{family}] Starting tuning: search_strategy={search_strategy}, "
            f"primary_metric={primary_metric}, cv_folds={strategy.cv_folds}, "
            f"n_trials={tuning_config.n_trials}, n_jobs={tuning_config.n_jobs}"
        )

        if search_strategy == "optuna" and OPTUNA_AVAILABLE:
            result = self._tune_with_optuna(
                family, df, config, tuning_config, profile,
                primary_metric, optuna_space, strategy, overrides,
            )
        else:
            result = self._tune_with_sklearn(
                family, df, config, tuning_config, profile,
                primary_metric, sklearn_space, strategy,
            )

        result.tuning_duration_seconds = time.time() - family_start

        if BEST_TRIAL_SCORE:
            BEST_TRIAL_SCORE.labels(model_id=self.model_id, model_family=family).set(result.best_score)
        if TUNING_DURATION:
            TUNING_DURATION.labels(model_id=self.model_id, model_family=family).observe(result.tuning_duration_seconds)

        logger.info(
            f"[{family}] Tuning complete: status={result.status}, "
            f"best_score={result.best_score:.6f}, best_std={result.best_std:.6f}, "
            f"trials_completed={result.n_trials_completed}, trials_pruned={result.n_trials_pruned}, "
            f"duration={result.tuning_duration_seconds:.2f}s"
        )
        if result.error_message:
            logger.warning(f"[{family}] Tuning error: {result.error_message}")

        return result

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        """Compute scale_pos_weight for XGBoost/LightGBM from class distribution.

        For binary classification: ratio of negative to positive samples.
        This tells the model how much more to weight the minority class.
        """
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        if n_positive == 0:
            return 1.0
        return float(n_negative / n_positive)

    def _tune_with_optuna(
        self, family:str, df: pd.DataFrame, config: TrainingConfig, tuning_config: TuningConfig, profile: DatasetProfile,
        primary_metric: str, optuna_space_fn: Any, strategy: ImbalanceStrategy, overrides: Dict[str, Any]
    )-> FamilyTuningResult:
        """Optuna-based hyperparameter optimization with pruning."""
        logger.info(
            f"[{family}] _tune_with_optuna entered: primary_metric={primary_metric}, "
            f"n_trials={tuning_config.n_trials}, timeout={tuning_config.timeout_seconds}s, "
            f"cv_folds={strategy.cv_folds}"
        )

        X = df[config.feature_columns].values
        y = df[config.target_column].values

        logger.debug(
            f"[{family}] Input data: X.shape={X.shape}, y.shape={y.shape}, "
            f"y.dtype={y.dtype}"
        )

        label_encoder = None
        if profile.is_classification and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(np.asarray(y))

        factory_info = MODEL_FACTORIES[family]
        model_cls = factory_info["classifier"] if profile.is_classification else factory_info["regressor"]
        if model_cls is None:
            return FamilyTuningResult(
                model_family=family, best_params={}, best_score=0.0, best_std=0.0,
                primary_metric=primary_metric, all_trials=[], tuning_duration_seconds=0,
                n_trials_completed=0, n_trials_pruned=0, status="failed",
                error_message=f"{family} does not support this problem type",
            )

        # Use strategy's CV fold count, but reduce when the minority class
        # is too small for stratified folds to contain positives in every fold.
        # Floor at 2 — 1-fold is not cross-validation.
        if y is None or len(np.asarray(y)) == 0:
            raise HyperparameterTuningError(
                f"[{family}] Target variable is empty or None.",
                details={"y": y}
            )
        cv_folds = strategy.cv_folds
        if profile.is_classification:
            minority_count = int(y.value_counts().min()) if hasattr(y, "value_counts") else len(np.asarray(y))
            max_feasible_folds = max(2, minority_count // 2)
            if cv_folds > max_feasible_folds:
                logger.info(
                    f"[{family}] Reducing CV folds from {cv_folds} to "
                    f"{max_feasible_folds} (minority_count={minority_count})"
                )
                cv_folds = max_feasible_folds
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=tuning_config.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=tuning_config.random_state)

        # sklearn has no "pr_auc" string alias — convert to a callable scorer.
        # average_precision_score defaults to average='binary' which raises on
        # multiclass targets, so select the averaging strategy from the profile.
        if primary_metric == "pr_auc":
            if profile.is_multiclass:
                scoring = make_scorer(average_precision_score, needs_proba=True, average="macro")
            else:
                scoring = make_scorer(average_precision_score, needs_proba=True)
        else:
            scoring = primary_metric

        all_trials: list[TrialResult] = []
        best_score = -np.inf
        best_params: dict[str, Any] = {}
        best_std = 0.0
        n_pruned = 0
        patience_counter = 0

        def objective(trial):
            nonlocal best_score, best_params, best_std, n_pruned, patience_counter
            params = optuna_space_fn(trial)
            params = {k: v for k, v in params.items() if v is not None}

            # Apply strategy overrides (e.g., force class_weight for extreme imbalance).
            #
            # Override value shapes:
            #   - single-element list [v]  → force this exact value
            #   - multi-element list [a, b, → replace the search space; Optuna
            #     samples from these strategy-specified candidates instead of
            #     the base search space defaults
            #   - scalar                  → force this exact value
            for key, value in overrides.items():
                if isinstance(value, list) and len(value) == 1:
                    # Single-value override forces the param (unwrap list)
                    params[key] = value[0]
                elif isinstance(value, list) and len(value) > 1:
                    # Multi-value override: the strategy is narrowing the
                    # search space to specific candidates.  Replace the
                    # Optuna-sampled value with a strategy value (cycle by
                    # trial number so different trials explore different
                    # candidates).
                    override_idx = trial.number % len(value)
                    params[key] = value[override_idx]

            try:
                safe_params = SearchSpace._safe_model_params(
                    model_cls, params, tuning_config.random_state
                )
                model = model_cls(**safe_params)
                trial_start = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                fit_time = time.time() - trial_start

                # average_precision_score returns nan when a fold has no
                # positive samples. Use nanmean so a single nan fold doesn't
                # poison the whole trial's mean.
                if np.all(np.isnan(scores)):
                    # All folds degenerate (no positives). Use a finite
                    # sentinel so the pruner can distinguish this from a
                    # crashed trial — -np.nan is still nan and gets pruned.
                    mean_score = 0.0
                    std_score = 0.0
                else:
                    mean_score = float(np.nanmean(scores))
                    std_score = float(np.nanstd(scores))

                trial.report(mean_score, step=0)
                if trial.should_prune():
                    n_pruned += 1
                    raise optuna.TrialPruned()

                secondary = {}
                if tuning_config.use_multi_objective and profile.is_classification:
                    # 1. Get the first fold tuple directly without indexing a slice
                    X_arr, y_arr = np.asarray(X), np.asarray(y)
                    train_idx, val_idx = next(cv.split(X_arr, y_arr))
                    model_clone = clone(model)
                    model_clone.fit(X_arr[train_idx], y_arr[train_idx])
                    y_pred = model_clone.predict(X_arr[val_idx])
                    y_true = y_arr[val_idx]
                    # Use profile's recommended average (binary for binary, macro for multiclass)
                    avg = profile.recommended_average
                    secondary["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
                    secondary["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    best_std = std_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= tuning_config.early_stopping_rounds:
                    trial.study.stop()

                all_trials.append(TrialResult(
                    trial_number=trial.number, model_family=family,
                    hyperparameters=params, cv_mean=mean_score, cv_std=std_score,
                    primary_metric=primary_metric, secondary_metrics=secondary,
                    fit_time_seconds=fit_time,
                ))

                if HYPERPARAMETER_TRIALS_TOTAL:
                    HYPERPARAMETER_TRIALS_TOTAL.labels(
                        model_id=self.model_id, model_family=family, status="completed"
                    ).inc()
                return mean_score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(
                    f"Optuna trial {trial.number} failed ({type(e).__name__}): {e}"
                )
                if HYPERPARAMETER_TRIALS_TOTAL:
                    HYPERPARAMETER_TRIALS_TOTAL.labels(
                        model_id=self.model_id, model_family=family, status="failed"
                    ).inc()
                return -np.inf

        logger.debug(
            f"[{family}] Creating Optuna study: direction=maximize, "
            f"sampler=TPE(multivariate=True), pruner=MedianPruner(n_startup=5), "
            f"seed={tuning_config.random_state}"
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=tuning_config.random_state, multivariate=True),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            study_name=f"{self.model_id}_{family}_{uuid.uuid4().hex[:8]}",
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective, n_trials=tuning_config.n_trials,
            timeout=tuning_config.timeout_seconds,
            show_progress_bar=False,
        )

        logger.info(
            f"[{family}] Optuna study completed: "
            f"n_trials={len(study.trials)}, "
            f"best_value={study.best_value if study.best_trial else 'N/A'}, "
            f"n_pruned={sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}"
        )

        return FamilyTuningResult(
            model_family=family, best_params=best_params,
            best_score=best_score if best_score != -np.inf else 0.0,
            best_std=best_std, primary_metric=primary_metric, all_trials=all_trials,
            tuning_duration_seconds=0,
            n_trials_completed=len(all_trials), n_trials_pruned=n_pruned,
            status="completed" if best_params else "failed",
        )

    def _tune_with_sklearn(
        self, family: str, df: pd.DataFrame, config: TrainingConfig, tuning_config: TuningConfig, profile,
        primary_metric: str, sklearn_space: Dict[str,Any], strategy: ImbalanceStrategy,
    ):
        """Sklearn-based hyperparameter search (fallback when Optuna unavailable)."""
        logger.info(
            f"[{family}] _tune_with_sklearn entered: primary_metric={primary_metric}, "
            f"search_space_keys={list(sklearn_space.keys())}, "
            f"halving_available={HALVING_AVAILABLE}"
        )

        X = df[config.feature_columns].values
        y = df[config.target_column].values

        logger.debug(
            f"[{family}] Input data: X.shape={X.shape}, y.shape={y.shape}, "
            f"y.dtype={y.dtype}, y.unique={np.unique(y)[:10]}"
        )

        if profile.is_classification and y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(np.asarray(y))
            logger.debug(
                f"[{family}] Label-encoded y: classes={label_encoder.classes_.tolist()}"
            )

        factory_info = MODEL_FACTORIES[family]
        model_cls = factory_info["classifier"] if profile.is_classification else factory_info["regressor"]
        if model_cls is None:
            return FamilyTuningResult(
                model_family=family, best_params={}, best_score=0.0, best_std=0.0,
                primary_metric=primary_metric, all_trials=[], tuning_duration_seconds=0,
                n_trials_completed=0, n_trials_pruned=0, status="failed",
                error_message=f"{family} does not support this problem type",
            )

        # sklearn has no "pr_auc" string alias — convert to a callable scorer.
        # average_precision_score defaults to average='binary' which raises on
        # multiclass targets, so select the averaging strategy from the profile.
        if primary_metric == "pr_auc":
            if profile.is_multiclass:
                scoring = make_scorer(average_precision_score, needs_proba=True, average="macro")
            else:
                scoring = make_scorer(average_precision_score, needs_proba=True)
        else:
            scoring = primary_metric

        base_model = model_cls(random_state=tuning_config.random_state)
        # Filter out params that this model doesn't understand
        import inspect
        valid_params = set(inspect.signature(base_model.__init__).parameters.keys())
        clean_space = {
            k: v for k, v in sklearn_space.items()
            if k in valid_params
            and not (k == "class_weight" and not profile.is_classification)
            and not (k == "scale_pos_weight" and "scale_pos_weight" not in valid_params)
        }
        n_iter = min(tuning_config.n_trials, 30)

        # Use strategy's CV fold count (reduced for extreme imbalance)
        cv_folds = strategy.cv_folds

        logger.debug(
            f"[{family}] Search config: n_iter={n_iter}, cv_folds={cv_folds}, "
            f"clean_space_keys={list(clean_space.keys())}, "
            f"n_jobs={tuning_config.n_jobs}, factor=3, min_resources=smallest"
        )

        # Guard: with extreme imbalance, even StratifiedKFold can produce
        # folds with 0 positives when the absolute minority count is tiny.
        # Ensure each fold contains at least MIN_FOLD_POSITIVES minority
        # samples. If not, reduce fold count until it does.
        MIN_FOLD_POSITIVES = 2
        if profile.is_classification and profile.has_imbalanced_classes:
            # Estimate minority count from class_balance_ratio if
            # minority_class_counts is empty or unreliable
            minority_count = 0
            if profile.minority_class_counts:
                minority_count = min(profile.minority_class_counts.values())
            if minority_count == 0 and profile.class_balance_ratio > 0:
                minority_count = max(1, int(profile.class_balance_ratio * profile.n_samples))
            max_safe_folds = max(2, minority_count // MIN_FOLD_POSITIVES)
            if cv_folds > max_safe_folds:
                logger.warning(
                    f"Reducing CV folds from {cv_folds} to {max_safe_folds} "
                    f"for {family}: minority_count={minority_count}, "
                    f"need >= {MIN_FOLD_POSITIVES} positives per fold."
                )
                cv_folds = max_safe_folds

        if HALVING_AVAILABLE:
            search = HalvingRandomSearchCV(
                estimator=base_model, param_distributions=clean_space,
                n_candidates=n_iter, cv=cv_folds,
                scoring=scoring, random_state=tuning_config.random_state,
                n_jobs=tuning_config.n_jobs, factor=3, min_resources="smallest",
                error_score=np.nan,
            )
        else:
            logger.warning("HalvingRandomSearchCV is not available!!")
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=base_model, param_distributions=clean_space,
                n_iter=n_iter, cv=cv_folds,
                scoring=scoring, random_state=tuning_config.random_state,
                n_jobs=tuning_config.n_jobs,
                error_score=np.nan,
            )

        logger.info(
            f"[{family}] Starting search.fit: search_type="
            f"{'HalvingRandomSearchCV' if HALVING_AVAILABLE else 'RandomizedSearchCV'}, "
            f"n_candidates={n_iter}, cv={cv_folds}, scoring={primary_metric}"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                search.fit(X, y)
            except Exception as e:
                logger.warning(
                    f"[{family}] Sklearn search failed: {e}. "
                    f"Returning empty result."
                )
                return FamilyTuningResult(
                    model_family=family, best_params={}, best_score=0.0,
                    best_std=0.0, primary_metric=primary_metric,
                    all_trials=[], tuning_duration_seconds=0,
                    n_trials_completed=0, n_trials_pruned=0,
                    status="failed", error_message=str(e),
                )

        all_trials = []
        cv_results = search.cv_results_
        n_trials_actual = len(cv_results["params"])
        for i in range(n_trials_actual):
            all_trials.append(TrialResult(
                trial_number=i, model_family=family,
                hyperparameters=cv_results["params"][i],
                cv_mean=float(cv_results["mean_test_score"][i]),
                cv_std=float(cv_results["std_test_score"][i]),
                primary_metric=primary_metric, secondary_metrics={},
                fit_time_seconds=float(cv_results.get("mean_fit_time", [0] * n_trials_actual)[i]),
            ))

        if HYPERPARAMETER_TRIALS_TOTAL:
            HYPERPARAMETER_TRIALS_TOTAL.labels(
                model_id=self.model_id, model_family=family, status="completed"
            ).inc(n_trials_actual)

        best_score = float(search.best_score_)
        logger.info(
            f"[{family}] search.fit completed: best_score={best_score:.6f}, "
            f"best_params={search.best_params_}, "
            f"n_trials_actual={len(cv_results['params'])}"
        )
        # If every trial's mean was nan (e.g. PR-AUC on folds with no
        # positives), sklearn returns nan as best_score. Report as failed.
        if np.isnan(best_score):
            logger.warning(
                f"[{family}] All trials produced nan scores "
                f"(likely single-class folds with PR-AUC scoring)."
            )
            return FamilyTuningResult(
                model_family=family, best_params={}, best_score=0.0,
                best_std=0.0, primary_metric=primary_metric,
                all_trials=all_trials, tuning_duration_seconds=0,
                n_trials_completed=n_trials_actual, n_trials_pruned=0,
                status="failed",
                error_message="All trials produced nan scores — folds may be too small for the minority class.",
            )

        return FamilyTuningResult(
            model_family=family, best_params=search.best_params_,
            best_score=float(search.best_score_),
            best_std=float(cv_results["std_test_score"][search.best_index_]),
            primary_metric=primary_metric, all_trials=all_trials,
            tuning_duration_seconds=0, n_trials_completed=n_trials_actual,
            n_trials_pruned=0, status="completed",
        )

    # -- Selection & Factory ------------------------------------------------

    def _select_best(self, family_results: Dict[str, FamilyTuningResult]):
        """Select the best model family from tuning results."""
        valid = {n: r for n, r in family_results.items() if r.status == "completed" and r.best_params}
        logger.info(
            f"_select_best: {len(family_results)} families evaluated, "
            f"{len(valid)} valid: {list(valid.keys())}"
        )
        for name, r in family_results.items():
            logger.debug(
                f"  {name}: status={r.status}, best_score={r.best_score:.6f}, "
                f"best_std={r.best_std:.6f}, n_trials={r.n_trials_completed}"
            )
        if not valid:
            logger.warning("_select_best: no valid results — all families failed")
            return "", None
        sorted_families = sorted(valid.items(), key=lambda x: (x[1].best_score, -x[1].best_std), reverse=True)
        logger.info(f"_select_best: winner={sorted_families[0][0]}, score={sorted_families[0][1].best_score:.6f}")
        return sorted_families[0]

    def _build_model_factory(self, family: str, is_classification:bool):
        """Build a model factory function for the given family."""
        factory_info = MODEL_FACTORIES[family]
        model_cls = factory_info["classifier"] if is_classification else factory_info["regressor"]
        if model_cls is None:
            raise HyperparameterTuningError(
                f"Model family '{family}' does not support "
                f"{'classification' if is_classification else 'regression'}."
            )   

        def factory(params: Dict[str, Any]) -> ModelProtocol:
            return model_cls(**{k: v for k, v in params.items() if v is not None})

        return factory
