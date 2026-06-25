# src/training/trainer.py
"""
Model trainer with full MLflow experiment tracking.
Every training run is logged, reproducible, and visible in MLflow UI.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable, Self

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from config.settings import get_settings
from src.common.enums import ModelStatus
from src.common.exceptions import TrainingError
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    TRAINING_RUNS_TOTAL, TRAINING_DURATION,
    MODEL_CV_SCORE, MODEL_VALIDATION_METRIC,
)

logger = logging.getLogger("ml_platform.training.trainer")

class ModelProtocol(BaseEstimator):
    """
    Structural type matching ANY sklearn-compatible estimator.
    Includes BaseEstimator interface + ML interface.
    """
    # --- ML methods (Pylance needs these) ---
    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs: Any) -> Self: ...
    def predict(self, X: ArrayLike) -> Any: ...
    def score(self, X: ArrayLike, y: ArrayLike) -> float: ...
    # --- BaseEstimator methods (runtime inheritance) ---
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Self: ...
    
   # Add sklearn-specific methods so they're not hidden
    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...
    def __getattr__(self, name: str) -> Any: ...


@dataclass
class TrainingConfig:
    model_type: str
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    target_column: str
    cv_folds: int = 10
    cv_scoring: str = "f1_weighted"
    random_state: int = 42
    test_size: float = 0.2
    stratify: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    is_imbalanced: bool = False
    class_weight: Optional[str] = None  # "balanced", "balanced_subsample", or None
    scale_pos_weight: float = 1.0  # For XGBoost/LightGBM: neg/pos ratio
    strategy: Optional[Any] = None  # ImbalanceStrategy — set by AutoTrainer
    # -- Phase 2: Advanced techniques -------------------------------------------
    enable_feature_engineering: bool = False  # Add interaction/stat features
    enable_calibration: bool = False          # Calibrate probabilities (binary)
    enable_ensemble: bool = False             # BalancedBagging for extreme imb
    feature_pipeline: Optional[list] = None    # Fitted (name, transformer) pairs
    optimal_threshold: Optional[float] = None  # F1-optimal decision threshold


@dataclass
class TrainingResult:
    """Complete result of a training run."""
    run_id: str
    model_id: str
    model_version: str
    mlflow_run_id: str
    status: ModelStatus
    training_config: TrainingConfig
    metrics: dict[str, float]
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    feature_importance: dict[str, float]
    training_duration_seconds: float
    training_samples: int
    validation_samples: int
    data_hash: str
    model_artifact_path: str
    preprocessing_artifact_path: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id":                    self.run_id,
            "model_id":                  self.model_id,
            "model_version":             self.model_version,
            "mlflow_run_id":             self.mlflow_run_id,
            "status":                    self.status.value,
            "metrics":                   {k: round(v, 6) for k, v in self.metrics.items()},
            "cv_mean":                   round(self.cv_mean, 6),
            "cv_std":                    round(self.cv_std, 6),
            "training_duration_seconds": round(self.training_duration_seconds, 2),
            "training_samples":          self.training_samples,
            "timestamp":                 self.timestamp.isoformat(),
        }


class ModelTrainer:
    """
    Production trainer with MLflow integration.
    Every run is tracked: params, metrics, artifacts, model signature.
    """

    def __init__(
        self,
        model_factory: Callable[[dict[str, Any]], ModelProtocol],
        preprocessing_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        model_id: str = "",
        pipeline_run_id: str = "",
    ):
        self.settings         = get_settings()
        self.model_factory    = model_factory
        self.preprocessing_fn = preprocessing_fn
        self.model_id         = model_id
        self.pipeline_run_id  = pipeline_run_id

        # Configure MLflow
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        self._experiment_name = (
            f"{self.settings.mlflow.experiment_prefix}/{model_id or 'default'}_v2"
        )
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        artifact_location = getattr(self.settings.mlflow, 'artifact_root', '/app/artifacts/mlflow')

        if experiment is None:
            # Create with correct artifact location
            mlflow.create_experiment(
                name=self._experiment_name,
                artifact_location=artifact_location
            )
        mlflow.set_experiment(self._experiment_name)

    def train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        model_id: str,
        model_version: str,
    ) -> TrainingResult:
        import uuid
        from sklearn.model_selection import train_test_split

        run_id     = str(uuid.uuid4())
        start_time = time.time()

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.TRAINING_STARTED,
            step_name="training",
            title="🏋️ Training Started",
            message=(
                f"Model: {model_id} v{model_version} | "
                f"Samples: {len(df)} | "
                f"Features: {len(config.feature_columns)}"
            ),
            model_id=model_id,
            model_version=model_version,
            data={
                "model_type":   config.model_type,
                "hyperparams":  config.hyperparameters,
                "cv_folds":     config.cv_folds,
                "feature_count": len(config.feature_columns),
                "sample_count": len(df),
            },
        ))

        logger.info(
            f"Training: run_id={run_id}, model={model_id}, version={model_version}"
        )

        mlflow_run_id = ""

        try:
            with mlflow.start_run(run_name=f"{model_id}_v{model_version}") as run:
                mlflow_run_id = run.info.run_id

                # Log everything to MLflow
                mlflow.log_params(config.hyperparameters)
                # Log training config as params
                mlflow.log_params({
                    "model_type":     config.model_type,
                    "model_version":  model_version,
                    "model_id":       model_id,
                    "cv_folds":       config.cv_folds,
                    "cv_scoring":     config.cv_scoring,
                    "random_state":   config.random_state,
                    "test_size":       config.test_size,
                    "feature_count":   len(config.feature_columns),
                    "features":        ",".join(config.feature_columns),
                })

                for k, v in config.tags.items():
                    mlflow.set_tag(k, v)
                mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)
                mlflow.set_tag("model_id", model_id)
                mlflow.set_tag("model_version", model_version)

                # Data hash
                data_hash = self._compute_data_hash(df)
                mlflow.log_param("data_hash", data_hash)

                # Preprocessing if not yet haven done
                if self.preprocessing_fn:
                    df = self.preprocessing_fn(df)

                X = df[config.feature_columns].copy()
                y = df[config.target_column].copy()

                # ── Imbalance-aware hyperparameter injection ──────────────────
                # When the dataset is imbalanced, inject class_weight / scale_pos_weight
                # into the model hyperparameters so the model learns to care about
                # the minority class instead of just predicting the majority.
                effective_hyperparams = dict(config.hyperparameters)
                strategy = config.strategy
                if strategy is not None:
                    # Log the full strategy to MLflow for reproducibility.
                    # NOTE: class_weight/scale_pos_weight are already in
                    # config.hyperparameters (logged at line 192). Only log
                    # them from the strategy if they were NOT already present,
                    # to avoid MLflow's "param already logged" error.
                    mlflow.log_param("dataset_category", strategy.category)
                    mlflow.log_param("is_imbalanced", strategy.needs_imbalance_handling)
                    if "class_weight" not in config.hyperparameters:
                        mlflow.log_param("class_weight", strategy.class_weight or "none")
                    if "scale_pos_weight" not in config.hyperparameters:
                        mlflow.log_param("scale_pos_weight", strategy.scale_pos_weight)
                    mlflow.log_param("use_sample_weight", strategy.use_sample_weight)
                    if strategy.class_weight and "class_weight" not in effective_hyperparams:
                        effective_hyperparams["class_weight"] = strategy.class_weight
                    if strategy.scale_pos_weight > 1.0 and "scale_pos_weight" not in effective_hyperparams:
                        effective_hyperparams["scale_pos_weight"] = strategy.scale_pos_weight
                    logger.info(
                        f"Strategy active: category={strategy.category}, "
                        f"class_weight={strategy.class_weight}, "
                        f"spw={strategy.scale_pos_weight:.1f}"
                    )
                elif config.is_imbalanced:
                    # Fallback: old boolean flag behavior
                    if config.class_weight and "class_weight" not in effective_hyperparams:
                        effective_hyperparams["class_weight"] = config.class_weight
                    if config.scale_pos_weight > 1.0 and "scale_pos_weight" not in effective_hyperparams:
                        effective_hyperparams["scale_pos_weight"] = config.scale_pos_weight
                    mlflow.log_param("is_imbalanced", True)
                    mlflow.log_param("class_weight", config.class_weight or "none")
                    mlflow.log_param("scale_pos_weight", config.scale_pos_weight)
                else:
                    mlflow.log_param("is_imbalanced", False)
                    mlflow.log_param("dataset_category", "balanced")

                # Stratify if classification and not too many classes
                stratify_col = (
                    y if config.stratify and y.nunique() < 50 else None
                )

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    shuffle = True,
                    stratify=stratify_col,
                )

                mlflow.log_param("training_samples",   len(X_train))
                mlflow.log_param("validation_samples", len(X_val))

                # ── Feature Engineering (Phase 2) ───────────────────────────
                # Add statistical and interaction features BEFORE model fitting.
                # Pipeline is fit on X_train only (no leakage), then applied to both.
                if config.enable_feature_engineering and strategy is not None:
                    from src.training.feature_engineering import build_feature_pipeline
                    try:
                        import pandas as pd
                        X_train_df = pd.DataFrame(X_train, columns=config.feature_columns)
                        X_val_df = pd.DataFrame(X_val, columns=config.feature_columns)
                        steps = build_feature_pipeline(
                            n_features=X_train.shape[1],
                            n_samples=X_train.shape[0],
                            category=strategy.category,
                        )
                        for name, transformer in steps:
                            X_train_df = transformer.fit_transform(X_train_df, y_train)
                            X_val_df = transformer.transform(X_val_df)
                        X_train = X_train_df.values
                        X_val = X_val_df.values
                        config.feature_pipeline = steps
                        mlflow.log_param("feature_engineering", True)
                        mlflow.log_param("n_features_after_fe", X_train.shape[1])
                        logger.info(
                            f"Feature engineering applied: {X_train.shape[1]} features "
                            f"(from {len(config.feature_columns)}), steps={[s[0] for s in steps]}"
                        )
                    except Exception as e:
                        logger.warning(f"Feature engineering failed, using raw features: {e}")

                # ── Build model with effective (possibly imbalance-adjusted) hyperparams ──
                # XGBoost and LightGBM do not accept class_weight in their
                # constructors — silently including it produces warnings or
                # errors.  Strip it for those model families.
                model_type_lower = config.model_type.lower()
                if model_type_lower in ("xgboost", "lightgbm"):
                    effective_hyperparams.pop("class_weight", None)
                model: ModelProtocol = self.model_factory(effective_hyperparams)

                # ── Ensemble Wrapper (Phase 2) ──────────────────────────────
                # For extreme/severe imbalance, wrap base model with BalancedBagging.
                # This trains multiple sub-models on balanced under-samples.
                if config.enable_ensemble and strategy is not None:
                    from src.training.ensemble import get_ensemble_for_category
                    try:
                        ensemble = get_ensemble_for_category(
                            strategy.category, model, random_state=config.random_state
                        )
                        if ensemble is not None:
                            model = ensemble
                            mlflow.log_param("ensemble", True)
                            logger.info(f"Ensemble wrapper applied: {type(ensemble).__name__}")
                    except Exception as e:
                        logger.warning(f"Ensemble wrapper failed, using single model: {e}")

                # Cross-validation: use strategy's fold count when available,
                # otherwise fall back to config with adaptive reduction
                if strategy is not None:
                    effective_cv_folds = strategy.cv_folds
                else:
                    effective_cv_folds = config.cv_folds
                    if config.is_imbalanced and y_train.nunique() == 2:
                        min_minority = int(y_train.value_counts().min())
                        max_folds = max(3, min_minority // 2)
                        if effective_cv_folds > max_folds:
                            effective_cv_folds = max_folds
                            logger.info(
                                f"Reduced CV folds from {config.cv_folds} to "
                                f"{effective_cv_folds} for imbalanced data "
                                f"(minority count={min_minority})"
                            )

                # sklearn has no "pr_auc" string alias — convert to
                # average_precision (which IS PR-AUC) as a callable scorer.
                cv_scoring = config.cv_scoring
                if cv_scoring == "pr_auc":
                    from sklearn.metrics import average_precision_score, make_scorer
                    cv_scoring = make_scorer(
                        average_precision_score,
                        response_method="predict_proba",
                    )

                # cross_val_score sees BaseEstimator at runtime (duck typing)
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=effective_cv_folds,
                    scoring=cv_scoring,
                    n_jobs=-1,
                )

                # Degenerate folds (no positives) produce nan scores.  Use
                # errstate to suppress the noisy All-NaN slice warning; the
                # nanmean path below handles them numerically.
                with np.errstate(invalid="ignore", divide="ignore"):
                    cv_mean = float(np.nanmean(cv_scores))
                    cv_std  = float(np.nanstd(cv_scores))

                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std",  cv_std)
                mlflow.log_param("effective_cv_folds", effective_cv_folds)

                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i+1}", float(score))


                # Full training — use sample_weight for imbalanced data on
                # models that support it (GradientBoosting, etc.)
                #
                # CRITICAL: Do NOT apply sample_weight if the model already
                # has built-in imbalance handling (scale_pos_weight for XGB/LGB,
                # class_weight for RF/GBM).  Applying both causes
                # over-correction: the model predicts EVERYTHING as positive,
                # giving recall=100% but precision=base_rate (disaster).
                fit_kwargs: dict[str, Any] = {}
                model_has_builtin_imbalance = (
                    "scale_pos_weight" in effective_hyperparams
                    or "class_weight" in effective_hyperparams
                )
                use_sw = (
                    not model_has_builtin_imbalance
                    and (
                        (strategy is not None and strategy.use_sample_weight)
                        or (strategy is None and config.is_imbalanced)
                    )
                    and y_train.nunique() == 2
                )
                if use_sw:
                    n_total = len(y_train)
                    # Handle non-0/1 labels: minority class gets higher weight
                    vc = y_train.value_counts()
                    n_minority = int(vc.min())
                    n_majority = int(vc.max())
                    minority_label = vc.idxmin()
                    if n_minority > 0 and n_majority > 0:
                        sample_weight = np.where(
                            y_train == minority_label,
                            n_total / (2.0 * n_minority),
                            n_total / (2.0 * n_majority),
                        )
                        fit_kwargs["sample_weight"] = sample_weight
                        logger.info(
                            f"Using sample_weight for imbalanced fit: "
                            f"minority_weight={n_total/(2*n_minority):.2f}, "
                            f"majority_weight={n_total/(2*n_majority):.2f}"
                        )
                    else:
                        logger.warning(
                            f"Skipping sample_weight: fold has only "
                            f"{'minority' if n_minority > 0 else 'majority'} "
                            f"samples (n_minority={n_minority}, n_majority={n_majority})."
                        )
                elif model_has_builtin_imbalance:
                    logger.info(
                        "Skipping sample_weight: model has builtin imbalance "
                        "handling (scale_pos_weight or class_weight)."
                    )

                # Merge any extra fit kwargs from the strategy
                if strategy is not None and strategy.fit_extras:
                    fit_kwargs.update(strategy.fit_extras)

                # XGBoost 2.0+ and LightGBM 4.0+ moved early_stopping_rounds
                # from .fit() to the constructor.  When it's present:
                # 1. Remove it from fit_kwargs (it's not accepted there).
                # 2. Add it to constructor params and rebuild the model.
                # 3. Add eval_set to fit_kwargs (required for early stopping).
                esr_value = fit_kwargs.pop("early_stopping_rounds", None)
                if esr_value is not None:
                    import inspect
                    fit_sig = inspect.signature(model.fit)
                    if "early_stopping_rounds" not in fit_sig.parameters:
                        effective_hyperparams["early_stopping_rounds"] = esr_value
                        model = self.model_factory(effective_hyperparams)
                    else:
                        # Old API (sklearn GradientBoosting) — put it back
                        fit_kwargs["early_stopping_rounds"] = esr_value
                    # Provide eval_set for models that need it (XGBoost, LightGBM)
                    if "eval_set" not in fit_kwargs:
                        fit_kwargs["eval_set"] = [(X_val, y_val)]

                model.fit(X_train, y_train, **fit_kwargs)

                # ── Probability Calibration (Phase 2) ───────────────────────
                # Calibrate predicted probabilities using validation data.
                # This improves threshold tuning and PR-AUC.
                if (config.enable_calibration
                    and hasattr(model, "predict_proba")
                    and y_train.nunique() == 2):
                    from src.training.calibration import calibrate_model
                    try:
                        calibrated = calibrate_model(model, X_val, y_val)
                        if calibrated is not model:
                            model = calibrated
                            mlflow.log_param("calibration", True)
                    except Exception as e:
                        logger.warning(f"Calibration failed, using uncalibrated model: {e}")

                # Threshold tuning for imbalanced classification.
                # The default 0.5 threshold is almost never optimal when
                # classes are imbalanced.  Scan over candidate thresholds
                # on the validation set and pick the one that maximizes F1.
                optimal_threshold = None
                if (
                    strategy is not None
                    and strategy.is_classification
                    and y_train.nunique() == 2
                    and hasattr(model, "predict_proba")
                ):
                    try:
                        from sklearn.metrics import f1_score
                        y_prob_val = model.predict_proba(X_val)[:, 1]
                        best_f1 = -1.0
                        # Search thresholds from 0.05 to 0.95
                        for t in np.arange(0.05, 0.96, 0.05):
                            y_pred_t = (y_prob_val >= t).astype(int)
                            f1_t = f1_score(y_val, y_pred_t, zero_division=0)
                            if f1_t > best_f1:
                                best_f1 = f1_t
                                optimal_threshold = t
                        if optimal_threshold is not None:
                            mlflow.log_param(
                                "optimal_threshold", optimal_threshold
                            )
                            logger.info(
                                f"Optimal threshold: {optimal_threshold:.2f} "
                                f"(val F1={best_f1:.4f})"
                            )
                    except Exception as e:
                        logger.warning(f"Threshold tuning failed: {e}")

                # Validation metrics
                metrics = self._compute_metrics(
                    model, X_val, y_val, config, strategy,
                    threshold=optimal_threshold,
                )
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Feature importance
                feature_importance = self._extract_feature_importance(
                    model, config.feature_columns
                )
                import os
                if feature_importance:
                    for fname, fval in feature_importance.items():
                        mlflow.log_metric(f"importance_{fname}", fval)
                        

                # Save model to MLflow (with signature)
                signature = None  # ensure fallback path always has a binding
                try:
                    from mlflow.models.signature import infer_signature
                    signature = infer_signature(X_train, model.predict(X_train))
                    # MLflow 3.x: use `name` instead of deprecated `artifact_path`,
                    # and drop `input_sample` (no longer accepted).
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        signature=signature,
                        registered_model_name=f"{model_id}",
                        metadata={
                            "model_id": model_id,
                            "model_version": model_version,
                            "cv_mean": str(cv_mean),
                            "cv_std": str(cv_std),
                        },
                    )
                    # Log the model URI for easy retrieval
                    mlflow.set_tag("model_uri", model_info.model_uri)
                    logger.info(f"Model registered: {model_info.model_uri}")
                except Exception as e:
                    logger.warning(f"MLflow model logging failed: {e}")
                    # Fallback: log model without registry
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=signature,
                    )
                # Stored artifacts at local path (for registry) - MLflow also has its own artifact storage    
                artifact_base = f"/app/artifacts/training"
                os.makedirs(artifact_base, exist_ok=True)
                # 1. Save feature importance as JSON artifact
                importance_path = f"{artifact_base}/{model_id}_{model_version}_feature_importance.json"
                import json
                with open(importance_path, "w") as f:
                    json.dump(feature_importance, f, indent=2)

                # Save local artifact
                model_path = self._save_model_artifact(model, model_id, model_version)

                preprocessing_path = self._save_preprocessing_artifact(
                    model_id, model_version
                )
                # Save training config as JSON artifact
                config_path = f"{artifact_base}/{model_id}_{model_version}_training_config.json"
                with open(config_path, "w") as f:
                    json.dump({
                        "model_type": config.model_type,
                        "hyperparameters": config.hyperparameters,
                        "feature_columns": config.feature_columns,
                        "target_column": config.target_column,
                        "cv_folds": config.cv_folds,
                        "cv_scoring": config.cv_scoring,
                        "random_state": config.random_state,
                        "test_size": config.test_size,
                    }, f, indent=2)

                # Save metrics summary as JSON artifact
                metrics_path = f"{artifact_base}/{model_id}_{model_version}_metrics_summary.json"
                with open(metrics_path, "w") as f:
                    json.dump({
                        "metrics": metrics,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "cv_scores": cv_scores.tolist(),
                        "training_samples": len(X_train),
                        "validation_samples": len(X_val),
                        "data_hash": data_hash,
                        "model_id": model_id,
                        "model_version": model_version,
                    }, f, indent=2)

                duration = time.time() - start_time
                mlflow.log_metric("training_duration_seconds", duration)

        except Exception as e:
            duration = time.time() - start_time
            TRAINING_RUNS_TOTAL.labels(model_id=model_id, status="failed").inc()
            TRAINING_DURATION.labels(model_id=model_id).observe(duration)

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.TRAINING_FAILED,
                step_name="training",
                title="❌ Training Failed",
                message=str(e),
                model_id=model_id,
                status="failed",
                severity="error",
                data={"error": str(e), "duration_seconds": round(duration, 2)},
            ))

            logger.error(f"Training failed: {e}", exc_info=True)
            raise TrainingError(
                f"Training failed for '{model_id}' v{model_version}: {e}",
                details={"run_id": run_id, "duration": duration},
            ) from e

        duration = time.time() - start_time

        # Prometheus
        TRAINING_RUNS_TOTAL.labels(model_id=model_id, status="success").inc()
        TRAINING_DURATION.labels(model_id=model_id).observe(duration)
        for metric_name, metric_value in metrics.items():
            MODEL_VALIDATION_METRIC.labels(
                model_id=model_id, version=model_version, metric=metric_name
            ).set(metric_value)
        MODEL_CV_SCORE.labels(
            model_id=model_id, version=model_version, metric="cv_mean"
        ).set(cv_mean)

        result = TrainingResult(
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            mlflow_run_id=mlflow_run_id,
            status=ModelStatus.TRAINING,
            training_config=config,
            metrics=metrics,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_mean,
            cv_std=cv_std,
            feature_importance=feature_importance,
            training_duration_seconds=duration,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            data_hash=data_hash,
            model_artifact_path=model_path,
            preprocessing_artifact_path=preprocessing_path,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "mlflow_experiment": self._experiment_name,
                "mlflow_run_id":     mlflow_run_id,
                "sklearn_version":   self._get_sklearn_version(),
                "feature_count":     len(config.feature_columns),
            },
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.TRAINING_COMPLETED,
            step_name="training",
            title="✅ Training Complete",
            message=(
                f"Model {model_id} v{model_version} | "
                f"CV: {cv_mean:.4f} ±{cv_std:.4f} | "
                f"Duration: {duration:.1f}s"
            ),
            model_id=model_id,
            model_version=model_version,
            duration_ms=duration * 1000,
            data={
                **result.to_dict(),
                "mlflow_run_id":     mlflow_run_id,
                "mlflow_ui_url":     f"{self.settings.mlflow.tracking_uri}/#/runs/{mlflow_run_id}",
                "feature_importance": dict(list(feature_importance.items())[:10]),
            },
        ))

        logger.info(
            f"Training complete: {model_id} v{model_version}, "
            f"cv={cv_mean:.4f}, duration={duration:.1f}s, "
            f"mlflow_run={mlflow_run_id}"
        )

        return result

    def _compute_metrics(
        self,
        model: ModelProtocol,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: TrainingConfig,
        strategy,
        threshold: Optional[float] = None,
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        )

        # Use tuned threshold for binary classification if available.
        # For imbalanced data, the default argmax (0.5) is suboptimal.
        if (
            threshold is not None
            and hasattr(model, "predict_proba")
            and y_val.nunique() == 2
        ):
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = model.predict(X_val)
        metrics: dict[str, float] = {}

        is_classification = hasattr(model, "predict_proba") or y_val.nunique() <= 20

        if is_classification:
            n_classes = y_val.nunique()
            is_binary  = n_classes == 2
            # For binary: use "binary" average (focuses on positive class)
            # For multiclass: use "macro" (equal weight to all classes)
            avg = "binary" if is_binary else "macro"

            metrics["accuracy"]  = float(accuracy_score(y_val, y_pred))
            metrics["f1_score"]  = float(
                f1_score(y_val, y_pred, average=avg, zero_division=0)
            )
            metrics["precision"] = float(
                precision_score(y_val, y_pred, average=avg, zero_division=0)
            )
            metrics["recall"]    = float(
                recall_score(y_val, y_pred, average=avg, zero_division=0)
            )
            # For imbalanced data, also report F1 with macro average
            # so the user sees per-class performance
            is_imbalanced = (
                (strategy is not None and strategy.needs_imbalance_handling)
                or (strategy is None and config.is_imbalanced)
            )
            if is_imbalanced and is_binary:
                metrics["f1_macro"] = float(
                    f1_score(y_val, y_pred, average="macro", zero_division=0)
                )
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_val)
                    if is_binary:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_val, y_prob[:, 1])
                        )
                        # PR-AUC is the gold standard for imbalanced data
                        # It focuses on minority class performance
                        from sklearn.metrics import average_precision_score
                        metrics["pr_auc"] = float(
                            average_precision_score(y_val, y_prob[:, 1])
                        )
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y_val, y_prob,
                                multi_class="ovr", average="macro"
                            )
                        )
                except Exception:
                    pass
        else:
            metrics["rmse"] = float(
                np.sqrt(mean_squared_error(y_val, y_pred))
            )
            metrics["mae"]  = float(mean_absolute_error(y_val, y_pred))
            metrics["r2"]   = float(r2_score(y_val, y_pred))

        return metrics

    def _extract_feature_importance(
        self, model: ModelProtocol, feature_names: list[str]
    ) -> dict[str, float]:
        importance: dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).flatten()
            if len(coefs) == len(feature_names):
                for name, coef in zip(feature_names, coefs):
                    importance[name] = float(coef)
        return dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

    def _save_model_artifact(
        self, model: BaseEstimator, model_id: str, version: str
    ) -> str:
        models_dir = self.settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"{model_id}_{version}.joblib"
        joblib.dump(model, path)
        logger.info(f"Artifact: {path}")
        return str(path)

    def _save_preprocessing_artifact(
        self, model_id: str, version: str
    ) -> str:
        # Save alongside the model artifact (same models_dir), not under
        # registry_dir which would create an invalid nested path.
        artifact_dir = self.settings.models_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{model_id}_{version}_preprocessing.joblib"
        if self.preprocessing_fn:
            joblib.dump(self.preprocessing_fn, path)
        return str(path)

    @staticmethod
    def _compute_data_hash(df: pd.DataFrame) -> str:
        content = np.array(pd.util.hash_pandas_object(df).values).tobytes()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def _get_sklearn_version() -> str:
        import sklearn
        return sklearn.__version__