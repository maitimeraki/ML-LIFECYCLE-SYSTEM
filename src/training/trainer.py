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
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
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

@runtime_checkable
class ModelProtocol(Protocol):
    """
    Structural type matching ANY sklearn-compatible estimator.
    Includes BaseEstimator interface + ML interface.
    """
    # --- ML methods (Pylance needs these) ---
    def fit(self, X: Any, y: Any, **kwargs: Any) -> ModelProtocol: ...
    def predict(self, X: Any) -> Any: ...
    def score(self, X: Any, y: Any) -> float: ...
    # --- BaseEstimator methods (runtime inheritance) ---
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> ModelProtocol: ...
    
    # Allow any attribute (silences Pylance for unknown attrs)
    def __getattr__(self, name:str)-> Any: ...

T = TypeVar('T', bound=ModelProtocol)

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
        model_factory: Callable[[dict[str, Any]], T],
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

        if experiment is None:
            # Create with correct artifact location
            mlflow.create_experiment(
                name=self._experiment_name,
                artifact_location="/tmp/mlflow-artifacts"  # ← Explicit
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
            with mlflow.start_run(run_name=f"{model_id}_{model_version}") as run:
                mlflow_run_id = run.info.run_id

                # Log everything to MLflow
                mlflow.log_params(config.hyperparameters)
                mlflow.log_param("model_type",     config.model_type)
                mlflow.log_param("model_version",  model_version)
                mlflow.log_param("cv_folds",       config.cv_folds)
                mlflow.log_param("cv_scoring",     config.cv_scoring)
                mlflow.log_param("random_state",   config.random_state)
                mlflow.log_param("test_size",       config.test_size)
                mlflow.log_param("feature_count",   len(config.feature_columns))
                mlflow.log_param("features",        ",".join(config.feature_columns))

                for k, v in config.tags.items():
                    mlflow.set_tag(k, v)
                mlflow.set_tag("pipeline_run_id", self.pipeline_run_id)

                # Data hash
                data_hash = self._compute_data_hash(df)
                mlflow.log_param("data_hash", data_hash)

                # Preprocessing if not yet haven done
                if self.preprocessing_fn:
                    df = self.preprocessing_fn(df)

                X = df[config.feature_columns].copy()
                y = df[config.target_column].copy()
                
                # Stratify if classification and not too many classes
                stratify_col = (
                    y if config.stratify and y.nunique() < 50 else None
                )

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=stratify_col,
                )

                mlflow.log_param("training_samples",   len(X_train))
                mlflow.log_param("validation_samples", len(X_val))

                # Cross-validation
                model: T = self.model_factory(config.hyperparameters)
                # cross_val_score sees BaseEstimator at runtime (duck typing)
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=config.cv_folds,
                    scoring=config.cv_scoring,
                    n_jobs=-1,
                )

                cv_mean = float(np.mean(cv_scores))
                cv_std  = float(np.std(cv_scores))

                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std",  cv_std)

                for i, score in enumerate(cv_scores):
                    mlflow.log_metric(f"cv_fold_{i+1}", float(score))
                
        
                # Pylance sees .fit(), .predict(), .score() via Protocol
                # Full training
                model.fit(X_train, y_train)

                # Validation metrics
                metrics = self._compute_metrics(model, X_val, y_val, config)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Feature importance
                feature_importance = self._extract_feature_importance(
                    model, config.feature_columns
                )
                if feature_importance:
                    for fname, fval in feature_importance.items():
                        mlflow.log_metric(f"importance_{fname}", fval)

                # Save model to MLflow (with signature)
                try:
                    from mlflow.models.signature import infer_signature
                    signature = infer_signature(X_train, model.predict(X_train))
                    mlflow.sklearn.log_model(model, name="model", signature=signature, registered_model_name=f"{model_id}")
                except Exception as e:
                    logger.warning(f"MLflow model logging failed: {e}")
                    mlflow.sklearn.log_model(model, name="model")

                # Save local artifact
                model_path = self._save_model_artifact(model, model_id, model_version)
                mlflow.log_artifact(model_path, artifact_path="local_artifacts")

                preprocessing_path = self._save_preprocessing_artifact(
                    model_id, model_version
                )

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
    ) -> dict[str, float]:
        from sklearn.metrics import (              
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        )

        y_pred  = model.predict(X_val)
        metrics: dict[str, float] = {}

        is_classification = hasattr(model, "predict_proba") or y_val.nunique() <= 20

        if is_classification:
            metrics["accuracy"]  = float(accuracy_score(y_val, y_pred))
            average              = "binary" if y_val.nunique() == 2 else "weighted"
            metrics["f1_score"]  = float(
                f1_score(y_val, y_pred, average=average, zero_division=0)
            )
            metrics["precision"] = float(          
                precision_score(y_val, y_pred, average=average, zero_division=0)
            )
            metrics["recall"]    = float(
                recall_score(y_val, y_pred, average=average, zero_division=0)
            )
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_val)
                    if y_val.nunique() == 2:
                        metrics["roc_auc"] = float(   
                            roc_auc_score(y_val, y_prob[:, 1])
                        )
                    else:
                        metrics["roc_auc"] = float( 
                            roc_auc_score(
                                y_val, y_prob,
                                multi_class="ovr", average="weighted"
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
        self, model: ModelProtocol, model_id: str, version: str
    ) -> str:
        artifact_dir = self.settings.registry_dir /self.settings.models_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{model_id}_{version}_model.joblib"
        # path = artifact_dir / model.joblib"
        joblib.dump(model, path)
        logger.info(f"Artifact: {path}")
        return str(path)

    def _save_preprocessing_artifact(
        self, model_id: str, version: str
    ) -> str:
        artifact_dir = self.settings.registry_dir / self.settings.models_dir
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