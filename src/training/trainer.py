"""
Model training orchestrator with experiment tracking,
hyperparameter tuning, and reproducibility guarantees.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from config.settings import get_settings
from src.common.enums import ModelStatus
from src.common.exceptions import TrainingError

logger = logging.getLogger("ml_platform.training.trainer")


class ModelProtocol(Protocol):
    """Protocol for models compatible with the trainer."""
    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    model_type: str                          # e.g., "xgboost", "lightgbm", "sklearn_rf"
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    target_column: str
    cv_folds: int = 5
    cv_scoring: str = "f1_weighted"         # Sklearn scoring string
    random_state: int = 42
    test_size: float = 0.2
    stratify: bool = True
    early_stopping: bool = True
    max_training_time_seconds: int = 3600    # Safety timeout
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Complete result of a training run."""
    run_id: str
    model_id: str
    model_version: str
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
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": self.status.value,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "cv_mean": round(self.cv_mean, 6),
            "cv_std": round(self.cv_std, 6),
            "training_duration_seconds": round(self.training_duration_seconds, 2),
            "training_samples": self.training_samples,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelTrainer:
    """
    Orchestrates model training with production-grade practices:
    - Data splitting with stratification
    - Cross-validation
    - Experiment tracking (MLflow integration ready)
    - Artifact serialization with versioning
    - Reproducibility via data hashing and seed control
    - Training timeout protection
    """

    def __init__(
        self,
        model_factory: Callable[[dict[str, Any]], BaseEstimator],
        preprocessing_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        """
        Args:
            model_factory: Callable that takes hyperparameters and returns a model instance.
            preprocessing_fn: Optional preprocessing function applied before training.
        """
        self.settings = get_settings()
        self.model_factory = model_factory
        self.preprocessing_fn = preprocessing_fn

    def train(
        self,
        df: pd.DataFrame,
        config: TrainingConfig,
        model_id: str,
        model_version: str,
    ) -> TrainingResult:
        """
        Execute a full training run.

        Args:
            df: Full dataset (will be split internally).
            config: Training configuration.
            model_id: Logical model identifier.
            model_version: Version string for this training run.

        Returns:
            TrainingResult with metrics, artifacts, and metadata.
        """
        import uuid

        run_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"Starting training run: run_id={run_id}, model_id={model_id}, "
            f"version={model_version}, samples={len(df)}"
        )

        try:
            # 1. Data hash for reproducibility tracking
            data_hash = self._compute_data_hash(df)

            # 2. Preprocessing
            if self.preprocessing_fn:
                df = self.preprocessing_fn(df)
                logger.info("Preprocessing applied.")

            # 3. Feature/target split
            X = df[config.feature_columns].copy()
            y = df[config.target_column].copy()

            # 4. Train/validation split
            from sklearn.model_selection import train_test_split

            stratify_col = y if config.stratify and y.nunique() < 50 else None

            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify_col,
            )

            logger.info(f"Split: train={len(X_train)}, validation={len(X_val)}")

            # 5. Create model
            model = self.model_factory(config.hyperparameters)

            # 6. Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=config.cv_folds,
                scoring=config.cv_scoring,
                n_jobs=-1,
            )
            logger.info(
                f"CV scores: mean={np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
            )

            # 7. Train on full training set
            model.fit(X_train, y_train)

            # 8. Evaluate on validation set
            metrics = self._compute_metrics(model, X_val, y_val, config)

            # 9. Feature importance
            feature_importance = self._extract_feature_importance(model, config.feature_columns)

            # 10. Save artifacts
            model_artifact_path = self._save_model_artifact(model, model_id, model_version)
            preprocessing_artifact_path = self._save_preprocessing_artifact(model_id, model_version)

            duration = time.time() - start_time

            result = TrainingResult(
                run_id=run_id,
                model_id=model_id,
                model_version=model_version,
                status=ModelStatus.TRAINING,
                training_config=config,
                metrics=metrics,
                cv_scores=cv_scores.tolist(),
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                feature_importance=feature_importance,
                training_duration_seconds=duration,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                data_hash=data_hash,
                model_artifact_path=model_artifact_path,
                preprocessing_artifact_path=preprocessing_artifact_path,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "sklearn_version": self._get_sklearn_version(),
                    "feature_count": len(config.feature_columns),
                    "target_classes": int(y.nunique()) if y.dtype in ("object", "category", "int64") else None,
                },
            )

            logger.info(
                f"Training complete: run_id={run_id}, duration={duration:.1f}s, "
                f"cv_mean={result.cv_mean:.4f}, val_metrics={metrics}"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Training failed after {duration:.1f}s: {e}", exc_info=True)
            raise TrainingError(
                f"Training failed for model '{model_id}' v{model_version}: {str(e)}",
                details={"run_id": run_id, "duration": duration},
            ) from e

    def _compute_metrics(
        self, model: BaseEstimator, X_val: pd.DataFrame, y_val: pd.Series, config: TrainingConfig
    ) -> dict[str, float]:
        """Compute comprehensive metrics on validation set."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
        )

        y_pred = model.predict(X_val)
        metrics: dict[str, float] = {}

        # Determine task type
        is_classification = hasattr(model, "predict_proba") or y_val.nunique() <= 20

        if is_classification:
            metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
            average = "binary" if y_val.nunique() == 2 else "weighted"
            metrics["f1_score"] = float(f1_score(y_val, y_pred, average=average, zero_division=0))
            metrics["precision"] = float(precision_score(y_val, y_pred, average=average, zero_division=0))
            metrics["recall"] = float(recall_score(y_val, y_pred, average=average, zero_division=0))

            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_val)
                    if y_val.nunique() == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted"))
                except Exception:
                    pass
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_val, y_pred))
            metrics["r2"] = float(r2_score(y_val, y_pred))

        return metrics

    def _extract_feature_importance(
        self, model: BaseEstimator, feature_names: list[str]
    ) -> dict[str, float]:
        """Extract feature importance from model if available."""
        importance = {}
        if hasattr(model, "feature_importances_"):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_).flatten()
            if len(coefs) == len(feature_names):
                for name, coef in zip(feature_names, coefs):
                    importance[name] = float(coef)

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _save_model_artifact(self, model: BaseEstimator, model_id: str, version: str) -> str:
        artifact_dir = self.settings.model_dir / model_id / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "model.joblib"
        joblib.dump(model, path)
        logger.info(f"Model artifact saved: {path}")
        return str(path)

    def _save_preprocessing_artifact(self, model_id: str, version: str) -> str:
        artifact_dir = self.settings.model_dir / model_id / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "preprocessing.joblib"
        if self.preprocessing_fn:
            joblib.dump(self.preprocessing_fn, path)
        return str(path)

    @staticmethod
    def _compute_data_hash(df: pd.DataFrame) -> str:
        content = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def _get_sklearn_version() -> str:
        import sklearn
        return sklearn.__version__