"""
Production prediction service.

Wired to:
  - ModelRegistry   → loads correct champion artifact (MLflow-first, file fallback)
  - ProductionDataProcessor → applies fitted processor at inference
  - Prometheus      → emits latency, error, cache metrics
  - EventBus        → emits prediction events for white-box visibility
  - LRU cache       → sha256-keyed, thread-safe

Champion hot-swap:
  Call predictor.reload_champion() to swap model without restart.
  Thread-safe via RLock — in-flight requests complete before swap.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.common.exceptions import (
    InputValidationError,
    ModelLoadError,
    PredictionError,
)
from src.data.processing import ProductionDataProcessor
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    PREDICTION_CACHE_HIT_RATE,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS_TOTAL,
)
from src.registry.model_registry import ModelRegistry

logger = logging.getLogger("ml_platform.serving.predictor")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionRequest:
    request_id:          str
    features:            dict[str, Any]
    model_version:       Optional[str] = None
    return_probabilities: bool = False
    timestamp:           datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class PredictionResponse:
    request_id:   str
    prediction:   Any
    probabilities: Optional[dict[str, float]]
    model_id:     str
    model_version: str
    latency_ms:   float
    cached:       bool
    timestamp:    datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id":    self.request_id,
            "prediction":    (
                self.prediction.item()
                if isinstance(self.prediction, np.generic)
                else self.prediction
            ),
            "probabilities": self.probabilities,
            "model_id":      self.model_id,
            "model_version": self.model_version,
            "latency_ms":    round(self.latency_ms, 2),
            "cached":        self.cached,
            "timestamp":     self.timestamp.isoformat(),
        }


@dataclass
class BatchPredictionResult:
    predictions:          list[Any]
    probabilities:        Optional[list[dict[str, float]]]
    model_id:             str
    model_version:        str
    total_latency_ms:     float
    per_sample_latency_ms: float
    batch_size:           int

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions":           self.predictions,
            "probabilities":         self.probabilities,
            "model_id":              self.model_id,
            "model_version":         self.model_version,
            "total_latency_ms":      round(self.total_latency_ms, 2),
            "per_sample_latency_ms": round(self.per_sample_latency_ms, 2),
            "batch_size":            self.batch_size,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LRU Cache
# ─────────────────────────────────────────────────────────────────────────────

class LRUCache:
    """
    Thread-safe LRU prediction cache.
    Key: SHA256 of sorted feature values + model_version.
    Invalidated on model hot-swap.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self.maxsize  = maxsize
        self._cache:  OrderedDict[str, Any] = OrderedDict()
        self._lock    = RLock()
        self._hits    = 0
        self._misses  = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits   = 0
            self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# Model Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ModelPredictor:
    """
    Production prediction service.

    Responsibilities:
      - Load champion from ModelRegistry (MLflow-first, file fallback)
      - Apply fitted processor at inference (zero data leakage)
      - Cache predictions (LRU, SHA256 key)
      - Emit Prometheus metrics on every request
      - Thread-safe model hot-swap
      - Log every prediction for monitoring
    """

    def __init__(
        self,
        model_id:          str,
        tag_version:       str,
        feature_columns:   list[str],
        model_dir:         Optional[Path] = None,
        processor_path:    Optional[str]  = None,
        cache_size:        int = 10_000,
        pipeline_run_id:   str = "",
    ) -> None:
        self.model_id        = model_id
        self.feature_columns = feature_columns
        self.model_dir       = model_dir or Path("/app/artifacts/models")
        self.tag_version     = tag_version
        self.pipeline_run_id = pipeline_run_id

        self._model:         Optional[Any] = None
        self._model_version: Optional[str] = None
        self._processor:     Optional[ProductionDataProcessor] = None
        self._lock           = RLock()
        self._cache          = LRUCache(maxsize=cache_size)

        # Metrics counters
        self._prediction_count  = 0
        self._error_count       = 0
        self._total_latency_ms  = 0.0

        # Load processor if provided (static, not tied to model version)
        if processor_path and Path(processor_path).exists():
            self._processor = ProductionDataProcessor.load(processor_path)
            logger.info(f"Processor loaded: {processor_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading — MLflow-first with file fallback
    # ─────────────────────────────────────────────────────────────────────────

    def load_champion(self) -> str:
        """
        Load current champion from ModelRegistry.
        MLflow-first: queries Production stage or champion alias.
        Falls back to local file registry if MLflow unavailable.
        
        Thread-safe — in-flight requests complete before swap.
        Returns loaded version string.
        """
        registry = ModelRegistry(model_id=self.model_id)
        
        # NEW: Use get_champion_artifact_path() — handles MLflow + fallback
        try:
            artifact_path, version = registry.get_champion_artifact_path(self.model_id, self.tag_version)
            logger.info(f"Champion artifact resolved: {artifact_path} (v{version})")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to resolve champion artifact for \'{self.model_id}\': {e}",
                details={"model_id": self.model_id, "error": str(e)},
            ) from e

        # Load model artifact
        self._load_artifact(artifact_path, version)

        # Load corresponding processor from shared volume
        self._load_processor_for_champion(registry, version)

        return version

    def _load_processor_for_champion(
        self, registry: ModelRegistry, version: str
    ) -> None:
        """Load processor matching the champion version."""
        # Try to get champion metadata for pipeline_run_id
        try:
            self.settings = get_settings()
            champion = registry.get_champion(self.model_id)
            pipeline_run_id = champion.tags.get("pipeline_run_id", "") if champion else ""
        except Exception:
            pipeline_run_id = ""

        # Build processor path
        processor_paths = [
            # Path from champion metadata
            Path(self.settings.processors_dir) / f"{self.model_id}_{self.tag_version}_processor.joblib",
            # Fallback: version-based naming
            Path("artifacts") / self.model_id / "processors" / f"{version}_processor.joblib",
            # Fallback: latest processor
            Path("artifacts") / self.model_id / "processors" / "latest_processor.joblib",
        ]

        for processor_path in processor_paths:
            if processor_path.exists():
                with self._lock:
                    self._processor = ProductionDataProcessor.load(str(processor_path))
                logger.info(f"Processor loaded for champion v{version}: {processor_path}")
                return

        logger.warning(
            f"No processor found for champion v{version}. "
            f"Tried: {[str(p) for p in processor_paths]}. "
            f"Raw features will be used directly."
        )

    def load_version(self, version: str) -> None:
        """Load a specific model version (for A/B testing or rollback)."""
        registry = ModelRegistry(model_id=self.model_id)
        artifact_path = registry.get_artifact_path(self.model_id, version)
        self._load_artifact(artifact_path, version)

    def _load_artifact(self, artifact_path: str, version: str) -> None:
        """Thread-safe model artifact loading with validation.
        
        What happens during the swap?
        - Request A (in-flight): Uses OLD model (already in predict())
        - Request B (waiting): Waits for lock, then uses NEW model
        - Request C (new): Uses NEW model
        Result: ZERO errors, ZERO dropped requests
        """
        if not Path(artifact_path).exists():
            raise ModelLoadError(
                f"Artifact not found: {artifact_path}",
                details={"model_id": self.model_id, "version": version},
            )

        try:
            new_model = joblib.load(artifact_path)

            if not hasattr(new_model, "predict"):
                raise ModelLoadError(
                    f"Artifact missing \'predict\' method: {artifact_path}"
                )

            with self._lock:
                self._model         = new_model
                self._model_version = version
                self._cache.clear()   # Invalidate cache on model change

            logger.info(
                f"Model loaded: {self.model_id} v{version} "
                f"from {artifact_path}"
            )

        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model: {exc}",
                details={
                    "model_id":  self.model_id,
                    "version":   version,
                    "path":      artifact_path,
                },
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Single prediction with caching, monitoring, and error handling.

        Flow:
          validate input
          → check cache
          → apply processor (if fitted)
          → model.predict()
          → cache result
          → emit metrics
        """
        start_time = time.perf_counter()

        try:
            with self._lock:
                if self._model is None:
                    raise PredictionError(
                        "No model loaded. Call load_champion() first."
                    )
                model         = self._model
                model_version = self._model_version or ""
                processor     = self._processor

            # Validate input
            self._validate_features(request.features)

            # Check cache
            cache_key     = self._cache_key(request.features, model_version)
            cached_result = self._cache.get(cache_key)

            if cached_result is not None:
                latency = (time.perf_counter() - start_time) * 1000
                self._emit_prediction_metrics(
                    model_version, latency, cached=True
                )
                return PredictionResponse(
                    request_id=request.request_id,
                    prediction=cached_result["prediction"],
                    probabilities=cached_result.get("probabilities"),
                    model_id=self.model_id,
                    model_version=model_version,
                    latency_ms=latency,
                    cached=True,
                )

            # Prepare features
            feature_df = pd.DataFrame([request.features])

            # Apply processor if available
            if processor is not None:
                processed_df, _ = processor.transform_single(feature_df)
                # Use processed feature columns
                feature_cols    = [
                    c for c in processed_df.columns
                    if c != processor.target_column
                ]
                X = processed_df[feature_cols]
            else:
                X = feature_df[self.feature_columns]

            # Predict
            prediction   = model.predict(X)[0]
            probabilities = None

            if request.return_probabilities and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                logger.info(f"Predicted probabilities: {proba}")
                if hasattr(model, "classes_"):
                    probabilities = {
                        str(cls): float(p)
                        for cls, p in zip(model.classes_, proba)
                    }
                else:
                    probabilities = {
                        str(i): float(p)
                        for i, p in enumerate(proba)
                    }

            # Convert numpy types
            if isinstance(prediction, np.generic):
                prediction = prediction.item()

            # Cache
            self._cache.put(
                cache_key,
                {"prediction": prediction, "probabilities": probabilities},
            )

            latency = (time.perf_counter() - start_time) * 1000
            self._prediction_count  += 1
            self._total_latency_ms  += latency

            self._emit_prediction_metrics(model_version, latency, cached=False)

            # Emit event for white-box monitoring
            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.PREDICTION_MADE,
                step_name="serving",
                title="🎯 Prediction",
                message=(
                    f"request={request.request_id} | "
                    f"v{model_version} | "
                    f"{latency:.1f}ms"
                ),
                model_id=self.model_id,
                model_version=model_version,
                data={
                    "latency_ms":    round(latency, 2),
                    "cached":        False,
                    "cache_hit_rate": self._cache.hit_rate,
                },
            ))

            return PredictionResponse(
                request_id=request.request_id,
                prediction=prediction,
                probabilities=probabilities,
                model_id=self.model_id,
                model_version=model_version,
                latency_ms=latency,
                cached=False,
            )

        except (InputValidationError, PredictionError):
            self._error_count += 1
            self._emit_error_metrics(
                self._model_version or "", time.perf_counter() - start_time
            )
            raise

        except Exception as exc:
            self._error_count += 1
            latency = (time.perf_counter() - start_time) * 1000
            self._emit_error_metrics(self._model_version or "", latency / 1000)

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.PREDICTION_ERROR,
                step_name="serving",
                title="❌ Prediction Error",
                message=str(exc),
                model_id=self.model_id,
                severity="error",
                data={"error": str(exc), "request_id": request.request_id},
            ))

            raise PredictionError(
                f"Prediction failed: {exc}",
                details={"request_id": request.request_id},
            ) from exc

    def predict_batch(
        self,
        features_list:        list[dict[str, Any]],
        return_probabilities: bool = False,
    ) -> BatchPredictionResult:
        """
        Batch prediction — efficient for bulk scoring.
        Does NOT use cache (batches are typically unique).
        """
        start_time = time.perf_counter()

        with self._lock:
            if self._model is None:
                raise PredictionError("No model loaded.")
            model         = self._model
            model_version = self._model_version or ""
            processor     = self._processor

        for features in features_list:
            self._validate_features(features)

        feature_df = pd.DataFrame(features_list)

        if processor is not None:
            processed_df, _ = processor.transform_single(feature_df)
            feature_cols    = [
                c for c in processed_df.columns
                if c != processor.target_column
            ]
            X = processed_df[feature_cols]
        else:
            X = feature_df[self.feature_columns]

        predictions   = model.predict(X).tolist()
        probabilities = None

        if return_probabilities and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if hasattr(model, "classes_"):
                probabilities = [
                    {str(cls): float(p) for cls, p in zip(model.classes_, row)}
                    for row in proba
                ]

        latency = (time.perf_counter() - start_time) * 1000

        self._prediction_count  += len(features_list)
        self._total_latency_ms  += latency

        PREDICTION_LATENCY.labels(
            model_id=self.model_id, model_version=model_version
        ).observe(latency)

        return BatchPredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            model_id=self.model_id,
            model_version=model_version,
            total_latency_ms=latency,
            per_sample_latency_ms=latency / len(features_list),
            batch_size=len(features_list),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_features(self, features: dict[str, Any]) -> None:
        """Validate input features against expected schema."""
        if self._processor is not None:
            # Processor handles schema — minimal validation here
            if not features:
                raise InputValidationError(
                    "Empty features dict",
                    details={"features": features},
                )
            return

        missing = set(self.feature_columns) - set(features.keys())
        if missing:
            raise InputValidationError(
                f"Missing features: {missing}",
                details={"missing": list(missing)},
            )

        for col in self.feature_columns:
            if features.get(col) is None:
                raise InputValidationError(
                    f"Feature \'{col}\' is null",
                    details={"feature": col},
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _emit_prediction_metrics(
        self,
        model_version: str,
        latency: float,
        cached: bool,
    ) -> None:
        status = "cached" if cached else "success"
        PREDICTION_REQUESTS_TOTAL.labels(
            model_id=self.model_id,
            model_version=model_version,
            status=status,
        ).inc()

        if not cached:
            PREDICTION_LATENCY.labels(
                model_id=self.model_id,
                model_version=model_version,
            ).observe(latency)

        PREDICTION_CACHE_HIT_RATE.labels(
            model_id=self.model_id
        ).set(self._cache.hit_rate)

    def _emit_error_metrics(
        self,
        model_version: str,
        latency_s: float,
    ) -> None:
        PREDICTION_REQUESTS_TOTAL.labels(
            model_id=self.model_id,
            model_version=model_version,
            status="error",
        ).inc()

    def _cache_key(
        self,
        features: dict[str, Any],
        model_version: str,
    ) -> str:
        """
        SHA256 cache key from features + model version.
        Includes model_version so cache is invalidated on hot-swap.
        """
        content = str(sorted(features.items())) + model_version
        return hashlib.sha256(content.encode()).hexdigest()

    # ─────────────────────────────────────────────────────────────────────────
    # Observability
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "model_id":         self.model_id,
            "model_version":    self._model_version,
            "prediction_count": self._prediction_count,
            "error_count":      self._error_count,
            "error_rate": (
                self._error_count
                / max(self._prediction_count + self._error_count, 1)
            ),
            "avg_latency_ms": (
                self._total_latency_ms / max(self._prediction_count, 1)
            ),
            "cache_hit_rate":   self._cache.hit_rate,
            "cache_size":       self._cache.size,
        }