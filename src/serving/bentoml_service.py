# src/serving/bentoml_service.py
"""
BentoML production serving service.

Wired to:
  ModelPredictor  → prediction logic + caching + metrics
  ModelRegistry   → champion discovery at startup
  Prometheus      → /metrics endpoint (BentoML built-in)
  EventBus        → white-box event emission

Endpoints:
  POST /predict           → single prediction
  POST /predict_batch     → batch prediction
  GET  /health            → liveness
  GET  /model/info        → current champion metadata
  GET  /model/metrics     → serving metrics
  POST /model/reload      → hot-swap champion (no restart)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.observability.event_bus import event_bus
from src.observability.prometheus_handler import register_prometheus_handler
from src.registry.model_registry import ModelRegistry
from src.serving.predictor import ModelPredictor, PredictionRequest

logger = logging.getLogger("ml_platform.serving.bentoml")

# Wire Prometheus handler once at module load
register_prometheus_handler(event_bus)

MODEL_ID = get_settings().model_id if hasattr(get_settings(), "model_id") else "customer_churn_model"


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictInput(BaseModel):
    features:             dict[str, Any] = Field(
        ...,
        description="Feature name → value",
        examples=[{
            "feature_1": 10.2,
            "feature_2": 4.8,
            "feature_3": 2.1,
            "feature_5": 55.0,
        }],
    )
    return_probabilities: bool = Field(False)
    request_id:           Optional[str] = Field(None)


class PredictOutput(BaseModel):
    request_id:   str
    prediction:   Any
    probabilities: Optional[dict[str, float]]
    model_id:     str
    model_version: str
    latency_ms:   float
    cached:       bool


class BatchPredictInput(BaseModel):
    instances:            list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=1000
    )
    return_probabilities: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# BentoML Service
# ─────────────────────────────────────────────────────────────────────────────

@bentoml.service(
    name="ml_platform_service",
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={
        "timeout":      30,
        "max_concurrency": 100,
    },
    monitoring={
        "enabled": True,
    },
)
class MLPlatformService:
    """
    BentoML production serving service.
    Loads champion model at startup via ModelRegistry.
    Supports hot-swap via /model/reload.
    """

    def __init__(self) -> None:
        self.settings  = get_settings()
        self.predictor = ModelPredictor(
            model_id=MODEL_ID,
            model_dir=self.settings.models_dir,
            feature_columns=[],   # Managed by processor
            cache_size=10_000,
        )

        # Load champion at startup
        try:
            version = self.predictor.load_champion()
            logger.info(f"Champion loaded at startup: v{version}")
        except Exception as exc:
            logger.error(
                f"Failed to load champion at startup: {exc}. "
                f"Service will return errors until model is loaded."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction endpoints
    # ─────────────────────────────────────────────────────────────────────────
    @bentoml.api(
        route="/predict",
        input_spec=PredictInput ,    # Fixed: Wrapped in JSON descriptor
        output_spec=PredictOutput,  # Fixed: Wrapped in JSON descriptor
    )
    def predict(self, features: dict[str, Any], request_id:str) -> PredictOutput:
        """Single prediction with caching and monitoring."""
        request_id = request_id or str(uuid.uuid4())

        request = PredictionRequest(
            request_id=request_id,
            features=features,
            return_probabilities=False,
        )

        response = self.predictor.predict(request)

        return PredictOutput(
            request_id=response.request_id,
            prediction=response.prediction,
            probabilities=response.probabilities,
            model_id=response.model_id,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
            cached=response.cached,
        )

    @bentoml.api(route="/predict/batch")
    def predict_batch(self, input_data: BatchPredictInput) -> dict[str, Any]:
        """Batch prediction for bulk scoring."""
        result = self.predictor.predict_batch(
            features_list=input_data.instances,
            return_probabilities=input_data.return_probabilities,
        )
        return result.to_dict()

    # ─────────────────────────────────────────────────────────────────────────
    # Model management
    # ─────────────────────────────────────────────────────────────────────────

    @bentoml.api(route="/model/reload")
    def reload_champion(self) -> dict[str, Any]:
        """
        Hot-swap champion model without restarting the service.
        Call this after promote_to_champion() in the DAG.
        """
        try:
            old_version = self.predictor._model_version
            new_version = self.predictor.load_champion()

            logger.info(
                f"Champion hot-swapped: v{old_version} → v{new_version}"
            )

            return {
                "status":      "success",
                "old_version": old_version,
                "new_version": new_version,
                "message":     f"Champion reloaded: v{new_version}",
            }

        except Exception as exc:
            logger.error(f"Hot-swap failed: {exc}")
            return {
                "status":  "error",
                "message": str(exc),
            }

    @bentoml.api(route="/model/info")
    def model_info(self) -> dict[str, Any]:
        """Current champion metadata from registry."""
        try:
            registry = ModelRegistry(model_id=MODEL_ID)
            champion = registry.get_champion(MODEL_ID)

            if champion is None:
                return {"error": "No champion registered"}

            return {
                "model_id":     MODEL_ID,
                "version":      champion.version,
                "stage":        champion.stage.value,
                "metrics":      champion.metrics,
                "mlflow_run_id": champion.mlflow_run_id,
                "created_at":   champion.created_at,
                "tags":         champion.tags,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @bentoml.api(route="/model/metrics")
    def serving_metrics(self) -> dict[str, Any]:
        """Live serving metrics from predictor."""
        return self.predictor.metrics

    @bentoml.api(route="/health")
    def health(self) -> dict[str, Any]:
        """Liveness check."""
        return {
            "status":        "healthy",
            "model_loaded":  self.predictor._model is not None,
            "model_version": self.predictor._model_version,
        }