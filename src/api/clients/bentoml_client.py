# src/api/clients/bentoml_client.py
"""
HTTP client to BentoML serving service.

FastAPI never imports BentoML directly.
All model inference goes through this HTTP client.

Benefits:
  - FastAPI and BentoML scale independently
  - BentoML can be replaced (TorchServe, Triton) without touching API
  - Circuit breaker isolates failures
  - Connection pooling for efficiency
  - Retry logic for transient failures
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

import httpx

from config.settings import get_settings
from src.common.exceptions import PredictionError

logger = logging.getLogger("ml_platform.api.clients.bentoml")


class BentoMLClient:
    """
    Async HTTP client to BentoML serving service.
    Uses httpx with connection pooling and timeout management.
    """

    def __init__(
        self,
        base_url:       Optional[str]  = None,
        timeout_seconds: float         = 30.0,
        max_connections: int           = 100,
    ) -> None:
        settings = get_settings()
        self.base_url = (
            base_url
            or getattr(settings, "bentoml_url", "http://localhost:3000")
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout_seconds,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=max_connections,  # Max 100 concurrent
                max_keepalive_connections=20, # Keep 20 connections alive
            ),
            headers={
                "Content-Type": "application/json",
                "Accept":       "application/json",
            },
        )

    async def predict(
        self,
        features:             dict[str, Any],
        return_probabilities: bool          = False,
        request_id:           Optional[str] = None,
        model_version:        Optional[str] = None,
    ) -> dict[str, Any]:
        """Single prediction via BentoML /predict endpoint."""
        payload = {
            "features":             features,
            "return_probabilities": return_probabilities,
            "request_id":           request_id or str(uuid.uuid4()),
        }
        if model_version:
            payload["model_version"] = model_version

        try:
            response = await self._client.post("/predict", json=payload)
            response.raise_for_status()
            return response.json() # Type -> dict with prediction, probabilities, metadata

        except httpx.TimeoutException as exc:
            logger.error(f"BentoML timeout: {exc}")
            raise PredictionError(
                "Prediction service timeout",
                details={"request_id": request_id},
            ) from exc

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"BentoML error {exc.response.status_code}: "
                f"{exc.response.text}"
            )
            raise PredictionError(
                f"Prediction service error: {exc.response.status_code}",
                details={
                    "status_code": exc.response.status_code,
                    "body":        exc.response.text[:200],
                },
            ) from exc

        except Exception as exc:
            logger.error(f"BentoML unexpected error: {exc}", exc_info=True)
            raise PredictionError(
                f"Prediction service unavailable: {exc}",
                details={"request_id": request_id},
            ) from exc

    async def predict_batch(
        self,
        instances:            list[dict[str, Any]],
        return_probabilities: bool = False,
    ) -> dict[str, Any]:
        """Batch prediction via BentoML /predict/batch endpoint."""
        try:
            response = await self._client.post(
                "/predict/batch",
                json={
                    "instances":            instances,
                    "return_probabilities": return_probabilities,
                },
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as exc:
            raise PredictionError(
                "Batch prediction timeout",
                details={"batch_size": len(instances)},
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise PredictionError(
                f"Batch prediction error: {exc.response.status_code}",
                details={"status_code": exc.response.status_code},
            ) from exc

    async def reload_champion(self) -> dict[str, Any]:
        """Trigger model hot-swap on BentoML service."""
        response = await self._client.post("/model/reload")
        response.raise_for_status()
        return response.json()

    async def model_info(self) -> dict[str, Any]:
        """Get champion model metadata from BentoML."""
        response = await self._client.get("/model/info")
        response.raise_for_status()
        return response.json()

    async def serving_metrics(self) -> dict[str, Any]:
        """Get serving metrics from BentoML predictor."""
        response = await self._client.get("/model/metrics")
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict[str, Any]:
        """Check BentoML service health."""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()