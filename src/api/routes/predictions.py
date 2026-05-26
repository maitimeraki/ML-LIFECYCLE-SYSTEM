# src/api/routes/predictions.py
"""
Prediction endpoints.

Flow for POST /predict:
  1. Auth     → get_auth_context()    [who is calling?]
  2. Scope    → require_predict_scope [do they have permission?]
  3. Rate     → rate_limiter.check()  [are they calling too fast?]
  4. Circuit  → circuit_breaker.check() [is BentoML up?]
  5. Validate → PredictionRequest     [is the input valid?]
  6. Call     → bentoml_client.predict() [get prediction]
  7. Record   → performance_monitor   [store for delayed label matching]
  8. Return   → PredictionResponse    [structured response]
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.dependencies import (
    BentoMLDep,
    CircuitBreakerDep,
    PerfMonitorDep,
    RateLimiterDep,
)
from src.api.middleware.auth import AuthContext, require_predict_scope
from src.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    PredictionRequest,
    PredictionResponse,                             
)
from src.api.schemas.common import ErrorResponse, SuccessResponse
from src.common.exceptions import InputValidationError, PredictionError

logger = logging.getLogger("ml_platform.api.predictions")
router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Prediction successful"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Single prediction",
    description=(
        "Make a single prediction.\n\n"
        "**Auth:** X-API-Key header or Bearer JWT\n"
        "**Rate limit:** 1000 req/min per client\n"
        "**Caching:** SHA256-keyed LRU cache in BentoML\n"
    ),
)
async def predict(
    body:            PredictionRequest,
    request:         Request,
    auth:            AuthContext         = Depends(require_predict_scope),
    rate_limiter:    RateLimiterDep      = None,
    circuit_breaker: CircuitBreakerDep   = None,
    bentoml:         BentoMLDep          = None,
    perf_monitor:    PerfMonitorDep      = None,
) -> PredictionResponse:
    """
    Single prediction endpoint.

    Complete flow:
      auth → rate_limit → circuit_check → validate → predict → record → return
    """
    request_id = (
        body.request_id
        or getattr(request.state, "request_id", None)
        or str(uuid.uuid4())
    )

    # ── 1. Rate limiting ───────────────────────────────────────────────────
    rate_limiter.check(
        client_id=auth.client_id,
        scope="predict",
    )

    # ── 2. Circuit breaker ─────────────────────────────────────────────────
    circuit_breaker.check()

    # ── 3. Call BentoML ────────────────────────────────────────────────────
    try:
        result = await bentoml.predict(
            features=body.features,
            return_probabilities=body.return_probabilities,
            request_id=request_id,
            model_version=body.model_version, # It is necessary because different users may have different model access permissions.
        )
        circuit_breaker.call_succeeded()

    except PredictionError as exc:
        circuit_breaker.call_failed()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error":      exc.message,
                "request_id": request_id,
            },
        )

    except Exception as exc:
        circuit_breaker.call_failed()
        logger.error(f"Unexpected error in predict: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error":      "Internal prediction error",
                "request_id": request_id,
            },
        )

    # ── 4. Record in PerformanceMonitor for delayed label matching ─────────
    try:
        perf_monitor.record_prediction(
            request_id=request_id,
            prediction=result.get("prediction"),
            model_version=result.get("model_version", ""),
        )
    except Exception as exc:
        # Non-fatal — don't fail the prediction because monitoring failed
        logger.warning(f"Failed to record prediction: {exc}")

    return PredictionResponse(
        request_id=request_id,
        prediction=result["prediction"],
        probabilities=result.get("probabilities"),
        model_id=result.get("model_id", ""),
        model_version=result.get("model_version", ""),
        latency_ms=result.get("latency_ms", 0.0),
        cached=result.get("cached", False),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description=(
        "Predict for up to 1000 instances in one call.\n\n"
        "**Rate:** counts as N requests (N = batch size)\n"
        "**Timeout:** 60 seconds (longer than single predict)\n"
    ),
)
async def predict_batch(
    body:            BatchPredictionRequest,
    request:         Request,
    auth:            AuthContext         = Depends(require_predict_scope),
    rate_limiter:    RateLimiterDep      = None,
    circuit_breaker: CircuitBreakerDep   = None,
    bentoml:         BentoMLDep          = None,
) -> BatchPredictionResponse:
    """Batch prediction — counts as N tokens against rate limit."""

    # Count batch size against rate limit
    rate_limiter.check(
        client_id=auth.client_id,
        scope="predict",
        tokens=float(len(body.instances)),
    )

    circuit_breaker.check()

    try:
        result = await bentoml.predict_batch(
            instances=body.instances,
            return_probabilities=body.return_probabilities,
        )
        circuit_breaker.call_succeeded()

    except PredictionError as exc:
        circuit_breaker.call_failed()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": exc.message},
        )

    return BatchPredictionResponse(**result)


@router.post(
    "/feedback",
    response_model=SuccessResponse,
    summary="Submit ground truth feedback",
    description=(
        "Submit actual label for a previous prediction.\n\n"
        "Used to compute live model performance metrics.\n"
        "Matches to stored prediction via request_id.\n"
    ),
)
async def submit_feedback(
    body:         FeedbackRequest,
    auth:         AuthContext      = Depends(require_predict_scope),
    perf_monitor: PerfMonitorDep   = None,
) -> SuccessResponse:
    """Match ground truth to a previous prediction."""
    matched = perf_monitor.record_ground_truth(
        request_id=body.request_id,
        ground_truth=body.ground_truth,
    )

    if not matched:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":      f"No prediction found for request_id: {body.request_id}",
                "request_id": body.request_id,
            },
        )

    return SuccessResponse(
        message="Feedback recorded",
        data={"request_id": body.request_id, "matched": True},
    )


@router.get(
    "/metrics",
    summary="Serving metrics",
    description="Live prediction metrics: count, error_rate, latency, cache.",
)
async def serving_metrics(
    auth:    AuthContext   = Depends(require_predict_scope),
    bentoml: BentoMLDep    = None,
) -> dict:
    """Get live serving metrics from BentoML predictor."""
    try:
        return await bentoml.serving_metrics()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": str(exc)},
        )