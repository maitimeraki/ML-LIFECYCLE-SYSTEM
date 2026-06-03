# src/api/routes/health.py
"""
Health, readiness, liveness endpoints.
No auth required — used by load balancers and Kubernetes.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.api.middleware.circuit_breaker import get_circuit_breaker
from src.api.schemas.common import HealthResponse
from config.settings import get_settings

router = APIRouter(tags=["Health"])
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic liveness check."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.version,
        environment=settings.environment.value,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=True,
        model_version=None,
    )


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """
    Readiness probe — checks BentoML is reachable.
    Returns 503 if BentoML is down (pod won't receive traffic).
    """
    checks = {
        "api":             True,
        "bentoml":         False,
        "circuit_breaker": get_circuit_breaker().state.value,
    }

    try:
        client = _get_bentoml_client()
        await client.health()
        checks["bentoml"] = True
    except Exception:
        checks["bentoml"] = False

    ready      = checks["api"] and checks["bentoml"]
    status_code = 200 if ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "ready":   ready,
            "checks":  checks,
            "uptime_seconds": round(time.time() - _startup_time, 1),
        },
    )


@router.get("/health/live")
async def liveness() -> dict:
    """Liveness probe — always 200 if process is alive."""
    return {
        "alive":          True,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


def _get_bentoml_client():
    from src.api.clients.bentoml_client import BentoMLClient
    from config.settings import get_settings
    settings = get_settings()
    url      = getattr(settings, "bentoml_url", "http://bentoml:3000")
    return BentoMLClient(base_url=url, timeout_seconds=5.0)