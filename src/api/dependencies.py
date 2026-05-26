# src/api/dependencies.py
"""
FastAPI dependency providers.
Single source of truth for all shared objects.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.api.clients.bentoml_client import BentoMLClient
from src.api.middleware.circuit_breaker import CircuitBreaker, get_circuit_breaker
from src.api.middleware.rate_limiter import RateLimiter, get_rate_limiter
from src.monitoring.performance_monitor import PerformanceMonitor
from config.settings import get_settings

logger = logging.getLogger("ml_platform.api.dependencies")


@lru_cache(maxsize=1)
def _get_bentoml_client() -> BentoMLClient:
    """Singleton BentoML HTTP client with connection pooling."""
    settings = get_settings()
    url      = getattr(settings, "bentoml_url", "http://localhost:3000")
    return BentoMLClient(base_url=url, timeout_seconds=30.0)


@lru_cache(maxsize=None)
def _get_performance_monitor(model_id: str) -> PerformanceMonitor:
    """Singleton PerformanceMonitor per model_id."""
    settings = get_settings()
    return PerformanceMonitor(
        model_id=model_id,
        primary_metric="f1_score",
        baseline_value=float(
            getattr(settings, "baseline_f1", 0.80)
        ),
        degradation_threshold=0.05,
        is_classification=True,
    )


async def get_bentoml_client() -> BentoMLClient:
    return _get_bentoml_client()


async def get_performance_monitor() -> PerformanceMonitor:
    settings = get_settings()
    model_id = getattr(settings, "model_id", "customer_churn_model")
    return _get_performance_monitor(model_id)


# Type aliases for cleaner route signatures
BentoMLDep       = Annotated[BentoMLClient,       Depends(get_bentoml_client)]
RateLimiterDep   = Annotated[RateLimiter,         Depends(get_rate_limiter)]
CircuitBreakerDep = Annotated[CircuitBreaker,     Depends(get_circuit_breaker)]
PerfMonitorDep   = Annotated[PerformanceMonitor,  Depends(get_performance_monitor)]