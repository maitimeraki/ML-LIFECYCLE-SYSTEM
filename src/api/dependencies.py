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
from monitoring.performance_monitor import PerformanceMonitor
from config.settings import get_settings

logger = logging.getLogger("ml_platform.api.dependencies")


@lru_cache(maxsize=1) # ← CRITICAL: Creates only ONE instance
def _get_bentoml_client() -> BentoMLClient:
    """Singleton BentoML HTTP client with connection pooling.
    
    Request 1 → Reuses Connection 1 from pool (no new TCP handshake)
    Request 2 → Reuses Connection 2 from pool
    Request 3 → Reuses Connection 1 again (keep-alive)
    ...
    Request 100 → Uses Connection 20 from pool
    Request 101 → Waits for free connection (or creates new one)"""
    settings = get_settings()
    url      = getattr(settings, "bentoml_url", "http://localhost:3000")
    return BentoMLClient(base_url=url, timeout_seconds=30.0)


@lru_cache(maxsize=None) # ← Cache per unique model_id
def _get_performance_monitor(model_id: str) -> PerformanceMonitor:
    """Singleton PerformanceMonitor per model_id.
    
    # maxsize=1: Only caches LAST call
    _get_bentoml_client()  # Always returns same object (no parameters)

    # maxsize=None: Caches ALL unique calls
    _get_performance_monitor("fraud_detector")     # Creates & caches
    _get_performance_monitor("recommendation")     # Creates & caches
    _get_performance_monitor("fraud_detector")     # Returns cached (same object)
    _get_performance_monitor("credit_scoring")     # Creates & caches
    # Cache size: 3 different monitors (one per model_id)"""
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

# Benefits:
# 1. Single import: from src.api.dependencies import BentoMLDep
# 2. Consistent across all endpoints
# 3. Easy to add new dependencies globally
# 4. IDE autocomplete works perfectly
# Type aliases for cleaner route signatures
BentoMLDep       = Annotated[BentoMLClient,       Depends(get_bentoml_client)]
RateLimiterDep   = Annotated[RateLimiter,         Depends(get_rate_limiter)]
CircuitBreakerDep = Annotated[CircuitBreaker,     Depends(get_circuit_breaker)]
PerfMonitorDep   = Annotated[PerformanceMonitor,  Depends(get_performance_monitor)]




# 100 concurrent requests arrive simultaneously

# Request 1: Gets cached BentoMLClient (no creation)
# Request 2: Gets cached BentoMLClient (same object)
# ...
# Request 100: Gets cached BentoMLClient (same object)

# All 100 requests share:
# - Same HTTP connection pool (20 keep-alive connections)
# - Same PerformanceMonitor instance
# - Same RateLimiter state
# - Same CircuitBreaker state

# Memory: Only 1 of each object (not 100)
# Performance: Connection pooling (not 100 separate TCP connections)