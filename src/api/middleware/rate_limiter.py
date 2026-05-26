# src/api/middleware/rate_limiter.py
"""
Token bucket rate limiter.

Per-client limits based on auth context.
Stores state in memory (single instance) or Redis (distributed).

Limits:
  predict scope:  1000 req/min, 50 req/sec burst
  admin scope:    60 req/min
  read scope:     300 req/min
  anonymous:      10 req/min
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from fastapi import HTTPException, Request, status

from src.api.middleware.auth import AuthContext

logger = logging.getLogger("ml_platform.api.rate_limiter")


@dataclass
class TokenBucket:
    """
    Token bucket algorithm.
    Allows burst up to capacity, refills at rate tokens/second.
    """
    capacity:    float
    rate:        float          # tokens per second
    tokens:      float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens      = self.capacity
        self.last_refill = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.
        Returns True if allowed, False if rate limited.
        """
        now      = time.monotonic()
        elapsed  = now - self.last_refill

        # Refill
        self.tokens      = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


# Scope → (capacity, rate_per_second)
RATE_LIMITS: dict[str, tuple[float, float]] = {
    "predict":     (50.0,  16.67),   # 50 burst, 1000/min
    "admin":       (10.0,  1.0),     # 10 burst, 60/min
    "read":        (20.0,  5.0),     # 20 burst, 300/min
    "development": (100.0, 100.0),   # No real limit in dev
    "anonymous":   (5.0,   0.17),    # 5 burst, 10/min
}


class RateLimiter:
    """
    In-memory rate limiter per client_id.
    Thread-safe via Lock.
    Replace with Redis in multi-instance deployments.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock    = Lock()

    def check(
        self,
        client_id:  str,
        scope:      str,
        tokens:     float = 1.0,
    ) -> None:
        """
        Check rate limit. Raises 429 if exceeded.
        """
        bucket_key   = f"{client_id}:{scope}"
        capacity, rate = RATE_LIMITS.get(
            scope, RATE_LIMITS["anonymous"]
        )

        with self._lock:
            if bucket_key not in self._buckets:
                self._buckets[bucket_key] = TokenBucket(
                    capacity=capacity,
                    rate=rate,
                )
            bucket  = self._buckets[bucket_key]
            allowed = bucket.consume(tokens)

        if not allowed:
            logger.warning(
                f"Rate limited: client={client_id}, scope={scope}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":       "Rate limit exceeded",
                    "client_id":   client_id,
                    "retry_after": f"{1.0 / rate:.1f}s",
                },
                headers={
                    "Retry-After":       str(int(1.0 / rate)),
                    "X-RateLimit-Limit": str(int(capacity)),
                },
            )


# Singleton
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter