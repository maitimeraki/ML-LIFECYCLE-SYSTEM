# src/api/middleware/circuit_breaker.py
"""
Circuit breaker for BentoML client calls.

States:
  CLOSED   → normal operation, requests pass through
  OPEN     → BentoML is down, requests fail fast (no waiting)
  HALF_OPEN → probe: allow one request to test recovery

Thresholds:
  failure_threshold:  5 consecutive failures → OPEN
  recovery_timeout:   30 seconds before HALF_OPEN attempt
  success_threshold:  2 successes in HALF_OPEN → CLOSED
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional

from fastapi import HTTPException, status

logger = logging.getLogger("ml_platform.api.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe circuit breaker for external service calls.
    """

    def __init__(
        self,
        service_name:      str,
        failure_threshold: int   = 5,
        recovery_timeout:  float = 30.0,
        success_threshold: int   = 2,
    ) -> None:
        self.service_name      = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self.success_threshold = success_threshold

        self._state:             CircuitState = CircuitState.CLOSED
        self._failure_count:     int          = 0
        self._success_count:     int          = 0
        self._last_failure_time: float        = 0.0
        self._lock               = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time
                >= self.recovery_timeout
            ):
                self._state         = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(
                    f"Circuit breaker [{self.service_name}]: "
                    f"OPEN → HALF_OPEN (probing)"
                )
            return self._state

    def call_succeeded(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state         = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker [{self.service_name}]: "
                        f"HALF_OPEN → CLOSED (recovered)"
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def call_failed(self) -> None:
        with self._lock:
            self._failure_count    += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker [{self.service_name}]: "
                    f"HALF_OPEN → OPEN (probe failed)"
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker [{self.service_name}]: "
                    f"CLOSED → OPEN "
                    f"({self._failure_count} failures)"
                )

    def check(self) -> None:
        """Raise 503 if circuit is open."""
        if self.state == CircuitState.OPEN:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error":    "Prediction service temporarily unavailable",
                    "service":  self.service_name,
                    "state":    "circuit_open",
                    "retry_after": f"{self.recovery_timeout}s",
                },
                headers={"Retry-After": str(int(self.recovery_timeout))},
            )

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "service":        self.service_name,
            "state":          self.state.value,
            "failure_count":  self._failure_count,
            "success_count":  self._success_count,
        }


# Singleton for BentoML circuit breaker
_bentoml_circuit_breaker = CircuitBreaker(
    service_name="bentoml",
    failure_threshold=5,
    recovery_timeout=30.0,
    success_threshold=2,
)


def get_circuit_breaker() -> CircuitBreaker:
    return _bentoml_circuit_breaker