# src/api/middleware/audit_logger.py
"""
Audit logger — records every API request with auth context.

Required for:
  - Compliance (who called what, when)
  - Debugging (full request trace)
  - Security (detect abuse patterns)
  - Cost attribution (per-client usage)
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ml_platform.api.audit")


class AuditLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every request with:
      - request_id (injected into response headers)
      - client_id (from auth context)
      - endpoint, method, status_code
      - latency_ms
      - user_agent, ip
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        request_id = request.headers.get(
            "X-Request-ID", str(uuid.uuid4())
        )

        # Attach request_id to request state for use in route handlers
        request.state.request_id = request_id
        request.state.started_at = time.perf_counter()

        start = time.perf_counter()

        try:
            response    = await call_next(request)
            latency_ms  = (time.perf_counter() - start) * 1000

            # Inject request_id into response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Latency-Ms"] = str(round(latency_ms, 1))

            # Audit log
            client_id = getattr(
                getattr(request.state, "auth_context", None),
                "client_id",
                "anonymous",
            )

            logger.info(
                "API request",
                extra={
                    "request_id":   request_id,
                    "client_id":    client_id,
                    "method":       request.method,
                    "path":         request.url.path,
                    "status_code":  response.status_code,
                    "latency_ms":   round(latency_ms, 1),
                    "user_agent":   request.headers.get("user-agent", ""),
                    "client_ip":    request.client.host if request.client else "",
                    "timestamp":    datetime.now(timezone.utc).isoformat(),
                },
            )

            return response

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "API request failed",
                extra={
                    "request_id": request_id,
                    "method":     request.method,
                    "path":       request.url.path,
                    "error":      str(exc),
                    "latency_ms": round(latency_ms, 1),
                },
                exc_info=True,
            )
            raise