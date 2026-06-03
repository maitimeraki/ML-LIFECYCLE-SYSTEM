# src/api/main.py
"""
FastAPI application factory.

Wires everything together:
  - Middleware: audit logging, CORS
  - Routes: predictions, models, drift, pipelines, health
  - Prometheus /metrics endpoint
  - Startup: BentoML client health check
  - Shutdown: graceful client close
  - Exception handlers: structured error responses
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

from config.logging_config import setup_logging
from config.settings import get_settings
from src.api.dependencies import _get_bentoml_client
from src.api.middleware.audit_logger import AuditLoggerMiddleware
from src.api.routes import drift, health, models, predictions
from src.common.exceptions import (
    InputValidationError,
    MLPlatformError,
    PredictionError,
)
from src.observability.event_bus import event_bus
from src.observability.prometheus_handler import register_prometheus_handler

logger = logging.getLogger("ml_platform.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    setup_logging()
    settings = get_settings()

    # Wire Prometheus to EventBus
    register_prometheus_handler(event_bus)

    # Verify BentoML is reachable at startup
    try:
        client = _get_bentoml_client()
        info   = await client.model_info()
        logger.info(
            f"BentoML connected: "
            f"model={info.get('model_id')}, "
            f"version={info.get('version')}"
        )
    except Exception as exc:
        logger.warning(
            f"BentoML not reachable at startup: {exc}. "
            f"Predictions will fail until BentoML is available."
        )

    logger.info(
        f"ML Platform API starting: "
        f"env={settings.environment.value}, "
        f"version={settings.version}"
    )

    yield

    # Shutdown: close HTTP client
    try:
        client = _get_bentoml_client()
        await client.aclose()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ML Lifecycle Platform API",
        description=(
            "Production ML serving API.\n\n"
            "**Authentication:** X-API-Key header or Bearer JWT\n"
            "**Rate limiting:** Per-client token bucket\n"
            "**Serving:** BentoML inference engine (behind this gateway)\n"
        ),
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # # ── Prometheus metrics ─────────────────────────────────────────────────
    # metrics_app = make_asgi_app()
    # app.mount("/metrics", metrics_app)
    
    # ── Auto-instrument FastAPI routes ────────────────────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Middleware (order matters — outermost first) ────────────────────────
    app.add_middleware(AuditLoggerMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Latency-Ms"],
    )

    # ── Exception handlers ─────────────────────────────────────────────────
    @app.exception_handler(MLPlatformError)
    async def platform_error_handler(
        request: Request, exc: MLPlatformError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error":      exc.message,
                "details":    exc.details,
                "type":       type(exc).__name__,
                "request_id": request_id,
            },
        )

    @app.exception_handler(InputValidationError)
    async def validation_error_handler(
        request: Request, exc: InputValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request, exc: PredictionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": exc.message, "details": exc.details},
        )

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predictions.router, prefix="/api/v1")
    app.include_router(models.router,      prefix="/api/v1")
    app.include_router(drift.router,       prefix="/api/v1")
    # app.include_router(pipelines.router,   prefix="/api/v1")

    return app


app = create_app()