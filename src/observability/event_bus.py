# src/observability/event_bus.py
"""
Event bus with OpenTelemetry tracing integration.
Every event becomes both a log entry AND a trace span attribute AND a metric.
This is the "white box" backbone.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("ml_platform.observability.event_bus")


class EventType(str, Enum):
    # Pipeline lifecycle
    PIPELINE_STARTED           = "pipeline.started"
    PIPELINE_STAGE_STARTED     = "pipeline.stage.started"
    PIPELINE_STAGE_COMPLETED   = "pipeline.stage.completed"
    PIPELINE_COMPLETED         = "pipeline.completed"
    PIPELINE_FAILED            = "pipeline.failed"
    PIPELINE_AWAITING_APPROVAL = "pipeline.awaiting_approval"

    # Data validation
    GE_SUITE_STARTED           = "ge.suite.started"
    GE_EXPECTATION_CHECKED     = "ge.expectation.checked"
    GE_SUITE_COMPLETED         = "ge.suite.completed"

    # Drift
    DRIFT_DETECTION_STARTED    = "drift.detection.started"
    DRIFT_FEATURE_CHECKED      = "drift.feature.checked"
    DRIFT_DETECTION_COMPLETED  = "drift.detection.completed"

    # Training
    TRAINING_STARTED           = "training.started"
    TRAINING_EPOCH_COMPLETED   = "training.epoch.completed"
    TRAINING_COMPLETED         = "training.completed"
    TRAINING_FAILED            = "training.failed"

    # Evaluation
    CHAMPION_CHALLENGER_STARTED   = "evaluation.champion_challenger.started"
    CHAMPION_CHALLENGER_COMPLETED = "evaluation.champion_challenger.completed"

    # Retrain decision
    RETRAIN_DECISION_MADE      = "retrain.decision.made"

    # Deployment
    DEPLOYMENT_STARTED         = "deployment.started"
    DEPLOYMENT_STAGE_CHANGED   = "deployment.stage.changed"
    DEPLOYMENT_COMPLETED       = "deployment.completed"
    DEPLOYMENT_ROLLBACK        = "deployment.rollback"

    # Serving
    PREDICTION_MADE            = "serving.prediction.made"
    PREDICTION_ERROR           = "serving.prediction.error"

    # Monitoring
    PERFORMANCE_DEGRADATION    = "monitoring.performance.degradation"
    ALERT_TRIGGERED            = "monitoring.alert.triggered"


@dataclass
class PlatformEvent:
    pipeline_run_id: str
    event_type: EventType
    step_name: str
    title: str
    message: str
    timestamp: str
    framework: str = ""
    status: str = "info"
    severity: str = "info"
    data: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    duration_ms: float = 0.0
    parent_step: str = ""
    model_id: str = ""
    model_version: str = ""


def make_event(
    pipeline_run_id: str,
    event_type: EventType,
    step_name: str,
    title: str,
    message: str,
    **kwargs: Any,
) -> PlatformEvent:
    return PlatformEvent(
        pipeline_run_id=pipeline_run_id,
        event_type=event_type,
        step_name=step_name,
        title=title,
        message=message,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **kwargs,
    )


class EventBus:
    """
    Central event bus.

    In production this fans out to:
    1. Structured logger → Loki
    2. OpenTelemetry span → Tempo/Jaeger
    3. Prometheus metrics (via metric callbacks)
    4. WebSocket → Frontend real-time feed
    
    Jaeger View:
    Request: POST /predict (Total: 500ms)
    │
    ├── API Gateway: 5ms
    │   └── Auth check: 3ms
    │
    ├── FastAPI Service: 495ms
    │   ├── Input Validation: 2ms
    │   ├── Feature Processing: 250ms  ← BOTTLENECK!
    │   │   ├── DB Query (user profile): 200ms
    │   │   └── Feature Engineering: 50ms
    │   ├── Model Inference: 200ms
    │   │   ├── Preprocessing: 50ms
    │   │   └── Predict: 150ms
    │   └── Response Formatting: 43ms
    │
    └── Audit Logging: 3ms
    """

    def __init__(self) -> None:
        self._handlers: list[Callable[[PlatformEvent], None]] = []
        self._otel_enabled = False
        self._tracer = None

    def enable_otel(self, service_name: str = "ml-platform") -> None:
        """Wire OpenTelemetry if the SDK is installed."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            self._otel_enabled = True
            logger.info("OpenTelemetry tracing enabled.")
        except ImportError:
            logger.warning("opentelemetry-sdk not installed. Tracing disabled.")

    def register_handler(self, handler: Callable[[PlatformEvent], None]) -> None:
        self._handlers.append(handler)

    def emit(self, event: PlatformEvent) -> None:
        # 1. Structured log
        log_level = (
            logging.ERROR if event.severity == "error"
            else logging.WARNING if event.severity == "warning"
            else logging.INFO
        )
        logger.log(
            log_level,
            event.message,
            extra={
                "event_type":       event.event_type.value,
                "pipeline_run_id":  event.pipeline_run_id,
                "step_name":        event.step_name,
                "status":           event.status,
                "progress":         event.progress,
                "model_id":         event.model_id,
                "model_version":    event.model_version,
                "duration_ms":      event.duration_ms,
                **{f"data_{k}": v for k, v in event.data.items()
                   if isinstance(v, (str, int, float, bool))},
            },
        )

        # 2. OTel span attribute (non-blocking)
        if self._otel_enabled and self._tracer:
            try:
                from opentelemetry import trace as otel_trace
                span = otel_trace.get_current_span()
                if span.is_recording():
                    span.set_attribute("event.type",   event.event_type.value)
                    span.set_attribute("event.status", event.status)
                    span.set_attribute("event.step",   event.step_name)
            except Exception:
                pass

        # 3. Custom handlers (Prometheus metrics, WebSocket, etc.)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.warning(f"Event handler failed: {exc}")


# Singleton
event_bus = EventBus()