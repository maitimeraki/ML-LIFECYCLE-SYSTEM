# src/observability/prometheus_handler.py
"""
Translates EventBus events → Prometheus metric updates.
Register this once at startup and every event auto-updates Grafana.
"""
from __future__ import annotations

from src.observability.event_bus import EventBus, EventType, PlatformEvent
from src.observability.metrics import (
    GE_EXPECTATIONS_TOTAL, GE_SUITE_SCORE,
    DRIFT_SCORE, DRIFT_FEATURES_DRIFTED, DRIFT_SEVERITY,
    DRIFT_PSI, DRIFT_KS_PVALUE, DRIFT_SEVERITY_MAP,
    TRAINING_RUNS_TOTAL, MODEL_CV_SCORE, MODEL_VALIDATION_METRIC,
    CHAMPION_CHALLENGER_COMPARISONS, CHALLENGER_IMPROVEMENT, CHAMPION_METRIC,
    DEPLOYMENT_STATUS_GAUGE, DEPLOYMENT_TRAFFIC_SPLIT, DEPLOYMENT_ROLLBACKS_TOTAL,
    PREDICTION_REQUESTS_TOTAL, PREDICTION_LATENCY, PREDICTION_CACHE_HIT_RATE,
    RETRAIN_DECISIONS_TOTAL, RETRAIN_PRIORITY_SCORE,
    PIPELINE_RUNS_TOTAL, PIPELINE_STAGE_DURATION,
)


def _handle_event(event: PlatformEvent) -> None:
    data = event.data
    model_id = event.model_id or data.get("model_id", "unknown")

    # ── GE Validation ────────────────────────────────────────────────────────
    if event.event_type == EventType.GE_EXPECTATION_CHECKED:
        result = "passed" if event.status == "success" else "failed"
        GE_EXPECTATIONS_TOTAL.labels(
            suite_name=data.get("suite_name", data.get("expectation_type", "unknown")),
            result=result,
        ).inc()

    elif event.event_type == EventType.GE_SUITE_COMPLETED:
        suite_name = data.get("suite_name", "unknown")
        stats = data.get("statistics", {})
        total = stats.get("evaluated_expectations", 1)
        passed = stats.get("successful_expectations", 0)
        GE_SUITE_SCORE.labels(suite_name=suite_name).set(
            passed / total if total > 0 else 0.0
        )

    # ── Drift ─────────────────────────────────────────────────────────────────
    elif event.event_type == EventType.DRIFT_DETECTION_COMPLETED:
        DRIFT_SCORE.labels(model_id=model_id).set(data.get("overall_drift_score", 0))
        DRIFT_FEATURES_DRIFTED.labels(model_id=model_id).set(
            data.get("features_drifted", 0)
        )
        severity_str = data.get("severity", "none")
        DRIFT_SEVERITY.labels(model_id=model_id).set(
            DRIFT_SEVERITY_MAP.get(severity_str, 0)
        )

    elif event.event_type == EventType.DRIFT_FEATURE_CHECKED:
        feature = data.get("feature", "unknown")
        if "psi" in data:
            DRIFT_PSI.labels(model_id=model_id, feature=feature).set(data["psi"])
        if "ks_p_value" in data:
            DRIFT_KS_PVALUE.labels(model_id=model_id, feature=feature).set(
                data["ks_p_value"]
            )

    # ── Training ──────────────────────────────────────────────────────────────
    elif event.event_type == EventType.TRAINING_COMPLETED:
        version = data.get("model_version", "unknown")
        TRAINING_RUNS_TOTAL.labels(model_id=model_id, status="success").inc()
        for metric, value in data.get("metrics", {}).items():
            MODEL_VALIDATION_METRIC.labels(
                model_id=model_id, version=version, metric=metric
            ).set(value)
        MODEL_CV_SCORE.labels(
            model_id=model_id, version=version, metric="cv_mean"
        ).set(data.get("cv_mean", 0))

    elif event.event_type == EventType.TRAINING_FAILED:
        TRAINING_RUNS_TOTAL.labels(model_id=model_id, status="failed").inc()

    # ── Champion-Challenger ───────────────────────────────────────────────────
    elif event.event_type == EventType.CHAMPION_CHALLENGER_COMPLETED:
        result = "approved" if data.get("challenger_is_better") else "rejected"
        CHAMPION_CHALLENGER_COMPARISONS.labels(model_id=model_id, result=result).inc()
        CHALLENGER_IMPROVEMENT.labels(model_id=model_id).set(
            data.get("improvement", 0)
        )
        CHAMPION_METRIC.labels(
            model_id=model_id, metric=data.get("primary_metric", "unknown")
        ).set(data.get("champion_metric_value", 0))

    # ── Deployment ────────────────────────────────────────────────────────────
    elif event.event_type == EventType.DEPLOYMENT_STARTED:
        DEPLOYMENT_STATUS_GAUGE.labels(
            model_id=model_id, strategy=data.get("strategy", "unknown")
        ).set(1)

    elif event.event_type == EventType.DEPLOYMENT_STAGE_CHANGED:
        for variant, pct in data.get("traffic_split", {}).items():
            DEPLOYMENT_TRAFFIC_SPLIT.labels(
                model_id=model_id, variant=variant
            ).set(pct)

    elif event.event_type == EventType.DEPLOYMENT_COMPLETED:
        DEPLOYMENT_STATUS_GAUGE.labels(
            model_id=model_id, strategy=data.get("strategy", "unknown")
        ).set(0)

    elif event.event_type == EventType.DEPLOYMENT_ROLLBACK:
        DEPLOYMENT_ROLLBACKS_TOTAL.labels(
            model_id=model_id,
            reason=data.get("reason", "unknown"),
        ).inc()

    # ── Serving ───────────────────────────────────────────────────────────────
    elif event.event_type == EventType.PREDICTION_MADE:
        version = event.model_version or "unknown"
        status = "cached" if data.get("cached") else "success"
        PREDICTION_REQUESTS_TOTAL.labels(
            model_id=model_id, model_version=version, status=status
        ).inc()
        latency = data.get("latency_ms", 0)
        if latency > 0:
            PREDICTION_LATENCY.labels(
                model_id=model_id, model_version=version
            ).observe(latency)
        PREDICTION_CACHE_HIT_RATE.labels(model_id=model_id).set(
            data.get("cache_hit_rate", 0)
        )

    elif event.event_type == EventType.PREDICTION_ERROR:
        version = event.model_version or "unknown"
        PREDICTION_REQUESTS_TOTAL.labels(
            model_id=model_id, model_version=version, status="error"
        ).inc()

    # ── Retrain Decision ──────────────────────────────────────────────────────
    elif event.event_type == EventType.RETRAIN_DECISION_MADE:
        RETRAIN_DECISIONS_TOTAL.labels(
            model_id=model_id, decision=data.get("decision", "unknown")
        ).inc()
        RETRAIN_PRIORITY_SCORE.labels(model_id=model_id).set(
            data.get("priority_score", 0)
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    elif event.event_type == EventType.PIPELINE_COMPLETED:
        PIPELINE_RUNS_TOTAL.labels(
            model_id=model_id, status=data.get("status", "unknown")
        ).inc()

    elif event.event_type == EventType.PIPELINE_STAGE_COMPLETED:
        duration = event.duration_ms / 1000.0
        if duration > 0:
            PIPELINE_STAGE_DURATION.labels(
                model_id=model_id, stage=event.step_name
            ).observe(duration)


def register_prometheus_handler(bus: EventBus) -> None:
    """Call once at application startup."""
    bus.register_handler(_handle_event)