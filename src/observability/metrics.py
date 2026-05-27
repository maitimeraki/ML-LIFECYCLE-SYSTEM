# src/observability/metrics.py
"""
Prometheus metrics registry.
Every ML operation emits a metric — this is your white-box backbone.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry

# Use default registry so FastAPI /metrics endpoint picks them up automatically

# ── Pipeline ──────────────────────────────────────────────────────────────────
PIPELINE_RUNS_TOTAL = Counter(
    "ml_pipeline_runs_total",
    "Total pipeline runs",
    ["model_id", "status"],
)

PIPELINE_STAGE_DURATION = Histogram(
    "ml_pipeline_stage_duration_seconds",
    "Duration of each pipeline stage",
    ["model_id", "stage"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],
)

PIPELINE_CURRENT_STAGE = Gauge(
    "ml_pipeline_current_stage",
    "Currently running pipeline stage (1=active)",
    ["model_id", "stage"],
)

# ── Data Validation ───────────────────────────────────────────────────────────
GE_EXPECTATIONS_TOTAL = Counter(
    "ml_ge_expectations_total",
    "Total Great Expectations checks",
    ["suite_name", "result"],          # result: passed | failed
)

GE_SUITE_SCORE = Gauge(
    "ml_ge_suite_score",
    "Overall GE suite success rate (0-1)",
    ["suite_name"],
)

DATA_ROWS_VALIDATED = Counter(
    "ml_data_rows_validated_total",
    "Total rows validated",
    ["model_id"],
)

# ── Drift ─────────────────────────────────────────────────────────────────────
DRIFT_SCORE = Gauge(
    "ml_drift_score",
    "Overall drift score (0-1)",
    ["model_id"],
)

DRIFT_FEATURES_DRIFTED = Gauge(
    "ml_drift_features_drifted",
    "Number of drifted features",
    ["model_id"],
)

DRIFT_SEVERITY = Gauge(
    "ml_drift_severity_numeric",
    "Drift severity as number (0=none,1=low,2=moderate,3=high,4=critical)",
    ["model_id"],
)

DRIFT_PSI = Gauge(
    "ml_drift_psi",
    "PSI value per feature",
    ["model_id", "feature"],
)

DRIFT_KS_PVALUE = Gauge(
    "ml_drift_ks_pvalue",
    "KS test p-value per feature",
    ["model_id", "feature"],
)

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_RUNS_TOTAL = Counter(
    "ml_training_runs_total",
    "Total training runs",
    ["model_id", "status"],
)

TRAINING_DURATION = Histogram(
    "ml_training_duration_seconds",
    "Training duration",
    ["model_id"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)

MODEL_CV_SCORE = Gauge(
    "ml_model_cv_score",
    "Cross-validation mean score",
    ["model_id", "version", "metric"],
)

MODEL_VALIDATION_METRIC = Gauge(
    "ml_model_validation_metric",
    "Validation set metric value",
    ["model_id", "version", "metric"],
)

# ── Champion-Challenger ───────────────────────────────────────────────────────
CHAMPION_METRIC = Gauge(
    "ml_champion_metric_value",
    "Current champion model metric",
    ["model_id", "metric"],
)

CHALLENGER_IMPROVEMENT = Gauge(
    "ml_challenger_improvement",
    "Relative improvement of challenger over champion",
    ["model_id"],
)

CHAMPION_CHALLENGER_COMPARISONS = Counter(
    "ml_champion_challenger_comparisons_total",
    "Total champion-challenger comparisons",
    ["model_id", "result"],            # result: approved | rejected
)

# ── Deployment ────────────────────────────────────────────────────────────────
DEPLOYMENT_STATUS_GAUGE = Gauge(
    "ml_deployment_status",
    "Active deployment status (1=running,0=idle)",
    ["model_id", "strategy"],
)

DEPLOYMENT_TRAFFIC_SPLIT = Gauge(
    "ml_deployment_traffic_split_pct",
    "Traffic split percentage",
    ["model_id", "variant"],           # variant: champion | challenger
)

DEPLOYMENT_ROLLBACKS_TOTAL = Counter(
    "ml_deployment_rollbacks_total",
    "Total deployment rollbacks",
    ["model_id", "reason"],
)

# ── Serving ───────────────────────────────────────────────────────────────────
PREDICTION_REQUESTS_TOTAL = Counter(
    "ml_prediction_requests_total",
    "Total prediction requests",
    ["model_id", "model_version", "status"],  # status: success | error | cached
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_ms",
    "Prediction latency in milliseconds",
    ["model_id", "model_version"],
    buckets=[1, 5, 10, 25, 50, 100, 200, 500, 1000],
)

PREDICTION_CACHE_HIT_RATE = Gauge(
    "ml_prediction_cache_hit_rate",
    "Cache hit rate (0-1)",
    ["model_id"],
)

# ── Retrain Decision ──────────────────────────────────────────────────────────
RETRAIN_DECISIONS_TOTAL = Counter(
    "ml_retrain_decisions_total",
    "Total retrain decisions",
    ["model_id", "decision"],
)

RETRAIN_PRIORITY_SCORE = Gauge(
    "ml_retrain_priority_score",
    "Retrain priority score (0-1)",
    ["model_id"],
)

# ── Model Performance (Live) ──────────────────────────────────────────────────
LIVE_MODEL_METRIC = Gauge(
    "ml_live_model_metric",
    "Live production model metric value",
    ["model_id", "model_version", "metric"],
)

MODEL_INFO = Info(
    "ml_model",
    "Current champion model metadata",
    ["model_id"],
)


# ── Severity mapping ──────────────────────────────────────────────────────────
DRIFT_SEVERITY_MAP = {
    "none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4
}