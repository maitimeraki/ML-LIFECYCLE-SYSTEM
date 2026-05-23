"""
Retrain decision engine.
Aggregates signals from drift detection, performance monitoring,
data readiness, and business rules to produce a retrain decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from config.settings import RetrainSettings, get_settings
from src.common.enums import DriftSeverity, RetrainDecision
from src.drift.detector import DriftReport

logger = logging.getLogger("ml_platform.decision.retrain_policy")


@dataclass
class PerformanceMetrics:
    """Current model performance metrics from production monitoring."""
    # There is the drawback of only tracking one primary metric, but it keeps the decision logic simpler. We can always add more metrics later if needed. But in produciton has different matrics for different use cases, so we need to be flexible.
    primary_metric_name: str           # e.g., "f1_score", "rmse"
    primary_metric_value: float
    baseline_metric_value: float       # Value at deployment time
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    evaluation_sample_size: int = 0
    evaluation_window_start: Optional[datetime] = None
    evaluation_window_end: Optional[datetime] = None


@dataclass
class DataReadinessReport:
    """Assessment of whether new data is sufficient and suitable for retraining."""
    new_samples_available: int
    min_samples_required: int
    is_sufficient: bool
    label_availability_rate: float  # What % of new data has ground truth labels
    data_freshness_days: int        # How recent is the latest data point
    quality_score: float            # 0-1, from data validation
    notes: list[str] = field(default_factory=list)


@dataclass
class RetrainAssessment:
    """Complete retrain decision with justification and evidence. 
    If data isn't ready, no amount of drift or performance degradation justifies retraining. It's like trying to bake a cake without flour - doesn't matter how hungry you are.
    """
    decision: RetrainDecision
    confidence: float                    # 0-1
    timestamp: datetime
    signals: dict[str, Any]              # All input signals
    reasons: list[str]                   # Human-readable reasons
    estimated_improvement: Optional[float] = None
    recommended_action: str = ""
    requires_human_approval: bool = True
    priority_score: float = 0.0          # 0-1, higher = more urgent

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "priority_score": round(self.priority_score, 3),
            "reasons": self.reasons,
            "recommended_action": self.recommended_action,
            "requires_human_approval": self.requires_human_approval,
            "signals": {
                k: v if not isinstance(v, datetime) else v.isoformat()
                for k, v in self.signals.items()
            },
        }


class RetrainPolicyEngine:
    """
    Evaluates multiple signals to decide whether a model should be retrained.

    Signal hierarchy (priority order):
    1. Performance degradation (direct evidence of model quality loss)
    2. Concept drift (target distribution shift)
    3. Data drift severity (input distribution shift)
    4. Staleness (time since last training)
    5. Data readiness (sufficient new data available)

    The engine produces a scored decision with justification.
    """

    def __init__(self, settings: Optional[RetrainSettings] = None):
        self.settings = settings or get_settings().retrain

    def evaluate(
        self,
        drift_report: Optional[DriftReport] = None,
        performance_metrics: Optional[PerformanceMetrics] = None,
        data_readiness: Optional[DataReadinessReport] = None,
        last_training_date: Optional[datetime] = None,
        model_id: Optional[str] = None,
    ) -> RetrainAssessment:
        """
        Evaluate all signals and produce a retrain decision.
        At least one signal source must be provided.
        
        Input Signals → Score Each Signal → Weighted Average → Make Decision → Format Output
        """
        logger.info(f"Evaluating retrain policy for model '{model_id or 'unknown'}'")

        signals: dict[str, Any] = {}
        reasons: list[str] = []
        score_components: list[tuple[float, float]] = []  # (score, weight)

        # --- Signal 1: Performance Degradation ---
        perf_score = 0.0
        if performance_metrics:
            perf_score = self._evaluate_performance(performance_metrics, reasons)
            signals["performance_degradation"] = {
                "current": performance_metrics.primary_metric_value,
                "baseline": performance_metrics.baseline_metric_value,
                "metric": performance_metrics.primary_metric_name,
                "signal_score": round(perf_score, 3),
            }
            score_components.append((perf_score, 0.35))  # Highest weight

        # --- Signal 2: Drift ---
        drift_score = 0.0
        if drift_report:
            drift_score = self._evaluate_drift(drift_report, reasons)
            signals["drift"] = {
                "severity": drift_report.overall_severity.value,
                "drift_proportion": drift_report.drift_proportion,
                "concept_drift": drift_report.concept_drift_detected,
                "signal_score": round(drift_score, 3),
            }
            score_components.append((drift_score, 0.30))

        # --- Signal 3: Staleness ---
        staleness_score = 0.0
        if last_training_date:
            staleness_score = self._evaluate_staleness(last_training_date, reasons)
            days_since = (datetime.now(timezone.utc) - last_training_date).days
            signals["staleness"] = {
                "days_since_training": days_since,
                "max_allowed_days": self.settings.max_staleness_days,
                "signal_score": round(staleness_score, 3),
            }
            score_components.append((staleness_score, 0.15))

        # --- Signal 4: Data Readiness ---
        data_ready = True
        if data_readiness:
            data_ready = data_readiness.is_sufficient
            signals["data_readiness"] = {
                "new_samples": data_readiness.new_samples_available,
                "min_required": data_readiness.min_samples_required,
                "is_sufficient": data_readiness.is_sufficient,
                "label_rate": data_readiness.label_availability_rate,
                "quality_score": data_readiness.quality_score,
            }
            if not data_ready:
                reasons.append(
                    f"Insufficient new data: {data_readiness.new_samples_available} "
                    f"available, {data_readiness.min_samples_required} required."
                )
            score_components.append((1.0 if data_ready else 0.0, 0.20))

        # --- Compute weighted priority score ---
        if score_components:
            total_weight = sum(w for _, w in score_components)
            priority_score = sum(s * w for s, w in score_components) / total_weight
        else:
            priority_score = 0.0

        # --- Make decision ---
        decision, confidence, recommended_action = self._make_decision(
            priority_score, perf_score, drift_score, staleness_score, data_ready,
            drift_report, performance_metrics
        )

        # Determine if human approval is needed
        requires_approval = (
            self.settings.approval_required
            and decision in (RetrainDecision.RECOMMENDED, RetrainDecision.REQUIRED)
        )

        # Urgent decisions bypass approval in production if auto-retrain is enabled
        if decision == RetrainDecision.URGENT and self.settings.auto_retrain_enabled:
            requires_approval = False

        assessment = RetrainAssessment(
            decision=decision,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            signals=signals,
            reasons=reasons,
            recommended_action=recommended_action,
            requires_human_approval=requires_approval,
            priority_score=priority_score,
        )

        logger.info(
            f"Retrain assessment for model '{model_id}': decision={decision.value}, "
            f"priority={priority_score:.3f}, confidence={confidence:.3f}"
        )

        return assessment

    def _evaluate_performance(
        self, metrics: PerformanceMetrics, reasons: list[str]
    ) -> float:
        """Score performance degradation signal (0 = fine, 1 = severe degradation)."""
        if metrics.baseline_metric_value == 0:
            return 0.0

        # Calculate relative degradation
        # Handle both higher-is-better and lower-is-better metrics
        lower_is_better = metrics.primary_metric_name.lower() in ("rmse", "mae", "mse", "loss", "error")

        if lower_is_better:
            relative_change = (metrics.primary_metric_value - metrics.baseline_metric_value) / abs(metrics.baseline_metric_value)
        else:
            relative_change = (metrics.baseline_metric_value - metrics.primary_metric_value) / abs(metrics.baseline_metric_value)

        if relative_change > self.settings.performance_degradation_threshold:
            reasons.append(
                f"Performance degradation detected: {metrics.primary_metric_name} "
                f"changed from {metrics.baseline_metric_value:.4f} to "
                f"{metrics.primary_metric_value:.4f} "
                f"(relative change: {relative_change:.2%})"
            )

        # Normalize to 0-1 score
        score = min(max(relative_change / (3 * self.settings.performance_degradation_threshold), 0), 1.0)
        return float(score)

    def _evaluate_drift(self, drift_report: DriftReport, reasons: list[str]) -> float:
        """Score drift signal."""
        severity_scores = {
            DriftSeverity.NONE: 0.0,
            DriftSeverity.LOW: 0.2,
            DriftSeverity.MODERATE: 0.5,
            DriftSeverity.HIGH: 0.8,
            DriftSeverity.CRITICAL: 1.0,
        }

        score = severity_scores.get(drift_report.overall_severity, 0.0)

        if drift_report.concept_drift_detected:
            score = max(score, 0.8)
            reasons.append("Concept drift detected in target variable distribution.")

        if drift_report.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            reasons.append(
                f"Data drift severity: {drift_report.overall_severity.value} "
                f"({drift_report.features_drifted}/{drift_report.features_total} features drifted)"
            )

        return score

    def _evaluate_staleness(
        self, last_training_date: datetime, reasons: list[str]
    ) -> float:
        """Score model staleness."""
        if last_training_date.tzinfo is None:
            last_training_date = last_training_date.replace(tzinfo=timezone.utc)

        days_since = (datetime.now(timezone.utc) - last_training_date).days
        staleness_ratio = days_since / self.settings.max_staleness_days

        if staleness_ratio >= 1.0:
            reasons.append(
                f"Model is stale: {days_since} days since last training "
                f"(max allowed: {self.settings.max_staleness_days} days)"
            )

        return float(min(staleness_ratio, 1.0))

    def _make_decision(
        self,
        priority_score: float,
        perf_score: float,
        drift_score: float,
        staleness_score: float,
        data_ready: bool,
        drift_report: Optional[DriftReport],
        performance_metrics: Optional[PerformanceMetrics],
    ) -> tuple[RetrainDecision, float, str]:
        """
        Make the final retrain decision based on aggregated signals.

        Returns (decision, confidence, recommended_action)
        """
        # Data not ready -> can't retrain regardless
        if not data_ready:
            return (
                RetrainDecision.NOT_NEEDED,
                0.7,
                "Collect more labeled data before considering retraining.",
            )

        # Urgent: Critical drift + performance degradation
        if perf_score > 0.7 and drift_score > 0.7:
            return (
                RetrainDecision.URGENT,
                0.95,
                "Immediate retraining required: significant performance degradation with critical drift.",
            )

        # Required: Clear performance degradation
        if perf_score > 0.5:
            return (
                RetrainDecision.REQUIRED,
                0.85,
                "Retrain model: measurable performance degradation detected.",
            )

        # Required: Concept drift
        if drift_report and drift_report.concept_drift_detected:
            return (
                RetrainDecision.REQUIRED,
                0.80,
                "Retrain model: concept drift detected in target distribution.",
            )

        # Recommended: Significant data drift
        if drift_score > 0.5:
            return (
                RetrainDecision.RECOMMENDED,
                0.70,
                "Retraining recommended: significant data drift detected.",
            )

        # Recommended: Model staleness
        if staleness_score >= 1.0:
            return (
                RetrainDecision.RECOMMENDED,
                0.60,
                "Retraining recommended: model exceeds maximum staleness threshold.",
            )

        # Priority score above threshold
        if priority_score > 0.5:
            return (
                RetrainDecision.RECOMMENDED,
                0.55,
                "Retraining recommended based on aggregated signal analysis.",
            )

        return (
            RetrainDecision.NOT_NEEDED,
            0.80,
            "Model is performing within acceptable parameters. Continue monitoring.",
        )