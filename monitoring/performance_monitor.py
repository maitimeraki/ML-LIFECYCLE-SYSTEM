# monitoring/performance_monitor.py
"""
Live model performance monitor.

Tracks predictions against delayed ground truth.
Detects performance degradation and triggers alerts.

Ground truth arrives with delay in production:
  t=0:   prediction made, stored with request_id    
  t+24h: label arrives (fraud confirmed, churn happened)
  monitor matches label to prediction, computes metrics
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import LIVE_MODEL_METRIC

logger = logging.getLogger("ml_platform.monitoring.performance")


@dataclass
class PredictionRecord:
    """Stored prediction awaiting ground truth."""
    request_id:    str
    prediction:    Any
    ground_truth:  Optional[Any]        = None
    timestamp:     datetime             = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    model_version: str                  = ""
    labeled_at:    Optional[datetime]   = None


@dataclass
class PerformanceWindow:
    """Metrics computed over a rolling time window."""
    window_start:  datetime
    window_end:    datetime
    sample_count:  int
    labeled_count: int
    label_rate:    float
    metrics:       dict[str, float]
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start":  self.window_start.isoformat(),
            "window_end":    self.window_end.isoformat(),
            "sample_count":  self.sample_count,
            "labeled_count": self.labeled_count,
            "label_rate":    round(self.label_rate, 4),
            "metrics":       {k: round(v, 4) for k, v in self.metrics.items()},
            "model_version": self.model_version,
        }


class PerformanceMonitor:
    """
    Tracks model performance against delayed ground truth.

    Usage:
        monitor = PerformanceMonitor(model_id="churn_model")

        # At prediction time:
        monitor.record_prediction(request_id, prediction, model_version)

        # When label arrives (hours/days later):
        monitor.record_ground_truth(request_id, actual_label)

        # Check if model is degrading:
        is_degraded, window = monitor.check_degradation()
    """

    def __init__(
        self,
        model_id:              str,
        primary_metric:        str   = "f1_score",
        baseline_value:        float = 0.0,
        degradation_threshold: float = 0.05,
        buffer_size:           int   = 100_000,
        is_classification:     bool  = True,
        pipeline_run_id:       str   = "",
    ) -> None:
        self.model_id              = model_id
        self.primary_metric        = primary_metric
        self.baseline_value        = baseline_value
        self.degradation_threshold = degradation_threshold
        self.is_classification     = is_classification
        self.pipeline_run_id       = pipeline_run_id

        self._buffer:         deque[PredictionRecord] = deque(maxlen=buffer_size)
        self._by_request_id:  dict[str, PredictionRecord] = {}
        self._window_history: list[PerformanceWindow] = []

    def record_prediction(
        self,
        request_id:    str,
        prediction:    Any,
        model_version: str = "",
    ) -> None:
        """Store prediction for later matching with ground truth."""
        record = PredictionRecord(
            request_id=request_id,
            prediction=prediction,
            model_version=model_version,
        )
        self._buffer.append(record)
        self._by_request_id[request_id] = record

    def record_ground_truth(
        self,
        request_id:   str,
        ground_truth: Any,
    ) -> bool:
        """
        Match incoming ground truth to a stored prediction.
        Returns True if matched.
        """
        record = self._by_request_id.get(request_id)
        if record:
            record.ground_truth = ground_truth
            record.labeled_at   = datetime.now(timezone.utc)
            return True
        return False

    def compute_window(
        self,
        window_hours:  int            = 24,
        model_version: Optional[str]  = None,
    ) -> Optional[PerformanceWindow]:
        """
        Compute metrics over rolling window of labeled predictions.
        Minimum 30 labeled samples required for reliable metrics.
        """
        now          = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=window_hours)

        labeled = [
            r for r in self._buffer
            if r.ground_truth is not None
            and r.timestamp >= window_start
            and (model_version is None or r.model_version == model_version)
        ]

        all_in_window = [
            r for r in self._buffer
            if r.timestamp >= window_start
            and (model_version is None or r.model_version == model_version)
        ]

        if len(labeled) < 30:
            logger.debug(
                f"Insufficient labeled data: {len(labeled)} < 30 "
                f"in {window_hours}h window"
            )
            return None

        y_pred = np.array([r.prediction for r in labeled])
        y_true = np.array([r.ground_truth for r in labeled])

        metrics = self._compute_metrics(y_pred, y_true)

        # Update Prometheus
        for metric_name, value in metrics.items():
            LIVE_MODEL_METRIC.labels(
                model_id=self.model_id,
                model_version=model_version or "unknown",
                metric=metric_name,
            ).set(value)

        label_rate = (
            len(labeled) / len(all_in_window)
            if all_in_window else 0.0
        )

        window = PerformanceWindow(
            window_start=window_start,
            window_end=now,
            sample_count=len(all_in_window),
            labeled_count=len(labeled),
            label_rate=label_rate,
            metrics=metrics,
            model_version=model_version or (
                labeled[0].model_version if labeled else ""
            ),
        )

        self._window_history.append(window)
        return window

    def check_degradation(
        self,
        window_hours:  int           = 24,
        model_version: Optional[str] = None,
    ) -> tuple[bool, Optional[PerformanceWindow]]:
        """
        Check if model performance has degraded below threshold.
        Returns (is_degraded, window).
        """
        window = self.compute_window(window_hours, model_version)
        if window is None:
            return False, None

        current = window.metrics.get(self.primary_metric, 0.0)

        lower_is_better = self.primary_metric in ("rmse", "mae", "log_loss")

        if lower_is_better:
            relative_change = (
                (current - self.baseline_value)
                / abs(self.baseline_value)
                if self.baseline_value != 0 else 0.0
            )
        else:
            relative_change = (
                (self.baseline_value - current)
                / abs(self.baseline_value)
                if self.baseline_value != 0 else 0.0
            )

        is_degraded = relative_change > self.degradation_threshold

        if is_degraded:
            logger.warning(
                f"Performance degradation: {self.model_id} | "
                f"{self.primary_metric}={current:.4f} "
                f"(baseline={self.baseline_value:.4f}, "
                f"change={relative_change:.2%})"
            )

            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.PERFORMANCE_DEGRADATION,
                step_name="monitoring",
                title=f"⚠️ Performance Degradation: {self.model_id}",
                message=(
                    f"{self.primary_metric}={current:.4f} "
                    f"(baseline={self.baseline_value:.4f}, "
                    f"change={relative_change:+.2%})"
                ),
                model_id=self.model_id,
                severity="warning",
                data=window.to_dict(),
            ))

        return is_degraded, window

    def _compute_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self.is_classification:
            n_classes = len(np.unique(y_true))
            average   = "binary" if n_classes == 2 else "weighted"

            metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
            metrics["f1_score"]  = float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            )
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
            metrics["recall"]    = float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"]  = float(mean_absolute_error(y_true, y_pred))

        return metrics

    @property
    def summary(self) -> dict[str, Any]:
        total   = len(self._buffer)
        labeled = sum(1 for r in self._buffer if r.ground_truth is not None)
        return {
            "model_id":       self.model_id,
            "total":          total,
            "labeled":        labeled,
            "label_rate":     labeled / total if total > 0 else 0.0,
            "baseline":       {
                "metric": self.primary_metric,
                "value":  self.baseline_value,
            },
            "windows_computed": len(self._window_history),
        }