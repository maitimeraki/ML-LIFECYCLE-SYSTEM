# monitoring/monitoring_pipeline.py
"""
Monitoring orchestrator.

Runs all monitors on a schedule and coordinates alerts.
Called by Airflow monitoring DAG (separate from training DAG).

Schedule: every 6 hours
  → data quality check on latest production batch
  → performance check against labeled predictions
  → drift check against reference

If any monitor triggers:
  → alert via AlertManager
  → set Airflow Variable to trigger retraining
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from monitoring.alerting import AlertManager
from monitoring.data_quality_monitor import DataQualityMonitor, DataQualityResult
from monitoring.drift_monitor import DriftMonitor, DriftMonitorResult
from monitoring.performance_monitor import PerformanceMonitor, PerformanceWindow
from src.data.validation import DatasetSchema

logger = logging.getLogger("ml_platform.monitoring.pipeline")


@dataclass
class MonitoringResult:
    """Combined result of all monitoring checks."""
    model_id:              str
    run_at:                datetime
    data_quality:          Optional[DataQualityResult]   = None
    performance:           Optional[PerformanceWindow]   = None
    drift:                 Optional[DriftMonitorResult]  = None
    any_alert_triggered:   bool                          = False
    retrain_recommended:   bool                          = False
    summary:               dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id":            self.model_id,
            "run_at":              self.run_at.isoformat(),
            "any_alert_triggered": self.any_alert_triggered,
            "retrain_recommended": self.retrain_recommended,
            "data_quality": (
                self.data_quality.to_dict() if self.data_quality else None
            ),
            "drift": (
                self.drift.to_dict() if self.drift else None
            ),
            "performance": (
                self.performance.to_dict() if self.performance else None
            ),
        }


class MonitoringPipeline:
    """
    Orchestrates all monitoring checks.

    Usage (called by Airflow monitoring DAG every 6 hours):
        pipeline = MonitoringPipeline(
            model_id="churn_model",
            schema=schema,
            baseline_f1=0.80,
        )
        result = pipeline.run(
            production_batch=recent_df,
            reference_df=reference_df,
            performance_monitor=monitor,
        )
    """

    def __init__(
        self,
        model_id:          str,
        schema:            DatasetSchema,
        baseline_metric:   float = 0.0,
        primary_metric:    str   = "f1_score",
        slack_webhook:     Optional[str] = None,
        pagerduty_key:     Optional[str] = None,
        pipeline_run_id:   str = "",
    ) -> None:
        self.model_id   = model_id

        self.quality_monitor = DataQualityMonitor(
            model_id=model_id,
            schema=schema,
            pipeline_run_id=pipeline_run_id,
        )

        self.drift_monitor = DriftMonitor(
            model_id=model_id,
            pipeline_run_id=pipeline_run_id,
        )

        self.alert_manager = AlertManager(
            model_id=model_id,
            slack_webhook=slack_webhook,
            pagerduty_key=pagerduty_key,
            pipeline_run_id=pipeline_run_id,
        )

        self.baseline_metric = baseline_metric
        self.primary_metric  = primary_metric

    def run(
        self,
        production_batch:    pd.DataFrame,
        reference_df:        pd.DataFrame,
        performance_monitor: Optional[PerformanceMonitor] = None,
        window_hours:        int = 24,
    ) -> MonitoringResult:
        """
        Run all monitors and route alerts.

        Parameters
        ----------
        production_batch:
            Recent production data (last N hours).
        reference_df:
            Reference data (what champion was trained on).
        performance_monitor:
            Shared monitor instance with stored predictions.
        """
        logger.info(
            f"Monitoring pipeline: {self.model_id} | "
            f"batch={len(production_batch)} | "
            f"reference={len(reference_df)}"
        )

        result = MonitoringResult(
            model_id=self.model_id,
            run_at=datetime.now(timezone.utc),
        )

        # ── 1. Data quality ────────────────────────────────────────────────
        try:
            result.data_quality = self.quality_monitor.check(
                batch_df=production_batch,
                batch_id=f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            if not result.data_quality.is_valid:
                result.any_alert_triggered = True
                logger.warning(
                    f"Data quality alert: "
                    f"{result.data_quality.failed_checks}"
                )
        except Exception as exc:
            logger.error(f"Data quality monitor failed: {exc}")

        # ── 2. Drift detection ─────────────────────────────────────────────
        try:
            result.drift = self.drift_monitor.run(
                reference_df=reference_df,
                current_df=production_batch,
            )

            if result.drift.dataset_drift:
                result.any_alert_triggered = True
                result.retrain_recommended = True
                self.alert_manager.drift_alert(
                    drift_share=result.drift.drift_share,
                    n_drifted=result.drift.n_features_drifted,
                    n_total=result.drift.n_features_total,
                    dataset_drift=result.drift.dataset_drift,
                )
        except Exception as exc:
            logger.error(f"Drift monitor failed: {exc}")

        # ── 3. Performance monitoring ──────────────────────────────────────
        if performance_monitor is not None:
            try:
                is_degraded, window = performance_monitor.check_degradation(
                    window_hours=window_hours
                )
                result.performance = window

                if is_degraded and window is not None:
                    result.any_alert_triggered = True
                    result.retrain_recommended = True

                    current = window.metrics.get(self.primary_metric, 0.0)
                    self.alert_manager.performance_degradation_alert(
                        current_value=current,
                        baseline_value=self.baseline_metric,
                        metric_name=self.primary_metric,
                        window_hours=window_hours,
                    )
            except Exception as exc:
                logger.error(f"Performance monitor failed: {exc}")

        logger.info(
            f"Monitoring complete: {self.model_id} | "
            f"alerts={result.any_alert_triggered} | "
            f"retrain_recommended={result.retrain_recommended}"
        )

        return result