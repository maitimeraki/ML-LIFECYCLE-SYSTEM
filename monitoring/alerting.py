# monitoring/alert_manager.py
"""
Alert routing for ML platform.

Sources:
  - PerformanceMonitor  → degradation alerts
  - DriftMonitor        → drift alerts
  - DataQualityMonitor  → quality alerts
  - Deployment          → rollback alerts

Channels:
  - Prometheus Alertmanager (primary)
  - Slack (human notification)
  - PagerDuty (on-call escalation)
  - Airflow Variable (triggers retraining pipeline)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.observability.event_bus import EventType, event_bus, make_event

logger = logging.getLogger("ml_platform.monitoring.alerts")


class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    model_id:    str
    title:       str
    message:     str
    severity:    AlertSeverity
    source:      str
    data:        dict[str, Any]
    triggered_at: datetime


class AlertManager:
    """
    Routes alerts to appropriate channels based on severity.

    INFO     → Grafana annotation only
    WARNING  → Slack notification + Grafana
    CRITICAL → PagerDuty + Slack + Airflow trigger
    """

    def __init__(
        self,
        model_id:         str,
        slack_webhook:    Optional[str] = None,
        pagerduty_key:    Optional[str] = None,
        pipeline_run_id:  str = "",
    ) -> None:
        self.model_id        = model_id
        self.slack_webhook   = slack_webhook
        self.pagerduty_key   = pagerduty_key
        self.pipeline_run_id = pipeline_run_id

    def fire(self, alert: Alert) -> None:
        """Route alert to appropriate channels."""
        logger.warning(
            f"ALERT [{alert.severity.value.upper()}] {alert.title}: "
            f"{alert.message}"
        )

        # Always emit to EventBus → Grafana
        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.ALERT_TRIGGERED,
            step_name="alert_manager",
            title=alert.title,
            message=alert.message,
            model_id=self.model_id,
            severity=alert.severity.value,
            status="warning" if alert.severity != AlertSeverity.CRITICAL else "failed",
            data=alert.data,
        ))

        if alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
            self._send_slack(alert)

        if alert.severity == AlertSeverity.CRITICAL:
            self._send_pagerduty(alert)
            self._trigger_airflow_pipeline(alert)

    def performance_degradation_alert(
        self,
        current_value: float,
        baseline_value: float,
        metric_name:   str,
        window_hours:  int,
    ) -> None:
        change = (baseline_value - current_value) / max(abs(baseline_value), 1e-8)

        severity = (
            AlertSeverity.CRITICAL if change > 0.15
            else AlertSeverity.WARNING
        )

        self.fire(Alert(
            model_id=self.model_id,
            title=f"Performance Degradation: {self.model_id}",
            message=(
                f"{metric_name}: {current_value:.4f} "
                f"(baseline={baseline_value:.4f}, "
                f"drop={change:.1%}, window={window_hours}h)"
            ),
            severity=severity,
            source="performance_monitor",
            data={
                "metric":       metric_name,
                "current":      current_value,
                "baseline":     baseline_value,
                "change_pct":   round(change * 100, 2),
                "window_hours": window_hours,
            },
            triggered_at=datetime.now(timezone.utc),
        ))

    def drift_alert(
        self,
        drift_share:     float,
        n_drifted:       int,
        n_total:         int,
        dataset_drift:   bool,
    ) -> None:
        severity = (
            AlertSeverity.CRITICAL if dataset_drift and drift_share > 0.5
            else AlertSeverity.WARNING if drift_share > 0.3
            else AlertSeverity.INFO
        )

        self.fire(Alert(
            model_id=self.model_id,
            title=f"Data Drift Detected: {self.model_id}",
            message=(
                f"{n_drifted}/{n_total} features drifted "
                f"({drift_share:.1%}) | "
                f"Dataset drift: {dataset_drift}"
            ),
            severity=severity,
            source="drift_monitor",
            data={
                "drift_share":   drift_share,
                "n_drifted":     n_drifted,
                "n_total":       n_total,
                "dataset_drift": dataset_drift,
            },
            triggered_at=datetime.now(timezone.utc),
        ))

    def _send_slack(self, alert: Alert) -> None:
        if not self.slack_webhook:
            logger.debug("Slack webhook not configured")
            return
        try:
            import requests
            emoji = "🔴" if alert.severity == AlertSeverity.CRITICAL else "⚠️"
            requests.post(
                self.slack_webhook,
                json={
                    "text": (
                        f"{emoji} *{alert.title}*\n"
                        f"{alert.message}\n"
                        f"Model: `{self.model_id}` | "
                        f"Source: `{alert.source}`"
                    )
                },
                timeout=5,
            )
        except Exception as exc:
            logger.warning(f"Slack alert failed: {exc}")

    def _send_pagerduty(self, alert: Alert) -> None:
        if not self.pagerduty_key:
            logger.debug("PagerDuty key not configured")
            return
        try:
            import requests
            requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json={
                    "routing_key":  self.pagerduty_key,
                    "event_action": "trigger",
                    "payload": {
                        "summary":   alert.title,
                        "source":    f"ml-platform/{self.model_id}",
                        "severity":  "critical",
                        "custom_details": alert.data,
                    },
                },
                timeout=5,
            )
        except Exception as exc:
            logger.warning(f"PagerDuty alert failed: {exc}")

    def _trigger_airflow_pipeline(self, alert: Alert) -> None:
        """
        Trigger Airflow DAG for critical alerts.
        Sets Airflow Variable → DAG sensor picks it up.
        """
        try:
            from airflow.models import Variable
            Variable.set(
                f"trigger_retrain_{self.model_id}",
                "true"
            )
            logger.info(
                f"Airflow retrain trigger set for {self.model_id}"
            )
        except Exception as exc:
            logger.warning(f"Airflow trigger failed: {exc}")