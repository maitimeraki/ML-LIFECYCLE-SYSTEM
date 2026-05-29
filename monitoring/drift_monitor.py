# monitoring/drift_monitor.py
"""
Continuous drift monitoring using Evidently.

Runs on a schedule (daily/weekly) against:
  reference = what current champion was trained on
  current   = recent production prediction requests

Produces:
  - Evidently HTML report (for human review)
  - Evidently JSON metrics (for Prometheus)
  - DriftReport (for retrain decision)
  - Events (for Grafana)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    DRIFT_FEATURES_DRIFTED,
    DRIFT_SCORE,
    DRIFT_SEVERITY,
    DRIFT_SEVERITY_MAP,
)

logger = logging.getLogger("ml_platform.monitoring.drift")


@dataclass
class DriftMonitorResult:
    """Result from one drift monitoring run."""
    model_id:            str
    run_at:              datetime
    reference_rows:      int
    current_rows:        int
    dataset_drift:       bool
    drift_share:         float          # Fraction of drifted features
    n_features_drifted:  int
    n_features_total:    int
    feature_details:     list[dict[str, Any]]
    report_html_path:    Optional[str]  = None
    report_json_path:    Optional[str]  = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id":           self.model_id,
            "run_at":             self.run_at.isoformat(),
            "reference_rows":     self.reference_rows,
            "current_rows":       self.current_rows,
            "dataset_drift":      self.dataset_drift,
            "drift_share":        round(self.drift_share, 4),
            "features_drifted":   self.n_features_drifted,
            "features_total":     self.n_features_total,
            "feature_details":    self.feature_details[:10],
            "report_html":        self.report_html_path,
        }


class DriftMonitor:
    """
    Evidently-based drift monitoring.

    Produces rich HTML reports for human inspection
    AND structured metrics for Prometheus/Grafana.

    Usage:
        monitor = DriftMonitor(model_id="churn_model")
        result  = monitor.run(reference_df, current_df)
        # HTML report saved to reports/drift/
        # Metrics pushed to Prometheus
        # Event emitted to EventBus
    """

    def __init__(
        self,
        model_id:        str,
        reports_dir:     str = "reports/drift",
        pipeline_run_id: str = "",
    ) -> None:
        self.model_id        = model_id
        self.reports_dir     = Path(reports_dir) / model_id
        self.pipeline_run_id = pipeline_run_id
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        reference_df:    pd.DataFrame,
        current_df:      pd.DataFrame,
        column_mapping:  Optional[Any] = None,
    ) -> DriftMonitorResult:
        """
        Run Evidently drift detection.

        Produces:
          1. HTML report → reports/drift/{model_id}/{timestamp}.html
          2. JSON metrics → reports/drift/{model_id}/{timestamp}.json
          3. DriftMonitorResult → for Prometheus + EventBus
        """
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset
        except ImportError:
            raise ImportError(
                "evidently not installed. "
                "Run: pip install evidently==0.7.21"
            )

        logger.info(
            f"Running Evidently drift monitor: {self.model_id} | "
            f"reference={len(reference_df)} | current={len(current_df)}"
        )

        # Build Evidently report
        report = Report(metrics=[
            DataDriftPreset(),
        ])

        report_eval= report.run(
            reference_data=reference_df,
            current_data=current_df,
        )

        # Save HTML report for human inspection
        timestamp     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        html_path     = str(self.reports_dir / f"drift_{timestamp}.html")
        json_path     = str(self.reports_dir / f"drift_{timestamp}.json")

        report_eval.save_html(html_path)
        
        # Save JSON report — use .dict() then dump to file
        report_dict = report_eval.dict()
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        # Extract metrics from Evidently result
        drift_result  = self._parse_evidently_result(report_dict)

        # Build monitor result
        n_features    = len(reference_df.columns)
        n_drifted     = int(drift_result.get("n_drifted_features", 0))
        drift_share   = drift_result.get("share_drifted_features", 0.0)
        dataset_drift = drift_result.get("dataset_drift", False)

        feature_details = self._extract_feature_details(report_dict)

        monitor_result = DriftMonitorResult(
            model_id=self.model_id,
            run_at=datetime.now(timezone.utc),
            reference_rows=len(reference_df),
            current_rows=len(current_df),
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            n_features_drifted=n_drifted,
            n_features_total=n_features,
            feature_details=feature_details,
            report_html_path=html_path,
            report_json_path=json_path,
        )

        # Update Prometheus
        DRIFT_SCORE.labels(
            model_id=self.model_id
        ).set(drift_share)

        DRIFT_FEATURES_DRIFTED.labels(
            model_id=self.model_id
        ).set(n_drifted)

        severity_val = "high" if dataset_drift else (
            "moderate" if drift_share > 0.3 else
            "low"      if drift_share > 0.1 else "none"
        )
        DRIFT_SEVERITY.labels(
            model_id=self.model_id
        ).set(DRIFT_SEVERITY_MAP.get(severity_val, 0))

        # Emit event
        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_DETECTION_COMPLETED,
            step_name="drift_monitoring",
            title=(
                f"{'🔴' if dataset_drift else '🟢'} "
                f"Evidently Drift: {self.model_id} | "
                f"{n_drifted}/{n_features} features"
            ),
            message=(
                f"Dataset drift: {dataset_drift} | "
                f"Drift share: {drift_share:.1%} | "
                f"Report: {html_path}"
            ),
            model_id=self.model_id,
            status="warning" if dataset_drift else "success",
            data=monitor_result.to_dict(),
        ))

        logger.info(
            f"Drift monitor complete: {self.model_id} | "
            f"dataset_drift={dataset_drift} | "
            f"drifted={n_drifted}/{n_features} | "
            f"report={html_path}"
        )

        return monitor_result

    def _parse_evidently_result(
        self, result_dict: dict
    ) -> dict[str, Any]:
        """Extract dataset-level drift metrics from Evidently result."""
        try:
            for metric in result_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    r = metric.get("result", {})
                    return {
                        "dataset_drift":           r.get("dataset_drift", False),
                        "n_drifted_features":      r.get("number_of_drifted_columns", 0),
                        "share_drifted_features":  r.get("share_of_drifted_columns", 0.0),
                        "n_features":              r.get("number_of_columns", 0),
                    }
        except Exception as exc:
            logger.warning(f"Could not parse Evidently result: {exc}")
        return {}

    def _extract_feature_details(
        self, result_dict: dict
    ) -> list[dict[str, Any]]:
        """Extract per-feature drift details."""
        details = []
        try:
            for metric in result_dict.get("metrics", []):
                if "ColumnDriftMetric" in str(metric.get("metric", "")):
                    r = metric.get("result", {})
                    details.append({
                        "column":    r.get("column_name", ""),
                        "drifted":   r.get("drift_detected", False),
                        "statistic": r.get("statistic", ""),
                        "p_value":   r.get("p_value"),
                        "drift_score": r.get("drift_score"),
                    })
        except Exception:
            pass
        return details