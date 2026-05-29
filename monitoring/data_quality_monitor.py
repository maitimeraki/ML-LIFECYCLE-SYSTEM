# monitoring/data_quality_monitor.py
"""
Continuous data quality monitoring using Great Expectations.

Runs GE validation on incoming production data batches.
Detects schema drift, null spikes, value range violations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.data.validation import DataValidator, DatasetSchema
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import GE_SUITE_SCORE, DATA_ROWS_VALIDATED

logger = logging.getLogger("ml_platform.monitoring.data_quality")


@dataclass
class DataQualityResult:
    """Result of one data quality monitoring run."""
    model_id:        str
    run_at:          datetime
    batch_rows:      int
    is_valid:        bool
    success_rate:    float
    failed_checks:   list[str]
    warnings:        list[str]
    suite_name:      str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id":      self.model_id,
            "run_at":        self.run_at.isoformat(),
            "batch_rows":    self.batch_rows,
            "is_valid":      self.is_valid,
            "success_rate":  round(self.success_rate, 4),
            "failed_checks": self.failed_checks[:10],
            "warnings":      self.warnings[:5],
        }


class DataQualityMonitor:
    """
    GE-based data quality monitoring on production batches.

    Detects:
      - Schema violations (missing columns, wrong types)
      - Null rate spikes (new nulls in previously clean columns)
      - Value range violations (negative ages, future dates)
      - Cardinality explosions (new categories appearing)

    Usage:
        monitor = DataQualityMonitor(model_id="churn_model", schema=schema)
        result  = monitor.check(incoming_batch_df)
        if not result.is_valid:
            alert(result.failed_checks)
    """

    def __init__(
        self,
        model_id:        str,
        schema:          DatasetSchema,
        pipeline_run_id: str = "",
    ) -> None:
        self.model_id        = model_id
        self.schema          = schema
        self.pipeline_run_id = pipeline_run_id
        self.validator       = DataValidator(schema=schema)

    def check(
        self,
        batch_df:   pd.DataFrame,
        batch_id:   str = "",
    ) -> DataQualityResult:
        """
        Run GE validation on an incoming production batch.
        Emits event and updates Prometheus.
        """
        logger.info(
            f"Data quality check: {self.model_id} | {len(batch_df)} rows"
        )

        suite_name = f"{self.model_id}_production_monitor"

        report = self.validator.validate(
            df=batch_df,
            dataset_id=batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pipeline_run_id=self.pipeline_run_id,
            model_id=self.model_id,
        )

        failed_checks = [
            e["expectation_type"]
            for e in report.expectation_results
            if not e.get("success", True)
        ]

        result = DataQualityResult(
            model_id=self.model_id,
            run_at=datetime.now(timezone.utc),
            batch_rows=len(batch_df),
            is_valid=report.is_valid,
            success_rate=report.success_rate,
            failed_checks=failed_checks,
            warnings=report.errors[:5],
            suite_name=suite_name,
        )

        # Update Prometheus
        GE_SUITE_SCORE.labels(suite_name=suite_name).set(report.success_rate)
        DATA_ROWS_VALIDATED.labels(model_id=self.model_id).inc(len(batch_df))

        # Emit event
        if not result.is_valid:
            event_bus.emit(make_event(
                pipeline_run_id=self.pipeline_run_id,
                event_type=EventType.ALERT_TRIGGERED,
                step_name="data_quality_monitoring",
                title=f"⚠️ Data Quality Alert: {self.model_id}",
                message=(
                    f"Batch quality: {report.success_rate:.1%} | "
                    f"Failed: {failed_checks[:3]}"
                ),
                model_id=self.model_id,
                severity="warning",
                data=result.to_dict(),
            ))

        return result