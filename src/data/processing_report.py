# src/data/processing_report.py
"""
Audit trail — unchanged from before.
Every transformation recorded for debugging + compliance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ColumnTransformation:
    column_name: str
    stage: str
    action: str
    original_dtype: str
    result_dtype: str
    rows_affected: int
    rows_total: int
    parameters: dict[str, Any] = field(default_factory=dict)
    before_stats: dict[str, Any] = field(default_factory=dict)
    after_stats: dict[str, Any] = field(default_factory=dict)
    warning: Optional[str] = None

    @property
    def pct_affected(self) -> float:
        return self.rows_affected / self.rows_total if self.rows_total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column":         self.column_name,
            "stage":          self.stage,
            "action":         self.action,
            "original_dtype": self.original_dtype,
            "result_dtype":   self.result_dtype,
            "rows_affected":  self.rows_affected,
            "pct_affected":   round(self.pct_affected, 4),
            "parameters":     self.parameters,
            "before_stats":   self.before_stats,
            "after_stats":    self.after_stats,
            "warning":        self.warning,
        }


@dataclass
class ProcessingReport:
    run_id: str
    model_id: str
    pipeline_run_id: str
    started_at: datetime

    input_rows: int = 0
    input_cols: int = 0
    output_rows: int = 0
    output_cols: int = 0

    dropped_columns: list[str] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)
    transformations: list[ColumnTransformation] = field(default_factory=list)

    duplicate_rows_removed: int = 0
    rows_dropped_null: int = 0
    rows_dropped_outlier: int = 0

    warnings: list[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None

    def add_transformation(self, t: ColumnTransformation) -> None:
        self.transformations.append(t)
        if t.warning:
            self.warnings.append(f"[{t.column_name}:{t.stage}] {t.warning}")

    def to_dict(self) -> dict[str, Any]:
        duration = (
            (self.completed_at - self.started_at).total_seconds()
            if self.completed_at else None
        )
        return {
            "run_id":           self.run_id,
            "model_id":         self.model_id,
            "pipeline_run_id":  self.pipeline_run_id,
            "started_at":       self.started_at.isoformat(),
            "completed_at":     self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
            "shape": {
                "input":        {"rows": self.input_rows,  "cols": self.input_cols},
                "output":       {"rows": self.output_rows, "cols": self.output_cols},
                "rows_removed": self.input_rows - self.output_rows,
                "cols_removed": self.input_cols - self.output_cols,
            },
            "row_audit": {
                "duplicates_removed":   self.duplicate_rows_removed,
                "rows_dropped_null":    self.rows_dropped_null,
                "rows_dropped_outlier": self.rows_dropped_outlier,
            },
            "column_audit": {
                "dropped": self.dropped_columns,
                "added":   self.added_columns,
                "renamed": self.renamed_columns,
            },
            "transformations": [t.to_dict() for t in self.transformations],
            "warnings":        self.warnings,
            "warning_count":   len(self.warnings),
        }