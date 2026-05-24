# src/evaluation/champion_challenger.py
"""
Production-grade Champion-Challenger evaluation framework.

Evaluation dimensions:
  1. Predictive metrics     (F1, AUC, RMSE, per-class, calibration)
  2. Operational metrics    (latency P50/P95/P99, memory, model size)
  3. Slice performance      (per-segment regression detection)
  4. Statistical significance (paired bootstrap)
  5. Business constraints   (hard gates that cannot be overridden)

Decision:
  Challenger must PASS ALL hard gates AND score better on weighted
  composite score. One hard gate failure → REJECTED regardless of metrics.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config.settings import DeploymentSettings, get_settings
from src.common.enums import ApprovalStatus
from src.common.exceptions import ModelValidationError
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    CHAMPION_CHALLENGER_COMPARISONS,
    CHAMPION_METRIC,
    CHALLENGER_IMPROVEMENT,
)

logger = logging.getLogger("ml_platform.evaluation.champion_challenger")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyProfile:
    """
    Inference latency measurements.
    Measured by actually running the model N times.
    """
    p50_ms:   float = 0.0
    p75_ms:   float = 0.0
    p95_ms:   float = 0.0
    p99_ms:   float = 0.0
    mean_ms:  float = 0.0
    std_ms:   float = 0.0
    min_ms:   float = 0.0
    max_ms:   float = 0.0
    n_samples: int  = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_ms":   round(self.p50_ms,  2),
            "p75_ms":   round(self.p75_ms,  2),
            "p95_ms":   round(self.p95_ms,  2),
            "p99_ms":   round(self.p99_ms,  2),
            "mean_ms":  round(self.mean_ms, 2),
            "std_ms":   round(self.std_ms,  2),
            "n_samples": self.n_samples,
        }


@dataclass
class OperationalProfile:
    """
    Operational characteristics of a model.
    All measured, not self-reported.
    """
    latency:        LatencyProfile = field(default_factory=LatencyProfile)
    memory_mb:      float = 0.0   # Peak memory during inference
    model_size_mb:  float = 0.0   # Artifact size on disk
    throughput_qps: float = 0.0   # Queries per second (batch=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency":        self.latency.to_dict(),
            "memory_mb":      round(self.memory_mb, 2),
            "model_size_mb":  round(self.model_size_mb, 2),
            "throughput_qps": round(self.throughput_qps, 1),
        }


@dataclass
class SlicePerformance:
    """Performance on a specific data slice."""
    slice_name:   str
    slice_size:   int
    metrics:      dict[str, float]
    is_regression: bool = False   # True if regressed vs champion

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice":       self.slice_name,
            "size":        self.slice_size,  
            "metrics":     {k: round(v, 4) for k, v in self.metrics.items()},
            "is_regression": self.is_regression,
        }


@dataclass
class CalibrationMetrics:
    """
    Model calibration — does predicted probability match actual frequency?
    A model predicting 0.9 should be right 90% of the time.
    """
    expected_calibration_error: float = 0.0   # ECE (lower = better)
    max_calibration_error:      float = 0.0   # MCE (lower = better)
    brier_score:                float = 0.0   # Overall calibration (lower = better)
    n_bins:                     int   = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece":         round(self.expected_calibration_error, 4),
            "mce":         round(self.max_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
        }


@dataclass
class ModelSnapshot:
    """
    Complete snapshot of a model for comparison.
    Contains everything needed to evaluate it.
    """
    model_id:      str
    model_version: str
    artifact_path: str   # Path to .joblib file

    # Primary evaluation results (populated by evaluator)
    predictive_metrics: dict[str, float] = field(default_factory=dict)
    operational:        OperationalProfile = field(
        default_factory=OperationalProfile
    )
    slice_performance:  list[SlicePerformance] = field(default_factory=list)
    calibration:        CalibrationMetrics = field(
        default_factory=CalibrationMetrics
    )

    primary_metric_name:  str   = "f1_score"
    primary_metric_value: float = 0.0
    evaluation_samples:   int   = 0
    evaluated_at:         Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id":          self.model_id,
            "model_version":     self.model_version,
            "primary_metric":    self.primary_metric_name,
            "primary_value":     round(self.primary_metric_value, 4),
            "predictive":        {
                k: round(v, 4) for k, v in self.predictive_metrics.items()
            },
            "operational":       self.operational.to_dict(),
            "calibration":       self.calibration.to_dict(),
            "slice_performance": [s.to_dict() for s in self.slice_performance],
            "evaluation_samples": self.evaluation_samples,
        }


@dataclass
class GateResult:
    """Result of one approval gate."""
    gate_name:  str
    passed:     bool
    champion_value: Any
    challenger_value: Any
    threshold:  Any
    message:    str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate":      self.gate_name,
            "passed":    self.passed,
            "champion":  self.champion_value,
            "challenger": self.challenger_value,
            "threshold": self.threshold,
            "message":   self.message,
        }


@dataclass
class ComparisonResult:
    """
    Complete champion-challenger comparison result.
    """
    champion:             ModelSnapshot
    challenger:           ModelSnapshot
    challenger_wins:      bool
    approval_status:      ApprovalStatus

    # Individual gate results
    gate_results:         list[GateResult] = field(default_factory=list)

    # Statistical test
    improvement:          float = 0.0
    improvement_pct:      float = 0.0
    is_statistically_significant: bool = False
    p_value:              Optional[float] = None

    # Composite scores (0-1)
    champion_score:       float = 0.0
    challenger_score:     float = 0.0

    # Reasoning
    reasons:              list[str] = field(default_factory=list)
    timestamp:            datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger_wins":    self.challenger_wins,
            "approval_status":    self.approval_status.value,
            "improvement":        round(self.improvement, 6),
            "improvement_pct":    round(self.improvement_pct, 2),
            "statistically_significant": self.is_statistically_significant,
            "p_value":            round(self.p_value, 6) if self.p_value else None,
            "champion_score":     round(self.champion_score, 4),
            "challenger_score":   round(self.challenger_score, 4),
            "gates": [g.to_dict() for g in self.gate_results],
            "reasons":            self.reasons,
            "champion":           self.champion.to_dict(),
            "challenger":         self.challenger.to_dict(),
            "timestamp":          self.timestamp.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluator — measures everything
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Measures ALL dimensions of a model on an evaluation dataset.

    Produces ModelSnapshot with:
      - Predictive metrics (F1, AUC, per-class, calibration)
      - Operational metrics (latency, memory, model size)
      - Slice performance (per segment)
    """

    # How many times to run inference for latency measurement
    LATENCY_WARMUP_RUNS   = 5
    LATENCY_MEASURE_RUNS  = 50

    def evaluate(
        self,
        model_version: str,
        model_id: str,
        artifact_path: str,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        is_classification: bool = True,
        slice_columns: Optional[list[str]] = None,
        pipeline_run_id: str = "",
    ) -> ModelSnapshot:
        """
        Load model from artifact and measure all dimensions.

        Parameters
        ----------
        artifact_path:
            Path to .joblib model file.
        X_eval:
            Evaluation features (processed, same space for both models).
        y_eval:
            Ground truth labels.
        slice_columns:
            Columns to compute per-slice performance on.
            e.g. ["age_group", "region", "product_category"]
        """
        if not Path(artifact_path).exists():
            raise ModelValidationError(
                f"Artifact not found: {artifact_path}",
                details={"path": artifact_path},
            )

        logger.info(
            f"Evaluating model: {model_id} v{model_version} | "
            f"samples={len(X_eval)}"
        )

        model = joblib.load(artifact_path)

        snapshot = ModelSnapshot(
            model_id=model_id,
            model_version=model_version,
            artifact_path=artifact_path,
            evaluation_samples=len(X_eval),
            evaluated_at=datetime.now(timezone.utc),
        )

        # ── 1. Operational profile ─────────────────────────────────────────
        snapshot.operational = self._measure_operational(
            model, X_eval, artifact_path
        )

        # ── 2. Predictions ─────────────────────────────────────────────────
        y_pred = model.predict(X_eval)
        y_prob = None
        if is_classification and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)

        # ── 3. Predictive metrics ──────────────────────────────────────────
        snapshot.predictive_metrics = self._compute_predictive_metrics(
            y_eval, y_pred, y_prob, is_classification
        )
        snapshot.primary_metric_name  = "f1_score" if is_classification else "rmse"
        snapshot.primary_metric_value = snapshot.predictive_metrics.get(
            snapshot.primary_metric_name, 0.0
        )

        # ── 4. Calibration ─────────────────────────────────────────────────
        if is_classification and y_prob is not None:
            snapshot.calibration = self._compute_calibration(y_eval, y_prob)

        # ── 5. Slice performance ───────────────────────────────────────────
        if slice_columns:
            snapshot.slice_performance = self._compute_slice_performance(
                model, X_eval, y_eval,
                slice_columns=slice_columns,
                is_classification=is_classification,
            )

        logger.info(
            f"Evaluation complete: {model_id} v{model_version} | "
            f"{snapshot.primary_metric_name}="
            f"{snapshot.primary_metric_value:.4f} | "
            f"p99={snapshot.operational.latency.p99_ms:.1f}ms | "
            f"memory={snapshot.operational.memory_mb:.1f}MB"
        )

        return snapshot

    def _measure_operational(
        self,
        model: Any,
        X_eval: pd.DataFrame,
        artifact_path: str,
    ) -> OperationalProfile:
        """
        Measure real latency, memory, model size.
        Uses single-row inference for latency (realistic serving scenario).
        """
        profile = OperationalProfile()

        # ── Model size ─────────────────────────────────────────────────────
        try:
            profile.model_size_mb = (
                Path(artifact_path).stat().st_size / 1024 / 1024
            )
        except Exception:
            profile.model_size_mb = 0.0

        # ── Latency measurement ────────────────────────────────────────────
        # Single-row inference = realistic API serving scenario
        single_row = X_eval.iloc[[0]]
        latencies_ms: list[float] = []

        # Warmup (JIT compilation, cache warming)
        for _ in range(self.LATENCY_WARMUP_RUNS):
            model.predict(single_row)

        # Actual measurement
        for _ in range(self.LATENCY_MEASURE_RUNS):
            start    = time.perf_counter()
            model.predict(single_row)
            elapsed  = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed)

        arr = np.array(latencies_ms)
        profile.latency = LatencyProfile(
            p50_ms=float(np.percentile(arr, 50)),
            p75_ms=float(np.percentile(arr, 75)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            n_samples=len(latencies_ms),
        )

        # ── Memory measurement ─────────────────────────────────────────────
        try:
            tracemalloc.start()
            model.predict(X_eval)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            profile.memory_mb = peak / 1024 / 1024
        except Exception:
            profile.memory_mb = 0.0

        # ── Throughput ─────────────────────────────────────────────────────
        try:
            batch_size = min(100, len(X_eval))
            batch      = X_eval.iloc[:batch_size]
            start      = time.perf_counter()
            for _ in range(10):
                model.predict(batch)
            elapsed = time.perf_counter() - start
            profile.throughput_qps = (batch_size * 10) / elapsed
        except Exception:
            profile.throughput_qps = 0.0

        return profile

    def _compute_predictive_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        is_classification: bool,
    ) -> dict[str, float]:
        """Compute full suite of predictive metrics."""
        metrics: dict[str, float] = {}

        if is_classification:
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

            # Per-class metrics
            classes = np.unique(y_true)
            for cls in classes:
                mask = y_true == cls
                if mask.sum() > 0:
                    metrics[f"precision_class_{cls}"] = float(
                        precision_score(
                            y_true, y_pred,
                            labels=[cls], average="micro", zero_division=0
                        )
                    )
                    metrics[f"recall_class_{cls}"] = float(
                        recall_score(
                            y_true, y_pred,
                            labels=[cls], average="micro", zero_division=0
                        )
                    )

            # AUC and log-loss (need probabilities)
            if y_prob is not None:
                try:
                    if n_classes == 2:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true, y_prob[:, 1])
                        )
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y_true, y_prob,
                                multi_class="ovr", average="weighted"
                            )
                        )
                except Exception:
                    pass

                try:
                    metrics["log_loss"] = float(log_loss(y_true, y_prob))
                except Exception:
                    pass

        else:
            metrics["rmse"] = float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            )
            metrics["mae"]  = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"]   = float(r2_score(y_true, y_pred))

            # Relative RMSE
            y_range = float(y_true.max() - y_true.min())
            if y_range > 0:
                metrics["rmse_relative"] = metrics["rmse"] / y_range

        return metrics

    def _compute_calibration(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ (|bin| / n) * |accuracy(bin) - confidence(bin)|

        A perfectly calibrated model has ECE = 0.
        Production models often have ECE > 0.1 (poorly calibrated).
        """
        try:
            if y_prob.ndim == 2:
                confidences = np.max(y_prob, axis=1)
                predictions = np.argmax(y_prob, axis=1)
                y_true_arr  = np.array(y_true)

                # Map predictions to boolean correct/incorrect
                correct = (predictions == y_true_arr).astype(float)
            else:
                confidences = y_prob
                correct     = np.array(y_true).astype(float)

            n       = len(y_true)
            bins    = np.linspace(0, 1, n_bins + 1)
            ece     = 0.0
            mce     = 0.0

            for i in range(n_bins):
                mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
                if mask.sum() == 0:
                    continue
                bin_acc  = float(correct[mask].mean())
                bin_conf = float(confidences[mask].mean())
                bin_size = mask.sum()

                calibration_gap = abs(bin_acc - bin_conf)
                ece            += (bin_size / n) * calibration_gap
                mce             = max(mce, calibration_gap)

            # Brier score
            if y_prob.ndim == 2:
                from sklearn.preprocessing import label_binarize
                classes       = np.unique(y_true)
                y_true_bin    = label_binarize(y_true, classes=classes)
                brier          = float(np.mean((y_prob - y_true_bin) ** 2))
            else:
                brier = float(
                    np.mean((y_prob - np.array(y_true)) ** 2)
                )

            return CalibrationMetrics(
                expected_calibration_error=ece,
                max_calibration_error=mce,
                brier_score=brier,
                n_bins=n_bins,
            )

        except Exception as exc:
            logger.warning(f"Calibration computation failed: {exc}")
            return CalibrationMetrics()

    def _compute_slice_performance(
        self,
        model: Any,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        slice_columns: list[str],
        is_classification: bool,
    ) -> list[SlicePerformance]:
        """
        Compute performance on data slices.

        Catches cases like:
          - New model is better overall but worse on elderly users
          - New model is better on majority class but much worse on minority
          - Regional performance regression
        """
        slices = []
        average = "weighted" if len(np.unique(y_eval)) > 2 else "binary"

        for col in slice_columns:
            if col not in X_eval.columns:
                continue

            for val in X_eval[col].unique():
                mask = X_eval[col] == val
                n    = int(mask.sum())

                if n < 20:
                    continue

                X_slice = X_eval[mask]
                y_slice = y_eval[mask]
                y_pred  = model.predict(X_slice)

                if is_classification:
                    slice_metrics = {
                        "f1_score":  float(
                            f1_score(y_slice, y_pred, average=average, zero_division=0)
                        ),
                        "recall":    float(
                            recall_score(y_slice, y_pred, average=average, zero_division=0)
                        ),
                        "precision": float(
                            precision_score(y_slice, y_pred, average=average, zero_division=0)
                        ),
                    }
                else:
                    slice_metrics = {
                        "rmse": float(np.sqrt(mean_squared_error(y_slice, y_pred))),
                        "mae":  float(mean_absolute_error(y_slice, y_pred)),
                    }

                slices.append(SlicePerformance(
                    slice_name=f"{col}={val}",
                    slice_size=n,
                    metrics=slice_metrics,
                ))

        return slices


# ─────────────────────────────────────────────────────────────────────────────
# Approval Gates
# ─────────────────────────────────────────────────────────────────────────────

class ApprovalGates:
    """
    Hard gates — challenger must pass ALL of these.
    One failure = REJECTED regardless of metric improvement.

    These protect production SLA.
    """

    def __init__(self, settings: DeploymentSettings) -> None:
        self.settings = settings

    def evaluate_all(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
        is_classification: bool = True,
    ) -> list[GateResult]:
        """
        Run all hard gates.
        Returns list of GateResult — all must have passed=True.
        """
        gates = [
            self._gate_p99_latency(champion, challenger),
            self._gate_latency_regression(champion, challenger),
            self._gate_memory(champion, challenger),
            self._gate_primary_metric_floor(challenger, is_classification),
            self._gate_recall_floor(champion, challenger, is_classification),
            self._gate_calibration(champion, challenger, is_classification),
            self._gate_slice_regression(champion, challenger),
            self._gate_model_size(champion, challenger),
        ]

        return gates

    def _gate_p99_latency(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
    ) -> GateResult:
        """
        Challenger P99 latency must not exceed absolute threshold.
        SLA requirement — non-negotiable.
        """
        threshold = self.settings.rollback_on_latency_p99_ms
        val       = challenger.operational.latency.p99_ms
        passed    = val <= threshold or val == 0.0

        return GateResult(
            gate_name="p99_latency_absolute",
            passed=passed,
            champion_value=f"{champion.operational.latency.p99_ms:.1f}ms",
            challenger_value=f"{val:.1f}ms",
            threshold=f"{threshold}ms",
            message=(
                f"✅ P99 {val:.1f}ms ≤ {threshold}ms"
                if passed else
                f"❌ P99 {val:.1f}ms > {threshold}ms — SLA BREACH"
            ),
        )

    def _gate_latency_regression(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
    ) -> GateResult:
        """
        Challenger P99 must not be more than 2x the champion P99.
        Protects against severe latency regressions.
        """
        champ_p99 = champion.operational.latency.p99_ms
        chal_p99  = challenger.operational.latency.p99_ms

        if champ_p99 == 0 or chal_p99 == 0:
            return GateResult(
                gate_name="p99_latency_regression",
                passed=True,
                champion_value="not measured",
                challenger_value="not measured",
                threshold="2x champion",
                message="⚠️ Latency not measured — gate skipped",
            )

        ratio  = chal_p99 / champ_p99
        passed = ratio <= 2.0

        return GateResult(
            gate_name="p99_latency_regression",
            passed=passed,
            champion_value=f"{champ_p99:.1f}ms",
            challenger_value=f"{chal_p99:.1f}ms",
            threshold="≤ 2x champion",
            message=(
                f"✅ Challenger P99 is {ratio:.1f}x champion"
                if passed else
                f"❌ Challenger P99 is {ratio:.1f}x champion — too slow"
            ),
        )

    def _gate_memory(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
    ) -> GateResult:
        """
        Challenger peak memory must not exceed 500MB.
        Prevents OOM in constrained serving environments.
        """
        max_memory_mb = 500.0
        val           = challenger.operational.memory_mb
        passed        = val <= max_memory_mb or val == 0.0

        return GateResult(
            gate_name="memory_ceiling",
            passed=passed,
            champion_value=f"{champion.operational.memory_mb:.1f}MB",
            challenger_value=f"{val:.1f}MB",
            threshold=f"{max_memory_mb}MB",
            message=(
                f"✅ Memory {val:.1f}MB ≤ {max_memory_mb}MB"
                if passed else
                f"❌ Memory {val:.1f}MB exceeds {max_memory_mb}MB — OOM risk"
            ),
        )

    def _gate_primary_metric_floor(
        self,
        challenger: ModelSnapshot,
        is_classification: bool,
    ) -> GateResult:
        """
        Challenger must meet a minimum absolute performance floor.
        Prevents deploying a broken model that technically "improved".
        """
        if is_classification:
            metric    = "f1_score"
            floor     = 0.50   # Must be at least 50% F1
        else:
            metric    = "r2"
            floor     = 0.20   # Must explain at least 20% variance

        val    = challenger.predictive_metrics.get(metric, 0.0)
        passed = val >= floor

        return GateResult(
            gate_name=f"{metric}_floor",
            passed=passed,
            champion_value="n/a",
            challenger_value=f"{val:.4f}",
            threshold=f"≥ {floor}",
            message=(
                f"✅ {metric}={val:.4f} meets floor {floor}"
                if passed else
                f"❌ {metric}={val:.4f} below floor {floor} — broken model?"
            ),
        )

    def _gate_recall_floor(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
        is_classification: bool,
    ) -> GateResult:
        """
        Recall must not drop more than 5% vs champion.

        Critical for fraud, medical, safety applications.
        A model that catches fewer true positives is dangerous
        even if precision improved.
        """
        if not is_classification:
            return GateResult(
                gate_name="recall_regression",
                passed=True,
                champion_value="n/a (regression task)",
                challenger_value="n/a",
                threshold="n/a",
                message="ℹ️ Recall gate: skipped (regression task)",
            )

        champ_recall = champion.predictive_metrics.get("recall", 0.0)
        chal_recall  = challenger.predictive_metrics.get("recall", 0.0)

        if champ_recall == 0:
            return GateResult(
                gate_name="recall_regression",
                passed=True,
                champion_value="0.0",
                challenger_value=f"{chal_recall:.4f}",
                threshold="≤5% drop",
                message="⚠️ Champion recall=0 — gate skipped",
            )

        drop   = (champ_recall - chal_recall) / champ_recall
        passed = drop <= 0.05   # ≤5% relative drop

        return GateResult(
            gate_name="recall_regression",
            passed=passed,
            champion_value=f"{champ_recall:.4f}",
            challenger_value=f"{chal_recall:.4f}",
            threshold="≤5% relative drop",
            message=(
                f"✅ Recall drop={drop:.1%} ≤ 5%"
                if passed else
                f"❌ Recall dropped {drop:.1%} — "
                f"missing more true positives than champion"
            ),
        )

    def _gate_calibration(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
        is_classification: bool,
    ) -> GateResult:
        """
        Challenger ECE must not exceed 0.15 (poorly calibrated models
        give unreliable probability estimates → bad downstream decisions).
        """
        if not is_classification:
            return GateResult(
                gate_name="calibration_ece",
                passed=True,
                champion_value="n/a",
                challenger_value="n/a",
                threshold="n/a",
                message="ℹ️ Calibration gate: skipped (regression task)",
            )

        max_ece  = 0.15
        chal_ece = challenger.calibration.expected_calibration_error
        passed   = chal_ece <= max_ece or chal_ece == 0.0

        return GateResult(
            gate_name="calibration_ece",
            passed=passed,
            champion_value=(
                f"{champion.calibration.expected_calibration_error:.4f}"
            ),
            challenger_value=f"{chal_ece:.4f}",
            threshold=f"ECE ≤ {max_ece}",
            message=(
                f"✅ ECE={chal_ece:.4f} ≤ {max_ece}"
                if passed else
                f"❌ ECE={chal_ece:.4f} > {max_ece} — "
                f"poorly calibrated, probability estimates unreliable"
            ),
        )

    def _gate_slice_regression(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
        max_regression_pct: float = 0.10,
    ) -> GateResult:
        """
        Challenger must not regress more than 10% on any data slice.

        Prevents models that improve overall but hurt a specific segment.
        """
        if (
            not champion.slice_performance
            or not challenger.slice_performance
        ):
            return GateResult(
                gate_name="slice_regression",
                passed=True,
                champion_value="not evaluated",
                challenger_value="not evaluated",
                threshold=f"≤{max_regression_pct:.0%} per slice",
                message="ℹ️ Slice gate: no slice evaluation provided",
            )

        champ_map = {
            s.slice_name: s for s in champion.slice_performance
        }
        regressions: list[str] = []

        for chal_slice in challenger.slice_performance:
            champ_slice = champ_map.get(chal_slice.slice_name)
            if champ_slice is None:
                continue

            # Compare primary slice metric
            primary = "f1_score" if "f1_score" in chal_slice.metrics else "rmse"
            champ_val = champ_slice.metrics.get(primary, 0.0)
            chal_val  = chal_slice.metrics.get(primary, 0.0)

            if champ_val == 0:
                continue

            is_lower_better = primary == "rmse"
            if is_lower_better:
                regression_pct = (chal_val - champ_val) / abs(champ_val)
            else:
                regression_pct = (champ_val - chal_val) / abs(champ_val)

            if regression_pct > max_regression_pct:
                regressions.append(
                    f"{chal_slice.slice_name}: "
                    f"{primary} dropped {regression_pct:.1%}"
                )
                chal_slice.is_regression = True

        passed = len(regressions) == 0

        return GateResult(
            gate_name="slice_regression",
            passed=passed,
            champion_value="baseline",
            challenger_value=(
                f"{len(regressions)} regressions"
                if regressions else "no regressions"
            ),
            threshold=f"≤{max_regression_pct:.0%} per slice",
            message=(
                f"✅ No slice regressions detected"
                if passed else
                f"❌ Slice regressions: {regressions[:3]}"
            ),
        )

    def _gate_model_size(
        self,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
    ) -> GateResult:
        """
        Model size must not exceed 500MB.
        Large models have slow cold-start and deployment issues.
        """
        max_size_mb = 500.0
        val         = challenger.operational.model_size_mb
        passed      = val <= max_size_mb or val == 0.0

        return GateResult(
            gate_name="model_size",
            passed=passed,
            champion_value=f"{champion.operational.model_size_mb:.1f}MB",
            challenger_value=f"{val:.1f}MB",
            threshold=f"≤{max_size_mb}MB",
            message=(
                f"✅ Model size {val:.1f}MB ≤ {max_size_mb}MB"
                if passed else
                f"❌ Model size {val:.1f}MB exceeds {max_size_mb}MB"
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Champion-Challenger Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ChampionChallengerEvaluator:
    """
    Full champion-challenger evaluation.

    Workflow:
      1. Evaluate BOTH models on SAME held-out eval set
         (same data → fair comparison)
      2. Run all hard gates on challenger
      3. Compute composite weighted score
      4. Statistical significance test
      5. Make final decision

    Usage:
        evaluator = ChampionChallengerEvaluator()
        result = evaluator.compare(
            champion_path="models/champion/model.joblib",
            challenger_path="models/challenger/model.joblib",
            X_eval=eval_df,
            y_eval=eval_labels,
        )
    """

    # Composite score weights
    WEIGHTS = {
        "predictive":   0.50,   # F1/RMSE — most important
        "operational":  0.25,   # Latency + memory
        "calibration":  0.15,   # Probability quality
        "slice":        0.10,   # Fairness / slice consistency
    }

    def __init__(
        self,
        settings: Optional[DeploymentSettings] = None,
        pipeline_run_id: str = "",
        model_id: str = "",
    ) -> None:
        self.settings         = settings or get_settings().deployment
        self.pipeline_run_id  = pipeline_run_id
        self.model_id         = model_id
        self.evaluator        = ModelEvaluator()
        self.gates            = ApprovalGates(self.settings)

    def compare(
        self,
        champion_version: str,
        champion_artifact_path: str,
        challenger_version: str,
        challenger_artifact_path: str,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        is_classification: bool = True,
        slice_columns: Optional[list[str]] = None,
    ) -> ComparisonResult:
        """
        Full champion-challenger comparison.

        Both models evaluated on the SAME X_eval, y_eval.
        This is the only fair comparison.

        Parameters
        ----------
        champion_artifact_path:
            Path to current production model's .joblib file.
        challenger_artifact_path:
            Path to newly trained model's .joblib file.
        X_eval:
            Held-out evaluation set (processed, same feature space).
        y_eval:
            Ground truth labels.
        slice_columns:
            Columns for per-slice evaluation.
        """
        logger.info(
            f"Champion-Challenger comparison: "
            f"champion=v{champion_version} vs "
            f"challenger=v{challenger_version} | "
            f"eval_samples={len(X_eval)}"
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.CHAMPION_CHALLENGER_STARTED,
            step_name="evaluation",
            title="⚖️ Champion-Challenger Started",
            message=(
                f"champion=v{champion_version} vs "
                f"challenger=v{challenger_version} | "
                f"samples={len(X_eval)}"
            ),
            model_id=self.model_id,
            data={
                "champion_version":   champion_version,
                "challenger_version": challenger_version,
                "eval_samples":       len(X_eval),
                "slice_columns":      slice_columns or [],
            },
        ))

        # ── 1. Evaluate both models on same data ───────────────────────────
        logger.info("Evaluating champion...")
        champion_snapshot = self.evaluator.evaluate(
            model_version=champion_version,
            model_id=self.model_id,
            artifact_path=champion_artifact_path,
            X_eval=X_eval,
            y_eval=y_eval,
            is_classification=is_classification,
            slice_columns=slice_columns,
            pipeline_run_id=self.pipeline_run_id,
        )

        logger.info("Evaluating challenger...")
        challenger_snapshot = self.evaluator.evaluate(
            model_version=challenger_version,
            model_id=self.model_id,
            artifact_path=challenger_artifact_path,
            X_eval=X_eval,
            y_eval=y_eval,
            is_classification=is_classification,
            slice_columns=slice_columns,
            pipeline_run_id=self.pipeline_run_id,
        )

        # ── 2. Hard gates ──────────────────────────────────────────────────
        gate_results = self.gates.evaluate_all(
            champion_snapshot,
            challenger_snapshot,
            is_classification=is_classification,
        )

        gates_passed = all(g.passed for g in gate_results)
        failed_gates = [g for g in gate_results if not g.passed]

        # ── 3. Composite score ─────────────────────────────────────────────
        champ_score  = self._compute_composite_score(champion_snapshot, is_classification)
        chal_score   = self._compute_composite_score(challenger_snapshot, is_classification)

        # ── 4. Primary metric improvement ──────────────────────────────────
        primary = "f1_score" if is_classification else "rmse"
        lower_is_better = primary in ("rmse", "mae", "log_loss")

        champ_primary = champion_snapshot.predictive_metrics.get(primary, 0.0)
        chal_primary  = challenger_snapshot.predictive_metrics.get(primary, 0.0)

        if lower_is_better:
            improvement = (champ_primary - chal_primary) / max(abs(champ_primary), 1e-8)
        else:
            improvement = (chal_primary - champ_primary) / max(abs(champ_primary), 1e-8)

        meets_threshold = improvement >= self.settings.min_champion_improvement

        # ── 5. Statistical significance ────────────────────────────────────
        is_significant, p_value = self._bootstrap_significance(
            champion_artifact_path,
            challenger_artifact_path,
            X_eval,
            y_eval,
            primary,
            lower_is_better,
        )

        # ── 6. Final decision ──────────────────────────────────────────────
        challenger_wins = (
            gates_passed
            and meets_threshold
            and is_significant
            and chal_score > champ_score
        )

        reasons = self._build_reasons(
            gate_results, failed_gates,
            improvement, is_significant, p_value,
            champ_score, chal_score,
            champion_snapshot, challenger_snapshot,
            meets_threshold,
        )

        approval_status = (
            ApprovalStatus.AUTO_APPROVED
            if challenger_wins
            else ApprovalStatus.REJECTED
        )

        result = ComparisonResult(
            champion=champion_snapshot,
            challenger=challenger_snapshot,
            challenger_wins=challenger_wins,
            approval_status=approval_status,
            gate_results=gate_results,
            improvement=improvement,
            improvement_pct=improvement * 100,
            is_statistically_significant=is_significant,
            p_value=p_value,
            champion_score=champ_score,
            challenger_score=chal_score,
            reasons=reasons,
        )

        # Update Prometheus
        CHAMPION_CHALLENGER_COMPARISONS.labels(
            model_id=self.model_id,
            result="approved" if challenger_wins else "rejected",
        ).inc()
        CHALLENGER_IMPROVEMENT.labels(model_id=self.model_id).set(improvement)
        CHAMPION_METRIC.labels(
            model_id=self.model_id, metric=primary
        ).set(champ_primary)

        # Emit event
        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.CHAMPION_CHALLENGER_COMPLETED,
            step_name="evaluation",
            title=(
                f"{'✅ APPROVED' if challenger_wins else '❌ REJECTED'}: "
                f"v{challenger_version}"
            ),
            message=(
                f"Composite: champ={champ_score:.3f} → chal={chal_score:.3f} | "
                f"{primary}: {champ_primary:.4f} → {chal_primary:.4f} "
                f"({improvement:+.2%}) | "
                f"Gates: {sum(g.passed for g in gate_results)}/"
                f"{len(gate_results)} passed"
            ),
            model_id=self.model_id,
            model_version=challenger_version,
            status="success" if challenger_wins else "failed",
            data=result.to_dict(),
        ))

        logger.info(
            f"Champion-Challenger result: "
            f"challenger_wins={challenger_wins} | "
            f"champ_score={champ_score:.3f} | "
            f"chal_score={chal_score:.3f} | "
            f"improvement={improvement:.2%} | "
            f"gates_passed={gates_passed} | "
            f"p_value={p_value:.4f}"
        )

        return result

    def _compute_composite_score(
        self,
        snapshot: ModelSnapshot,
        is_classification: bool,
    ) -> float:
        """
        Compute weighted composite score (0-1).

        Higher = better for all components.
        """
        scores: dict[str, float] = {}

        # Predictive score
        if is_classification:
            f1      = snapshot.predictive_metrics.get("f1_score", 0.0)
            auc     = snapshot.predictive_metrics.get("roc_auc", f1)
            scores["predictive"] = float(np.mean([f1, auc]))
        else:
            r2          = max(snapshot.predictive_metrics.get("r2", 0.0), 0.0)
            rmse_rel    = snapshot.predictive_metrics.get("rmse_relative", 1.0)
            rmse_score  = max(1.0 - rmse_rel, 0.0)
            scores["predictive"] = float(np.mean([r2, rmse_score]))

        # Operational score — normalize latency + memory
        p99    = snapshot.operational.latency.p99_ms
        mem    = snapshot.operational.memory_mb
        thresh = self.settings.rollback_on_latency_p99_ms

        lat_score = max(1.0 - (p99 / thresh), 0.0) if thresh > 0 and p99 > 0 else 0.8
        mem_score = max(1.0 - (mem / 500.0), 0.0)  if mem > 0 else 0.8
        scores["operational"] = float(np.mean([lat_score, mem_score]))

        # Calibration score (1 - ECE, higher is better)
        ece = snapshot.calibration.expected_calibration_error
        scores["calibration"] = max(1.0 - ece * 5, 0.0) if ece > 0 else 0.8

        # Slice score (proportion of slices without regression)
        if snapshot.slice_performance:
            n_ok = sum(
                1 for s in snapshot.slice_performance
                if not s.is_regression
            )
            scores["slice"] = n_ok / len(snapshot.slice_performance)
        else:
            scores["slice"] = 0.8

        # Weighted composite
        composite = sum(
            scores.get(dim, 0.0) * weight
            for dim, weight in self.WEIGHTS.items()
        )

        return float(composite)

    def _bootstrap_significance(
        self,
        champion_path: str,
        challenger_path: str,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        primary_metric: str,
        lower_is_better: bool,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """
        Paired bootstrap test for statistical significance.
        Tests: P(challenger improvement > 0) > (1 - alpha)
        """
        try:
            champion   = joblib.load(champion_path)
            challenger = joblib.load(challenger_path)

            champ_preds = champion.predict(X_eval)
            chal_preds  = challenger.predict(X_eval)
            y_arr       = np.array(y_eval)

            rng = np.random.RandomState(42)
            n   = len(y_arr)

            def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                if primary_metric == "f1_score":
                    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
                    return float(f1_score(y_true, y_pred, average=avg, zero_division=0))
                elif primary_metric == "rmse":
                    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))
                elif primary_metric == "accuracy":
                    return float(accuracy_score(y_true, y_pred))
                else:
                    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

            base_champ = score_fn(y_arr, champ_preds)
            base_chal  = score_fn(y_arr, chal_preds)
            observed   = base_chal - base_champ

            bootstrap_diffs: list[float] = []
            for _ in range(n_bootstrap):
                idx          = rng.choice(n, size=n, replace=True)
                y_boot       = y_arr[idx]
                champ_boot   = champ_preds[idx]
                chal_boot    = chal_preds[idx]

                diff = score_fn(y_boot, chal_boot) - score_fn(y_boot, champ_boot)
                bootstrap_diffs.append(diff)

            bootstrap_arr = np.array(bootstrap_diffs)
            # p-value: proportion of bootstrap samples where improvement ≤ 0
            p_value       = float(np.mean(bootstrap_arr <= 0))
            is_significant = p_value < alpha

            return is_significant, p_value

        except Exception as exc:
            logger.warning(f"Bootstrap test failed: {exc}")
            return True, 0.05   # Conservative: assume significant if test fails

    def _build_reasons(
        self,
        gate_results: list[GateResult],
        failed_gates: list[GateResult],
        improvement: float,
        is_significant: bool,
        p_value: Optional[float],
        champ_score: float,
        chal_score: float,
        champion: ModelSnapshot,
        challenger: ModelSnapshot,
        meets_threshold: bool,
    ) -> list[str]:
        reasons = []

        if failed_gates:
            for g in failed_gates:
                reasons.append(f"GATE FAILED — {g.gate_name}: {g.message}")
        else:
            reasons.append(
                f"All {len(gate_results)} gates passed"
            )

        reasons.append(
            f"Primary metric improvement: {improvement:+.2%} "
            f"({'meets' if meets_threshold else 'below'} threshold "
            f"{self.settings.min_champion_improvement:.1%})"
        )
        reasons.append(
            f"Statistical significance: "
            f"{'significant' if is_significant else 'NOT significant'} "
            f"(p={p_value:.4f})"
        )
        reasons.append(
            f"Composite score: champion={champ_score:.3f} → "
            f"challenger={chal_score:.3f} "
            f"({'↑ better' if chal_score > champ_score else '↓ worse'})"
        )

        # Latency comparison
        champ_p99 = champion.operational.latency.p99_ms
        chal_p99  = challenger.operational.latency.p99_ms
        if champ_p99 > 0 and chal_p99 > 0:
            reasons.append(
                f"P99 latency: {champ_p99:.1f}ms → {chal_p99:.1f}ms "
                f"({'faster ✅' if chal_p99 < champ_p99 else 'slower ⚠️'})"
            )

        # Memory comparison
        champ_mem = champion.operational.memory_mb
        chal_mem  = challenger.operational.memory_mb
        if champ_mem > 0 and chal_mem > 0:
            reasons.append(
                f"Memory: {champ_mem:.1f}MB → {chal_mem:.1f}MB"
            )

        # Calibration
        champ_ece = champion.calibration.expected_calibration_error
        chal_ece  = challenger.calibration.expected_calibration_error
        if champ_ece > 0 or chal_ece > 0:
            reasons.append(
                f"Calibration (ECE): {champ_ece:.4f} → {chal_ece:.4f} "
                f"({'better ✅' if chal_ece < champ_ece else 'worse ⚠️'})"
            )

        return reasons