# src/drift/detector.py
"""
Drift detector — modified to accept ALREADY PROCESSED DataFrames.

Key change:
  Before: raw DataFrames passed in, dropna() called internally
  Now:    processed DataFrames passed in (both in same feature space)
          No dropna() needed — processor already handled nulls
          Feature columns auto-detected from processed DataFrame

Why this gives better drift results:
  Both DataFrames are in the SAME sklearn-transformed feature space.
  KS test on standardized features → scale-invariant comparison.
  PSI on frequency-encoded categoricals → proper probability comparison.
  OHE columns → binary drift detection per category.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import DriftSettings, get_settings
from src.common.enums import DriftSeverity
from src.drift.statistical_tests import StatisticalTestSuite
from src.observability.event_bus import EventType, event_bus, make_event
from src.observability.metrics import (
    DRIFT_FEATURES_DRIFTED,
    DRIFT_KS_PVALUE,
    DRIFT_PSI,
    DRIFT_SCORE,
    DRIFT_SEVERITY,
    DRIFT_SEVERITY_MAP,
)

logger = logging.getLogger("ml_platform.drift.detector")


@dataclass
class FeatureDriftResult:
    """Drift result for one feature."""
    feature_name: str
    test_name: str
    statistic: float
    p_value: Optional[float]
    drift_score: float
    is_drifted: bool
    ref_mean: float = 0.0
    cur_mean: float = 0.0
    ref_std: float = 0.0
    cur_std: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature":     self.feature_name,
            "test":        self.test_name,
            "statistic":   round(self.statistic, 4),
            "p_value":     round(self.p_value, 6) if self.p_value is not None else None,
            "drift_score": round(self.drift_score, 4),
            "is_drifted":  self.is_drifted,
            "ref_mean":    round(self.ref_mean, 4),
            "cur_mean":    round(self.cur_mean, 4),
            "ref_std":     round(self.ref_std, 4),
            "cur_std":     round(self.cur_std, 4),
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""
    report_id: str
    timestamp: datetime
    reference_dataset_id: str
    current_dataset_id: str
    overall_severity: DriftSeverity
    overall_drift_score: float
    features_drifted: int
    features_total: int
    drift_proportion: float
    feature_results: list[FeatureDriftResult]
    concept_drift_detected: bool = False
    concept_drift_details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id":              self.report_id,
            "timestamp":              self.timestamp.isoformat(),
            "overall_severity":       self.overall_severity.value,
            "overall_drift_score":    round(self.overall_drift_score, 4),
            "features_drifted":       self.features_drifted,
            "features_total":         self.features_total,
            "drift_proportion":       round(self.drift_proportion, 4),
            "concept_drift_detected": self.concept_drift_detected,
            "feature_results": [
                fr.to_dict() for fr in self.feature_results
            ],
            "recommendations": self.recommendations,
        }


class DriftDetector:
    """
    Detects drift between processed reference and production datasets.

    Input: PROCESSED DataFrames (same sklearn feature space).
    Auto-detects feature columns (no manual list needed).

    Numeric processed features  → KS test + PSI
    Binary OHE features         → Chi-square proportion test
    Frequency-encoded features  → KS test (already numeric)
    Target column               → concept drift detection
    """

    def __init__(
        self,
        settings: Optional[DriftSettings] = None,
        model_id: str = "",
        pipeline_run_id: str = "",
    ) -> None:
        self.settings        = settings or get_settings().drift
        self.test_suite      = StatisticalTestSuite()
        self.model_id        = model_id
        self.pipeline_run_id = pipeline_run_id

    def detect(
        self,
        reference_processed: pd.DataFrame,
        production_processed: pd.DataFrame,
        target_column: Optional[str] = None,
        report_id: Optional[str] = None,
        reference_dataset_id: str = str(uuid.uuid4()),
        current_dataset_id: str = str(uuid.uuid4()),
    ) -> DriftReport:
        """
        Run drift detection on PROCESSED DataFrames.

        Features auto-detected from DataFrame columns.
        No manual feature lists required.

        Parameters
        ----------
        reference_processed:
            Processed reference DataFrame (sklearn-transformed).
        production_processed:
            Processed production DataFrame (same sklearn space).
        target_column:
            Target column name for concept drift (optional).
        """
        report_id = report_id or str(uuid.uuid4())

        # Auto-detect features — exclude target
        all_cols     = [
            c for c in reference_processed.columns
            if c != target_column
            and c in production_processed.columns
        ]

        # Split into numeric and binary-OHE
        numeric_features = [
            c for c in all_cols
            if pd.api.types.is_numeric_dtype(reference_processed[c])
        ]

        logger.info(
            f"Drift detection: {len(numeric_features)} features | "
            f"ref={len(reference_processed)} rows | "
            f"prod={len(production_processed)} rows"
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_DETECTION_STARTED,
            step_name="drift_detection",
            title="🔍 Drift Detection Started (Processed Features)",
            message=(
                f"{len(numeric_features)} features | "
                f"ref={len(reference_processed)} | "
                f"prod={len(production_processed)}"
            ),
            model_id=self.model_id,
            data={
                "n_features":   len(numeric_features),
                "ref_rows":     len(reference_processed),
                "prod_rows":    len(production_processed),
            },
        ))

        feature_results: list[FeatureDriftResult] = []

        # ── Test each feature ──────────────────────────────────────────────
        for feature in numeric_features:
            result = self._test_feature(
                feature,
                reference_processed,
                production_processed,
            )
            if result is not None:
                feature_results.append(result)

        # ── Concept drift ──────────────────────────────────────────────────
        concept_drift        = False
        concept_drift_details: dict[str, Any] = {}

        if (
            target_column
            and target_column in reference_processed.columns
            and target_column in production_processed.columns
        ):
            concept_drift, concept_drift_details = self._detect_concept_drift(
                reference_processed,
                production_processed,
                target_column,
            )

        # ── Aggregate ──────────────────────────────────────────────────────
        features_drifted     = sum(1 for r in feature_results if r.is_drifted)
        features_total       = len(feature_results)
        drift_proportion     = (
            features_drifted / features_total
            if features_total > 0 else 0.0
        )
        overall_drift_score  = float(
            np.mean([r.drift_score for r in feature_results])
            if feature_results else 0.0
        )
        overall_severity     = self._classify_severity(
            drift_proportion, overall_drift_score, concept_drift
        )
        recommendations      = self._generate_recommendations(
            overall_severity, drift_proportion,
            concept_drift, feature_results,
        )

        # Update Prometheus
        DRIFT_SCORE.labels(model_id=self.model_id).set(overall_drift_score)
        DRIFT_FEATURES_DRIFTED.labels(model_id=self.model_id).set(
            features_drifted
        )
        DRIFT_SEVERITY.labels(model_id=self.model_id).set(
            DRIFT_SEVERITY_MAP.get(overall_severity.value, 0)
        )

        report = DriftReport(
            report_id=report_id,
            timestamp=datetime.now(timezone.utc),
            reference_dataset_id=reference_dataset_id,
            current_dataset_id=current_dataset_id,
            overall_severity=overall_severity,
            overall_drift_score=overall_drift_score,
            features_drifted=features_drifted,
            features_total=features_total,
            drift_proportion=drift_proportion,
            feature_results=feature_results,
            concept_drift_detected=concept_drift,
            concept_drift_details=concept_drift_details,
            recommendations=recommendations,
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_DETECTION_COMPLETED,
            step_name="drift_detection",
            title=(
                f"{'🔴' if overall_severity.value != 'none' else '🟢'} "
                f"Drift: {overall_severity.value.upper()} | "
                f"{features_drifted}/{features_total} features"
            ),
            message=(
                f"Score: {overall_drift_score:.3f} | "
                f"Concept drift: {concept_drift}"
            ),
            model_id=self.model_id,
            status=(
                "success"
                if overall_severity.value == "none"
                else "warning"
            ),
            data=report.to_dict(),
        ))

        logger.info(
            f"Drift complete: severity={overall_severity.value}, "
            f"drifted={features_drifted}/{features_total}, "
            f"score={overall_drift_score:.3f}"
        )

        return report

    # ─────────────────────────────────────────────────────────────────────
    # Feature-level tests
    # ─────────────────────────────────────────────────────────────────────

    def _test_feature(
        self,
        feature: str,
        ref_df: pd.DataFrame,
        prod_df: pd.DataFrame,
    ) -> Optional[FeatureDriftResult]:
        """
        Test one feature for drift.
        Since data is processed, no NaN handling needed (processor did it).
        But we still guard for safety.
        """
        ref_vals  = ref_df[feature].dropna().values.astype(float)
        prod_vals = prod_df[feature].dropna().values.astype(float)

        if len(ref_vals) < 10 or len(prod_vals) < 10:
            logger.debug(f"Skipping '{feature}': insufficient samples")
            return None

        # Detect if this is a binary OHE column (0/1 only)
        is_binary = (
            set(np.unique(np.asarray(ref_vals))).issubset({0.0, 1.0})
            and set(np.unique(np.asarray(prod_vals))).issubset({0.0, 1.0})
        )

        if is_binary:
            return self._test_binary_feature(feature, np.asarray(ref_vals), np.asarray(prod_vals))
        else:
            return self._test_continuous_feature(feature, np.asarray(ref_vals), np.asarray(prod_vals))

    def _test_continuous_feature(
        self,
        feature: str,
        ref_vals: np.ndarray,
        prod_vals: np.ndarray,
    ) -> FeatureDriftResult:
        """KS test + PSI for continuous/ordinal processed features."""
        ks_stat, ks_p = self.test_suite.ks_test(ref_vals, prod_vals)
        psi           = self.test_suite.psi_numerical(ref_vals, prod_vals, bins=20)
        wasserstein   = self.test_suite.wasserstein_distance(ref_vals, prod_vals)

        drift_score = self._compute_drift_score(ks_stat, ks_p, psi)
        is_drifted  = (
            ks_p < self.settings.ks_p_value_threshold
            or psi > self.settings.psi_threshold
        )

        # Update Prometheus per-feature
        DRIFT_PSI.labels(model_id=self.model_id, feature=feature).set(psi)
        DRIFT_KS_PVALUE.labels(
            model_id=self.model_id, feature=feature
        ).set(ks_p)

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_FEATURE_CHECKED,
            step_name="drift_detection",
            title=f"{'🔴' if is_drifted else '🟢'} {feature}",
            message=(
                f"PSI={psi:.3f} KS_p={ks_p:.4f} W={wasserstein:.3f} "
                f"→ {'DRIFT' if is_drifted else 'stable'}"
            ),
            model_id=self.model_id,
            status="warning" if is_drifted else "success",
            data={
                "feature": feature,
                "psi":     round(psi, 4),
                "ks_p":    round(ks_p, 6),
                "ks_stat": round(ks_stat, 4),
            },
        ))

        return FeatureDriftResult(
            feature_name=feature,
            test_name="ks+psi",
            statistic=ks_stat,
            p_value=ks_p,
            drift_score=drift_score,
            is_drifted=is_drifted,
            ref_mean=float(np.mean(ref_vals)),
            cur_mean=float(np.mean(prod_vals)),
            ref_std=float(np.std(ref_vals)),
            cur_std=float(np.std(prod_vals)),
            details={
                "psi":         round(psi, 4),
                "ks_stat":     round(ks_stat, 4),
                "ks_p_value":  round(ks_p, 6),
                "wasserstein": round(wasserstein, 4),
            },
        )

    def _test_binary_feature(
        self,
        feature: str,
        ref_vals: np.ndarray,
        prod_vals: np.ndarray,
    ) -> FeatureDriftResult:
        """Proportion test for binary OHE features."""
        ref_prop  = float(np.mean(ref_vals))
        prod_prop = float(np.mean(prod_vals))

        # PSI for binary
        eps       = 1e-8
        psi_val   = abs(
            (prod_prop + eps) - (ref_prop + eps)
        ) * abs(
            np.log((prod_prop + eps) / (ref_prop + eps))
        )

        diff          = abs(prod_prop - ref_prop)
        drift_score   = min(diff * 2, 1.0)
        is_drifted    = (
            psi_val > self.settings.psi_threshold
            or diff > 0.10  # >10% proportion shift
        )

        DRIFT_PSI.labels(model_id=self.model_id, feature=feature).set(psi_val)

        return FeatureDriftResult(
            feature_name=feature,
            test_name="proportion+psi",
            statistic=diff,
            p_value=None,
            drift_score=drift_score,
            is_drifted=is_drifted,
            ref_mean=ref_prop,
            cur_mean=prod_prop,
            ref_std=0.0,
            cur_std=0.0,
            details={
                "ref_proportion":  round(ref_prop, 4),
                "prod_proportion": round(prod_prop, 4),
                "difference":      round(diff, 4),
                "psi":             round(psi_val, 4),
            },
        )

    def _detect_concept_drift(
        self,
        ref_df: pd.DataFrame,
        prod_df: pd.DataFrame,
        target_column: str,
    ) -> tuple[bool, dict[str, Any]]:
        ref_target  = ref_df[target_column].dropna()
        prod_target = prod_df[target_column].dropna()

        if pd.api.types.is_numeric_dtype(ref_target):
            ks_stat, ks_p = self.test_suite.ks_test(
                np.asarray(ref_target),
                np.asarray(prod_target),
            )
            details = {
                "test":      "ks_2samp",
                "statistic": round(ks_stat, 4),
                "p_value":   round(ks_p, 6),
                "ref_mean":  round(float(np.mean(ref_target)), 4),
                "prod_mean": round(float(np.mean(prod_target)), 4),
            }
            drifted = ks_p < 0.01
        else:
            psi = self.test_suite.psi_categorical(ref_target, prod_target)
            details = {
                "test":             "psi",
                "psi_value":        round(psi, 4),
                "ref_distribution": (
                    ref_target.value_counts(normalize=True)
                    .head(10).to_dict()
                ),
                "prod_distribution": (
                    prod_target.value_counts(normalize=True)
                    .head(10).to_dict()
                ),
            }
            drifted = psi > 0.15

        return drifted, details

    # ─────────────────────────────────────────────────────────────────────
    # Scoring and severity
    # ─────────────────────────────────────────────────────────────────────

    def _compute_drift_score(
        self, ks_stat: float, ks_p: float, psi: float
    ) -> float:
        ks_score  = min(ks_stat * 2, 1.0)
        psi_score = min(psi / 0.5, 1.0)
        return float(0.4 * ks_score + 0.6 * psi_score)

    def _classify_severity(
        self,
        drift_proportion: float,
        drift_score: float,
        concept_drift: bool,
    ) -> DriftSeverity:
        if concept_drift and drift_proportion > 0.5:
            return DriftSeverity.CRITICAL
        if concept_drift or drift_proportion > 0.6:
            return DriftSeverity.HIGH
        if drift_proportion > 0.3 or drift_score > 0.5:
            return DriftSeverity.MODERATE
        if drift_proportion > 0.1:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        drift_proportion: float,
        concept_drift: bool,
        feature_results: list[FeatureDriftResult],
    ) -> list[str]:
        recs = []
        if severity == DriftSeverity.CRITICAL:
            recs.append(
                "URGENT: Critical drift. Immediate retraining required."
            )
        if concept_drift:
            recs.append(
                "Concept drift in target variable. Model predictions degraded."
            )
        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            recs.append(
                "Retrain on current production data to capture new trends."
            )
        if severity == DriftSeverity.MODERATE:
            recs.append(
                "Monitor performance closely. Schedule retraining if metrics degrade."
            )

        drifted = sorted(
            [r for r in feature_results if r.is_drifted],
            key=lambda x: x.drift_score,
            reverse=True,
        )
        if drifted:
            recs.append(
                f"Top drifted features: "
                f"{[r.feature_name for r in drifted[:5]]}"
            )

        return recs