# src/drift/detector.py
"""
Production drift detection.
Fix: NaN handling is now explicit and consistent across all features.
Events are emitted per-feature so Grafana shows real-time drift heatmap.
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
    DRIFT_PSI, DRIFT_KS_PVALUE,
    DRIFT_SCORE, DRIFT_FEATURES_DRIFTED, DRIFT_SEVERITY,
    DRIFT_SEVERITY_MAP,
)

logger = logging.getLogger("ml_platform.drift.detector")


@dataclass
class FeatureDriftResult:
    feature_name: str
    test_name: str
    statistic: float
    p_value: Optional[float]
    drift_score: float
    is_drifted: bool
    ref_null_rate: float = 0.0   # NEW: explicit null tracking
    cur_null_rate: float = 0.0   # NEW: explicit null tracking
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
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

    def to_dict(self) -> dict:
        return {
            "report_id":             self.report_id,
            "timestamp":             self.timestamp.isoformat(),
            "overall_severity":      self.overall_severity.value,
            "overall_drift_score":   round(self.overall_drift_score, 4),
            "features_drifted":      self.features_drifted,
            "features_total":        self.features_total,
            "drift_proportion":      round(self.drift_proportion, 4),
            "concept_drift_detected": self.concept_drift_detected,
            "feature_results": [
                {
                    "feature":     fr.feature_name,
                    "test":        fr.test_name,
                    "statistic":   round(fr.statistic, 4),
                    "p_value":     round(fr.p_value, 6) if fr.p_value is not None else None,
                    "drift_score": round(fr.drift_score, 4),
                    "is_drifted":  fr.is_drifted,
                    "ref_null_rate": fr.ref_null_rate,
                    "cur_null_rate": fr.cur_null_rate,
                }
                for fr in self.feature_results
            ],
            "recommendations": self.recommendations,
        }


class DriftDetector:
    """
    Detects data drift and concept drift.

    Key fix applied:
    - dropna() is called once per feature, null rates are RECORDED
      and emitted as events so dashboards can show null-drift separately.
    - If null rate itself changed significantly, that IS drift.
    """

    def __init__(
        self,
        settings: Optional[DriftSettings] = None,
        model_id: str = "",
        pipeline_run_id: str = "",
    ):
        self.settings       = settings or get_settings().drift
        self.test_suite     = StatisticalTestSuite()
        self.model_id       = model_id
        self.pipeline_run_id = pipeline_run_id

    def detect(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numerical_features: list[str],
        categorical_features: list[str],
        target_column: Optional[str] = None,
        report_id: Optional[str] = None,
        reference_dataset_id: str = "reference",
        current_dataset_id: str = "current",
    ) -> DriftReport:
        report_id = report_id or str(uuid.uuid4())

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_DETECTION_STARTED,
            step_name="drift_detection",
            title="🔍 Drift Detection Started",
            message=(
                f"Comparing {len(reference_df)} reference rows vs "
                f"{len(current_df)} current rows across "
                f"{len(numerical_features)+len(categorical_features)} features"
            ),
            model_id=self.model_id,
            data={
                "reference_rows": len(reference_df),
                "current_rows":   len(current_df),
                "numerical_features":   len(numerical_features),
                "categorical_features": len(categorical_features),
            },
        ))

        logger.info(
            f"Drift detection: report_id={report_id}, "
            f"ref={len(reference_df)}, cur={len(current_df)}"
        )

        if len(current_df) < self.settings.min_samples_for_drift:
            logger.warning(
                f"Current dataset ({len(current_df)} rows) below minimum "
                f"({self.settings.min_samples_for_drift}). Results may be unreliable."
            )

        feature_results: list[FeatureDriftResult] = []

        # ── Numerical features ────────────────────────────────────────────────
        for feature in numerical_features:
            result = self._test_numerical_feature(
                feature, reference_df, current_df
            )
            if result:
                feature_results.append(result)

        # ── Categorical features ──────────────────────────────────────────────
        for feature in categorical_features:
            result = self._test_categorical_feature(
                feature, reference_df, current_df
            )
            if result:
                feature_results.append(result)

        # ── Concept drift ─────────────────────────────────────────────────────
        concept_drift_detected   = False
        concept_drift_details: dict[str, Any] = {}

        if (target_column
                and target_column in reference_df.columns
                and target_column in current_df.columns):
            concept_drift_detected, concept_drift_details = \
                self._detect_concept_drift(reference_df, current_df, target_column)

        # ── Aggregate ─────────────────────────────────────────────────────────
        features_drifted  = sum(1 for fr in feature_results if fr.is_drifted)
        features_total    = len(feature_results)
        drift_proportion  = features_drifted / features_total if features_total > 0 else 0.0
        overall_drift_score = float(
            np.mean([fr.drift_score for fr in feature_results]) if feature_results else 0.0
        )
        overall_severity = self._classify_severity(
            drift_proportion, overall_drift_score, concept_drift_detected
        )
        recommendations = self._generate_recommendations(
            overall_severity, drift_proportion, features_drifted,
            concept_drift_detected, feature_results,
        )

        # Update Prometheus
        DRIFT_SCORE.labels(model_id=self.model_id).set(overall_drift_score)
        DRIFT_FEATURES_DRIFTED.labels(model_id=self.model_id).set(features_drifted)
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
            concept_drift_detected=concept_drift_detected,
            concept_drift_details=concept_drift_details,
            recommendations=recommendations,
        )

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_DETECTION_COMPLETED,
            step_name="drift_detection",
            title=(
                f"{'🟢' if overall_severity == DriftSeverity.NONE else '🔴'} "
                f"Drift: {overall_severity.value.upper()}"
            ),
            message=(
                f"{features_drifted}/{features_total} features drifted | "
                f"Score: {overall_drift_score:.3f} | "
                f"Concept drift: {concept_drift_detected}"
            ),
            model_id=self.model_id,
            status="success" if overall_severity == DriftSeverity.NONE else "warning",
            severity="info" if overall_severity in (DriftSeverity.NONE, DriftSeverity.LOW) else "error",
            data=report.to_dict(),
        ))

        logger.info(
            f"Drift complete: severity={overall_severity.value}, "
            f"drifted={features_drifted}/{features_total}"
        )

        return report

    def _test_numerical_feature(
        self,
        feature: str,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Optional[FeatureDriftResult]:
        if feature not in reference_df.columns or feature not in current_df.columns:
            logger.warning(f"Feature '{feature}' missing, skipping.")
            return None

        # Compute null rates BEFORE dropping — emit as part of drift signal
        ref_null_rate = float(reference_df[feature].isnull().mean())
        cur_null_rate = float(current_df[feature].isnull().mean())

        ref_vals = reference_df[feature].dropna().to_numpy(dtype=float)
        cur_vals = current_df[feature].dropna().to_numpy(dtype=float)

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            logger.warning(
                f"Skipping '{feature}': insufficient non-null values "
                f"(ref={len(ref_vals)}, cur={len(cur_vals)})"
            )
            return None

        ks_stat, ks_p      = self.test_suite.ks_test(ref_vals, cur_vals)
        psi_value          = self.test_suite.psi_numerical(ref_vals, cur_vals, bins=20)
        wasserstein        = self.test_suite.wasserstein_distance(ref_vals, cur_vals)

        drift_score = self._compute_numerical_drift_score(ks_stat, ks_p, psi_value)
        is_drifted  = (
            ks_p < self.settings.ks_p_value_threshold
            or psi_value > self.settings.psi_threshold
        )

        # Null rate drift is ALSO drift
        null_rate_delta = abs(cur_null_rate - ref_null_rate)
        if null_rate_delta > 0.10:
            is_drifted  = True
            drift_score = max(drift_score, null_rate_delta)

        # Update Prometheus per-feature gauges
        DRIFT_PSI.labels(model_id=self.model_id, feature=feature).set(psi_value)
        DRIFT_KS_PVALUE.labels(model_id=self.model_id, feature=feature).set(ks_p)

        details = {
            "ks_statistic":  round(ks_stat, 4),
            "ks_p_value":    round(ks_p, 6),
            "psi":           round(psi_value, 4),
            "wasserstein":   round(wasserstein, 4),
            "ref_mean":      round(float(np.mean(ref_vals)), 4),
            "cur_mean":      round(float(np.mean(cur_vals)), 4),
            "ref_std":       round(float(np.std(ref_vals)), 4),
            "cur_std":       round(float(np.std(cur_vals)), 4),
            "ref_null_rate": round(ref_null_rate, 4),
            "cur_null_rate": round(cur_null_rate, 4),
            "null_rate_delta": round(null_rate_delta, 4),
        }

        # Emit per-feature event
        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_FEATURE_CHECKED,
            step_name="drift_detection",
            title=f"{'🔴' if is_drifted else '🟢'} Drift: {feature}",
            message=(
                f"'{feature}': PSI={psi_value:.3f}, KS p={ks_p:.4f}, "
                f"null_delta={null_rate_delta:.3f} → "
                f"{'DRIFTED' if is_drifted else 'stable'}"
            ),
            model_id=self.model_id,
            status="warning" if is_drifted else "success",
            severity="warning" if is_drifted else "info",
            data={**details, "feature": feature},
        ))

        return FeatureDriftResult(
            feature_name=feature,
            test_name="ks+psi",
            statistic=ks_stat,
            p_value=ks_p,
            drift_score=drift_score,
            is_drifted=is_drifted,
            ref_null_rate=ref_null_rate,
            cur_null_rate=cur_null_rate,
            details=details,
        )

    def _test_categorical_feature(
        self,
        feature: str,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Optional[FeatureDriftResult]:
        if feature not in reference_df.columns or feature not in current_df.columns:
            return None

        ref_null_rate = float(reference_df[feature].isnull().mean())
        cur_null_rate = float(current_df[feature].isnull().mean())

        ref_vals = reference_df[feature].dropna()
        cur_vals = current_df[feature].dropna()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            return None

        psi_value          = self.test_suite.psi_categorical(ref_vals, cur_vals)
        chi2_stat, chi2_p  = self.test_suite.chi_square_test(ref_vals, cur_vals)

        drift_score = self._compute_categorical_drift_score(psi_value, chi2_p)
        is_drifted  = (
            psi_value > self.settings.psi_threshold
            or chi2_p < self.settings.ks_p_value_threshold
        )

        null_rate_delta = abs(cur_null_rate - ref_null_rate)
        if null_rate_delta > 0.10:
            is_drifted  = True
            drift_score = max(drift_score, null_rate_delta)

        new_categories     = set(cur_vals.unique()) - set(ref_vals.unique())
        missing_categories = set(ref_vals.unique()) - set(cur_vals.unique())

        details = {
            "psi":                round(psi_value, 4),
            "chi2_statistic":     round(chi2_stat, 4),
            "chi2_p_value":       round(chi2_p, 6),
            "new_categories":     list(new_categories)[:10],
            "missing_categories": list(missing_categories)[:10],
            "ref_null_rate":      round(ref_null_rate, 4),
            "cur_null_rate":      round(cur_null_rate, 4),
        }

        event_bus.emit(make_event(
            pipeline_run_id=self.pipeline_run_id,
            event_type=EventType.DRIFT_FEATURE_CHECKED,
            step_name="drift_detection",
            title=f"{'🔴' if is_drifted else '🟢'} Drift: {feature}",
            message=(
                f"'{feature}': PSI={psi_value:.3f}, χ²p={chi2_p:.4f} → "
                f"{'DRIFTED' if is_drifted else 'stable'}"
            ),
            model_id=self.model_id,
            status="warning" if is_drifted else "success",
            severity="warning" if is_drifted else "info",
            data={**details, "feature": feature},
        ))

        DRIFT_PSI.labels(model_id=self.model_id, feature=feature).set(psi_value)

        return FeatureDriftResult(
            feature_name=feature,
            test_name="psi+chi2",
            statistic=chi2_stat,
            p_value=chi2_p,
            drift_score=drift_score,
            is_drifted=is_drifted,
            ref_null_rate=ref_null_rate,
            cur_null_rate=cur_null_rate,
            details=details,
        )

    def _compute_numerical_drift_score(
        self, ks_stat: float, ks_p: float, psi: float
    ) -> float:
        ks_score  = min(ks_stat * 2, 1.0)
        psi_score = min(psi / 0.5, 1.0)
        return float(0.4 * ks_score + 0.6 * psi_score)

    def _compute_categorical_drift_score(self, psi: float, chi2_p: float) -> float:
        psi_score = min(psi / 0.5, 1.0)
        p_score   = 1.0 - min(chi2_p, 1.0)
        return float(0.6 * psi_score + 0.4 * p_score)

    def _detect_concept_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: str,
    ) -> tuple[bool, dict[str, Any]]:
        ref_target = reference_df[target_column].dropna()
        cur_target = current_df[target_column].dropna()

        if pd.api.types.is_numeric_dtype(ref_target):
            ks_stat, ks_p = self.test_suite.ks_test(
                np.asarray(ref_target), np.asarray(cur_target)
            )
            details = {
                "test":      "ks_2samp",
                "statistic": round(ks_stat, 4),
                "p_value":   round(ks_p, 6),
                "ref_mean":  round(float(np.mean(ref_target)), 4),
                "cur_mean":  round(float(np.mean(cur_target)), 4),
            }
            drifted = ks_p < 0.01
        else:
            psi = self.test_suite.psi_categorical(ref_target, cur_target)
            details = {
                "test":             "psi",
                "psi_value":        round(psi, 4),
                "ref_distribution": ref_target.value_counts(normalize=True).head(10).to_dict(),
                "cur_distribution": cur_target.value_counts(normalize=True).head(10).to_dict(),
            }
            drifted = psi > 0.15

        return drifted, details

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
        features_drifted: int,
        concept_drift: bool,
        feature_results: list[FeatureDriftResult],
    ) -> list[str]:
        recs = []
        if severity == DriftSeverity.CRITICAL:
            recs.append("URGENT: Critical drift. Immediate retraining required.")
        if concept_drift:
            recs.append("Concept drift in target. Model predictions degraded.")
        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            recs.append("Schedule retraining with latest production data.")
        if severity == DriftSeverity.MODERATE:
            recs.append("Monitor closely. Consider retraining if metrics degrade.")

        drifted = sorted(
            [fr for fr in feature_results if fr.is_drifted],
            key=lambda x: x.drift_score, reverse=True,
        )
        if drifted:
            recs.append(
                f"Top drifted: {[f.feature_name for f in drifted[:5]]}"
            )

        # Highlight null drift
        null_drifted = [
            fr for fr in feature_results
            if abs(fr.cur_null_rate - fr.ref_null_rate) > 0.10
        ]
        if null_drifted:
            recs.append(
                f"Null rate shift detected in: "
                f"{[f.feature_name for f in null_drifted]}"
                f" — check upstream data pipeline."
            )

        return recs