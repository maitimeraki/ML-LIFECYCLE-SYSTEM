"""
Production drift detection engine.
Runs statistical tests across features to detect data and concept drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import DriftSettings, get_settings
from src.common.enums import DriftSeverity
from src.drift.statistical_tests import StatisticalTestSuite

logger = logging.getLogger("ml_platform.drift.detector")


@dataclass
class FeatureDriftResult:
    """A single diagnostic test result for one feature.
    When to use each field:
    is_drifted → Your monitoring dashboard (red/green light),,
    drift_score → Priority ranking (which features need attention first),
    p_value → Statistical rigor (prevent false alarms in noisy data),
    details → Root cause analysis (debugging why drift occurred)."""
    feature_name: str
    test_name: str
    statistic: float
    p_value: Optional[float]
    drift_score: float       # Normalized 0-1
    is_drifted: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """The complete health assessment of your ML system."""
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
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_severity": self.overall_severity.value,
            "overall_drift_score": round(self.overall_drift_score, 4),
            "features_drifted": self.features_drifted,
            "features_total": self.features_total,
            "drift_proportion": round(self.drift_proportion, 4),
            "concept_drift_detected": self.concept_drift_detected,
            "feature_results": [
                {
                    "feature": fr.feature_name,
                    "test": fr.test_name,
                    "statistic": round(fr.statistic, 4),
                    "p_value": round(fr.p_value, 6) if fr.p_value else None,
                    "drift_score": round(fr.drift_score, 4),
                    "is_drifted": fr.is_drifted,
                }
                for fr in self.feature_results
            ],
            "recommendations": self.recommendations,
        }


class DriftDetector:
    """
    Detects data drift and concept drift between a reference dataset
    (used for training) and current production data.

    Workflow:
    1. Feature-level drift detection (KS, PSI, Chi-Square, Wasserstein)
    2. Aggregate drift scoring
    3. Concept drift detection (if target labels available)
    4. Severity classification
    5. Actionable recommendations
    """

    def __init__(self, settings: Optional[DriftSettings] = None):
        self.settings = settings or get_settings().drift
        self.test_suite = StatisticalTestSuite()

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
        """
        Run comprehensive drift detection.

        Args:
            reference_df: Reference (training) dataset.
            current_df: Current (production) dataset.
            numerical_features: List of numerical feature column names.
            categorical_features: List of categorical feature column names.
            target_column: Optional target column for concept drift.
            report_id: Optional report identifier.
            reference_dataset_id: ID of reference dataset.
            current_dataset_id: ID of current dataset.

        Returns:
            DriftReport with per-feature and aggregate results.
        """
        import uuid

        report_id = report_id or str(uuid.uuid4())
        logger.info(
            f"Running drift detection: report_id={report_id}, "
            f"reference_rows={len(reference_df)}, current_rows={len(current_df)}"
        )

        # Validate minimum sample size
        if len(current_df) < self.settings.min_samples_for_drift:
            logger.warning(
                f"Current dataset has {len(current_df)} rows, below minimum "
                f"{self.settings.min_samples_for_drift}. Drift results may be unreliable."
            )

        feature_results: list[FeatureDriftResult] = []

        # --- Numerical feature drift ---
        for feature in numerical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature '{feature}' not found in one of the datasets, skipping.")
                continue

            ref_values = reference_df[feature].dropna().values  # Here we directly drop rows from each feature which makes probmetics in production when we have missing values. We can impute them with mean or median or some other strategy but for now we will just drop them.
            cur_values = current_df[feature].dropna().values

            if len(ref_values) < 10 or len(cur_values) < 10:
                continue

            # KS Test
            ks_stat, ks_p = self.test_suite.ks_test(np.array(ref_values), np.array(cur_values))

            # PSI
            psi_value = self.test_suite.psi_numerical(np.array(ref_values), np.array(cur_values), bins=20)

            # Wasserstein distance (normalized)
            wasserstein = self.test_suite.wasserstein_distance(np.array(ref_values), np.array(cur_values))

            # Combined drift score
            drift_score = self._compute_numerical_drift_score(ks_stat, ks_p, psi_value)
            is_drifted = (
                ks_p < self.settings.ks_p_value_threshold
                or psi_value > self.settings.psi_threshold
            )

            feature_results.append(FeatureDriftResult(
                feature_name=feature,
                test_name="ks+psi",
                statistic=ks_stat,
                p_value=ks_p,
                drift_score=drift_score,
                is_drifted=is_drifted,
                details={
                    "ks_statistic": round(ks_stat, 4),
                    "ks_p_value": round(ks_p, 6),
                    "psi": round(psi_value, 4),
                    "wasserstein": round(wasserstein, 4),
                    "ref_mean": round(float(np.mean(np.array(ref_values))), 4),
                    "cur_mean": round(float(np.mean(np.array(cur_values))), 4),
                    "ref_std": round(float(np.std(np.array(ref_values))), 4),
                    "cur_std": round(float(np.std(np.array(cur_values))), 4),
                },
            ))

        # --- Categorical feature drift ---
        for feature in categorical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue

            ref_values = reference_df[feature].dropna()
            cur_values = current_df[feature].dropna()

            psi_value = self.test_suite.psi_categorical(ref_values, cur_values)
            chi2_stat, chi2_p = self.test_suite.chi_square_test(ref_values, cur_values)

            drift_score = self._compute_categorical_drift_score(psi_value, chi2_p)
            is_drifted = psi_value > self.settings.psi_threshold or chi2_p < self.settings.ks_p_value_threshold

            new_categories = set(cur_values.unique()) - set(ref_values.unique())
            missing_categories = set(ref_values.unique()) - set(cur_values.unique())

            feature_results.append(FeatureDriftResult(
                feature_name=feature,
                test_name="psi+chi2",
                statistic=chi2_stat,
                p_value=chi2_p,
                drift_score=drift_score,
                is_drifted=is_drifted,
                details={
                    "psi": round(psi_value, 4),
                    "chi2_statistic": round(chi2_stat, 4),
                    "chi2_p_value": round(chi2_p, 6),
                    "new_categories": list(new_categories)[:10],
                    "missing_categories": list(missing_categories)[:10],
                },
            ))

        # --- Concept drift ---
        concept_drift_detected = False
        concept_drift_details: dict[str, Any] = {}
        if target_column and target_column in reference_df.columns and target_column in current_df.columns:
            concept_drift_detected, concept_drift_details = self._detect_concept_drift(
                reference_df, current_df, target_column
            )

        # --- Aggregate ---
        features_drifted = sum(1 for fr in feature_results if fr.is_drifted)
        features_total = len(feature_results)
        drift_proportion = features_drifted / features_total if features_total > 0 else 0.0

        overall_drift_score = float(
            np.mean([fr.drift_score for fr in feature_results]) if feature_results else 0.0
        )

        overall_severity = self._classify_severity(
            drift_proportion, overall_drift_score, concept_drift_detected
        )

        recommendations = self._generate_recommendations(
            overall_severity, drift_proportion, features_drifted,
            concept_drift_detected, feature_results
        )

        report = DriftReport(
            report_id=report_id,
            timestamp=datetime.now(timezone.utc),
            reference_dataset_id=reference_dataset_id,
            current_dataset_id=current_dataset_id,
            overall_severity=overall_severity,
            overall_drift_score=float(overall_drift_score),
            features_drifted=features_drifted,
            features_total=features_total,
            drift_proportion=drift_proportion,
            feature_results=feature_results,
            concept_drift_detected=concept_drift_detected,
            concept_drift_details=concept_drift_details,
            recommendations=recommendations,
        )

        logger.info(
            f"Drift detection complete: severity={overall_severity.value}, "
            f"drifted={features_drifted}/{features_total}, concept_drift={concept_drift_detected}"
        )

        return report

    def _compute_numerical_drift_score(
        self, ks_stat: float, ks_p: float, psi: float
    ) -> float:
        ks_score = min(ks_stat * 2, 1.0)
        psi_score = min(psi / 0.5, 1.0)  # Normalize: 0.5 PSI -> score 1.0
        return float(0.4 * ks_score + 0.6 * psi_score)

    def _compute_categorical_drift_score(self, psi: float, chi2_p: float) -> float:
        psi_score = min(psi / 0.5, 1.0)
        p_score = 1.0 - min(chi2_p, 1.0)
        return float(0.6 * psi_score + 0.4 * p_score)

    def _detect_concept_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: str,
    ) -> tuple[bool, dict[str, Any]]:
        ref_target = reference_df[target_column].dropna()
        cur_target = current_df[target_column].dropna()

        details: dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(ref_target):
            ks_stat, ks_p = self.test_suite.ks_test(np.asarray(ref_target.values), np.asarray(cur_target.values))
            details = {
                "test": "ks_2samp",
                "statistic": round(ks_stat, 4),
                "p_value": round(ks_p, 6),
                "ref_mean": round(float(np.mean(ref_target.to_numpy())), 4),
                "cur_mean": round(float(np.mean(cur_target.to_numpy())), 4),
            }
            drifted = ks_p < 0.01
        else:
            psi = self.test_suite.psi_categorical(ref_target, cur_target)
            details = {
                "test": "psi",
                "psi_value": round(psi, 4),
                "ref_distribution": ref_target.value_counts(normalize=True).head(10).to_dict(),
                "cur_distribution": cur_target.value_counts(normalize=True).head(10).to_dict(),
            }
            drifted = psi > 0.15

        return drifted, details

    def _classify_severity(
        self, drift_proportion: float, drift_score: float, concept_drift: bool
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
        recommendations = []

        if severity == DriftSeverity.CRITICAL:
            recommendations.append("URGENT: Critical drift detected. Immediate model retraining recommended.")
        if concept_drift:
            recommendations.append("Concept drift detected in target variable. Model predictions may be degraded.")
        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            recommendations.append("Schedule model retraining with latest production data.")
            recommendations.append("Review top drifted features for upstream data pipeline issues.")
        if severity == DriftSeverity.MODERATE:
            recommendations.append("Monitor closely. Consider retraining if performance metrics degrade.")

        # Top drifted features
        drifted = sorted(
            [fr for fr in feature_results if fr.is_drifted],
            key=lambda x: x.drift_score,
            reverse=True,
        )
        if drifted:
            top_features = [f.feature_name for f in drifted[:5]]
            recommendations.append(f"Top drifted features: {top_features}")

        return recommendations