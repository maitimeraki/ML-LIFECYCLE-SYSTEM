"""
Champion-Challenger model comparison framework.
Determines whether a newly trained model should replace the current production model.

Why This Exists:
You've trained a new model. Its metrics look good. But should you deploy it?
The Naive Mistake: "New model has higher accuracy → deploy it."
The Reality: New models can be:
Actually better (genuine improvement)
Apparently better (random chance on test set)
Better on one metric, worse on others (regression)
Better but slower (latency tradeoff)
Better on test, worse in production (overfitting)
First Principle: You're not just comparing two numbers. You're making a business decision with consequences. A bad model in production costs real money, damages trust, and is hard to roll back.

The Core Question:
"Is the challenger's improvement real, meaningful, and safe enough to justify the deployment risk?"
This requires multiple independent checks, not just metric comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DeploymentSettings, get_settings
from src.common.enums import ApprovalStatus
from src.common.exceptions import ModelValidationError

logger = logging.getLogger("ml_platform.evaluation.champion_challenger")


@dataclass
class ModelPerformanceSnapshot:
    """ A complete performance fingerprint of one model version."""
    model_id: str
    model_version: str
    metrics: dict[str, float]
    primary_metric_name: str
    primary_metric_value: float
    evaluation_dataset_id: str
    evaluation_samples: int
    prediction_latency_p50_ms: float = 0.0
    prediction_latency_p99_ms: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class ComparisonResult:
    """Result of champion vs challenger comparison."""
    champion: ModelPerformanceSnapshot
    challenger: ModelPerformanceSnapshot
    challenger_is_better: bool
    improvement: float                   # Relative improvement on primary metric
    improvement_significant: bool        # Statistically significant?
    approval_status: ApprovalStatus
    comparison_details: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "challenger_is_better": self.challenger_is_better,
            "improvement": round(self.improvement, 6),
            "improvement_significant": self.improvement_significant,
            "approval_status": self.approval_status.value,
            "champion": {
                "model_id": self.champion.model_id,
                "version": self.champion.model_version,
                "primary_metric": self.champion.primary_metric_value,
            },
            "challenger": {
                "model_id": self.challenger.model_id,
                "version": self.challenger.model_version,
                "primary_metric": self.challenger.primary_metric_value,
            },
            "reasons": self.reasons,
            "timestamp": self.timestamp.isoformat(),
        }


class ChampionChallengerEvaluator:
    """
    Compares a challenger model against the current champion(New) using:
    1. Primary metric improvement (must exceed minimum threshold)
    2. Secondary metric non-regression (no significant degradation)
    3. Statistical significance testing (paired bootstrap or permutation test)
    4. Latency and resource constraints
    5. Automated approval gate

    This ensures only genuinely better models proceed to deployment.
    """

    def __init__(self, settings: Optional[DeploymentSettings] = None):
        self.settings = settings or get_settings().deployment

    def compare(
        self,
        champion: ModelPerformanceSnapshot,
        challenger: ModelPerformanceSnapshot,
        champion_predictions: Optional[np.ndarray] = None,
        challenger_predictions: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        """
        Run full champion-challenger comparison.

        Args:
            champion: Performance snapshot of current production model.
            challenger: Performance snapshot of newly trained model.
            champion_predictions: Optional predictions for statistical testing.
            challenger_predictions: Optional predictions for statistical testing.
            ground_truth: Optional ground truth for statistical testing.

        Returns:
            ComparisonResult with approval decision.
        """
        logger.info(
            f"Comparing champion ({champion.model_id} v{champion.model_version}) "
            f"vs challenger ({challenger.model_id} v{challenger.model_version})"
        )

        reasons: list[str] = []
        details: dict[str, Any] = {}

        # 1. Primary metric comparison               
        lower_is_better = champion.primary_metric_name.lower() in (
            "rmse", "mae", "mse", "loss", "error", "log_loss"
        )

        if lower_is_better:
            # For "lower is better" metrics (RMSE, MAE, MSE, etc.)
            improvement = (champion.primary_metric_value - challenger.primary_metric_value) / abs(
                champion.primary_metric_value
            ) if champion.primary_metric_value != 0 else 0.0
        else:
            # For "higher is better" metrics (accuracy, F1, precision, recall)
            improvement = (challenger.primary_metric_value - champion.primary_metric_value) / abs(
                champion.primary_metric_value
            ) if champion.primary_metric_value != 0 else 0.0
        
        """Why a threshold and not just "better"?:
        Statistical noise: A 0.1% improvement on 1000 samples is meaningless,
        Deployment risk: Every deployment has risk; improvement must justify it,
        Practical significance: 0.01% better accuracy doesn't change business outcomes.                                
        Typical threshold values:
        High-stakes (healthcare, finance): 1-2% minimum
        Standard applications: 0.5-1%
        Low-stakes (recommendations): 0.1-0.5%"""

        meets_threshold = improvement >= self.settings.min_champion_improvement
        details["primary_metric_improvement"] = round(improvement, 6)
        details["meets_improvement_threshold"] = meets_threshold

        if meets_threshold:
            reasons.append(
                f"Challenger improves {champion.primary_metric_name} by "
                f"{improvement:.2%} (threshold: {self.settings.min_champion_improvement:.2%})"
            )
        else:
            reasons.append(
                f"Challenger improvement ({improvement:.2%}) below threshold "
                f"({self.settings.min_champion_improvement:.2%})"
            )

        # 2. Secondary metrics non-regression
        secondary_regression = self._check_secondary_regression(champion, challenger) # Here we customize the regression according to the problem domain and business needs. For example, in a fraud detection model, we might prioritize recall over precision, so we'd check that recall doesn't drop significantly even if F1 improves.
        details["secondary_metric_regression"] = secondary_regression
        if secondary_regression:
            reasons.append(f"Secondary metric regression detected: {secondary_regression}")

        # 3. Statistical significance
        is_significant = False
        if (
            champion_predictions is not None
            and challenger_predictions is not None
            and ground_truth is not None
        ):
            is_significant, p_value = self._statistical_significance_test(
                champion_predictions, challenger_predictions, ground_truth,
                lower_is_better
            )
            details["significance_test_p_value"] = round(p_value, 6)
            details["is_statistically_significant"] = is_significant
            if is_significant:
                reasons.append(f"Improvement is statistically significant (p={p_value:.4f})")
            else:
                reasons.append(f"Improvement is NOT statistically significant (p={p_value:.4f})")
        else:
            is_significant = meets_threshold  # If no predictions available, use threshold alone
            details["significance_test_p_value"] = None
            reasons.append("Statistical significance test skipped (predictions not provided)")

        # 4. Latency check
        latency_ok = True
        if challenger.prediction_latency_p99_ms > 0:
            latency_ok = challenger.prediction_latency_p99_ms <= self.settings.rollback_on_latency_p99_ms
            details["latency_check_passed"] = latency_ok
            if not latency_ok:
                reasons.append(
                    f"Challenger latency ({challenger.prediction_latency_p99_ms:.1f}ms p99) "
                    f"exceeds limit ({self.settings.rollback_on_latency_p99_ms:.1f}ms)"
                )

        # 5. Overall decision
        challenger_is_better = meets_threshold and is_significant and not secondary_regression and latency_ok

        if challenger_is_better:
            approval_status = ApprovalStatus.AUTO_APPROVED
            reasons.append("Challenger APPROVED: all gates passed.")
        else:
            approval_status = ApprovalStatus.REJECTED
            reasons.append("Challenger REJECTED: one or more gates failed.")

        result = ComparisonResult(
            champion=champion,
            challenger=challenger,
            challenger_is_better=challenger_is_better,
            improvement=improvement,
            improvement_significant=is_significant,
            approval_status=approval_status,
            comparison_details=details,
            reasons=reasons,
        )

        logger.info(
            f"Comparison result: challenger_better={challenger_is_better}, "
            f"improvement={improvement:.4f}, approved={approval_status.value}"
        )

        return result

    def _check_secondary_regression(
        self,
        champion: ModelPerformanceSnapshot,
        challenger: ModelPerformanceSnapshot,
        max_regression: float = 0.02,
    ) -> dict[str, float]:
        """Check if challenger significantly regresses on any secondary metric.
        
        What this prevents:
        Scenario: New fraud model improves F1 from 0.85 to 0.87 (+2.3%)
        But recall drops from 0.92 to 0.82 (-10.9%)
        
        The problem: F1 improved because precision went way up, but recall crashed. You're now missing 10% more fraud cases. This is catastrophic despite the "improvement."
        """
        regressions = {}
        for metric_name, champion_value in champion.metrics.items():
            if metric_name == champion.primary_metric_name:
                continue

            challenger_value = challenger.metrics.get(metric_name)
            if challenger_value is None:
                continue

            lower_is_better = metric_name.lower() in ("rmse", "mae", "mse", "loss", "error")

            if lower_is_better:
                regression = (challenger_value - champion_value) / abs(champion_value) if champion_value != 0 else 0
            else:
                regression = (champion_value - challenger_value) / abs(champion_value) if champion_value != 0 else 0

            if regression > max_regression:
                regressions[metric_name] = round(regression, 4)

        return regressions

    def _statistical_significance_test(
        self,
        champion_preds: np.ndarray,
        challenger_preds: np.ndarray,
        ground_truth: np.ndarray,
        lower_is_better: bool,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> tuple[bool, float]:
        """
        Paired bootstrap test for statistical significance of model comparison.
        Tests whether the challenger's performance improvement is significant.
        
        Is this improvement real, or did we just get lucky on this test set?
        
        Why this matters:
        Test set of 1000 samples: 2% improvement could be random
        Test set of 1,000,000 samples: 0.1% improvement is probably real
        Sample size determines reliability
        """
        rng = np.random.RandomState(42)
        n = len(ground_truth)

        from sklearn.metrics import mean_squared_error, accuracy_score

        # Choose metric function
        if lower_is_better:
            metric_fn = lambda y, p: -mean_squared_error(y, p)  # Negate so higher is better
        else:
            metric_fn = accuracy_score

        champion_score = metric_fn(ground_truth, champion_preds)
        challenger_score = metric_fn(ground_truth, challenger_preds)
        observed_diff = challenger_score - champion_score

        # Bootstrap
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            gt_boot = ground_truth[indices]
            champ_boot = champion_preds[indices]
            chall_boot = challenger_preds[indices]

            champ_score_boot = metric_fn(gt_boot, champ_boot)
            chall_score_boot = metric_fn(gt_boot, chall_boot)
            bootstrap_diffs.append(chall_score_boot - champ_score_boot)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value: proportion of bootstrap samples where diff <= 0
        p_value = float(np.mean(bootstrap_diffs <= 0))
        is_significant = p_value < alpha

        return is_significant, p_value