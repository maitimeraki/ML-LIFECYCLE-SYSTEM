"""
ImbalanceStrategy dataclass — encapsulates all training decisions for a
given dataset category.

Instead of a simple boolean flag, this carries the complete strategy:
which models to try, which metric to optimize, how many CV folds,
what hyperparameter overrides to apply, and what fit-time extras
(sample_weight, early stopping) to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ImbalanceStrategy:
    """Complete training strategy derived from a DatasetCategory.

    Every field is a decision the system makes automatically based on
    the dataset classification. This replaces the old boolean
    ``is_imbalanced`` flag with a rich, multi-dimensional strategy.

    Attributes
    ----------
    category : str
        The dataset category string this strategy is designed for.
    model_families : list[str]
        Model families to evaluate, ordered by priority.
    scoring_metric : str
        Primary metric to optimize during HPO.
    cv_folds : int
        Number of cross-validation folds.
    class_weight : str or None
        sklearn class_weight parameter: 'balanced', 'balanced_subsample', or None.
    scale_pos_weight : float
        For XGBoost/LightGBM: neg/pos ratio. 1.0 = no adjustment.
    use_sample_weight : bool
        Whether to compute and pass sample_weight at fit time.
    is_unbalanced_lgbm : bool
        For LightGBM: set is_unbalanced=True.
    search_space_overrides : dict
        Per-algorithm hyperparameter overrides keyed by model family.
    fit_extras : dict
        Extra kwargs passed to model.fit().
    """

    category: str = ""
    model_families: list[str] = field(default_factory=list)
    scoring_metric: str = "accuracy"
    cv_folds: int = 5
    class_weight: Optional[str] = None
    scale_pos_weight: float = 1.0
    use_sample_weight: bool = False
    is_unbalanced_lgbm: bool = False
    search_space_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    fit_extras: dict[str, Any] = field(default_factory=dict)

    # -- Convenience properties ------------------------------------------------

    @property
    def is_extreme(self) -> bool:
        return self.category == "extreme_imbalance"

    @property
    def is_severe(self) -> bool:
        return self.category == "severe_imbalance"

    @property
    def is_classification(self) -> bool:
        return self.category not in {
            "regression_small",
            "regression_medium",
            "regression_large",
        }

    @property
    def needs_imbalance_handling(self) -> bool:
        return (
            self.class_weight is not None
            or self.scale_pos_weight > 1.0
            or self.use_sample_weight
            or self.is_unbalanced_lgbm
        )

    def get_overrides(self, family: str) -> dict[str, Any]:
        """Get hyperparameter overrides for a specific model family."""
        return self.search_space_overrides.get(family, {})

    def summary(self) -> dict[str, Any]:
        """Human-readable summary for logging."""
        return {
            "category": self.category,
            "model_families": self.model_families,
            "scoring_metric": self.scoring_metric,
            "cv_folds": self.cv_folds,
            "class_weight": self.class_weight,
            "scale_pos_weight": round(self.scale_pos_weight, 1),
            "use_sample_weight": self.use_sample_weight,
            "is_unbalanced_lgbm": self.is_unbalanced_lgbm,
        }
