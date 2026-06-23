"""Tests for strategy_factory.py — the decision engine."""

import pytest

from src.training.dataset_profile import DatasetProfile, DatasetCategory
from src.training.strategy_factory import get_strategy


def _make_profile(category, n_samples=10000, n_classes=2, balance=0.5, spw=1.0):
    return DatasetProfile(
        n_samples=n_samples, n_features=10, n_classes=n_classes,
        is_classification=category.is_classification,
        class_balance_ratio=balance,
        has_imbalanced_classes=balance < 0.45 if category.is_classification else False,
        feature_types={}, memory_mb=50,
        recommended_families=["random_forest"],
        recommended_scoring="f1" if category.is_classification else "neg_root_mean_squared_error",
        problem_subtype="binary" if n_classes == 2 and category.is_classification else "regression",
        category=category, scale_pos_weight=spw,
    )


class TestExtremeImbalanceStrategy:
    def setup_method(self):
        self.strategy = get_strategy(
            _make_profile(DatasetCategory.EXTREME_IMBALANCE, spw=999.0))

    def test_scoring_metric(self):
        assert self.strategy.scoring_metric == "pr_auc"

    def test_cv_folds(self):
        assert self.strategy.cv_folds == 3

    def test_class_weight(self):
        assert self.strategy.class_weight == "balanced_subsample"

    def test_scale_pos_weight(self):
        assert self.strategy.scale_pos_weight == 999.0

    def test_use_sample_weight(self):
        assert self.strategy.use_sample_weight is True

    def test_is_unbalanced_lgbm(self):
        assert self.strategy.is_unbalanced_lgbm is True

    def test_needs_imbalance_handling(self):
        assert self.strategy.needs_imbalance_handling is True

    def test_xgboost_overrides(self):
        assert "scale_pos_weight" in self.strategy.get_overrides("xgboost")


class TestSevereImbalanceStrategy:
    def setup_method(self):
        self.strategy = get_strategy(
            _make_profile(DatasetCategory.SEVERE_IMBALANCE, spw=33.0))

    def test_scoring_metric(self):
        assert self.strategy.scoring_metric == "f1"

    def test_cv_folds(self):
        assert self.strategy.cv_folds == 4

    def test_use_sample_weight(self):
        assert self.strategy.use_sample_weight is True


class TestModerateImbalanceStrategy:
    def setup_method(self):
        self.strategy = get_strategy(
            _make_profile(DatasetCategory.MODERATE_IMBALANCE, balance=0.15))

    def test_scoring_metric(self):
        assert self.strategy.scoring_metric == "f1_weighted"

    def test_cv_folds(self):
        assert self.strategy.cv_folds == 5

    def test_class_weight(self):
        assert self.strategy.class_weight == "balanced"

    def test_use_sample_weight(self):
        assert self.strategy.use_sample_weight is False


class TestBalancedBinaryStrategy:
    def setup_method(self):
        self.strategy = get_strategy(
            _make_profile(DatasetCategory.BALANCED_BINARY, balance=0.50))

    def test_scoring_metric(self):
        assert self.strategy.scoring_metric == "accuracy"

    def test_class_weight(self):
        assert self.strategy.class_weight is None

    def test_needs_imbalance_handling(self):
        assert self.strategy.needs_imbalance_handling is False


class TestRegressionStrategies:
    def test_small(self):
        s = get_strategy(_make_profile(DatasetCategory.REGRESSION_SMALL, n_samples=1000, n_classes=None))
        assert s.scoring_metric == "neg_root_mean_squared_error"
        assert s.cv_folds == 5

    def test_medium(self):
        s = get_strategy(_make_profile(DatasetCategory.REGRESSION_MEDIUM, n_samples=10000, n_classes=None))
        assert s.cv_folds == 5

    def test_large(self):
        s = get_strategy(_make_profile(DatasetCategory.REGRESSION_LARGE, n_samples=100000, n_classes=None))
        assert s.cv_folds == 3


class TestMulticlassStrategies:
    def test_balanced(self):
        s = get_strategy(_make_profile(DatasetCategory.MULTICLASS_BALANCED, n_classes=5, balance=0.60))
        assert s.scoring_metric == "f1_macro"
        assert s.cv_folds == 5

    def test_imbalanced(self):
        s = get_strategy(_make_profile(DatasetCategory.MULTICLASS_IMBALANCED, n_classes=5, balance=0.20, spw=5.0))
        assert s.scoring_metric == "f1_macro"
        assert s.cv_folds == 4
        assert s.class_weight == "balanced"
        assert s.use_sample_weight is True


class TestStrategySummary:
    def test_summary_fields(self):
        s = get_strategy(_make_profile(DatasetCategory.EXTREME_IMBALANCE, spw=500.0))
        summary = s.summary()
        assert "category" in summary
        assert "scoring_metric" in summary
        assert "cv_folds" in summary
        assert summary["category"] == "extreme_imbalance"
