"""Tests for dataset_profile.py — classification taxonomy and profile builder."""

import numpy as np
import pandas as pd
import pytest

from src.training.dataset_profile import (
    DatasetProfile,
    DatasetCategory,
    RegressionSize,
    classify_dataset,
)


class TestClassifyDataset:
    """Test the classify_dataset function with boundary values."""

    def test_extreme_imbalance(self):
        assert classify_dataset(True, 2, 0.001, 10000) == DatasetCategory.EXTREME_IMBALANCE
        assert classify_dataset(True, 2, 0.009, 10000) == DatasetCategory.EXTREME_IMBALANCE

    def test_severe_imbalance(self):
        assert classify_dataset(True, 2, 0.01, 10000) == DatasetCategory.SEVERE_IMBALANCE
        assert classify_dataset(True, 2, 0.03, 10000) == DatasetCategory.SEVERE_IMBALANCE
        assert classify_dataset(True, 2, 0.049, 10000) == DatasetCategory.SEVERE_IMBALANCE

    def test_moderate_imbalance(self):
        assert classify_dataset(True, 2, 0.05, 10000) == DatasetCategory.MODERATE_IMBALANCE
        assert classify_dataset(True, 2, 0.15, 10000) == DatasetCategory.MODERATE_IMBALANCE
        assert classify_dataset(True, 2, 0.19, 10000) == DatasetCategory.MODERATE_IMBALANCE

    def test_mild_imbalance(self):
        assert classify_dataset(True, 2, 0.20, 10000) == DatasetCategory.MILD_IMBALANCE
        assert classify_dataset(True, 2, 0.30, 10000) == DatasetCategory.MILD_IMBALANCE
        assert classify_dataset(True, 2, 0.44, 10000) == DatasetCategory.MILD_IMBALANCE

    def test_balanced_binary(self):
        assert classify_dataset(True, 2, 0.45, 10000) == DatasetCategory.BALANCED_BINARY
        assert classify_dataset(True, 2, 0.50, 10000) == DatasetCategory.BALANCED_BINARY
        assert classify_dataset(True, 2, 1.0, 10000) == DatasetCategory.BALANCED_BINARY

    def test_multiclass_balanced(self):
        assert classify_dataset(True, 5, 0.60, 10000) == DatasetCategory.MULTICLASS_BALANCED

    def test_multiclass_imbalanced(self):
        assert classify_dataset(True, 5, 0.20, 10000) == DatasetCategory.MULTICLASS_IMBALANCED
        assert classify_dataset(True, 3, 0.10, 10000) == DatasetCategory.MULTICLASS_IMBALANCED

    def test_regression_small(self):
        assert classify_dataset(False, None, 1.0, 1000) == DatasetCategory.REGRESSION_SMALL
        assert classify_dataset(False, None, 1.0, 4999) == DatasetCategory.REGRESSION_SMALL

    def test_regression_medium(self):
        assert classify_dataset(False, None, 1.0, 5000) == DatasetCategory.REGRESSION_MEDIUM
        assert classify_dataset(False, None, 1.0, 10000) == DatasetCategory.REGRESSION_MEDIUM
        assert classify_dataset(False, None, 1.0, 49999) == DatasetCategory.REGRESSION_MEDIUM

    def test_regression_large(self):
        assert classify_dataset(False, None, 1.0, 50000) == DatasetCategory.REGRESSION_LARGE
        assert classify_dataset(False, None, 1.0, 100000) == DatasetCategory.REGRESSION_LARGE


class TestDatasetCategoryProperties:
    def test_is_classification(self):
        assert DatasetCategory.EXTREME_IMBALANCE.is_classification is True
        assert DatasetCategory.BALANCED_BINARY.is_classification is True
        assert DatasetCategory.REGRESSION_SMALL.is_classification is False

    def test_is_regression(self):
        assert DatasetCategory.REGRESSION_MEDIUM.is_regression is True
        assert DatasetCategory.EXTREME_IMBALANCE.is_regression is False

    def test_imbalance_severity(self):
        assert DatasetCategory.EXTREME_IMBALANCE.imbalance_severity == 4
        assert DatasetCategory.SEVERE_IMBALANCE.imbalance_severity == 3
        assert DatasetCategory.MODERATE_IMBALANCE.imbalance_severity == 2
        assert DatasetCategory.MILD_IMBALANCE.imbalance_severity == 1
        assert DatasetCategory.BALANCED_BINARY.imbalance_severity == 0
        assert DatasetCategory.REGRESSION_SMALL.imbalance_severity == -1


class TestDatasetProfileFromDataFrame:
    def _make_df(self, n, n_features, target_col, target_values):
        data = {f"f{i}": np.random.randn(n) for i in range(n_features)}
        data[target_col] = target_values
        return pd.DataFrame(data)

    def test_extreme_imbalance_fraud(self):
        n = 10000
        target = np.zeros(n, dtype=int)
        target[:10] = 1
        np.random.shuffle(target)
        df = self._make_df(n, 5, "target", target)
        profile = DatasetProfile.from_dataframe(df, [f"f{i}" for i in range(5)], "target")
        assert profile.category == DatasetCategory.EXTREME_IMBALANCE
        assert profile.is_classification is True
        assert profile.has_imbalanced_classes is True
        assert profile.scale_pos_weight > 100

    def test_balanced_binary(self):
        n = 1000
        target = np.random.choice([0, 1], n, p=[0.5, 0.5])
        df = self._make_df(n, 3, "target", target)
        profile = DatasetProfile.from_dataframe(df, ["f0", "f1", "f2"], "target")
        assert profile.category == DatasetCategory.BALANCED_BINARY
        assert profile.class_balance_ratio >= 0.45

    def test_regression_medium(self):
        n = 10000
        df = self._make_df(n, 5, "price", np.random.randn(n) * 1000 + 50000)
        profile = DatasetProfile.from_dataframe(df, [f"f{i}" for i in range(5)], "price")
        assert profile.category == DatasetCategory.REGRESSION_MEDIUM
        assert profile.is_classification is False
        assert profile.regression_size == RegressionSize.MEDIUM

    def test_multiclass_balanced(self):
        n = 5000
        target = np.random.choice([0, 1, 2, 3, 4], n)
        df = self._make_df(n, 4, "target", target)
        profile = DatasetProfile.from_dataframe(df, ["f0", "f1", "f2", "f3"], "target")
        assert profile.category == DatasetCategory.MULTICLASS_BALANCED
        assert profile.is_multiclass is True
        assert profile.n_classes == 5

    def test_minority_class_counts(self):
        n = 1000
        target = np.array([0] * 800 + [1] * 150 + [2] * 50)
        np.random.shuffle(target)
        df = self._make_df(n, 2, "target", target)
        profile = DatasetProfile.from_dataframe(df, ["f0", "f1"], "target")
        assert len(profile.minority_class_counts) == 3
        assert sum(profile.minority_class_counts.values()) == n

    def test_per_class_scale_weights(self):
        n = 1000
        target = np.array([0] * 700 + [1] * 200 + [2] * 100)
        np.random.shuffle(target)
        df = self._make_df(n, 2, "target", target)
        profile = DatasetProfile.from_dataframe(df, ["f0", "f1"], "target")
        assert len(profile.per_class_scale_weights) == 3
        assert profile.per_class_scale_weights["0"] == pytest.approx(1.0, abs=0.01)
        assert profile.per_class_scale_weights["2"] == pytest.approx(7.0, abs=0.1)
