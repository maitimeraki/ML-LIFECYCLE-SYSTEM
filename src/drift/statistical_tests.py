"""
Statistical test implementations for drift detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance as scipy_wasserstein


class StatisticalTestSuite:
    """Collection of statistical tests for distribution comparison."""

    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        """Kolmogorov-Smirnov two-sample test."""
        ref_array = np.asarray(reference).flatten()
        cur_array = np.asarray(current).flatten()
        statistic, pvalue = stats.ks_2samp(ref_array, cur_array, alternative="two-sided")
        # Access as attributes - this satisfies Pylance's type checker
        return float(np.asarray(statistic)), float(np.asarray(pvalue))

    @staticmethod
    def psi_numerical(
        reference: np.ndarray, current: np.ndarray, bins: int = 20
    ) -> float:
        """
        Population Stability Index for numerical features.
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift
        """
        eps = 1e-8

        # Use reference quantiles for binning to ensure consistent bins
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
        cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum() + eps
        cur_pct = cur_counts / cur_counts.sum() + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    @staticmethod
    def psi_categorical(
        reference: pd.Series, current: pd.Series
    ) -> float:
        """Population Stability Index for categorical features."""
        eps = 1e-8

        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)

        all_categories = set(ref_counts.index) | set(cur_counts.index)

        psi = 0.0
        for cat in all_categories:
            ref_pct = ref_counts.get(cat, 0) + eps
            cur_pct = cur_counts.get(cat, 0) + eps
            psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

        return float(psi)

    @staticmethod
    def chi_square_test(
        reference: pd.Series, current: pd.Series
    ) -> tuple[float, float]:
        """Chi-square test for categorical distributions."""
        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()

        all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))

        ref_freq = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=float)
        cur_freq = np.array([cur_counts.get(c, 0) for c in all_categories], dtype=float)

        # Scale reference to match current sample size
        ref_expected = ref_freq / ref_freq.sum() * cur_freq.sum()

        # Avoid zero expected values
        mask = ref_expected > 0
        if mask.sum() < 2:
            return 0.0, 1.0

        stat, p_value = stats.chisquare(cur_freq[mask], ref_expected[mask])
        return float(stat), float(p_value)

    @staticmethod
    def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
        """Wasserstein (Earth Mover's) distance, normalized by reference std."""
        dist = scipy_wasserstein(reference, current)
        ref_std = np.std(reference)
        if ref_std > 0:
            dist = dist / ref_std  # Normalize
        return float(dist)

    @staticmethod
    def jensen_shannon_divergence(
        reference: np.ndarray, current: np.ndarray, bins: int = 20
    ) -> float:
        """Jensen-Shannon divergence (symmetric KL divergence)."""
        eps = 1e-8
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_hist = np.histogram(reference, bins=breakpoints)[0].astype(float) + eps
        cur_hist = np.histogram(current, bins=breakpoints)[0].astype(float) + eps

        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()

        m = 0.5 * (ref_hist + cur_hist)

        jsd = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(cur_hist * np.log(cur_hist / m))
        return float(jsd)