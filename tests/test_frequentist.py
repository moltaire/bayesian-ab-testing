"""Tests for frequentist module."""

import pytest

from src.bayesian_ab import frequentist, simulation


class TestPowerAnalysis:
    """Tests for power analysis."""

    def test_power_analysis_returns_result(self):
        """Power analysis returns a result object."""
        result = frequentist.power_analysis(p_control=0.10, relative_lift=0.20)
        assert result.n_per_group > 0
        assert result.total_n == 2 * result.n_per_group

    def test_power_analysis_larger_effect_needs_fewer_samples(self):
        """Larger effect size requires fewer samples."""
        small_effect = frequentist.power_analysis(p_control=0.10, relative_lift=0.10)
        large_effect = frequentist.power_analysis(p_control=0.10, relative_lift=0.50)
        assert large_effect.n_per_group < small_effect.n_per_group

    def test_power_analysis_higher_power_needs_more_samples(self):
        """Higher power requires more samples."""
        low_power = frequentist.power_analysis(p_control=0.10, relative_lift=0.20, power=0.70)
        high_power = frequentist.power_analysis(p_control=0.10, relative_lift=0.20, power=0.90)
        assert high_power.n_per_group > low_power.n_per_group


class TestProportionTest:
    """Tests for proportion test."""

    def test_proportion_test_significant(self):
        """Test detects significant difference with large effect."""
        data = simulation.simulate_ab_test(
            n_days=20, daily_n=500, p_a=0.10, p_b=0.15, seed=42
        )
        result = frequentist.proportion_test(data)
        assert result.significant

    def test_proportion_test_not_significant(self):
        """Test does not detect difference when there is none."""
        data = simulation.simulate_ab_test(
            n_days=5, daily_n=100, p_a=0.10, p_b=0.10, seed=42
        )
        result = frequentist.proportion_test(data)
        # With no true difference, should usually not be significant
        # (can fail by chance ~5% of time)
        assert result.p_value > 0.01

    def test_proportion_test_rates(self):
        """Test computes sensible rates."""
        data = simulation.simulate_ab_test(
            n_days=10, daily_n=1000, p_a=0.10, p_b=0.12, seed=42
        )
        result = frequentist.proportion_test(data)
        # Rates should be close to true values with large samples
        assert 0.08 < result.rate_a < 0.12
        assert 0.10 < result.rate_b < 0.14
