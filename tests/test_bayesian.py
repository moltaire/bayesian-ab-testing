"""Tests for bayesian subpackage."""

import numpy as np
import pytest

from src.bayesian_ab import bayesian, simulation


class TestMetrics:
    """Tests for bayesian.metrics functions."""

    def test_prob_b_better_obvious_winner(self):
        """When B samples are always higher, P(B>A) = 1."""
        s_a = np.array([0.1, 0.1, 0.1])
        s_b = np.array([0.2, 0.2, 0.2])
        assert bayesian.prob_b_better(s_a, s_b) == 1.0

    def test_prob_b_better_obvious_loser(self):
        """When A samples are always higher, P(B>A) = 0."""
        s_a = np.array([0.2, 0.2, 0.2])
        s_b = np.array([0.1, 0.1, 0.1])
        assert bayesian.prob_b_better(s_a, s_b) == 0.0

    def test_expected_loss_values(self):
        """Expected loss is non-negative and sums correctly."""
        rng = np.random.default_rng(42)
        s_a = rng.beta(100, 900, 10_000)
        s_b = rng.beta(120, 880, 10_000)
        loss_a, loss_b = bayesian.expected_loss(s_a, s_b)
        assert loss_a >= 0
        assert loss_b >= 0
        # Winner (B) should have lower loss
        assert loss_b < loss_a

    def test_prob_above_rope(self):
        """ROPE probability is between 0 and 1."""
        rng = np.random.default_rng(42)
        s_a = rng.beta(100, 900, 10_000)
        s_b = rng.beta(100, 900, 10_000)  # same distribution
        prob = bayesian.prob_above_rope(s_a, s_b, rope=0.01)
        assert 0 <= prob <= 1
        # With same distribution and 1% ROPE, should be low
        assert prob < 0.5

    def test_hdi_contains_mean(self):
        """HDI should contain the distribution mean."""
        rng = np.random.default_rng(42)
        samples = rng.beta(100, 100, 10_000)  # mean = 0.5
        lo, hi = bayesian.hdi(samples, prob=0.94)
        assert lo < 0.5 < hi

    def test_hdi_width(self):
        """HDI with higher prob should be wider."""
        rng = np.random.default_rng(42)
        samples = rng.beta(10, 10, 10_000)
        lo_90, hi_90 = bayesian.hdi(samples, prob=0.90)
        lo_50, hi_50 = bayesian.hdi(samples, prob=0.50)
        assert (hi_90 - lo_90) > (hi_50 - lo_50)


class TestConjugate:
    """Tests for bayesian.conjugate functions."""

    def test_beta_prior_uniform(self):
        """Uniform prior is Beta(1, 1)."""
        prior = bayesian.BetaPrior.uniform()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0

    def test_beta_prior_from_history(self):
        """Prior from history has correct parameters."""
        prior = bayesian.BetaPrior.from_history(100, 1000)
        assert prior.alpha == 101.0
        assert prior.beta == 901.0

    def test_beta_posterior_mean(self):
        """Posterior mean is between prior and MLE."""
        prior = bayesian.BetaPrior.from_history(100, 1000)  # prior mean = 0.10
        post = bayesian.beta_posterior(n=200, conversions=40, prior=prior)  # MLE = 0.20
        assert 0.10 < post.mean() < 0.20

    def test_beta_posterior_uniform_prior(self):
        """With uniform prior, posterior mean ≈ MLE for large n."""
        post = bayesian.beta_posterior(n=10_000, conversions=1200)
        assert abs(post.mean() - 0.12) < 0.01

    def test_sequential_analysis_returns_dataframe(self):
        """Sequential analysis returns DataFrame with expected columns."""
        data = simulation.simulate_ab_test(n_days=3, seed=42)
        result = bayesian.sequential_analysis(data, seed=42)
        assert len(result) == 3
        assert "prob_b_better" in result.columns
        assert "expected_loss_a" in result.columns
        assert "expected_loss_b" in result.columns
        assert "prob_above_rope" in result.columns

    def test_sequential_analysis_monotonic_uncertainty(self):
        """Expected loss should generally decrease over time (more data = less uncertainty)."""
        data = simulation.simulate_ab_test(n_days=10, p_a=0.10, p_b=0.15, seed=42)
        result = bayesian.sequential_analysis(data, seed=42)
        # Not strictly monotonic due to randomness, but end should be lower than start
        assert result["expected_loss_b"].iloc[-1] < result["expected_loss_b"].iloc[0]


class TestSimulation:
    """Tests for simulation module."""

    def test_simulate_ab_test_shape(self):
        """Simulation returns correct number of rows."""
        data = simulation.simulate_ab_test(n_days=5, daily_n=100, seed=42)
        assert len(data) == 10  # 5 days x 2 variants

    def test_simulate_ab_test_columns(self):
        """Simulation has expected columns."""
        data = simulation.simulate_ab_test(n_days=3, seed=42)
        assert set(data.columns) == {"day", "variant", "conversions", "n"}

    def test_simulate_ab_test_reproducible(self):
        """Same seed gives same results."""
        data1 = simulation.simulate_ab_test(n_days=5, seed=123)
        data2 = simulation.simulate_ab_test(n_days=5, seed=123)
        assert data1["conversions"].tolist() == data2["conversions"].tolist()
