"""Conjugate Beta-Binomial model for A/B testing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import expected_loss, prob_above_rope, prob_b_better


@dataclass
class BetaPrior:
    """Beta distribution prior for a conversion rate.

    alpha and beta are pseudo-counts of successes and failures.
    """

    alpha: float = 1.0
    beta: float = 1.0

    @classmethod
    def uniform(cls):
        """Non-informative prior: Beta(1, 1)."""
        return cls(1.0, 1.0)

    @classmethod
    def from_history(cls, conversions, total):
        """Informative prior from historical data."""
        return cls(conversions + 1.0, (total - conversions) + 1.0)


def beta_posterior(n, conversions, prior=None):
    """Compute posterior Beta distribution.

    Returns scipy.stats.beta with updated parameters.
    """
    if prior is None:
        prior = BetaPrior.uniform()
    return stats.beta(prior.alpha + conversions, prior.beta + (n - conversions))


def sequential_analysis(data, prior_a=None, prior_b=None, rope=0.01,
                        n_samples=100_000, seed=None):
    """Run Bayesian sequential analysis over daily data.

    Returns DataFrame with metrics (prob_b_better, expected_loss, ROPE) per day.
    Expects columns: day, variant ('A'/'B'), n, conversions.
    Cumulative columns (n_cum, conversions_cum) are computed if missing.
    """
    rng = np.random.default_rng(seed)

    if "n_cum" not in data.columns:
        data = data.copy()
        data["n_cum"] = data.groupby("variant")["n"].cumsum()
        data["conversions_cum"] = data.groupby("variant")["conversions"].cumsum()

    results = []
    for day in sorted(data["day"].unique()):
        day_data = data.query("day == @day")
        n_a = day_data.query("variant == 'A'")["n_cum"].values[0]
        n_b = day_data.query("variant == 'B'")["n_cum"].values[0]
        c_a = day_data.query("variant == 'A'")["conversions_cum"].values[0]
        c_b = day_data.query("variant == 'B'")["conversions_cum"].values[0]

        post_a = beta_posterior(n_a, c_a, prior_a)
        post_b = beta_posterior(n_b, c_b, prior_b)
        s_a = post_a.rvs(n_samples, random_state=rng)
        s_b = post_b.rvs(n_samples, random_state=rng)

        loss_a, loss_b = expected_loss(s_a, s_b)
        results.append({
            "day": day,
            "prob_b_better": prob_b_better(s_a, s_b),
            "expected_loss_a": loss_a,
            "expected_loss_b": loss_b,
            "prob_above_rope": prob_above_rope(s_a, s_b, rope),
        })

    return pd.DataFrame(results)
