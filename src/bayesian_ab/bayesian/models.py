"""PyMC model builders for A/B testing. Requires PyMC."""

try:
    import pymc as pm
except ImportError:
    raise ImportError("PyMC is required for bayesian.models. Install with: pip install pymc")

from .conjugate import BetaPrior


def build_binomial_model(n_a, conv_a, n_b, conv_b, prior_a=None, prior_b=None):
    """Build a PyMC Beta-Binomial model for a two-variant A/B test.

    Returns a pm.Model ready for sampling.
    """
    if prior_a is None:
        prior_a = BetaPrior.uniform()
    if prior_b is None:
        prior_b = BetaPrior.uniform()

    with pm.Model() as model:
        p_a = pm.Beta("p_control", alpha=prior_a.alpha, beta=prior_a.beta)
        p_b = pm.Beta("p_treatment", alpha=prior_b.alpha, beta=prior_b.beta)
        pm.Binomial("obs_control", n=n_a, p=p_a, observed=conv_a)
        pm.Binomial("obs_treatment", n=n_b, p=p_b, observed=conv_b)
        pm.Deterministic("diff", p_b - p_a)

    return model
