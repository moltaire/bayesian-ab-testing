"""Bayesian A/B testing: conjugate models, metrics, and PyMC model builders."""

from . import conjugate, metrics, plotting
from .conjugate import BetaPrior, beta_posterior, sequential_analysis
from .metrics import expected_loss, hdi, prob_above_rope, prob_b_better
from .plotting import plot_sequential_metrics

try:
    from . import models
except ImportError:
    pass
