"""Bayesian metrics from posterior samples.

Framework-agnostic: works with samples from scipy, PyMC, or any other source.
"""

import numpy as np


def prob_b_better(samples_a, samples_b):
    """P(B > A) from posterior samples."""
    return float(np.mean(np.asarray(samples_b) > np.asarray(samples_a)))


def expected_loss(samples_a, samples_b):
    """Expected loss (regret) for choosing each variant.

    Returns (loss_a, loss_b) where loss_b = E[max(p_A - p_B, 0)].
    The variant with lower expected loss is preferred.
    See Stucchio (VWO whitepaper) for details.
    """
    a, b = np.asarray(samples_a), np.asarray(samples_b)
    return float(np.mean(np.maximum(b - a, 0))), float(np.mean(np.maximum(a - b, 0)))


def prob_above_rope(samples_a, samples_b, rope=0.01):
    """P(p_B - p_A > rope) — probability difference exceeds ROPE threshold."""
    return float(np.mean((np.asarray(samples_b) - np.asarray(samples_a)) > rope))


def hdi(samples, prob=0.94):
    """Highest Density Interval — narrowest interval containing `prob` mass."""
    samples = np.sort(np.asarray(samples))
    n = len(samples)
    k = int(np.ceil(prob * n))
    widths = samples[k:] - samples[: n - k]
    i = np.argmin(widths)
    return float(samples[i]), float(samples[i + k])
