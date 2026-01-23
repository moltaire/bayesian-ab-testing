"""Bayesian A/B Testing Package."""

from .frequentist import (
    PowerAnalysisResult,
    ProportionTestResult,
    power_analysis,
    proportion_test,
)
from .simulation import simulate_ab_test

__all__ = [
    # Frequentist analysis
    "PowerAnalysisResult",
    "ProportionTestResult",
    "power_analysis",
    "proportion_test",
    # Simulation
    "simulate_ab_test",
]
