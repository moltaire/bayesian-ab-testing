"""Frequentist A/B testing utilities.

This module provides traditional frequentist methods for A/B testing,
including power analysis and hypothesis testing for proportions.
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import (
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)


@dataclass
class PowerAnalysisResult:
    """Result of power analysis.

    Attributes
    ----------
    n_per_group : int
        Required sample size per group
    total_n : int
        Total required sample size (both groups)
    effect_size : float
        Cohen's h effect size
    power : float
        Statistical power (1 - β)
    alpha : float
        Significance level (α)
    p_control : float
        Expected control conversion rate
    p_treatment : float
        Expected treatment conversion rate
    relative_lift : float
        Relative lift (treatment/control - 1)
    """

    n_per_group: int
    total_n: int
    effect_size: float
    power: float
    alpha: float
    p_control: float
    p_treatment: float
    relative_lift: float

    def __repr__(self):
        return "\n".join(
            [
                f"PowerAnalysisResult(n_per_group={self.n_per_group}, ",
                f"total_n={self.total_n}, effect_size={self.effect_size:.4f}, ",
                f"power={self.power:.2f}, alpha={self.alpha:.2f}, ",
                f"p_control={self.p_control:.4f}, p_treatment={self.p_treatment:.4f}, ",
                f"relative_lift={self.relative_lift:.4f})",
            ]
        )


@dataclass
class ProportionTestResult:
    """Result of two-proportion z-test.

    Attributes
    ----------
    z_statistic : float
        Z-test statistic
    p_value : float
        P-value for the test
    significant : bool
        Whether result is statistically significant
    alpha : float
        Significance level used
    rate_a : float
        Observed conversion rate for variant A
    rate_b : float
        Observed conversion rate for variant B
    absolute_lift : float
        Absolute difference (B - A)
    relative_lift : float
        Relative lift (B/A - 1)
    ci_a : Tuple[float, float]
        Confidence interval for variant A
    ci_b : Tuple[float, float]
        Confidence interval for variant B
    n_a : int
        Sample size for variant A
    n_b : int
        Sample size for variant B
    alternative : str
        Alternative hypothesis ('two-sided', 'larger', 'smaller')
    """

    z_statistic: float
    p_value: float
    significant: bool
    alpha: float
    rate_a: float
    rate_b: float
    absolute_lift: float
    relative_lift: float
    ci_a: Tuple[float, float]
    ci_b: Tuple[float, float]
    n_a: int
    n_b: int
    alternative: str


def power_analysis(
    p_control: float,
    p_treatment: float = None,
    relative_lift: float = None,
    power: float = 0.80,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> PowerAnalysisResult:
    """
    Calculate required sample size for two-proportion z-test.

    Determines how many samples are needed per group to detect a difference
    between two proportions with specified power and significance level.

    Parameters
    ----------
    p_control : float
        Expected conversion rate for control group (e.g., 0.05 for 5%)
    p_treatment : float
        Expected conversion rate for treatment group (e.g., 0.06 for 6%)
    power : float, default=0.80
        Statistical power (1 - β), probability of detecting true effect
        Common values: 0.80 (80%), 0.90 (90%)
    alpha : float, default=0.05
        Significance level (α), probability of Type I error
        Common values: 0.05 (5%), 0.01 (1%)
    ratio : float, default=1.0
        Ratio of treatment to control sample size
        1.0 = equal allocation, 2.0 = 2x more in treatment
    alternative : str, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': Treatment ≠ Control
        - 'larger': Treatment > Control
        - 'smaller': Treatment < Control

    Returns
    -------
    PowerAnalysisResult
    """

    # calculate p_treatment or relative_lift
    if p_treatment is None:
        p_treatment = p_control * (1 + relative_lift)
    else:
        relative_lift = (p_treatment / p_control) - 1

    # Calculate Cohen's h effect size
    effect_size = proportion_effectsize(p_treatment, p_control)

    # Calculate required sample size
    n_per_group = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )

    # Round up to ensure sufficient power
    n_per_group = int(np.ceil(n_per_group))
    total_n = int(np.ceil(n_per_group * (1 + ratio)))

    return PowerAnalysisResult(
        n_per_group=n_per_group,
        total_n=total_n,
        effect_size=effect_size,
        power=power,
        alpha=alpha,
        p_control=p_control,
        p_treatment=p_treatment,
        relative_lift=relative_lift,
    )


def proportion_test(
    data: pd.DataFrame,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
    ci_method: str = "normal",
) -> ProportionTestResult:
    """
    Run two-proportion z-test on A/B test data.

    Tests whether the conversion rates of two variants are significantly different
    using a z-test for proportions.

    Parameters
    ----------
    data : pd.DataFrame
        A/B test data with columns: variant, conversions, n
        variant should have values 'A' (control) and 'B' (treatment)
    alpha : float, default=0.05
        Significance level for hypothesis test and confidence intervals
    alternative : str, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': Treatment ≠ Control (most common)
        - 'larger': Treatment > Control (one-sided)
        - 'smaller': Treatment < Control (one-sided)
    ci_method : str, default='normal'
        Method for confidence interval calculation
        Options: 'normal', 'wilson', 'agresti_coull', 'beta', 'jeffreys'

    Returns
    -------
    ProportionTestResult
    """
    # Aggregate data by variant
    agg_data = data.groupby("variant")[["conversions", "n"]].sum()

    # Extract values
    conversions_a = agg_data.loc["A", "conversions"]
    n_a = agg_data.loc["A", "n"]
    rate_a = conversions_a / n_a

    conversions_b = agg_data.loc["B", "conversions"]
    n_b = agg_data.loc["B", "n"]
    rate_b = conversions_b / n_b

    # Run z-test (order: B, A for testing if B > A)
    z_stat, p_value = proportions_ztest(
        count=[conversions_b, conversions_a],
        nobs=[n_b, n_a],
        alternative=alternative,
    )

    # Calculate confidence intervals
    ci_a = proportion_confint(
        count=conversions_a, nobs=n_a, alpha=alpha, method=ci_method
    )
    ci_b = proportion_confint(
        count=conversions_b, nobs=n_b, alpha=alpha, method=ci_method
    )

    # Calculate lifts
    absolute_lift = rate_b - rate_a
    relative_lift = (rate_b / rate_a) - 1 if rate_a > 0 else np.inf

    # Determine significance
    significant = p_value < alpha

    return ProportionTestResult(
        z_statistic=z_stat,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        rate_a=rate_a,
        rate_b=rate_b,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        ci_a=ci_a,
        ci_b=ci_b,
        n_a=int(n_a),
        n_b=int(n_b),
        alternative=alternative,
    )
