"""Minimal A/B test data simulation."""

import numpy as np
import pandas as pd


def simulate_ab_test(
    n_days: int = 14,
    daily_n: int = 100,
    p_a: float = 0.10,
    p_b: float = 0.12,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulate A/B test data.
    
    Parameters
    ----------
    n_days : int
        Number of days
    daily_n : int
        Sample size per variant per day
    p_a : float
        True conversion rate for variant A
    p_b : float
        True conversion rate for variant B
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Columns: day, variant, conversions, n
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    for day in range(1, n_days + 1):
        for variant, p in [('A', p_a), ('B', p_b)]:
            conversions = np.random.binomial(daily_n, p)
            data.append({
                'day': day,
                'variant': variant,
                'conversions': conversions,
                'n': daily_n,
            })
    
    return pd.DataFrame(data)
