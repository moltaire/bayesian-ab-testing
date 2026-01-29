"""Generate the metrics figure for README."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.bayesian_ab as bab

# Simulate A/B test data
data = bab.simulation.simulate_ab_test(
    n_days=10,
    daily_n=50,
    p_a=0.10,
    p_b=0.12,
    seed=1763,  # Year Bayes' theorem was published
)

# Run Bayesian sequential analysis
rope = 0.005
results = bab.bayesian.sequential_analysis(data, rope=rope, seed=1763)

# Plot metrics over time
fig, axs = bab.bayesian.plot_sequential_metrics(results, rope=rope)
fig.savefig("figures/metrics.png", dpi=150, bbox_inches="tight")
print("Saved figures/metrics.png")
