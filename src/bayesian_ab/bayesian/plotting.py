"""Plotting for Bayesian A/B testing results."""

import matplotlib.pyplot as plt

# Import myplotlib to set default styling
import myplotlib as my
from matplotlib.ticker import MaxNLocator


def plot_sequential_metrics(results, thresholds=None, figsize=None, rope=None):
    """Plot HDI, P(B>A), ROPE probability, and expected loss over time.

    Args:
        results: DataFrame from sequential_analysis with columns:
            day, prob_b_better, prob_above_rope, expected_loss_b,
            mean_a, mean_b, hdi_a_lo, hdi_a_hi, hdi_b_lo, hdi_b_hi
        thresholds: dict with keys 'prob' (default 0.95) and 'loss' (default 0.001)
        figsize: Figure size tuple in inches (default 16x3.75 cm)
        rope: ROPE value for y-axis label (optional)

    Returns:
        fig, axs: Matplotlib figure and axes (1x4 grid)
    """
    if thresholds is None:
        thresholds = {"prob": 0.95, "loss": 0.001}
    if figsize is None:
        figsize = my.utilities.cm2inch(16, 3.75)

    colors = plt.cm.plasma([0.1, 0.8])

    fig, axs = plt.subplots(1, 4, figsize=figsize, sharex=True)

    days = results["day"].values

    # Panel 0: HDI for both variants
    ax = axs[0]
    for i, variant in enumerate(["A", "B"]):
        mean = results[f"mean_{variant.lower()}"].values
        lo = results[f"hdi_{variant.lower()}_lo"].values
        hi = results[f"hdi_{variant.lower()}_hi"].values
        ax.plot(days, mean, "-", color=colors[i], lw=1, zorder=1)
        ax.fill_between(days, lo, hi, color=colors[i], alpha=0.3, zorder=0)
        ax.scatter(days, mean, color=colors[i], edgecolor="k", zorder=2, label=variant)
    ax.set_title("Conversion rate")
    ax.set_ylabel("Rate (94% HDI)")
    ax.set_xlabel("Day")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.25, lw=0.25, fillstyle="top")

    # Panel 1: P(B > A)
    ax = axs[1]
    ax.axhline(
        thresholds["prob"],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"{thresholds['prob']:.0%}",
    )
    ax.plot(days, results["prob_b_better"], "-", color="k", lw=1, zorder=-1)
    ax.scatter(
        days,
        results["prob_b_better"],
        color=colors[
            (results["prob_b_better"].values > thresholds["prob"]).astype(int)
        ],
        edgecolor="k",
        clip_on=False,
        zorder=2,
    )
    ax.set_title("P(B > A)")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Day")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, lw=0.25, fillstyle="top")

    # Panel 2: ROPE
    ax = axs[2]
    ax.axhline(
        thresholds["prob"],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"{thresholds['prob']:.0%}",
    )
    ax.plot(days, results["prob_above_rope"], "-", color="k", lw=1, zorder=-1)
    ax.scatter(
        days,
        results["prob_above_rope"],
        color=colors[
            (results["prob_above_rope"].values > thresholds["prob"]).astype(int)
        ],
        edgecolor="k",
        clip_on=False,
        zorder=2,
    )
    title = "P(B > A + ROPE)" if rope is None else f"P(B > A + {rope:.1%})"
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Day")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, lw=0.25, fillstyle="top")

    # Panel 3: Expected loss
    ax = axs[3]
    loss_pct = results["expected_loss_b"].values * 100
    threshold_pct = thresholds["loss"] * 100
    ax.plot(days, loss_pct, "-", color="k", lw=1, zorder=-1)
    ax.scatter(
        days,
        loss_pct,
        color=colors[(loss_pct < threshold_pct).astype(int)],
        edgecolor="k",
        clip_on=False,
        zorder=2,
    )
    ax.axhline(
        threshold_pct,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"{threshold_pct}%",
    )
    ax.set_title("Expected loss")
    ax.set_ylabel("Loss (%)")
    ax.set_xlabel("Day")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, lw=0.25, fillstyle="top")

    # Integer x-ticks for all axes
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(w_pad=1.5)
    return fig, axs
