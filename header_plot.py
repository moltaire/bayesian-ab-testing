import matplotlib.pyplot as plt
import myplotlib as my
import numpy as np
from scipy.stats import beta

if __name__ == "__main__":

    np.random.seed(1234)

    true_rate = 0.4
    n_trials = 10
    n_updates = 5
    successes = np.cumsum(np.random.binomial(n_trials, true_rate, n_updates))
    trials = np.cumsum([n_trials] * n_updates)

    # Figure: exact size, no frame
    fig = plt.figure(figsize=my.utilities.cm2inch(8, 1), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure

    x = np.linspace(0, 1, 1000)

    for s, t in zip(successes, trials):
        p_a = beta(s + 1, t - s + 1)

        ax.plot(x, p_a.pdf(x), lw=0.75, color=plt.cm.plasma(t / trials[-1]), alpha=0.8)

    # Remove everything visual except data
    ax.set_axis_off()

    # Crucial: make data touch image edges
    ax.set_xlim(x.min(), x.max())

    # Save
    fig.savefig(
        "figures/header.png",
        dpi=300,
        transparent=True,
        bbox_inches=None,  # IMPORTANT: do NOT use "tight" here
        pad_inches=0,
    )

    plt.close(fig)
