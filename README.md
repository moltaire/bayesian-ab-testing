<p align="left">
  <img src="figures/header.png" alt="Bayesian A/B Testing" width="600">
</p>

# Bayesian A/B Testing

Flexible metrics for better decisions. A companion repository to the blog post.

**[Read the blog post →](https://moltaire.github.io/bayesian-ab-testing)**

## Setup

```bash
uv sync
```

## Project structure

```
.
├── index.ipynb              # Main notebook / blog post
├── src/bayesian_ab/         # Python package
│   ├── simulation.py        #   A/B test data simulation
│   ├── frequentist.py       #   Power analysis & z-tests
│   └── bayesian/            #   Bayesian analysis
│       ├── metrics.py       #     P(B>A), expected loss, ROPE, HDI
│       ├── conjugate.py     #     Beta-Binomial model
│       └── models.py        #     PyMC model builders
└── tests/                   # pytest tests
```
