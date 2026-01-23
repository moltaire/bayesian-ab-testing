# Bayesian Sequential A/B Testing

Demonstrate how Bayesian sequential analysis enables faster A/B testing decisions.

**[View the blog post →](https://moltaire.github.io/bayesian-ab-testing)**

## The Idea

**Traditional testing:** Pre-specify sample size, wait, then analyze
**Bayesian sequential:** Check anytime, stop when confident

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync --extra dev

# Run Jupyter notebook
uv run jupyter lab index.ipynb
```

### Build and Preview Site Locally

```bash
# Install Quarto: https://quarto.org/docs/get-started/

# Render the site
quarto preview

# Or just render without preview
quarto render
```

## Structure

```
index.ipynb              # Main analysis notebook (becomes the blog post)
src/bayesian_ab/
  simulation.py          # Generate test data
  plots.py               # Visualization utilities
_quarto.yml              # Quarto configuration
.github/workflows/       # Auto-publish to GitHub Pages
```

## To-Do

- [x] src: Implement frequentist test
- [x] implement GitHub action to automatically render this
- [x] nb: Run frequentist test
- [ ] src: Implement Baysian testing
- [ ] nb: Run Baysian test
- [ ] nb: Visualize Bayesian test result
- [ ] nb: Write wrap-up and outlook