---
title: "Changelog"
description: "forecastbox version history — all releases with key changes, migration notes, and breaking changes."
---

# Changelog

All notable changes to forecastbox are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and forecastbox adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Sections**: Added, Changed, Fixed, Deprecated, Removed, Security, Performance.

---

## [Unreleased]

_No unreleased changes._

---

## [0.1.0] — 2026-04-16

### Summary

**Initial Release — Forecasting Engine for the NodesEcon Ecosystem**

forecastbox v0.1.0 is the first public release, providing a complete toolkit for automatic forecasting, forecast combination, statistical evaluation, nowcasting, and production pipelines.

### Added

#### Auto-Forecast Models

- **AutoARIMA** — Automatic ARIMA model selection with information criteria (AIC, BIC, HQIC); seasonal and non-seasonal; integration with chronobox
- **AutoETS** — Automatic exponential smoothing with error-trend-seasonal decomposition; additive, multiplicative, and damped variants
- **AutoVAR** — Automatic Vector Autoregression with lag selection and stability checks
- **AutoSelect** — Meta-selector that runs multiple models and chooses the best by cross-validation
- **ModelZoo** — Extensible registry for forecasting models with unified interface

#### Forecast Combination (7 Methods)

- **Simple Average** — Equal-weight combination (baseline)
- **Fixed Weights** — User-specified weight combination
- **OLS Combination** — Granger-Ramanathan (1984) regression-based combination
- **Stacking** — Meta-learner combination with cross-validated base forecasts
- **Bayesian Model Averaging (BMA)** — Posterior probability weights via marginal likelihoods
- **Time-Varying Weights** — Exponentially decaying weights adapting to recent performance
- **Optimal Combination** — Bates-Granger (1969) variance-minimizing weights

#### Statistical Evaluation Tests

- **Diebold-Mariano (DM) Test** — Pairwise predictive accuracy comparison with HAC standard errors
- **Model Confidence Set (MCS)** — Hansen, Lunde & Nason (2011) elimination procedure for identifying superior models
- **Giacomini-White (GW) Test** — Conditional predictive ability test for nested and non-nested models
- **Mincer-Zarnowitz Regression** — Forecast unbiasedness and efficiency test
- **Forecast Encompassing Test** — Harvey, Leybourne & Newbold (1998) test for redundant forecasts
- **Cross-Validation** — Expanding window, rolling window, and combinatorial purged cross-validation

#### Nowcasting

- **Dynamic Factor Model (DFM)** — Kalman filter-based extraction of common factors from mixed-frequency data; integration with kalmanbox
- **Bridge Equations** — High-frequency indicator aggregation for GDP nowcasting
- **MIDAS** — Mixed Data Sampling regression with Almon, Beta, and exponential lag polynomials
- **News Decomposition** — Banbura & Modugno (2014) decomposition of nowcast revisions into news from individual data releases
- **Vintage Management** — Real-time data vintage tracking and pseudo real-time evaluation

#### Scenarios and Conditional Forecasting

- **Conditional Forecast** — Constrained VAR forecasting with hard and soft constraints
- **Scenario Builder** — Programmatic scenario definition with parameter sweeps
- **Monte Carlo Simulation** — Stochastic scenario generation with correlated shocks
- **Fan Charts** — Probability-weighted forecast visualization
- **Stress Testing** — Adverse scenario construction with historical and hypothetical shocks

#### Pipeline and Experiment Tracking

- **ForecastPipeline** — End-to-end pipeline: data loading, model fitting, combination, evaluation, reporting
- **Pipeline Monitor** — Real-time monitoring of forecast accuracy with alert thresholds
- **Experiment Tracking** — MLflow-style experiment logging for forecast experiments

#### CLI

- **`forecastbox forecast`** — Run forecasts from the command line
- **`forecastbox combine`** — Combine forecasts with specified methods
- **`forecastbox evaluate`** — Evaluate forecast accuracy
- **`forecastbox pipeline`** — Run full pipeline from YAML configuration
- **`forecastbox nowcast`** — Generate nowcasts from mixed-frequency data

#### Documentation

- Complete documentation with MkDocs Material
- Getting Started guide with installation, quickstart, and core concepts
- User Guide covering all modules with examples
- Theory section with mathematical foundations
- Diagnostics guide with interpretation guidelines
- Tutorials from fundamentals to complete workflows
- Visualization gallery
- Full API Reference
- FAQ and Benchmarks

---

## Versioning Policy

forecastbox uses [Semantic Versioning](https://semver.org/):

| Component | When incremented |
|---|---|
| **Major** (X.0.0) | Incompatible API changes |
| **Minor** (0.X.0) | New features, backward compatible |
| **Patch** (0.0.X) | Bug fixes, backward compatible |

---

## Template for Future Releases

```markdown
## [X.Y.Z] — YYYY-MM-DD

### Summary

**Brief one-line summary of the release.**

### Added
- New feature 1
- New feature 2

### Changed
- Changed behavior 1

### Fixed
- Bug fix 1

### Deprecated
- Deprecated feature 1

### Removed
- Removed feature 1

### Security
- Security fix 1

### Performance
- Performance improvement 1
```

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
