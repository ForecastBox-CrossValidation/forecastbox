---
title: "Roadmap"
description: "forecastbox development roadmap — planned features, priorities, and release schedule."
---

# Roadmap

forecastbox aims to be the most comprehensive Python forecasting engine for econometric applications. This roadmap outlines planned features, organized by version.

!!! info "Current Status"
    forecastbox v0.1.0 is the current release with auto-forecast models (AutoARIMA, AutoETS, AutoVAR), 7 combination methods, statistical evaluation tests (DM, MCS, GW), nowcasting (DFM, Bridge, MIDAS), pipelines, and CLI.

---

## v0.2.0 — Deep Learning and Prediction Intervals

**Timeline**: Q3 2026

### Deep Learning Models

- **N-BEATS** — Neural Basis Expansion Analysis for interpretable time series forecasting (Oreshkin et al., 2020)
- **Temporal Fusion Transformer (TFT)** — Multi-horizon attention-based forecasting with variable selection (Lim et al., 2021)
- **DeepAR** — Autoregressive RNN with probabilistic output (Salinas et al., 2020)
- **PatchTST** — Channel-independent Transformer with patching (Nie et al., 2023)
- **Integration with ModelZoo** — Unified interface for classical and deep learning models

### Conformal Prediction Intervals

- **Split conformal prediction** — Distribution-free prediction intervals with finite-sample coverage guarantees
- **Conformalized quantile regression** — Adaptive intervals via quantile regression
- **EnbPI** — Ensemble batch prediction intervals for time series
- **Coverage diagnostics** — Tools to evaluate empirical coverage and interval width

---

## v0.3.0 — Hierarchical Forecasting

**Timeline**: Q4 2026

### Hierarchical Reconciliation

- **Bottom-up aggregation** — Forecast at the lowest level and aggregate
- **Top-down disaggregation** — Forecast at the top level and distribute proportionally
- **Middle-out** — Forecast at an intermediate level with both aggregation and disaggregation
- **Optimal reconciliation (MinT)** — Wickramasuriya, Athanasopoulos & Hyndman (2019) minimum trace reconciliation
- **ERM reconciliation** — Empirical Risk Minimization for reconciliation weights

### Grouped Time Series

- **Cross-temporal reconciliation** — Reconcile across both hierarchical and temporal dimensions
- **Temporal aggregation** — Forecast at multiple temporal granularities simultaneously
- **Coherent probabilistic forecasts** — Reconciled prediction intervals

---

## v0.4.0 — Online Learning and Streaming

**Timeline**: Q1 2027

### Online Forecasting

- **Online model updating** — Incremental parameter updates as new data arrives
- **Adaptive combination weights** — Real-time weight adjustment via online gradient descent
- **Regime detection** — Automatic detection of structural breaks and regime changes
- **Forgetting factor models** — Exponentially weighted estimation for non-stationary environments

### Streaming Nowcast

- **Real-time data ingestion** — Streaming interface for high-frequency data feeds
- **Incremental DFM** — Online factor extraction without full re-estimation
- **Event-driven updates** — Trigger nowcast revision on individual data releases
- **Latency monitoring** — Track and optimize nowcast update latency

---

## v0.5.0 — Explainability and Reporting

**Timeline**: Q2 2027

### Forecast Explainability

- **SHAP for forecasts** — Shapley value decomposition of forecast contributions by variable and lag
- **Feature importance** — Permutation-based and model-specific importance measures
- **Counterfactual analysis** — "What-if" analysis showing forecast sensitivity to input changes
- **Decomposition plots** — Visual breakdown of forecast drivers over time

### Automated Reporting

- **Report templates** — Pre-built templates for periodic forecast reports (weekly, monthly, quarterly)
- **Narrative generation** — Automatic text summaries of forecast changes, key drivers, and risk flags
- **PDF/HTML export** — Publication-ready reports with charts, tables, and commentary
- **Scheduled reports** — Cron-based automatic report generation and delivery

---

## v1.0.0 — Production Ready

**Timeline**: Q4 2027

### Stable API

- **API freeze** — No breaking changes after v1.0.0 within the 1.x series
- **Comprehensive deprecation policy** — Minimum 2 minor versions before removal
- **Full backward compatibility guarantees** — Semantic versioning strictly enforced

### Full Test Coverage

- **95%+ line coverage** — All public API paths tested
- **Validation against R forecast ecosystem** — Coefficients, p-values, and intervals match within tolerance
- **Validation against Stata** — Cross-platform numerical agreement
- **Performance regression tests** — Automated benchmarks in CI

### Production Features

- **Docker images** — Pre-built containers for deployment
- **REST API** — FastAPI-based forecast serving endpoint
- **Monitoring dashboard** — Grafana-compatible metrics for forecast health
- **Cloud integration** — AWS Lambda, GCP Cloud Functions, Azure Functions deployment guides

---

## Beyond v1.0.0

Exploratory features for future major releases.

### GPU Acceleration

- **JAX backend** — GPU-accelerated matrix operations for large-scale forecasting
- **Batch model fitting** — Parallel estimation across multiple series
- **Automatic backend selection** — Transparent CPU/GPU switching

### Distributed Computing

- **Dask integration** — Out-of-core processing for very large datasets
- **Ray support** — Distributed cross-validation and ensemble training
- **Cloud-native workflows** — Scalable pipelines on managed infrastructure

### Interactive Dashboard

- **Streamlit app** — Point-and-click forecasting interface
- **Scenario explorer** — Interactive scenario analysis with real-time visualization
- **Model comparison dashboard** — Side-by-side model evaluation tool

---

## Documentation Roadmap

### Completed

- [x] Phase 1: Documentation infrastructure and MkDocs setup
- [x] Phase 2: Getting Started and User Guide pages
- [x] Phase 3: Theory, Diagnostics, and Visualization guides
- [x] Phase 4: Tutorials (fundamentals to complete workflows)
- [x] Phase 5: API Reference, FAQ, and Benchmarks
- [x] Phase 6: Contributing, Changelog, Roadmap, and Final Review

### Planned

- [ ] Community-contributed tutorials and case studies
- [ ] Video walkthroughs for key workflows
- [ ] Translations: Spanish, Portuguese

---

## How to Influence the Roadmap

### Feature Requests

Open a [GitHub Issue](https://github.com/nodesecon/forecastbox/issues) with the `[Feature]` label. Include:

1. **Use case**: What problem does it solve?
2. **Description**: What should the feature do?
3. **References**: Academic papers, existing implementations (R, Python, Stata)
4. **Priority justification**: Why is this important for forecasting workflows?

### Community Voting

React with a thumbs-up on existing feature request issues to signal demand. Features with more community interest are prioritized higher.

### Contributions

The fastest way to get a feature is to implement it yourself! See the [Contributing Guide](contributing.md) for templates and process. We provide mentorship for first-time contributors.

---

## Release Schedule

### Versioning

forecastbox follows [Semantic Versioning](https://semver.org/):

| Version | Cadence | Content |
|---|---|---|
| **Major** (X.0.0) | As needed | Breaking API changes |
| **Minor** (0.X.0) | Every 2-3 months | New features, backward compatible |
| **Patch** (0.0.X) | As needed | Bug fixes, documentation updates |

### Release Process

1. Feature freeze 1 week before release
2. Release candidate published for testing
3. Final release after validation
4. Changelog and migration notes published

### Support Policy

- **Current minor version**: Full support (bug fixes, security patches, new features)
- **Previous minor version**: Bug fixes only for 3 months after new minor release
- **Older versions**: Community support only

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute code and documentation
- [Changelog](changelog.md) — Version history
- [Code of Conduct](code-of-conduct.md) — Community standards
- [API Reference](../api/index.md) — Full API documentation
