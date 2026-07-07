# Changelog

All notable changes to `carbon-ets` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned for v0.2
- Real TNAC nowcast with validated accuracy against Commission publications, built from actual EUTL per-year free-allocation and auction data
- Monthly emissions nowcast from ENTSO-E power generation mix + Eurostat industrial output
- UK ETS (UKA) and California CCA data adapters using the same architecture

## [0.1.0] — Unreleased

### Added
- EEX primary-auction clearing price fetcher covering 2012 to present
- Eurostat industrial production fetcher via SDMX bulk endpoint with legacy JSON-stat fallback
- TNAC reference series 2016-2025 with primary-source citation per year
- MSR regime-context helper (`get_regime_context()`) — returns policy regime for any TNAC value
- Feature engineering utilities: log returns, z-scores, IP YoY, realized volatility
- V1 causal-chain backtest (long-only, IP + Stoxx momentum)
- Markov-switching regression wrapper (descriptive, not predictive)
- Standard plot helpers for equity curves and TNAC history
- Full test suite for TNAC reference data (10 known values as hard assertions)
- MIT license

### Removed before v0.1 ship (rather than shipping broken)
- **Monthly TNAC nowcast** — internal validation showed 20% median error against Commission-published TNAC values with systematic bias. Planned to return in v0.2 with corrected accounting.
- **Compliance-buy timing signal** — empirically failed to persistently outperform naive uniform-monthly baseline.
- **Sector transmission regressions** — no Rotterdam-cluster equity ticker cleared 95% significance for EUA sensitivity after controlling for equity-market factors.

### Design principle
Ship less rather than ship broken. If a feature does not clear internal validation, it stays out until it does.
