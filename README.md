# carbon-ets

[![CI](https://github.com/causalsystems/carbon-ets/actions/workflows/ci.yml/badge.svg)](https://github.com/causalsystems/carbon-ets/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/carbon-ets/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/causalsystems/carbon-ets/blob/main/examples/quickstart.ipynb)

Open, reproducible pipeline for **EU Emissions Trading System** (EU ETS) analysis from free public data.

Everything runs on published, non-paywalled sources: EEX primary-auction clearing prices, Eurostat industrial production, European Commission MSR Communications, and Open-Meteo weather. **Free replacement for ~€20-50k/year of paid EU ETS data infrastructure.**

Companion code to Causal Systems research report [Factories to carbon](https://causalsystems.co/research/factories-to-carbon).

**Try it in 30 seconds:** click the "Open in Colab" badge above.

## What's in the box

- **Historical EUA prices 2012-present** — daily primary auction clearing prices scraped from EEX archive
- **Eurostat industrial production** — via SDMX bulk endpoint with JSON-stat fallback
- **TNAC reference series 2016-2025** — verified values from every Commission Communication PDF, with primary-source citations
- **MSR mechanism helpers** — thresholds, invalidation history, regime-context lookup for any TNAC value
- **Feature engineering** — log returns, z-scores, momentum, realised volatility, IP YoY
- **V1 baseline backtest** — long-only causal-chain strategy from the CS/RES/05 report
- **Markov-switching regression** — two-regime model for EUA return dynamics (descriptive, not predictive)
- **Standard plots** — equity curves, TNAC history with MSR thresholds, regime diagnostics

### What's NOT in v0.1

Being explicit about scope. We removed features that failed our internal validation rather than shipping them broken:

- **Monthly TNAC nowcast** — validated at 20%+ median error against Commission-published values, well outside a defensible confidence band. Planned for v0.2 once we wire in real EUTL per-year data.
- **Compliance-buy timing signal** — tested empirically and does not persistently outperform naive strategies.
- **Sector transmission model** — regressions of Rotterdam-cluster equities on EUA did not clear statistical significance.

Choosing to ship less rather than ship broken is the design principle here.

## Install

```bash
pip install carbon-ets              # core (no yfinance)
pip install carbon-ets[yfinance]    # with Yahoo Finance for equity/gas proxies
pip install carbon-ets[all]         # everything including Jupyter
```

Or from source:

```bash
git clone https://github.com/causalsystems/carbon-ets
cd carbon-ets
pip install -e .[all]
```

## Quickstart

```python
from carbon_ets.data import build_full_panel
from carbon_ets.features import engineer
from carbon_ets.models import backtest_v1
from carbon_ets.tnac import nowcast_monthly

panel = build_full_panel(start="2012-01-01")
feats = engineer(panel)
stats, equity = backtest_v1(feats)
print(f"Sharpe = {stats['sharpe']:.2f}, CAGR = {stats['cagr']:+.1%}")

# Current TNAC nowcast — updated as new data arrives
tnac = nowcast_monthly()
print(f"Estimated TNAC ({tnac.as_of.date()}): {tnac.tnac_estimate:,.0f} allowances")
```

See `examples/quickstart.py` for the full end-to-end run.

## What's honest about this

This toolkit does not claim to be a trading system. The backtest returns you see are:

- **In-sample and window-dependent** — Sharpe 0.8-1.0 on 2015-2024 EUA settlement data, but not tested out-of-sample
- **Sensitive to feature specification** — different regime interpretations arise from different feature sets
- **Not corrected for realistic transaction costs, slippage, or execution frictions**

The toolkit is designed for **research and policy analysis**, not live trading. If you want to trade EUA, you need paid data feeds, a proper execution stack, and independent validation.

The core empirical finding — that the demand-side chain explains ~5% of monthly variance on average and up to 63% in specific windows — is reproducible from the pipeline and is the piece worth citing.

## Architecture

```
carbon_ets/
├── data.py         — EEX auction fetcher, Eurostat SDMX, panel builder
├── features.py     — engineered features (returns, z-scores, momentum, vol)
├── models.py       — backtest_v1, Markov-switching MS-2 wrapper
├── tnac.py         — hardcoded TNAC series + monthly nowcast + validation
├── plots.py        — equity curves, TNAC diagnostics
└── __init__.py

examples/
└── quickstart.py   — end-to-end reproducibility

pyproject.toml      — package config, dependencies
```

## Contributing

Issues and pull requests welcome, especially:

- **Data gap fills** — missing pre-2013 auction data, alternative EUA proxies (KEUA, CFI), or additional emissions data sources
- **Feature ideas** — CBAM import volumes, ETS2 preparation data, sector-specific compliance proxies
- **Method extensions** — regime-switching alternatives, structural VAR, natural experiments around MSR events
- **Cross-market ports** — California CCA (CARB), UK ETS, RGGI applications of the same architecture

## Citing

If you use this in published work, please cite:

> Causal Systems (2026). *Factories to carbon: a demand-side model of EU allowance prices.* CS/RES/05.
> https://causalsystems.co/research/factories-to-carbon

## License

MIT. See LICENSE.

## Contact

hello@causalsystems.co
