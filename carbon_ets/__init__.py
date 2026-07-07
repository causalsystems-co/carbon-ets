"""
carbon-ets — open, reproducible pipeline for EU Emissions Trading System analysis.

Everything in this package can be reproduced from free public data:
    - EEX primary-auction clearing prices (2012+)
    - Eurostat industrial production (via SDMX API)
    - European Commission MSR / TNAC Communications
    - Open-Meteo weather data
    - Yahoo Finance equity + fuel proxies

Quick start:

    from carbon_ets.data import build_full_panel
    from carbon_ets.models import backtest_v1

    panel = build_full_panel(start="2012-01-01")
    stats, equity = backtest_v1(panel)
    print(f"Sharpe = {stats['sharpe']:.2f}, CAGR = {stats['cagr']:+.1%}")

See `examples/quickstart.py` for a full end-to-end example.

Package structure:
    carbon_ets.data      — fetchers for EEX, Eurostat, TNAC, weather
    carbon_ets.features  — feature engineering (returns, z-scores, lags)
    carbon_ets.models    — OLS baseline, MS-2 regime model, backtest
    carbon_ets.tnac      — TNAC nowcast (monthly estimate ahead of Commission publication)
    carbon_ets.plots     — standard visualisations
"""

__version__ = "0.1.0"
__all__ = ["data", "features", "models", "tnac", "plots"]
