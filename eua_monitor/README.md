# EUA Regime Monitor

Live companion to [CS/RES/05 — Factories to Carbon](https://causalsystems.co/research/factories-to-carbon):
a self-contained dashboard that classifies the current EU carbon pricing
regime from public data and rebuilds itself daily.

What it shows:

- **Regime light** — rolling 24-month R² of the demand model (monthly EUA
  returns on euro-area IP YoY growth, Stoxx 50 momentum, HDD anomaly),
  classified demand-driven / transitional / policy-driven.
- **EUA price** shaded by regime.
- **TNAC vs the MSR thresholds** (24% intake ≥ 1,096M, partial intake
  > 833M, release < 400M) from the Commission's annual communications.
- **Auction calendar** (indicative EEX weekday pattern, override via
  `data/auction_overrides.csv`).
- **Digest** — `site/digest.md` is written only when the regime flips or a
  new TNAC print changes the MSR response; CI emails it if SMTP secrets
  are configured.

## Run

```bash
pip install numpy
python -m eua_monitor build   # writes site/index.html, site/state.json
```

Every fetcher snapshots its raw payload into `data/cache/` and falls back
to the snapshot when offline, so the build never hard-fails on a flaky
source. Sources: EEX primary-auction reports (price + cover ratio),
Eurostat `sts_inpr_m`, Yahoo Finance (^STOXX50E; CO2.L as recency
fallback), Open-Meteo archive, EC MSR communications (`data/tnac.csv`).

The EUA series is the paper's own: EEX auction clearing prices 2012→now.
`data/eua_history.csv` holds the imported archive (regenerate with
`python -m eua_monitor.eex_import <zip>` from EEX's
`emission-spot-primary-market-auction-report-2012-2025-data.zip`); the
current year tops up live from the public EEX report at build time.

## Deploy

`.github/workflows/eua-monitor.yml` rebuilds on weekday mornings, commits
the refreshed `site/`, and (optionally) emails the digest. Point GitHub
Pages at the `site/` directory to serve the dashboard.

Model constants (window, regime cutoffs, HDD basket, MSR thresholds) live
in `config.py`.
