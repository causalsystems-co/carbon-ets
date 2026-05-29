# Maurizio — onboarding to the Carbon_ETS chain

## In 30 seconds

```bash
git clone <repo-url>          # or unzip Carbon_ETS.zip
cd Carbon_ETS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_fetch_prices.py     --start 2018-01-01
python scripts/02_fetch_emissions.py  --start 2018-01-01
python scripts/03_build_dataset.py
python scripts/04_analyze_chain.py
python scripts/05_backtest.py
```

You'll get `data/panel_features.parquet`, two PNGs in `plots/`, and a
printed Sharpe / max-DD on stdout.

## The intended workflow

1. Read `README.md` — the causal chain and why it should be tradable.
2. Run scripts 01-05 once unchanged. You should get out of the box:
   - `plots/leadlag.png` — driver lead-lag diagnostics
   - `plots/equity_curve.png` — **Sharpe ≈ 1.5, ret ≈ 21%, vol ≈ 13%, max-DD ≈ -14%** on KRBN 2020-08 → present
   This is a *working* baseline, not a placeholder. Two features
   (Eurostat IP YoY + Stoxx 50 momentum), equal-weighted, long-biased.
3. Pick one upgrade from the "Things for Maurizio to try" list in the
   README. Build it in `scripts/06_<your_idea>.py` so the baseline stays
   intact as a reference.
4. When you have an extension that beats the baseline on Sharpe, propose
   merging it.

## Repo conventions (match the rest of Causal Trading/)

- One folder per chain. Don't put EUA-specific code outside `Carbon_ETS/`.
- Scripts are numbered `NN_verb_object.py` and run top-to-bottom.
- Data files go in `data/` and are gitignored (they re-fetch).
- Plots go in `plots/`. Commit them — they're the artifacts other people
  look at first.
- One causal mechanism per script. If you're tempted to add a second,
  start a new numbered script.

## Where to put your own ideas

- **New driver?** Add a fetcher to `02_fetch_emissions.py` (or a new
  `02b_fetch_<source>.py` if it's heavy). Add the engineered feature
  to `03_build_dataset.py`. Everything downstream picks it up.
- **New signal?** Copy `05_backtest.py` to `05b_<idea>.py`. Don't mutate
  the strawman.
- **New asset?** EUA → UKA, CCA, RGGI: same code, different ticker. Try
  it as a cross-section before going single-name.

## What I'd love feedback on

- The KRBN proxy is the weakest link. If you have access to ICE EUA
  settlements (any channel — Refinitiv, Bloomberg, or scraping EEX),
  that single upgrade probably doubles signal-to-noise.
- The compliance calendar (April 30 surrender, May 15 MSR, December
  squaring) isn't modelled yet. Pure seasonality — should be easy.
- CBAM and ETS2 are coming live. The framework currently has no policy
  channel at all.

Questions: ping me on Signal / WhatsApp.
