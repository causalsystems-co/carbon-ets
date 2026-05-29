# Maurizio — start here

Welcome. This repo trades the **EU ETS (carbon allowances)** using a
causal chain. The thesis is in `README.md`. This file is just your
on-ramp.

## Your first 5 minutes

```bash
git clone <repo-url>
cd carbon-ets-causal-chain
make setup
make run
open plots/equity_curve.png
```

That's it. The baseline strategy runs end-to-end on free public data,
fetches everything itself, and produces a plot with **Sharpe ≈ 1.5**.

If `make` is missing on your machine: `xcode-select --install` on Mac.

## Your first 30 minutes

Open `notebooks/01_explore.ipynb` (`make notebook`). The notebook
walks you through the panel:

1. What's in `panel_features.parquet` (~30 columns: prices, fundamentals,
   engineered features).
2. KRBN price chart — the EUA proxy we're trading.
3. IP YoY vs EUA price — visual sanity check on the causal chain.
4. The baseline equity curve, loaded from `data/backtest_trades.parquet`.
5. An empty cell labelled "your turn". Pick a feature, plot it against
   EUA returns, look for something predictive.

## Your first PR

Don't touch `scripts/05_backtest.py` — it's the baseline reference.
Instead:

```bash
cp scripts/06_TEMPLATE_your_idea.py scripts/06_<your_idea>.py
# edit my_features() to add or swap a signal
python scripts/06_<your_idea>.py
```

The template prints the same stats block as the baseline, so you can
compare apples-to-apples. Open a PR when your variant beats Sharpe 1.5,
or when it shows something new even if it doesn't beat the baseline.

## What to try (ranked by expected lift)

1. **Replace KRBN with real ICE EUA front-month.** KRBN is a basket
   (EUA + RGGI + CCA) and loses ~15% of signal-to-noise. EEX publishes
   primary-auction clearing prices daily, free, scrapeable. This single
   upgrade probably moves Sharpe by +0.3.
2. **Add transaction costs.** Currently zero. Realistic: 5 bps per side
   on KRBN, 1 tick on EUA futures. The baseline rebalances daily — also
   add a turnover cap (`|Δposition| ≤ 0.2/day`) and see how Sharpe holds up.
3. **Walk-forward weights.** Right now the two features are equal-weighted
   in-sample. Fit weights on a rolling 3-year window, re-estimate yearly.
4. **Vol-target the position** so realised 20d vol ≈ 15% annualised.
   The `vol_target()` function is already in `05_backtest.py`, just
   set `TARGET_VOL_ANN = 0.15`.
5. **Compliance-calendar overlay.** Long bias into April 30 surrender
   deadline, flat around May 15 MSR announcement, December squaring.
   Pure seasonality, well-documented, almost certainly free alpha.
6. **Gas leg properly.** TTF momentum should help via fuel-switching
   (gas up → coal-to-gas more expensive → buy more EUAs). Daily signal
   is too noisy — try weekly resampling first.
7. **Policy-news classifier.** EU Commission press feed publishes
   CBAM / ETS2 / free-allowance updates. NLP topic + sentiment on those
   headlines is a strong feature with no obvious crowding.

## Repo conventions

- `scripts/NN_verb_object.py` — numbered, run top-to-bottom.
- `data/` and `plots/` are gitignored for `.parquet` / `.csv`. Commit
  PNGs only if they're headline results.
- One mechanism per script. Don't add a second signal to the same file —
  start `07_<your_idea>.py`.
- Branch + PR for anything beyond a small fix. Commit messages: imperative,
  one line, no emoji.

## Where Halis lives in the chain

This is part of a larger `Causal Trading/` repo with sister chains
(`EU_Power_Model/`, `Spark_Spread/`, `TTF_Gas/`). EUA is the carbon
leg that overlays all of them. If you want to see how other chains
are structured, ask Halis for the parent repo.

## Stuck?

- Pipeline won't run on fresh clone → check `python --version` ≥ 3.10
  and that `make setup` actually finished installing.
- KRBN fetch fails → yfinance occasionally rate-limits; wait 60s, retry.
- `02_fetch_emissions.py` Fraunhofer call times out → expected from
  some networks. The script automatically falls back to a
  temperature-derived load proxy. The baseline backtest doesn't depend
  on `load_eu5_mw` anyway.
- Anything else → ping Halis.
