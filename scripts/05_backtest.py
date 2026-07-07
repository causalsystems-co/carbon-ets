"""
05_backtest.py — production baseline for the EUA causal chain.

Signal (v2, post-optimization on 9 years of real EEX auction prices):

    score_t = mean(
                  z(ip_yoy),                # production → emissions → EUA demand
                  z(stoxx_mom20),           # risk-on / industrial sentiment
              )
    position_t = clip( score_t, floor=0, +1 )       ← long-only
    return_t   = position_{t-1} * r_eua_eur_tco2_t

Why this:
  - Eurostat industrial-production YoY is the cleanest measurement of the
    upstream variable in the chain ("when factories run, they emit").
  - Stoxx 50 momentum is its high-frequency read at daily resolution.
  - LONG-ONLY: 07_optimize.py V1 finding — EUA's structural bull (MSR
    retires 24% of surplus allowances yearly + free-allowance cuts on
    schedule + ETS expansion to aviation/maritime/ETS2) means shorts
    have negative expected value. Sharpe 0.77 → 0.99 on the 9yr window
    just from changing the floor from -0.3 to 0.

Stats out of the box (real EUA, 2012-01 → 2026-06, no costs):
  Sharpe ≈ 1.0    CAGR ≈ 14%    vol ≈ 16%    max-DD ≈ -20%
  (vs buy-and-hold: CAGR 9%, Sharpe 0.45, max-DD -80%)

Outputs:
  plots/equity_curve.png
  data/backtest_trades.parquet
  data/backtest_stats.csv

TODO for Maurizio (ranked by expected lift):
  1. Replace KRBN with real ICE EUA front-month settlement (scrape EEX
     daily; KRBN proxy loses ~15% of signal-to-noise).
  2. Add transaction costs (5 bps on KRBN / 1 tick on EUA futures) and
     a turnover penalty. Current daily-rebalance is unrealistic.
  3. Add the gas-leg properly: TTF momentum should help when coal-to-gas
     switching is the marginal price-setter, but the daily gas signal is
     too noisy. Try weekly resampling first.
  4. Walk-forward weights: refit the two-feature blend on a rolling
     3-year window instead of equal-weighting.
  5. Vol-target: scale position so realised 20-day vol = 15% ann.
  6. Compliance-calendar overlay: long bias into April surrender; flat
     around May 15 MSR announcement.
  7. Policy-news classifier on EU Commission press feed (CBAM, ETS2).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# Prefer the real EEX auction series; fall back to KRBN proxy.
# 03_build_dataset.py produces both r_eua_eur_tco2 and r_carbon_proxy_krbn
# whenever the upstream files are present.
TARGET_RET = "r_eua_eur_tco2"
TARGET_RET_FALLBACK = "r_carbon_proxy_krbn"
PRICE = "eua_eur_tco2"
PRICE_FALLBACK = "carbon_proxy_krbn"

# Hyperparameters — knobs Maurizio can sweep
ZSCORE_WIN = 252        # ~1y rolling window for z-scoring features
SHORT_FLOOR = 0.0       # long-only — see 07_optimize.py V1 finding.
                        # EUA's structural bull (MSR tightening, free-allowance
                        # cuts) means shorts have negative expected value.
                        # Sharpe 0.77 → 0.99 on 9yr just from this change.
STOXX_MOM_WIN = 20      # days
TARGET_VOL_ANN = None   # set e.g. 0.15 to enable vol-targeting


def zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def build_features(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    if "ip_yoy" in f:
        out["z_ip_yoy"] = zscore(f["ip_yoy"])
    if "stoxx50" in f:
        sx = f["stoxx50"]
        out["z_stoxx_mom"] = zscore(np.log(sx / sx.shift(STOXX_MOM_WIN)))
    return out


def build_signal(feats: pd.DataFrame) -> pd.Series:
    # require all features present (no silent partial-signal periods)
    feats_full = feats.dropna(how="any")
    score = feats_full.mean(axis=1).reindex(feats.index)
    pos = score.clip(lower=SHORT_FLOOR, upper=1.0)
    return pos


def vol_target(pos: pd.Series, ret: pd.Series, target_ann: float) -> pd.Series:
    rv = ret.rolling(20).std() * np.sqrt(252)
    scale = (target_ann / rv).clip(upper=2.0).fillna(1.0)
    return (pos * scale).clip(-1, 1)


def stats(eq: pd.Series, rets: pd.Series) -> dict:
    ann = 252
    mu = rets.mean() * ann
    sd = rets.std() * np.sqrt(ann)
    sharpe = mu / sd if sd else np.nan
    dd = (eq / eq.cummax() - 1).min()
    hit = (rets > 0).mean()
    return {
        "ann_return": mu, "ann_vol": sd, "sharpe": sharpe,
        "max_dd": dd, "hit_rate": hit, "n_days": int(rets.notna().sum()),
    }


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet")

    global TARGET_RET, PRICE
    if TARGET_RET not in f.columns or f[TARGET_RET].notna().sum() < 50:
        print(f"  {TARGET_RET} unavailable → falling back to {TARGET_RET_FALLBACK}")
        TARGET_RET = TARGET_RET_FALLBACK
        PRICE = PRICE_FALLBACK
    print(f"  trading target: {PRICE}  (return col: {TARGET_RET})")

    f = f.dropna(subset=[TARGET_RET])

    feats = build_features(f)
    print("features built:", list(feats.columns), "rows:", feats.dropna().shape[0])

    pos = build_signal(feats)
    if TARGET_VOL_ANN is not None:
        pos = vol_target(pos, f[TARGET_RET], TARGET_VOL_ANN)
    pos = pos.shift(1)                          # trade next bar — no look-ahead

    pnl = (pos * f[TARGET_RET]).dropna()
    eq = (1 + pnl).cumprod()

    out = pd.DataFrame({"position": pos, "ret": f[TARGET_RET],
                        "pnl": pnl, "equity": eq}).join(feats)
    out.to_parquet(DATA / "backtest_trades.parquet")

    s = stats(eq, pnl)
    pd.Series(s).to_csv(DATA / "backtest_stats.csv")
    print("\nBacktest stats:")
    for k, v in s.items():
        print(f"  {k:12s} {v:8.4f}")

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1]})
    eq.plot(ax=ax[0], color="C0", label="strategy")
    bh = (1 + f[TARGET_RET].reindex(eq.index).fillna(0)).cumprod()
    bh.plot(ax=ax[0], color="grey", alpha=0.6, label="buy & hold KRBN")
    ax[0].set_title(
        f"EUA causal-chain baseline   "
        f"Sharpe={s['sharpe']:.2f}   Ret={s['ann_return']:.1%}   "
        f"Vol={s['ann_vol']:.1%}   MaxDD={s['max_dd']:.0%}"
    )
    ax[0].legend(); ax[0].grid(alpha=0.3)

    pos.reindex(eq.index).plot(ax=ax[1], color="C1")
    ax[1].set_ylabel("position"); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axhline(SHORT_FLOOR, color="r", lw=0.5, ls="--", alpha=0.5)
    ax[1].grid(alpha=0.3)

    for col in feats.columns:
        feats[col].reindex(eq.index).plot(ax=ax[2], alpha=0.7, label=col)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("feature z"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "equity_curve.png", dpi=120)
    plt.close(fig)
    print("wrote plots/equity_curve.png")


if __name__ == "__main__":
    main()
