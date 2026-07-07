"""
07_optimize.py — strategy variant sweep on the 9-year real EUA window.

Baseline (05_backtest.py) scored Sharpe 0.77 / CAGR +14% / DD -27% on
2015-09 → 2024-08 of real EEX auction prices. The hypothesis: most of
that mediocre Sharpe is concentrated in periods where the demand-driven
chain doesn't apply (sideways IP, policy-driven moves like the 2018 MSR
rally). This script tests 12 variants of overlay / filter / sizing that
might recover the lost edge.

What each variant does is documented inline. Run it, look at the printed
table, and the best variant's equity curve is saved to:
    plots/equity_curve_best.png

The table is also written to:
    data/optimization_results.csv

Run with:  python scripts/07_optimize.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

TARGET_RET = "r_eua_eur_tco2"
TARGET_PRICE = "eua_eur_tco2"


# ─────────────────────────  helpers  ─────────────────────────

def zscore(s: pd.Series, win: int = 252) -> pd.Series:
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def stats(eq: pd.Series, pnl: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    return {
        "CAGR":     cagr,
        "Sharpe":   pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() else np.nan,
        "Vol":      pnl.std() * np.sqrt(252),
        "MaxDD":    (eq / eq.cummax() - 1).min(),
        "HitRate":  (pnl > 0).mean(),
        "Years":    yrs,
        "Days":     int(pnl.notna().sum()),
    }


def run(label: str, position_fn: Callable[[pd.DataFrame], pd.Series],
        f: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    """Apply position_fn to f, compute pnl, return stats + equity + pnl."""
    pos = position_fn(f).shift(1)
    pnl = (pos * f[TARGET_RET]).dropna()
    eq = (1 + pnl).cumprod()
    s = stats(eq, pnl)
    s["Name"] = label
    return s, eq, pnl


# ─────────────────────────  data  ─────────────────────────

def load_panel() -> pd.DataFrame:
    f = pd.read_parquet(DATA / "panel_features.parquet")
    f = f.dropna(subset=[TARGET_RET]).copy()

    # base features
    f["z_ip"]  = zscore(f["ip_yoy"], 252)
    f["z_stx"] = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)

    # EUA own-momentum (mean-reversion vs trend tests)
    p = f[TARGET_PRICE]
    f["eua_mom20"] = np.log(p / p.shift(20))
    f["eua_mom60"] = np.log(p / p.shift(60))
    f["eua_mom120"] = np.log(p / p.shift(120))
    f["z_eua_mom60"] = zscore(f["eua_mom60"], 252)

    # gas leg (TTF momentum lagged 5 days)
    g = f["ttf_gas_eur_mwh"]
    f["z_gas_mom60"] = zscore(np.log(g / g.shift(60)).shift(5), 252)

    # realized vol on EUA
    f["eua_rv20"] = f[TARGET_RET].rolling(20).std()
    f["eua_rv_pct"] = f["eua_rv20"].rolling(252).rank(pct=True)

    # calendar features for MSR overlay
    f["month"] = f.index.month
    f["dom"]   = f.index.day
    # April surrender deadline (compliance buying) → +1 in April
    f["surrender_season"] = (f["month"] == 4).astype(float)
    # May 15 MSR announcement (uncertainty around → reduce risk)
    f["msr_week"] = ((f["month"] == 5) & (f["dom"] >= 10) & (f["dom"] <= 20)).astype(float)
    # December position-squaring (often weak)
    f["dec_squaring"] = ((f["month"] == 12) & (f["dom"] >= 15)).astype(float)

    return f


# ─────────────────────────  position functions  ─────────────────────────

def baseline(f):
    """V0: current production strategy."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    return score.clip(lower=-0.3, upper=1.0)

def long_only(f):
    """V1: never short."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    return score.clip(lower=0.0, upper=1.0)

def conditional_trade(f):
    """V2: only trade when |z_ip| > 0.5 (clear expansion/contraction)."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    mask = f["z_ip"].abs() > 0.5
    return score.where(mask, 0.0).clip(lower=-0.3, upper=1.0)

def regime_long_only_in_bull(f):
    """V3: long-only when EUA is in a 120d uptrend; baseline otherwise."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    in_bull = f["eua_mom120"] > 0
    sig = np.where(in_bull, score.clip(lower=0), score.clip(lower=-0.3))
    return pd.Series(sig, index=f.index, name="pos").clip(upper=1.0)

def add_eua_momentum(f):
    """V4: baseline + EUA own momentum (trend-following overlay)."""
    score = (f["z_ip"] + f["z_stx"] + f["z_eua_mom60"]) / 3
    return score.clip(lower=-0.3, upper=1.0)

def add_gas_leg(f):
    """V5: baseline + gas momentum (fuel-switching channel)."""
    score = (f["z_ip"] + f["z_stx"] + f["z_gas_mom60"]) / 3
    return score.clip(lower=-0.3, upper=1.0)

def msr_calendar_overlay(f):
    """V6: baseline + MSR/compliance calendar tilts."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    tilt = 0.3 * f["surrender_season"] - 0.5 * f["msr_week"] - 0.2 * f["dec_squaring"]
    return (score + tilt).clip(lower=-0.3, upper=1.0)

def vol_target(f, target_ann=0.15):
    """V7: scale baseline so realized 20d vol matches target."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    pos = score.clip(lower=-0.3, upper=1.0)
    rv = f[TARGET_RET].rolling(20).std() * np.sqrt(252)
    scale = (target_ann / rv).clip(upper=2.0).fillna(1.0)
    return (pos * scale).clip(lower=-1.0, upper=2.0)

def low_vol_only(f):
    """V8: only trade in lower-vol regimes (top vol-decile → flat)."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    pos = score.clip(lower=-0.3, upper=1.0)
    return pos.where(f["eua_rv_pct"] < 0.85, 0.0)

def kitchen_sink(f):
    """V9: all overlays combined."""
    score = (f["z_ip"] + f["z_stx"] + 0.5 * f["z_eua_mom60"]) / 2.5
    in_bull = f["eua_mom120"] > 0
    sig = np.where(in_bull, score.clip(lower=0), score.clip(lower=-0.3))
    sig = pd.Series(sig, index=f.index)
    # calendar tilts
    tilt = 0.3 * f["surrender_season"] - 0.5 * f["msr_week"]
    sig = sig + tilt
    # skip top vol decile
    sig = sig.where(f["eua_rv_pct"] < 0.85, 0.0)
    return sig.clip(lower=-0.3, upper=1.5)

def walk_forward(f):
    """V10: refit feature weights on rolling 3yr window via simple in-sample IR.

    For each day, look back 750 days, compute IR of each feature alone (sign · mean/std),
    use those as weights for that day. Re-fit every 60 days to reduce noise.
    """
    feats = ["z_ip", "z_stx", "z_eua_mom60"]
    out = pd.Series(np.nan, index=f.index)
    win = 750
    refit_every = 60
    weights = pd.Series([1.0]*len(feats), index=feats)
    for i in range(len(f)):
        if i < win:
            continue
        if i % refit_every == 0:
            past = f.iloc[i-win:i]
            w = {}
            for c in feats:
                x = past[c].shift(1).dropna()
                r = past[TARGET_RET].reindex(x.index)
                m = (np.sign(x) * r).dropna()
                w[c] = m.mean() / m.std() if m.std() else 0
            ws = pd.Series(w)
            ws = ws / ws.abs().sum() if ws.abs().sum() else ws
            weights = ws
        row = f.iloc[i][feats]
        out.iloc[i] = (row * weights).sum()
    return out.clip(lower=-0.3, upper=1.0)

def best_guess(f):
    """V11: my prior on what should actually work — long-bias trend +
    demand signal active only when IP is moving + skip high-vol periods."""
    demand = (f["z_ip"] + f["z_stx"]) / 2
    demand_active = demand.where(f["z_ip"].abs() > 0.3, 0.0)
    trend = (f["eua_mom60"] > 0).astype(float) * 0.5
    # always slightly long (structural EUA bull)
    base = 0.2
    sig = base + 0.6*demand_active + 0.4*trend
    sig = sig.where(f["eua_rv_pct"] < 0.9, sig * 0.3)  # de-risk when vol is extreme
    # calendar: pull back during MSR week
    sig = sig - 0.3 * f["msr_week"]
    return sig.clip(lower=-0.3, upper=1.5)


# ─── MSR-augmented variants ──────────────────────────────────────────

def _msr_feat(f):
    """Return MSR supply-tightness feature, or a series of zeros if unavailable."""
    if "z_msr_supply_bull" in f and f["z_msr_supply_bull"].notna().any():
        return f["z_msr_supply_bull"]
    return pd.Series(0.0, index=f.index)

def add_msr(f):
    """V13: baseline (IP+Stoxx) + MSR supply-tightness. Three equal-weighted features."""
    z_msr = _msr_feat(f)
    score = (f["z_ip"] + f["z_stx"] + z_msr) / 3
    return score.clip(lower=-0.3, upper=1.0)

def msr_only(f):
    """V14: MSR alone — is the policy-supply signal predictive without demand?"""
    z_msr = _msr_feat(f)
    return z_msr.clip(lower=-0.3, upper=1.0)

def long_only_plus_msr(f):
    """V15: V1 (long-only) + MSR overlay. Best V1 plus supply-side context."""
    z_msr = _msr_feat(f)
    score = (f["z_ip"] + f["z_stx"] + z_msr) / 3
    return score.clip(lower=0.0, upper=1.0)

def demand_gated_by_msr(f):
    """V16: only trade demand signal when MSR says supply is tightening.
    Otherwise flat. Directly targets the 2014-17 dead-signal era where
    demand was mildly positive but oversupply pinned prices."""
    z_msr = _msr_feat(f)
    demand = (f["z_ip"] + f["z_stx"]) / 2
    tight = z_msr > -0.5   # allow modest oversupply
    return demand.where(tight, 0.0).clip(lower=0.0, upper=1.0)


def long_only_vol_targeted(f, target_ann=0.15):
    """V12: V1 long-only + V7 vol-targeted — combine the two winners."""
    score = (f["z_ip"] + f["z_stx"]) / 2
    pos = score.clip(lower=0.0, upper=1.0)
    rv = f[TARGET_RET].rolling(20).std() * np.sqrt(252)
    scale = (target_ann / rv).clip(upper=2.0).fillna(1.0)
    return (pos * scale).clip(lower=0.0, upper=2.0)


VARIANTS = [
    ("V0_baseline",              baseline),
    ("V1_long_only",             long_only),
    ("V2_only_when_IP_moves",    conditional_trade),
    ("V3_bull_regime_long_only", regime_long_only_in_bull),
    ("V4_add_eua_momentum",      add_eua_momentum),
    ("V5_add_gas_leg",           add_gas_leg),
    ("V6_calendar_overlay",      msr_calendar_overlay),
    ("V7_vol_targeted_15pct",    vol_target),
    ("V8_skip_high_vol",         low_vol_only),
    ("V9_kitchen_sink",          kitchen_sink),
    ("V10_walk_forward",         walk_forward),
    ("V11_best_guess",           best_guess),
    ("V12_long_only_voltgt",     long_only_vol_targeted),
    ("V13_add_msr",              add_msr),
    ("V14_msr_only",             msr_only),
    ("V15_long_only_plus_msr",   long_only_plus_msr),
    ("V16_demand_gated_by_msr",  demand_gated_by_msr),
]


# ─────────────────────────  run  ─────────────────────────

def main() -> None:
    f = load_panel()
    print(f"panel: {len(f)} rows  {f.index.min().date()} → {f.index.max().date()}")
    print(f"trading target: {TARGET_RET}\n")

    results = []
    equities = {}
    for name, fn in VARIANTS:
        try:
            s, eq, pnl = run(name, fn, f)
            results.append(s)
            equities[name] = eq
            print(f"  {name:30s} "
                  f"CAGR {s['CAGR']:+6.1%}  "
                  f"Sharpe {s['Sharpe']:+5.2f}  "
                  f"Vol {s['Vol']:5.1%}  "
                  f"DD {s['MaxDD']:6.1%}  "
                  f"Hit {s['HitRate']:.1%}  "
                  f"n={s['Days']}")
        except Exception as e:
            print(f"  {name:30s} ERR {e}")

    # buy & hold reference
    bh_pnl = f[TARGET_RET].dropna()
    bh_eq = (1 + bh_pnl).cumprod()
    bh_stats = stats(bh_eq, bh_pnl)
    bh_stats["Name"] = "BH_buy_and_hold"
    results.append(bh_stats)
    equities["BH_buy_and_hold"] = bh_eq
    print(f"\n  {'BH_buy_and_hold':30s} "
          f"CAGR {bh_stats['CAGR']:+6.1%}  "
          f"Sharpe {bh_stats['Sharpe']:+5.2f}  "
          f"Vol {bh_stats['Vol']:5.1%}  "
          f"DD {bh_stats['MaxDD']:6.1%}")

    df_results = pd.DataFrame(results).set_index("Name")
    df_results = df_results.sort_values("Sharpe", ascending=False)
    df_results.to_csv(DATA / "optimization_results.csv")
    print("\nranked by Sharpe:")
    print(df_results[["CAGR","Sharpe","Vol","MaxDD","HitRate","Days"]].to_string())

    # plot top 5 + baseline + BH
    top5 = df_results.head(5).index.tolist()
    must_include = ["V0_baseline", "BH_buy_and_hold"]
    plot_set = list(dict.fromkeys(top5 + must_include))

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in plot_set:
        eq = equities[name]
        ax.plot(eq.index, eq.values, label=name, alpha=0.85, lw=1.5)
    ax.set_yscale("log")
    ax.set_title(f"EUA strategy variants on {len(f)} days of real EEX auction prices "
                 f"({f.index.min().date()} → {f.index.max().date()})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "optimization_equity.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote plots/optimization_equity.png")

    best = df_results.index[0]
    print(f"\nbest variant: {best}")
    print(f"  CAGR   {df_results.loc[best,'CAGR']:+.1%}")
    print(f"  Sharpe {df_results.loc[best,'Sharpe']:+.2f}")
    print(f"  MaxDD  {df_results.loc[best,'MaxDD']:.0%}")


if __name__ == "__main__":
    main()
