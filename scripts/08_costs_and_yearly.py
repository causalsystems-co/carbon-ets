"""
08_costs_and_yearly.py — does V1 survive real-world frictions?

Two analyses on the winning V1 (long-only) strategy:

(1) TRANSACTION COST SWEEP
    Sweep round-trip costs from 0 to 20 bps. Each daily position change
    incurs cost = |Δposition| × bps. Shows where the strategy breaks even.

    Realistic anchors:
      ~3 bps  — EUA front-month futures (1 tick spread + exchange fee)
      ~5-10 bps — KRBN ETF retail
      ~15-30 bps — small-account broker with markup

(2) YEAR-BY-YEAR DECOMPOSITION
    Where does the Sharpe come from? Is it a few great years carrying
    a bunch of mediocre ones, or is the edge stable across regimes?

Output:
    data/cost_sweep.csv
    data/yearly_pnl.csv
    plots/costs_and_yearly.png
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

TARGET_RET = "r_eua_eur_tco2"
TARGET_PRICE = "eua_eur_tco2"


def zscore(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def v1_signal(f):
    """V1 long-only — the winning baseline from 07_optimize.py."""
    z_ip  = zscore(f["ip_yoy"], 252)
    z_stx = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)
    score = (z_ip + z_stx) / 2
    return score.clip(lower=0.0, upper=1.0)


def run_with_costs(f, signal, cost_bps_roundtrip):
    """Apply costs per unit of |Δposition|. cost in basis points = 1e-4."""
    pos = signal.shift(1)
    pnl_gross = (pos * f[TARGET_RET])
    turnover = pos.diff().abs()
    cost = turnover * (cost_bps_roundtrip * 1e-4)
    pnl_net = (pnl_gross - cost).dropna()
    eq = (1 + pnl_net).cumprod()
    return eq, pnl_net, turnover.dropna()


def stats(eq, pnl):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "CAGR": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
        "Sharpe": pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() else np.nan,
        "Vol":    pnl.std() * np.sqrt(252),
        "MaxDD":  (eq / eq.cummax() - 1).min(),
        "Years":  yrs,
    }


def main():
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_RET])
    sig = v1_signal(f)

    # ─── (1) COST SWEEP ───
    print("=== Transaction cost sweep (V1_long_only) ===\n")
    print(f"{'bps/RT':>8s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'Vol':>6s}  {'MaxDD':>7s}  {'avg|Δpos|':>10s}")
    cost_rows = []
    eqs_costs = {}
    for bps in [0, 1, 3, 5, 10, 15, 20, 30, 50]:
        eq, pnl, turn = run_with_costs(f, sig, bps)
        s = stats(eq, pnl)
        s["bps"] = bps
        s["avg_turnover_daily"] = turn.mean()
        s["ann_turnover"] = turn.sum() / s["Years"]
        cost_rows.append(s)
        eqs_costs[bps] = eq
        print(f"{bps:>8d}  {s['CAGR']:>+6.1%}  {s['Sharpe']:>+6.2f}  {s['Vol']:>5.1%}  {s['MaxDD']:>+6.0%}  {turn.mean():>10.4f}")

    pd.DataFrame(cost_rows).to_csv(DATA / "cost_sweep.csv", index=False)

    # breakeven
    sweep = pd.DataFrame(cost_rows)
    pos = sweep[sweep["CAGR"] > 0]
    if not pos.empty and (sweep["CAGR"] <= 0).any():
        breakeven = pos["bps"].max()
        print(f"\n  → strategy breaks even around {breakeven*2:.0f} bps roundtrip")
    else:
        print("\n  → strategy still positive at 50 bps/RT — very robust")

    # ─── (2) YEAR-BY-YEAR ───
    print("\n=== Yearly PnL (V1, no costs) ===\n")
    eq_clean, pnl_clean, _ = run_with_costs(f, sig, 0)
    yearly = pnl_clean.groupby(pnl_clean.index.year).agg(
        ret=("sum"), days=("count"),
        sharpe=lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() else np.nan,
        maxdd=lambda x: ((1+x).cumprod() / (1+x).cumprod().cummax() - 1).min(),
    )
    yearly["ret_pct"] = yearly["ret"]
    yearly_view = yearly[["ret_pct", "sharpe", "maxdd", "days"]]
    yearly_view.to_csv(DATA / "yearly_pnl.csv")
    print(yearly_view.to_string())

    # which year is the worst, which is the best
    print(f"\n  best year:  {yearly_view['ret_pct'].idxmax()}  "
          f"return {yearly_view['ret_pct'].max():+.1%}")
    print(f"  worst year: {yearly_view['ret_pct'].idxmin()}  "
          f"return {yearly_view['ret_pct'].min():+.1%}")
    print(f"  positive years: {(yearly_view['ret_pct'] > 0).sum()} / {len(yearly_view)}")

    # ─── PLOTS ───
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))

    # cost sweep equity curves
    for bps, eq in eqs_costs.items():
        ax[0].plot(eq.index, eq.values, label=f"{bps} bps/RT", alpha=0.8)
    ax[0].set_yscale("log")
    ax[0].set_title("V1 long-only equity curves under increasing transaction costs")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(alpha=0.3)

    # yearly returns bar chart
    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in yearly_view["ret_pct"]]
    ax[1].bar(yearly_view.index, yearly_view["ret_pct"], color=colors, alpha=0.85)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("V1 yearly returns (no costs)")
    ax[1].set_ylabel("yearly return")
    ax[1].grid(alpha=0.3, axis="y")
    for x, y in zip(yearly_view.index, yearly_view["ret_pct"]):
        ax[1].text(x, y + (0.005 if y > 0 else -0.015), f"{y:+.0%}",
                   ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS / "costs_and_yearly.png", dpi=120)
    plt.close(fig)
    print(f"\nwrote plots/costs_and_yearly.png")


if __name__ == "__main__":
    main()
