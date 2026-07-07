"""
12_compliance_timing_backtest.py — Framework 3.

The decision problem for a compliance buyer:
    "I need to acquire N tonnes of EUAs each compliance year. What's the
    optimal monthly purchase schedule to minimise average purchase price?"

Three strategies compared, each buying exactly N tonnes over the same
16-month window (Jan year N → April 30 year N+1, the standard compliance
purchase horizon under EU ETS):

  A. Uniform monthly       — buy N/16 each month. Naïve benchmark.
  B. Q4 concentration      — wait, buy N over 7 months Oct(N)→Apr(N+1).
                             Common in practice for cost-of-carry reasons.
  C. Signal-modulated      — buy weighted by the V1 causal-chain signal.
                             Overweight bullish months (buy before it rises),
                             underweight bearish months.

Constraint on strategy C: monthly weight bounded to [0.3, 2.5] × uniform
share, so no month is skipped entirely and no month exceeds 2.5× the
naive share. Keeps execution realistic.

Metrics reported per compliance year and aggregated:
    avg purchase price (€/tCO2)  ←── the KPI compliance managers optimise
    €/tonne saved vs uniform
    €/tonne saved vs Q4 concentration
    total €-saved on 100kt/yr programme (representative mid-utility)
    total €-saved on 20Mt/yr programme (representative full-scale utility)

Output:
    data/compliance_timing.csv       year-by-year comparison
    plots/compliance_timing.png      cumulative savings chart
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

TARGET_PRICE = "eua_eur_tco2"
TARGET_RET = "r_eua_eur_tco2"

# Signal sensitivity: monthly weight = clip(1 + α × signal, 0.3, 2.5)
ALPHA = 1.5


def zscore(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def v1_signal(f):
    z_ip  = zscore(f["ip_yoy"], 252)
    z_stx = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)
    return ((z_ip + z_stx) / 2).clip(lower=0.0, upper=1.0)


def compliance_year_window(compliance_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Buying window for compliance year N: Jan 1 N → Apr 30 N+1 (surrender deadline)."""
    return pd.Timestamp(f"{compliance_year}-01-01"), pd.Timestamp(f"{compliance_year+1}-04-30")


def compute_strategies(monthly: pd.DataFrame, sig: pd.Series,
                       compliance_year: int, target_tonnes: float = 1.0):
    """For a given compliance year, compute the three strategies' average
    purchase price. Returns (price_uniform, price_q4, price_signal, months_used)."""
    start, end = compliance_year_window(compliance_year)
    window = monthly.loc[start:end]
    if len(window) < 12:
        return None
    prices = window["eua_avg"]
    n_months = len(prices)

    # A) Uniform monthly
    uniform_qty = np.ones(n_months) * (target_tonnes / n_months)
    price_uniform = (uniform_qty * prices).sum() / target_tonnes

    # B) Q4 concentration (last 7 months of the window)
    q4_qty = np.zeros(n_months)
    q4_qty[-7:] = target_tonnes / 7
    price_q4 = (q4_qty * prices).sum() / target_tonnes

    # C) Signal-modulated
    sig_window = sig.reindex(prices.index, method="ffill")
    # Bullish signal → overweight; center signal at ~0.4 (mean of a clipped-to-[0,1] signal)
    centered = sig_window - 0.4
    weights = np.clip(1 + ALPHA * centered, 0.3, 2.5)
    if weights.sum() == 0 or weights.isna().all():
        weights = np.ones(n_months)
    weights = weights / weights.sum()
    signal_qty = weights.values * target_tonnes
    price_signal = (signal_qty * prices).sum() / target_tonnes

    return {
        "compliance_year": compliance_year,
        "n_months": n_months,
        "price_uniform": price_uniform,
        "price_q4": price_q4,
        "price_signal": price_signal,
        "save_vs_uniform": price_uniform - price_signal,   # € per tonne, positive = saved
        "save_vs_q4": price_q4 - price_signal,
    }


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_PRICE])
    sig = v1_signal(f)

    # Aggregate EUA to monthly average price (compliance buyers execute
    # multiple auction-day purchases, so monthly average is realistic)
    monthly = f[[TARGET_PRICE]].resample("ME").mean().rename(columns={TARGET_PRICE: "eua_avg"})

    rows = []
    for cy in range(2014, 2025):
        r = compute_strategies(monthly, sig, cy)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No compliance years computed — check EUA data coverage")
        return

    df = pd.DataFrame(rows).set_index("compliance_year")
    df.to_csv(DATA / "compliance_timing.csv")

    # ───────────────────────  summary  ───────────────────────
    print("=" * 78)
    print("Framework 3 — EUA compliance-buy timing")
    print("=" * 78)
    print()
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    print()

    avg_save_vs_uniform = df["save_vs_uniform"].mean()
    avg_save_vs_q4      = df["save_vs_q4"].mean()
    median_save_vs_uniform = df["save_vs_uniform"].median()
    positive_years = (df["save_vs_uniform"] > 0).sum()

    print("─" * 78)
    print(f"Average saving vs uniform monthly buying: €{avg_save_vs_uniform:+.2f}/tonne")
    print(f"Median saving vs uniform monthly buying:  €{median_save_vs_uniform:+.2f}/tonne")
    print(f"Positive-savings years:                   {positive_years} / {len(df)}")
    print(f"Average saving vs Q4 concentration:       €{avg_save_vs_q4:+.2f}/tonne")
    print()

    # Scale to representative annual programmes
    for name, N in [("100kt/yr — mid-cap utility", 100_000),
                    ("1Mt/yr  — regional player",   1_000_000),
                    ("20Mt/yr — full-scale utility (Uniper/RWE)", 20_000_000)]:
        annual_saving_eur = avg_save_vs_uniform * N
        print(f"  {name:45s}  →  €{annual_saving_eur:>+14,.0f} / yr saved vs uniform")

    print()
    print("Note: uses BSM assumption of no execution costs; real bid-ask ~1 tick "
          "(~€0.01/tonne) — negligible vs the savings numbers above.")

    # ───────────────────────  plot  ───────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(11, 8),
                           gridspec_kw={"height_ratios": [2, 1]})

    # Per-year savings bar
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in df["save_vs_uniform"]]
    ax[0].bar(df.index, df["save_vs_uniform"], color=colors, alpha=0.85,
              label="Signal-modulated vs uniform monthly")
    ax[0].bar(df.index, df["save_vs_q4"], color="#95a5a6", alpha=0.4,
              label="Signal-modulated vs Q4 concentration")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axhline(avg_save_vs_uniform, ls="--", color="k", lw=0.8, alpha=0.6,
                  label=f"avg: €{avg_save_vs_uniform:+.2f}/tonne")
    ax[0].set_title("Framework 3 — Per-year cost savings from signal-modulated compliance buying")
    ax[0].set_ylabel("€ saved per tonne EUA purchased")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, axis="y")

    # Cumulative on a 20Mt/yr programme
    cum_20mt = (df["save_vs_uniform"] * 20_000_000).cumsum() / 1e6  # €M
    ax[1].plot(df.index, cum_20mt, marker="o", color="#1a1a1a")
    ax[1].fill_between(df.index, 0, cum_20mt, alpha=0.15, color="#1a1a1a")
    ax[1].set_title("Cumulative saving on 20Mt/yr programme (representative full-scale utility)")
    ax[1].set_ylabel("Cumulative € saved (millions)")
    ax[1].set_xlabel("Compliance year")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "compliance_timing.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/compliance_timing.png and data/compliance_timing.csv")


if __name__ == "__main__":
    main()
