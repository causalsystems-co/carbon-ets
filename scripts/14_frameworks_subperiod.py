"""
14_frameworks_subperiod.py — subperiod diagnostics for Frameworks 2 & 3.

Two tests, both answering "when does the signal actually work?"

TEST A — Framework 3 vol-regime split.
  Hypothesis: signal-modulated compliance buying delivers real savings
  in high-vol regimes but is neutral in calm regimes. If true, we
  reframe the pitch from "cost optimization" to "volatility hedge".

TEST B — Framework 2 pre/post-2024 (maritime ETS activation).
  Hypothesis: the EU ETS maritime extension (Jan 2024) created a
  clean, statistically significant EUA transmission channel for
  shipping equities that didn't exist before. If true, this is a
  publishable finding by itself.

Outputs:
    data/subperiod_framework3.csv
    data/subperiod_framework2_shipping.csv
    plots/subperiod_analysis.png
"""

from __future__ import annotations
from pathlib import Path
import warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

TARGET_PRICE = "eua_eur_tco2"
TARGET_RET   = "r_eua_eur_tco2"


# ─── shared helpers ──────────────────────────────────────────────────
def zscore(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

def v1_signal(f):
    z_ip  = zscore(f["ip_yoy"], 252)
    z_stx = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)
    return ((z_ip + z_stx) / 2).clip(lower=0.0, upper=1.0)

def compliance_window(y):
    return pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-04-30")

def yfclose(ticker, start="2014-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty: return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)


# ─── TEST A: Framework 3 vol regimes ─────────────────────────────────
def run_test_a(f) -> pd.DataFrame:
    sig = v1_signal(f)
    monthly = f[[TARGET_PRICE]].resample("ME").mean().rename(columns={TARGET_PRICE: "eua_avg"})
    # annualised realised vol per compliance year (Jan → Dec of year N)
    annual_vol = f[TARGET_RET].resample("YE").std() * np.sqrt(252)
    annual_vol.index = annual_vol.index.year

    rows = []
    ALPHA = 1.5
    for cy in range(2014, 2025):
        start, end = compliance_window(cy)
        window = monthly.loc[start:end]
        if len(window) < 12:
            continue
        prices = window["eua_avg"]
        n = len(prices)

        uniform_qty = np.ones(n) / n
        price_uniform = (uniform_qty * prices).sum()

        sig_window = sig.reindex(prices.index, method="ffill")
        centered = sig_window - 0.4
        weights = np.clip(1 + ALPHA * centered, 0.3, 2.5)
        if weights.sum() == 0 or weights.isna().all():
            weights = pd.Series(np.ones(n), index=prices.index)
        weights = weights / weights.sum()
        price_signal = (weights.values * prices.values).sum()

        rows.append({
            "compliance_year":  cy,
            "eua_ann_vol":      annual_vol.get(cy, np.nan),
            "price_uniform":    price_uniform,
            "price_signal":     price_signal,
            "save_vs_uniform":  price_uniform - price_signal,
        })
    df = pd.DataFrame(rows).set_index("compliance_year")
    df["vol_regime"] = np.where(
        df["eua_ann_vol"] >= df["eua_ann_vol"].median(),
        "high_vol", "low_vol"
    )
    return df


# ─── TEST B: Framework 2 pre/post 2024 for shipping ─────────────────
SHIPPING_TICKERS = {
    "Maersk":       "MAERSK-B.CO",
    "Hapag-Lloyd":  "HLAG.DE",
    "MSC (proxy: Kuehne+Nagel)": "KNIN.SW",
}

def run_test_b(f):
    # build controls at weekly frequency
    stoxx = f["stoxx50"].resample("W-FRI").last()
    ttf   = f["ttf_gas_eur_mwh"].resample("W-FRI").last()
    eua   = f[TARGET_PRICE].resample("W-FRI").last()
    r_ctrl = pd.DataFrame({
        "r_eua":   np.log(eua / eua.shift(1)),
        "r_stoxx": np.log(stoxx / stoxx.shift(1)),
        "r_ttf":   np.log(ttf / ttf.shift(1)),
    }).dropna()

    rows = []
    for label, ticker in SHIPPING_TICKERS.items():
        print(f"  fetch {ticker:15s}", end=" ")
        p = yfclose(ticker)
        if p.empty:
            print("failed"); continue
        pw = p.resample("W-FRI").last()
        r_sec = np.log(pw / pw.shift(1)).dropna()

        joined = r_ctrl.join(r_sec.rename("r_sec"), how="inner").dropna()
        pre  = joined.loc[:"2023-12-31"]
        post = joined.loc["2024-01-01":]

        for name, sub in [("pre_2024", pre), ("post_2024", post)]:
            if len(sub) < 30:
                continue
            X = sm.add_constant(sub[["r_eua", "r_stoxx", "r_ttf"]])
            y = sub["r_sec"]
            m = sm.OLS(y, X).fit()
            rows.append({
                "ticker":   ticker,
                "sector":   label,
                "window":   name,
                "n":        len(sub),
                "beta_eua": m.params["r_eua"],
                "t_stat":   m.tvalues["r_eua"],
                "r2":       m.rsquared,
                "sig":      "***" if abs(m.tvalues['r_eua']) > 2.58 else "**" if abs(m.tvalues['r_eua']) > 1.96 else "*" if abs(m.tvalues['r_eua']) > 1.64 else " ",
            })
        print(f"  pre n={len(pre):3d}, post n={len(post):3d}")

    return pd.DataFrame(rows)


# ─── main ────────────────────────────────────────────────────────────
def main():
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_PRICE])

    print("=" * 80)
    print("TEST A — Framework 3 vol-regime split")
    print("=" * 80)
    df_a = run_test_a(f)
    df_a.to_csv(DATA / "subperiod_framework3.csv")

    high = df_a[df_a["vol_regime"] == "high_vol"]
    low  = df_a[df_a["vol_regime"] == "low_vol"]

    print()
    print(df_a[["eua_ann_vol", "price_uniform", "price_signal", "save_vs_uniform", "vol_regime"]].to_string(
        float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))
    print()
    print(f"HIGH-VOL years (vol ≥ {df_a['eua_ann_vol'].median():.1%}, n={len(high)}):")
    print(f"  mean save vs uniform: €{high['save_vs_uniform'].mean():+.2f}/tonne")
    print(f"  median:               €{high['save_vs_uniform'].median():+.2f}/tonne")
    print(f"  positive years:       {(high['save_vs_uniform']>0).sum()}/{len(high)}")
    print()
    print(f"LOW-VOL years (vol < {df_a['eua_ann_vol'].median():.1%}, n={len(low)}):")
    print(f"  mean save vs uniform: €{low['save_vs_uniform'].mean():+.2f}/tonne")
    print(f"  median:               €{low['save_vs_uniform'].median():+.2f}/tonne")
    print(f"  positive years:       {(low['save_vs_uniform']>0).sum()}/{len(low)}")

    # ────────────────────
    print()
    print("=" * 80)
    print("TEST B — Framework 2 pre/post 2024 (shipping, maritime ETS extension)")
    print("=" * 80)
    df_b = run_test_b(f)
    if not df_b.empty:
        df_b.to_csv(DATA / "subperiod_framework2_shipping.csv", index=False)
        print()
        print(df_b.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)))
        print()
        # comparison
        for tkr in df_b["ticker"].unique():
            sub = df_b[df_b["ticker"] == tkr]
            pre  = sub[sub["window"] == "pre_2024"].iloc[0] if not sub[sub["window"] == "pre_2024"].empty else None
            post = sub[sub["window"] == "post_2024"].iloc[0] if not sub[sub["window"] == "post_2024"].empty else None
            if pre is not None and post is not None:
                print(f"  {tkr:15s}  β: {pre['beta_eua']:+.3f}(t={pre['t_stat']:+.2f}) → {post['beta_eua']:+.3f}(t={post['t_stat']:+.2f})")

    # ─── plots ───
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))

    # A: vol-regime bars
    colors = ["#e74c3c" if r == "high_vol" else "#3498db" for r in df_a["vol_regime"]]
    ax[0].bar(df_a.index, df_a["save_vs_uniform"], color=colors, alpha=0.85)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axhline(high["save_vs_uniform"].mean(), color="#e74c3c", ls="--", lw=0.8,
                  label=f"high-vol mean €{high['save_vs_uniform'].mean():+.2f}/t")
    ax[0].axhline(low["save_vs_uniform"].mean(), color="#3498db", ls="--", lw=0.8,
                  label=f"low-vol mean €{low['save_vs_uniform'].mean():+.2f}/t")
    ax[0].set_title("Framework 3 by vol regime — is the signal a volatility hedge?")
    ax[0].set_ylabel("€ saved per tonne EUA (signal vs uniform)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, axis="y")

    # B: pre/post betas
    if not df_b.empty:
        x = np.arange(len(df_b))
        colors_b = ["#95a5a6" if w == "pre_2024" else "#27ae60" for w in df_b["window"]]
        ax[1].bar(x, df_b["beta_eua"], color=colors_b, alpha=0.85)
        ax[1].axhline(0, color="k", lw=0.5)
        labels = [f"{r['sector'].split()[0]}\n{r['window']}" for _, r in df_b.iterrows()]
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, fontsize=8)
        # significance markers
        for xi, (_, row) in zip(x, df_b.iterrows()):
            if row["sig"].strip():
                ax[1].text(xi, row["beta_eua"] + (0.005 if row["beta_eua"] > 0 else -0.005),
                           row["sig"], ha="center", fontsize=11, fontweight="bold")
        ax[1].set_title("Framework 2 shipping betas: did maritime ETS (Jan 2024) create a signal?")
        ax[1].set_ylabel("β on EUA weekly return")
        ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS / "subperiod_analysis.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/subperiod_analysis.png")


if __name__ == "__main__":
    main()
