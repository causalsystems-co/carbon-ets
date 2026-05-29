"""
04_analyze_chain.py — does the chain actually hold?

Three checks, each saved as a PNG + a CSV:

(A) lead-lag cross-correlation: do load / IP / HDD / TTF lead EUA returns,
    and at what horizon?
(B) Granger causality: do the leading indicators carry incremental info
    about future EUA returns beyond EUA's own lags?
(C) regime split: does the chain look different in expansion vs. recession
    months (IP YoY > 0 vs. < 0)?

This is *diagnostics*, not signal generation — Maurizio looks at the
output and decides which features to take into the backtest.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

TARGET = "r_carbon_proxy_krbn"
DRIVERS = [
    "load_eu5_yoy",
    "ip_yoy",
    "z_hdd_frankfurt_60",
    "z_ttf_gas_eur_mwh_60",
    "r_stoxx50",
    "r_wti_usd_bbl",
]
MAX_LAG = 30


def lead_lag(f: pd.DataFrame, driver: str) -> pd.Series:
    """Corr(driver_{t-k}, target_t) for k = -MAX_LAG..MAX_LAG."""
    if driver not in f or TARGET not in f:
        return pd.Series(dtype=float)
    s = f[[driver, TARGET]].dropna()
    out = {}
    for k in range(-MAX_LAG, MAX_LAG + 1):
        out[k] = s[driver].shift(k).corr(s[TARGET])
    return pd.Series(out, name=driver)


def plot_lead_lag(f: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in DRIVERS:
        s = lead_lag(f, d)
        if not s.empty:
            ax.plot(s.index, s.values, label=d, alpha=0.85)
    ax.axvline(0, ls="--", color="k", alpha=0.4)
    ax.axhline(0, ls=":",  color="k", alpha=0.4)
    ax.set_xlabel("lag k  (driver leads EUA at k>0)")
    ax.set_ylabel("corr( driver_{t-k}, EUA_return_t )")
    ax.set_title("Lead-lag: candidate drivers vs. EUA daily return")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS / "leadlag.png", dpi=120)
    plt.close(fig)
    print("wrote plots/leadlag.png")


def granger(f: pd.DataFrame, driver: str, maxlag: int = 10) -> dict:
    s = f[[TARGET, driver]].dropna()
    if len(s) < 100:
        return {"driver": driver, "pmin": np.nan, "lag_at_min": np.nan}
    try:
        res = grangercausalitytests(s[[TARGET, driver]], maxlag=maxlag, verbose=False)
        ps = {k: v[0]["ssr_ftest"][1] for k, v in res.items()}
        kmin = min(ps, key=ps.get)
        return {"driver": driver, "pmin": ps[kmin], "lag_at_min": kmin}
    except Exception as e:
        return {"driver": driver, "pmin": np.nan, "lag_at_min": np.nan, "err": str(e)}


def regime_split(f: pd.DataFrame) -> pd.DataFrame:
    if "ip_yoy" not in f:
        return pd.DataFrame()
    f = f.dropna(subset=[TARGET, "ip_yoy"])
    f = f.assign(regime=np.where(f["ip_yoy"] >= 0, "expansion", "contraction"))
    grp = f.groupby("regime")[TARGET].agg(["mean", "std", "count"])
    grp["sharpe_ann"] = grp["mean"] / grp["std"] * np.sqrt(252)
    return grp


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet")

    plot_lead_lag(f)

    print("\nGranger causality on EUA daily returns:")
    gc = pd.DataFrame([granger(f, d) for d in DRIVERS])
    gc.to_csv(DATA / "granger.csv", index=False)
    print(gc.to_string(index=False))

    print("\nRegime split (expansion vs contraction by IP YoY):")
    rs = regime_split(f)
    rs.to_csv(DATA / "regime_split.csv")
    print(rs.to_string())


if __name__ == "__main__":
    main()
