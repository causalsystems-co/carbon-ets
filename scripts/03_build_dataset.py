"""
03_build_dataset.py — merge prices + emissions drivers to a daily panel.

Outputs:
  data/panel_daily.parquet     ← left-join on prices.index, ffill demand-side
  data/panel_features.parquet  ← engineered features (returns, z-scores, lags)

Maurizio: this is the central data contract. Everything downstream
(analysis, backtest, monitoring) reads panel_features. Add a column
here once and every script picks it up.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

EUA_COL = "carbon_proxy_krbn"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    # log returns
    for c in ["eua_eur_tco2", "carbon_proxy_krbn", "carbon_proxy_grn",
              "ttf_gas_eur_mwh", "stoxx50", "wti_usd_bbl"]:
        if c in f:
            f[f"r_{c}"] = np.log(f[c]).diff()

    # MSR-derived features (if TNAC merged)
    if "msr_tightness" in f:
        # Sign convention: LOW msr_tightness = tight supply = bullish EUA.
        # We negate so higher z_msr_supply_bull → bullish signal.
        f["z_msr_supply_bull"] = -f["msr_tightness"].rolling(504).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() else 0
        )
        # 1-year delta in TNAC (negative = tightening = bullish)
        f["tnac_delta_1y"] = f["tnac"] - f["tnac"].shift(252)
        f["z_tnac_tightening"] = -f["tnac_delta_1y"].rolling(504).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() else 0
        )

    # demand-side aggregate
    load_cols = [c for c in f.columns if c.startswith("load_") and c.endswith("_mw")]
    if load_cols:
        f["load_eu5_mw"] = f[load_cols].sum(axis=1, min_count=1)
        f["load_eu5_yoy"] = f["load_eu5_mw"] / f["load_eu5_mw"].shift(365) - 1

    if "ip_ea19" in f:
        f["ip_yoy"] = f["ip_ea19"] / f["ip_ea19"].shift(365) - 1

    # rolling z-scores for the headline drivers
    def zscore(s: pd.Series, win: int = 60) -> pd.Series:
        return (s - s.rolling(win).mean()) / s.rolling(win).std()

    for c in ["ttf_gas_eur_mwh", "load_eu5_mw", "hdd_frankfurt", "ip_ea19"]:
        if c in f:
            f[f"z_{c}_60"] = zscore(f[c], 60)

    # lagged versions of demand drivers — they lead EUA
    for c in ["load_eu5_yoy", "ip_yoy", "z_ttf_gas_eur_mwh_60", "z_hdd_frankfurt_60"]:
        if c in f:
            for lag in (5, 10, 20):
                f[f"{c}_lag{lag}"] = f[c].shift(lag)

    return f


def main() -> None:
    prices = pd.read_parquet(DATA / "prices_daily.parquet")
    drivers = pd.read_parquet(DATA / "emissions_drivers.parquet")

    # Prefer real EUA settlement (from 01b_fetch_eua_auctions.py) when present.
    # Falls back to KRBN proxy if the EEX file isn't there yet.
    eua_path = DATA / "eua_daily.parquet"
    if eua_path.exists():
        eua = pd.read_parquet(eua_path)
        prices = prices.join(eua, how="left")
        print(f"  using real EEX EUA auction prices ({len(eua)} rows, "
              f"{eua.index.min().date()} → {eua.index.max().date()})")
    else:
        print("  no eua_daily.parquet found — using KRBN proxy")

    # Merge annual TNAC / MSR data if 02c has been run.
    # TNAC is published every May 15 of year N+1 for year N, so we ffill from
    # each May-15 publication date rather than the year-end index.
    tnac_path = DATA / "tnac_annual.parquet"
    if tnac_path.exists():
        tnac = pd.read_parquet(tnac_path)
        # index at year-end → shift to the following May 15 (public release)
        tnac.index = tnac.index + pd.DateOffset(months=4, days=15)
        prices = prices.join(tnac[["tnac", "msr_tightness"]], how="left")
        prices[["tnac", "msr_tightness"]] = prices[["tnac", "msr_tightness"]].ffill()
        print(f"  merged TNAC/MSR ({len(tnac)} annual obs, "
              f"ffilled from May-15 release dates)")

    # if only the temperature-proxy is available, alias it to load_eu5_mw
    if "load_eu5_mw_proxy" in drivers.columns and "load_eu5_mw" not in drivers.columns:
        drivers = drivers.rename(columns={"load_eu5_mw_proxy": "load_eu5_mw"})

    panel = prices.join(drivers, how="left").sort_index()
    panel = panel.ffill(limit=7)
    panel.to_parquet(DATA / "panel_daily.parquet")
    print(f"panel_daily  rows={len(panel)}  cols={panel.shape[1]}")

    feats = engineer(panel)
    feats.to_parquet(DATA / "panel_features.parquet")
    print(f"panel_features  rows={len(feats)}  cols={feats.shape[1]}")
    print("EUA col:", EUA_COL, "non-null:", feats[EUA_COL].notna().sum())


if __name__ == "__main__":
    main()
