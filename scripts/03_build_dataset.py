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
    for c in ["carbon_proxy_krbn", "carbon_proxy_grn", "ttf_gas_eur_mwh",
              "stoxx50", "wti_usd_bbl"]:
        if c in f:
            f[f"r_{c}"] = np.log(f[c]).diff()

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

    # if only the temperature-proxy is available, alias it to load_eu5_mw
    # so downstream features ("load_eu5_yoy") still get built.
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
