"""
13_sector_sensitivity.py — Framework 2.

Question this framework answers for Deltalinqs members:
    "How much does my sector's equity price move per €10/tonne EUA move,
    and how much of my sector's daily variation is *actually explained*
    by EUA vs by broader market factors?"

Method:
  1. Pull daily prices for one representative equity per sector present
     in the Rotterdam cluster (power, refining, chemicals, cement,
     steel, shipping).
  2. Compute weekly log-returns (daily is too noisy for EUA transmission).
  3. Regress sector returns on:
       - EUA weekly return              ← the coefficient we care about
       - Broad market return (Stoxx 50) ← control for common factor
       - TTF gas return                 ← control for energy price
  4. Report β_EUA, R² (with and without EUA), t-stat, sample size.
  5. Translate β into "expected sector-equity move per €10/tCO2 EUA change."

This is the analysis Deltalinqs's Programme Manager can quote to members:
"a €10 EUA move ≈ X% impact on your equity — with confidence Y%."

Output:
    data/sector_sensitivity.csv
    plots/sector_sensitivity.png
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

# Rotterdam-cluster representative equities per sector.
# One ticker per sector. If a ticker fails we skip cleanly.
SECTORS = {
    "Power (Uniper)":               "UN01.DE",
    "Power (RWE)":                  "RWE.DE",
    "Refining (Shell)":             "SHEL.L",
    "Refining (BP)":                "BP.L",
    "Chemicals (LyondellBasell)":   "LYB",
    "Chemicals (BASF)":             "BAS.DE",
    "Cement (HeidelbergMaterials)": "HEI.DE",
    "Cement (Holcim)":              "HOLN.SW",
    "Steel (ArcelorMittal)":        "MT.AS",
    "Steel (Salzgitter)":           "SZG.DE",
    "Shipping (Maersk)":            "MAERSK-B.CO",
    "Shipping (Hapag-Lloyd)":       "HLAG.DE",
}


def yfclose(ticker: str, start: str = "2014-01-01") -> pd.Series:
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_PRICE])
    print(f"loaded panel: {len(f)} rows, {f.index.min().date()} → {f.index.max().date()}")

    # Weekly-Friday resample of all necessary controls
    stoxx = f["stoxx50"].resample("W-FRI").last()
    ttf   = f["ttf_gas_eur_mwh"].resample("W-FRI").last()
    eua   = f[TARGET_PRICE].resample("W-FRI").last()

    r_stx = np.log(stoxx / stoxx.shift(1))
    r_ttf = np.log(ttf   / ttf.shift(1))
    r_eua = np.log(eua   / eua.shift(1))

    controls = pd.DataFrame({
        "r_eua":   r_eua,
        "r_stoxx": r_stx,
        "r_ttf":   r_ttf,
    }).dropna()

    results = []
    for sector, ticker in SECTORS.items():
        print(f"  fetch {ticker:15s}  → ", end="")
        p = yfclose(ticker)
        if p.empty:
            print("failed, skip")
            continue

        pw = p.resample("W-FRI").last()
        r_sec = np.log(pw / pw.shift(1)).dropna()

        # Align returns
        df = controls.join(r_sec.rename("r_sec"), how="inner").dropna()
        if len(df) < 100:
            print(f"only {len(df)} weeks, skip")
            continue

        # Two regressions: with and without EUA
        X_full = sm.add_constant(df[["r_eua", "r_stoxx", "r_ttf"]])
        X_ctrl = sm.add_constant(df[["r_stoxx", "r_ttf"]])
        y = df["r_sec"]

        m_full = sm.OLS(y, X_full).fit()
        m_ctrl = sm.OLS(y, X_ctrl).fit()

        beta_eua = m_full.params["r_eua"]
        t_eua    = m_full.tvalues["r_eua"]
        r2_full  = m_full.rsquared
        r2_ctrl  = m_ctrl.rsquared
        r2_delta = r2_full - r2_ctrl        # EUA's marginal contribution to R²

        # Translate: current EUA ~€78. €10 move = 12.8% weekly return.
        # Sector move per €10 EUA = β × 0.128
        current_eua = float(f[TARGET_PRICE].iloc[-1])
        eua_10_pct  = 10.0 / current_eua
        sec_move_per_10_eua = beta_eua * eua_10_pct

        results.append({
            "sector":            sector,
            "ticker":            ticker,
            "n_weeks":           len(df),
            "beta_eua":          beta_eua,
            "t_stat":            t_eua,
            "r2_with_eua":       r2_full,
            "r2_without_eua":    r2_ctrl,
            "r2_marginal_eua":   r2_delta,
            "sensitivity_pct_per_10eur":  sec_move_per_10_eua * 100,
        })
        stars = "***" if abs(t_eua) > 2.58 else "**" if abs(t_eua) > 1.96 else "*" if abs(t_eua) > 1.64 else " "
        print(f"n={len(df):4d}  β={beta_eua:+.3f} {stars}  R²Δ={r2_delta:+.3f}")

    if not results:
        print("\nNo sectors successfully processed — check network / yfinance access.")
        return

    df = pd.DataFrame(results)
    df.to_csv(DATA / "sector_sensitivity.csv", index=False)

    print()
    print("=" * 90)
    print("Framework 2 — Rotterdam cluster sector sensitivity to EUA")
    print("=" * 90)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print()
    print("Reading the table:")
    print("  beta_eua = weekly-return elasticity of sector equity to EUA return")
    print("  t_stat   = |t| > 1.96 → 95% significant, > 2.58 → 99%")
    print("  R² Δ     = incremental variance explained by EUA vs Stoxx+TTF only")
    print("  sens_pct_per_10eur = expected % move in sector equity per €10/tCO₂ move in EUA")
    print()
    print("Direct interpretation for a Rotterdam member:")
    print("  If sensitivity is +2%, a €10 EUA rise implies +2% expected weekly equity return")
    print("  If sensitivity is -2%, a €10 EUA rise implies -2% expected weekly equity return")

    # ─────────────  plot  ─────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    # Sensitivity bars — sorted
    df_sorted = df.sort_values("sensitivity_pct_per_10eur")
    colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in df_sorted["sensitivity_pct_per_10eur"]]
    ax[0].barh(df_sorted["sector"], df_sorted["sensitivity_pct_per_10eur"],
               color=colors, alpha=0.85)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_title("Sector equity sensitivity to a €10/tCO₂ EUA move")
    ax[0].set_xlabel("Expected sector return per €10 EUA (%)")
    ax[0].grid(alpha=0.3, axis="x")

    # Marginal R² bar
    df_sorted2 = df.sort_values("r2_marginal_eua")
    ax[1].barh(df_sorted2["sector"], df_sorted2["r2_marginal_eua"] * 100,
               color="#3498db", alpha=0.85)
    ax[1].set_title("Incremental variance in sector return explained by EUA")
    ax[1].set_xlabel("Δ R² (%) — after controlling for Stoxx + TTF")
    ax[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(PLOTS / "sector_sensitivity.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/sector_sensitivity.png and data/sector_sensitivity.csv")


if __name__ == "__main__":
    main()
