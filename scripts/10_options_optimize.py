"""
10_options_optimize.py — sweep of three improvements on top of 09_options_overlay.

Variants tested:
  M0 — original monthly, narrow CSP, pure long-call           (baseline from 09)
  M1 — biweekly rolls (DTE=14), narrow CSP, pure long-call
  M2 — monthly rolls, widened CSP band (0.05-0.5), pure long-call
  M3 — monthly rolls, narrow CSP, covered-call (instead of long-call) at s≥0.7
  M4 — biweekly + widened CSP + covered-call (all three combined)

For each, we report CAGR / Sharpe / Vol / MaxDD and structure usage. The
hypothesis: M4 dominates M0 on both Sharpe and CAGR. M1 alone should
recover most of the gap from the monthly-sampling-loses-signal problem.

Outputs:
  data/options_variants.csv
  plots/options_variants.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

TARGET_RET   = "r_eua_eur_tco2"
TARGET_PRICE = "eua_eur_tco2"

R_F        = 0.01
IV_MARKUP  = 1.20
CSP_STRIKE_PCT = 0.95
CC_STRIKE_PCT  = 1.05   # covered-call short strike: 5% OTM


# ─── Black-Scholes ────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─── V1 signal ────────────────────────────────────────────────────────
def zscore(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

def v1_signal(f):
    z_ip  = zscore(f["ip_yoy"], 252)
    z_stx = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)
    return ((z_ip + z_stx) / 2).clip(lower=0.0, upper=1.0)


# ─── variant configs ──────────────────────────────────────────────────
@dataclass
class VariantCfg:
    name: str
    dte_days: int
    roll_freq: str       # pandas offset alias: "ME" or "2W-FRI"
    csp_lo: float        # signal threshold to enter CSP
    csp_hi: float        # signal threshold above which we go spot/options instead
    use_covered_call: bool   # True → use CC at high signal instead of pure long call


VARIANTS = [
    VariantCfg("M0_original",        30, "ME",     0.1,  0.4,  False),
    VariantCfg("M1_biweekly",        14, "2W-FRI", 0.1,  0.4,  False),
    VariantCfg("M2_wider_csp",       30, "ME",     0.05, 0.5,  False),
    VariantCfg("M3_covered_call",    30, "ME",     0.1,  0.4,  True),
    VariantCfg("M4_all_three",       14, "2W-FRI", 0.05, 0.5,  True),
]


# ─── structure mapping + pricing ──────────────────────────────────────
def pick_structure(signal: float, cfg: VariantCfg) -> str:
    if signal >= 0.7:  return "covered_call" if cfg.use_covered_call else "long_call"
    if signal >= cfg.csp_hi:  return "long_spot"
    if signal >= cfg.csp_lo:  return "csp"
    return "cash"


def trade(structure: str, S0: float, S_T: float, sigma: float, dte_days: int) -> tuple[float, str]:
    T = dte_days / 365
    iv = sigma * IV_MARKUP

    if structure == "cash":
        return 0.0, "cash"

    if structure == "long_spot":
        return (S_T - S0) / S0, "long_spot"

    if structure == "long_call":
        K = S0
        prem = bs_call(S0, K, T, R_F, iv) / S0
        payoff = max(S_T - K, 0.0) / S0
        return payoff - prem, "long_call"

    if structure == "covered_call":
        # long the stock + short OTM call
        stock_pnl = (S_T - S0) / S0
        K = S0 * CC_STRIKE_PCT
        prem = bs_call(S0, K, T, R_F, iv) / S0
        short_call_pnl = prem - max(S_T - K, 0.0) / S0
        return stock_pnl + short_call_pnl, "covered_call"

    if structure == "csp":
        K = S0 * CSP_STRIKE_PCT
        prem = bs_put(S0, K, T, R_F, iv) / S0
        loss = max(K - S_T, 0.0) / S0
        return prem - loss, "csp"

    raise ValueError(structure)


# ─── per-variant backtest ─────────────────────────────────────────────
def run_variant(f, sig, rv60, cfg: VariantCfg) -> pd.DataFrame:
    # roll points = last actual trading day in each window
    roll_points = (
        f.groupby(pd.Grouper(freq=cfg.roll_freq))
         .apply(lambda g: g.index[-1] if len(g) else pd.NaT)
         .dropna()
         .tolist()
    )
    trades = []
    for i, d0 in enumerate(roll_points[:-1]):
        d1 = roll_points[i+1]
        if pd.isna(sig.loc[d0]) or pd.isna(rv60.loc[d0]):
            continue
        s_now = float(sig.loc[d0])
        S0 = float(f.loc[d0, TARGET_PRICE])
        S_T = float(f.loc[d1, TARGET_PRICE])
        sigma = float(rv60.loc[d0])
        structure = pick_structure(s_now, cfg)
        pnl, _ = trade(structure, S0, S_T, sigma, cfg.dte_days)
        trades.append({"open": d0, "close": d1, "signal": s_now,
                       "struct": structure, "pnl": pnl})
    return pd.DataFrame(trades).set_index("open")


# ─── stats ────────────────────────────────────────────────────────────
def stats(returns: pd.Series, periods_per_year: float) -> dict:
    if len(returns) < 2 or returns.std() == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, Vol=np.nan, MaxDD=np.nan)
    eq = (1 + returns).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(
        CAGR=(eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1,
        Sharpe=returns.mean() / returns.std() * np.sqrt(periods_per_year),
        Vol=returns.std() * np.sqrt(periods_per_year),
        MaxDD=(eq / eq.cummax() - 1).min(),
    )


def main():
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_RET, TARGET_PRICE])
    sig = v1_signal(f)
    rv60 = f[TARGET_RET].rolling(60).std() * np.sqrt(252)

    results = []
    eqs = {}
    usages = {}

    print(f"{'variant':<25s} {'rolls':>6s} {'CAGR':>7s} {'Sharpe':>7s} {'Vol':>6s} {'MaxDD':>7s}")
    print("─" * 70)

    for cfg in VARIANTS:
        df = run_variant(f, sig, rv60, cfg)
        ppy = 252 / cfg.dte_days
        s = stats(df["pnl"], ppy)
        s["Name"] = cfg.name
        s["Rolls"] = len(df)
        results.append(s)
        eqs[cfg.name] = (1 + df["pnl"]).cumprod()
        usages[cfg.name] = df["struct"].value_counts()
        print(f"{cfg.name:<25s} {len(df):>6d} "
              f"{s['CAGR']:>+6.1%} {s['Sharpe']:>+6.2f} "
              f"{s['Vol']:>5.1%} {s['MaxDD']:>+6.0%}")

    # V1 spot reference at the same monthly cadence
    v1_pos = sig.clip(lower=0, upper=1).shift(1)
    v1_d = (v1_pos * f[TARGET_RET]).dropna()
    v1_eq_d = (1 + v1_d).cumprod()
    v1_m = v1_eq_d.resample("ME").last().pct_change().dropna()
    s_v1 = stats(v1_m, 12)
    s_v1["Name"] = "V1_spot_monthly"
    s_v1["Rolls"] = len(v1_m)
    results.append(s_v1)
    eqs["V1_spot_monthly"] = (1 + v1_m).cumprod()
    print(f"{'V1_spot_monthly':<25s} {len(v1_m):>6d} "
          f"{s_v1['CAGR']:>+6.1%} {s_v1['Sharpe']:>+6.2f} "
          f"{s_v1['Vol']:>5.1%} {s_v1['MaxDD']:>+6.0%}")

    df_res = pd.DataFrame(results).set_index("Name").sort_values("Sharpe", ascending=False)
    df_res.to_csv(DATA / "options_variants.csv")
    print("\nranked by Sharpe:")
    print(df_res[["CAGR","Sharpe","Vol","MaxDD","Rolls"]].to_string())

    print("\nstructure usage per variant:")
    for name in [v.name for v in VARIANTS]:
        print(f"  {name}: {dict(usages[name])}")

    # plot
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {"M0_original":"#888", "M1_biweekly":"#3498db",
               "M2_wider_csp":"#9b59b6", "M3_covered_call":"#27ae60",
               "M4_all_three":"#e74c3c", "V1_spot_monthly":"#f39c12"}
    for name, eq in eqs.items():
        ax.plot(eq.index, eq.values, label=name, color=palette.get(name), lw=1.6, alpha=0.9)
    ax.set_yscale("log")
    ax.set_title("Options-overlay variants on V1 signal (equity, log scale)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "options_variants.png", dpi=120)
    plt.close(fig)
    print("\nwrote plots/options_variants.png")
    print("wrote data/options_variants.csv")

    best = df_res.index[0]
    print(f"\nbest variant: {best}")
    print(f"  CAGR    {df_res.loc[best,'CAGR']:+.1%}")
    print(f"  Sharpe  {df_res.loc[best,'Sharpe']:+.2f}")
    print(f"  Vol     {df_res.loc[best,'Vol']:.1%}")
    print(f"  MaxDD   {df_res.loc[best,'MaxDD']:.0%}")


if __name__ == "__main__":
    main()
