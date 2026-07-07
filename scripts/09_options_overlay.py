"""
09_options_overlay.py — translate the V1 signal into a monthly options rotation.

Why this exists
---------------
V1 (long-only delta-1) earns Sharpe ~1.0 on EUA. The asymmetric payoff
that drives the V1 finding — structural EUA bull, shorts have negative
expected value — is natural to express with options instead of spot:

  - long-call captures the upside with capped downside
  - cash-secured-put pays you to wait at a lower entry
  - covered-call collects premium when conviction is medium-high

The mapping below is a clean first-pass. It rolls one monthly structure
per regime. Maurizio can fork it and play with strike selection, DTE,
multi-leg spreads, etc.

Regime mapping (sampled at each monthly roll)
---------------------------------------------
  signal ≥ 0.7  →  long ATM call         (uncapped upside, premium at risk)
  0.4 ≤ s < 0.7 →  long spot (delta-1)    (medium conviction, no options)
  0.1 ≤ s < 0.4 →  cash-secured put @95%  (collect premium, willing to be assigned)
  s < 0.1       →  cash                   (no position)

Pricing
-------
EUA option chains aren't free historical data, so we mark them with
Black-Scholes using:
  IV = realized_vol_60d × 1.2     (typical RV→IV premium for EUA)
  r  = 0.01                       (10y average euro short rate)
  T  = 30 / 365                   (monthly expiry)

This is approximation, not truth. Real ICE EUA options have skew,
term-structure, and bid-ask spreads. But the relative comparison
(delta-1 vs options-overlay) is what we care about, and BSM is
adequate for that.

Outputs
-------
  data/options_overlay_monthly.csv   one row per roll, decision + PnL
  plots/options_vs_v1.png            equity curves side-by-side
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# ─── inputs ────────────────────────────────────────────────────────────
TARGET_RET   = "r_eua_eur_tco2"
TARGET_PRICE = "eua_eur_tco2"

R_F          = 0.01      # risk-free rate (10y avg euro short rate)
IV_MARKUP    = 1.20      # IV = RV60 × markup
DTE_DAYS     = 30        # monthly options
CSP_STRIKE_PCT = 0.95    # CSP strike at 95% of spot

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


# ─── data + V1 signal ─────────────────────────────────────────────────
def zscore(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

def v1_signal(f):
    z_ip  = zscore(f["ip_yoy"], 252)
    z_stx = zscore(np.log(f["stoxx50"] / f["stoxx50"].shift(20)), 252)
    return ((z_ip + z_stx) / 2).clip(lower=0.0, upper=1.0)


# ─── regime mapping ───────────────────────────────────────────────────
def pick_structure(signal: float) -> str:
    if signal >= 0.7:  return "long_call"
    if signal >= 0.4:  return "long_spot"
    if signal >= 0.1:  return "csp"
    return "cash"


# ─── one monthly trade ────────────────────────────────────────────────
def trade(structure: str, S0: float, S_T: float, sigma: float) -> tuple[float, dict]:
    """Return (PnL_per_1_eur_notional, details_dict).

    All PnLs normalized so notional = 1 €. For long-call, that means
    one call on €1 of underlying. Real-world this would be scaled by
    contract size and capital allocation.
    """
    T = DTE_DAYS / 365
    iv = sigma * IV_MARKUP

    if structure == "cash":
        return 0.0, {"struct": "cash", "premium": 0.0, "payoff": 0.0}

    if structure == "long_spot":
        # buy at S0, sell at S_T → PnL = (S_T - S0) / S0
        pnl = (S_T - S0) / S0
        return pnl, {"struct": "long_spot", "premium": 0.0, "payoff": pnl}

    if structure == "long_call":
        K = S0  # ATM
        premium = bs_call(S0, K, T, R_F, iv) / S0   # premium as fraction of notional
        payoff  = max(S_T - K, 0.0) / S0
        return payoff - premium, {"struct": "long_call", "K": K,
                                  "premium": premium, "payoff": payoff,
                                  "iv": iv}

    if structure == "csp":
        K = S0 * CSP_STRIKE_PCT
        premium = bs_put(S0, K, T, R_F, iv) / S0
        # we sold the put → we get premium, lose if S_T < K
        loss_if_assigned = max(K - S_T, 0.0) / S0
        return premium - loss_if_assigned, {"struct": "csp", "K": K,
                                            "premium": premium,
                                            "loss": loss_if_assigned,
                                            "iv": iv}
    raise ValueError(structure)


# ─── main loop ────────────────────────────────────────────────────────
def main():
    f = pd.read_parquet(DATA / "panel_features.parquet").dropna(subset=[TARGET_RET, TARGET_PRICE])
    sig = v1_signal(f)
    rv60 = f[TARGET_RET].rolling(60).std() * np.sqrt(252)   # annualized

    # last *actual trading day* of each month (not calendar month-end)
    month_ends = (
        f.groupby(pd.Grouper(freq="ME"))
         .apply(lambda g: g.index[-1] if len(g) else pd.NaT)
         .dropna()
         .tolist()
    )
    trades = []
    for i, d0 in enumerate(month_ends[:-1]):
        d1 = month_ends[i+1]
        if pd.isna(sig.loc[d0]) or pd.isna(rv60.loc[d0]):
            continue
        s_now = float(sig.loc[d0])
        S0 = float(f.loc[d0, TARGET_PRICE])
        S_T = float(f.loc[d1, TARGET_PRICE])
        sigma = float(rv60.loc[d0])

        structure = pick_structure(s_now)
        pnl, det = trade(structure, S0, S_T, sigma)
        trades.append({
            "open":  d0, "close": d1,
            "signal": s_now, "struct": structure,
            "S0": S0, "S_T": S_T, "rv60": sigma,
            "pnl": pnl,
            **{k: v for k, v in det.items() if k not in {"struct"}}
        })

    df = pd.DataFrame(trades).set_index("open")
    df["eq_options"] = (1 + df["pnl"]).cumprod()
    df.to_csv(DATA / "options_overlay_monthly.csv")

    # comparison: V1 delta-1 monthly returns over the same months
    v1_pos = sig.clip(lower=0, upper=1).shift(1)
    v1_pnl_d = (v1_pos * f[TARGET_RET]).dropna()
    v1_eq = (1 + v1_pnl_d).cumprod()
    v1_monthly = v1_eq.resample("ME").last().pct_change().dropna()

    # buy & hold
    bh_d = f[TARGET_RET].dropna()
    bh_monthly = (1 + bh_d).cumprod().resample("ME").last().pct_change().dropna()

    # stats
    def s(r):
        if len(r) < 2 or r.std() == 0:
            return dict(CAGR=np.nan, Sharpe=np.nan, Vol=np.nan, MaxDD=np.nan)
        eq = (1 + r).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        return dict(
            CAGR=(eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1,
            Sharpe=r.mean()/r.std() * np.sqrt(12),
            Vol=r.std() * np.sqrt(12),
            MaxDD=(eq/eq.cummax() - 1).min(),
        )

    print("\n=== Monthly comparison ===")
    print(f"window: {df.index.min().date()} → {df.index.max().date()}  ({len(df)} months)")
    rows = []
    rows.append(("V1_spot_delta1",   s(v1_monthly)))
    rows.append(("Options_overlay",  s(df["pnl"])))
    rows.append(("EUA_buy_and_hold", s(bh_monthly)))
    for name, r in rows:
        print(f"  {name:20s}  CAGR {r['CAGR']:+6.1%}  "
              f"Sharpe {r['Sharpe']:+5.2f}  "
              f"Vol {r['Vol']:5.1%}  "
              f"DD {r['MaxDD']:+6.0%}")

    # structure usage
    usage = df["struct"].value_counts()
    print("\n=== Structure usage (months) ===")
    print(usage.to_string())

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))

    # equity curves
    eq_opt = (1 + df["pnl"]).cumprod()
    eq_v1m = (1 + v1_monthly).cumprod()
    eq_bhm = (1 + bh_monthly).cumprod()
    eq_opt.plot(ax=ax[0], label="Options overlay", color="C0", lw=2)
    eq_v1m.plot(ax=ax[0], label="V1 delta-1", color="C1", lw=1.5)
    eq_bhm.plot(ax=ax[0], label="EUA buy & hold", color="grey", alpha=0.6, lw=1)
    ax[0].set_yscale("log")
    ax[0].set_title("V1 signal as options overlay vs delta-1 (monthly equity, log scale)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # structure timeline
    colors = {"cash":"#ccc", "csp":"#3498db", "long_spot":"#f39c12", "long_call":"#27ae60"}
    for struct, c in colors.items():
        sub = df[df["struct"] == struct]
        ax[1].scatter(sub.index, sub["signal"], s=40, color=c, label=struct, alpha=0.85)
    ax[1].set_title("Monthly structure choice (color) and V1 signal value")
    ax[1].set_ylabel("V1 signal")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].grid(alpha=0.3)
    ax[1].axhline(0.7, ls="--", color="k", alpha=0.4)
    ax[1].axhline(0.4, ls="--", color="k", alpha=0.4)
    ax[1].axhline(0.1, ls="--", color="k", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOTS / "options_vs_v1.png", dpi=120)
    plt.close(fig)
    print("\nwrote plots/options_vs_v1.png")
    print("wrote data/options_overlay_monthly.csv")


if __name__ == "__main__":
    main()
