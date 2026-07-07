"""
18_regime_crossmarket.py — Cross-market validation of the two-regime structure.

Question: does the same MS-2 architecture recover a coherent two-regime
structure in *other* emissions-trading markets? Or is the EU ETS finding
specific to EU ETS institutional design?

Markets tested:
    - California CCA via KCCA ETF (KraneShares California Carbon Allowance)
    - UK ETS via KUKA/GRN.L proxy (verify availability)
    - RGGI (Regional Greenhouse Gas Initiative) via GRNSTRATEGY or similar

Setup per market:
    - Fetch daily prices from yfinance
    - Aggregate to monthly returns
    - Fetch appropriate demand and sentiment features:
        US: S&P 500 momentum (US_STX), US industrial production if possible
        UK: FTSE 100 momentum, UK IP if possible
    - Fit MS-2 with the two features + switching variance
    - Compare: does each market show the same two-regime pattern?
    - Report: regime coefficients, persistence, R² by regime

Output:
    data/crossmarket_summary.csv
    plots/crossmarket_regime_comparison.png
"""

from __future__ import annotations
from pathlib import Path
import warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


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


def build_market_panel(carbon_ticker: str, equity_ticker: str,
                       label: str) -> pd.DataFrame | None:
    """Build a monthly panel for one carbon market."""
    print(f"\n  fetching {label}: carbon={carbon_ticker}, equity={equity_ticker}")
    carbon = yfclose(carbon_ticker)
    equity = yfclose(equity_ticker)
    if carbon.empty:
        print(f"    {carbon_ticker} unavailable, skipping")
        return None
    if equity.empty:
        print(f"    {equity_ticker} unavailable, skipping")
        return None

    carbon_m = carbon.ffill(limit=7).resample("ME").last()
    equity_m = equity.ffill(limit=7).resample("ME").last()

    m = pd.DataFrame(index=carbon_m.index)
    m["ret"]     = np.log(carbon_m / carbon_m.shift(1))
    m["eq_mom"]  = np.log(equity_m / equity_m.shift(1))

    def z(s, w=24):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()
    m["z_eq"] = z(m["eq_mom"])

    m = m.dropna(subset=["ret", "z_eq"])
    print(f"    panel: {len(m)} monthly obs, "
          f"{m.index.min().date() if len(m) else 'n/a'} → "
          f"{m.index.max().date() if len(m) else 'n/a'}")
    return m


def fit_and_summarize(m: pd.DataFrame, label: str) -> dict:
    """Fit MS-2 on the panel; return regime-conditional statistics."""
    if m is None or len(m) < 40:
        return {"market": label, "status": "insufficient data"}
    y = m["ret"].values
    X = m[["z_eq"]].values
    try:
        model = MarkovRegression(y, k_regimes=2, exog=X, switching_variance=True)
        res = model.fit(disp=False, maxiter=300)
    except Exception as e:
        return {"market": label, "status": f"fit failed: {e}"}

    # single-regime baseline
    baseline = sm.OLS(y, sm.add_constant(X)).fit()

    # LR test
    from scipy.stats import chi2
    lr = 2 * (res.llf - baseline.llf)
    df_diff = len(res.params) - len(baseline.params)
    lr_p = 1 - chi2.cdf(lr, df_diff) if df_diff > 0 else 1

    # identify high-vol regime
    param_names = res.model.param_names
    sig0_idx = next((i for i, n in enumerate(param_names) if "sigma2[0]" in n), None)
    sig1_idx = next((i for i, n in enumerate(param_names) if "sigma2[1]" in n), None)
    var0 = res.params[sig0_idx] if sig0_idx is not None else 0
    var1 = res.params[sig1_idx] if sig1_idx is not None else 0
    high_regime = 1 if var1 > var0 else 0

    # smoothed probabilities
    smoothed = res.smoothed_marginal_probabilities
    if hasattr(smoothed, "iloc"):
        p0 = np.asarray(smoothed.iloc[:, 0].values)
        p1 = np.asarray(smoothed.iloc[:, 1].values)
    else:
        arr = np.asarray(smoothed)
        p0, p1 = (arr[0], arr[1]) if arr.shape[0] == 2 else (arr[:, 0], arr[:, 1])
    prob_high = p1 if high_regime == 1 else p0

    # transition matrix + persistence
    tp = res.regime_transition[:, :, 0]
    persistence_low  = tp[1 - high_regime, 1 - high_regime]
    persistence_high = tp[high_regime, high_regime]
    dur_low  = 1 / (1 - persistence_low)  if persistence_low  < 1 else 100
    dur_high = 1 / (1 - persistence_high) if persistence_high < 1 else 100

    # within-regime OLS (post-hoc labelling)
    idx = m.index[-len(prob_high):]
    reg_series = pd.Series((prob_high > 0.5).astype(int), index=idx)

    r0_mask = reg_series != high_regime
    r1_mask = reg_series == high_regime
    r0_m = m.reindex(idx).loc[r0_mask].dropna(subset=["ret", "z_eq"])
    r1_m = m.reindex(idx).loc[r1_mask].dropna(subset=["ret", "z_eq"])

    def _reg(sub):
        if len(sub) < 10:
            return {"beta_eq": np.nan, "t_eq": np.nan, "R2": np.nan,
                    "mean_ret_ann": np.nan, "vol_ann": np.nan, "n": 0}
        X_ = sm.add_constant(sub[["z_eq"]])
        y_ = sub["ret"]
        r = sm.OLS(y_, X_).fit()
        return {
            "beta_eq": r.params.get("z_eq", np.nan),
            "t_eq":    r.tvalues.get("z_eq", np.nan),
            "R2":      r.rsquared,
            "mean_ret_ann": sub["ret"].mean() * 12,
            "vol_ann":      sub["ret"].std() * np.sqrt(12),
            "n":            len(sub),
        }

    low_stats  = _reg(r0_m)
    high_stats = _reg(r1_m)

    return {
        "market":         label,
        "status":         "ok",
        "n":              len(m),
        "llf_baseline":   baseline.llf,
        "llf_ms2":        res.llf,
        "lr_stat":        lr,
        "lr_pvalue":      lr_p,
        "baseline_R2":    baseline.rsquared,
        "dur_low_months":  dur_low,
        "dur_high_months": dur_high,
        "low_R2":         low_stats["R2"],
        "low_beta_eq":    low_stats["beta_eq"],
        "low_t_eq":       low_stats["t_eq"],
        "low_vol_ann":    low_stats["vol_ann"],
        "low_n":          low_stats["n"],
        "high_R2":        high_stats["R2"],
        "high_beta_eq":   high_stats["beta_eq"],
        "high_t_eq":      high_stats["t_eq"],
        "high_vol_ann":   high_stats["vol_ann"],
        "high_n":         high_stats["n"],
        "prob_high_index": idx,
        "prob_high_series": prob_high,
    }


def main() -> None:
    # ─── EU ETS (from local panel) for reference ────────────────────
    print("=" * 80)
    print("Fitting MS-2 (one equity feature only, for like-for-like cross-market comparison)")
    print("=" * 80)

    # markets to test
    markets = [
        # label, carbon_ticker, equity_ticker
        ("EU_ETS (via KRBN proxy)",         "KRBN",    "^STOXX50E"),
        ("California CCA (via KCCA)",       "KCCA",    "^GSPC"),
        ("EU ETS pure (via GRN)",           "GRN",     "^STOXX50E"),
        ("Cross-market carbon basket",      "KRBN",    "^GSPC"),  # sanity check
    ]

    results = []
    panels_by_market = {}
    for label, carbon, equity in markets:
        m = build_market_panel(carbon, equity, label)
        if m is None:
            results.append({"market": label, "status": "no data"})
            continue
        stats = fit_and_summarize(m, label)
        results.append(stats)
        panels_by_market[label] = m
        if stats.get("status") != "ok":
            print(f"  {label}: {stats.get('status')}")
            continue
        print(f"  {label} ─────────────────────────────────────────────")
        print(f"    n={stats['n']}  llf_ms2={stats['llf_ms2']:.2f}  vs OLS {stats['llf_baseline']:.2f}")
        print(f"    LR stat={stats['lr_stat']:.2f}  p={stats['lr_pvalue']:.4f}")
        print(f"    baseline pooled R² = {stats['baseline_R2']:.3f}")
        print(f"    LOW-VOL regime: n={stats['low_n']} β_eq={stats['low_beta_eq']:+.4f}"
              f" (t={stats['low_t_eq']:+.2f})  R²={stats['low_R2']:.3f}  vol={stats['low_vol_ann']:.1%}")
        print(f"    HIGH-VOL regime: n={stats['high_n']} β_eq={stats['high_beta_eq']:+.4f}"
              f" (t={stats['high_t_eq']:+.2f})  R²={stats['high_R2']:.3f}  vol={stats['high_vol_ann']:.1%}")
        print(f"    persistence: low={stats['dur_low_months']:.1f}mo,"
              f" high={stats['dur_high_months']:.1f}mo")

    # Save summary
    summary_rows = []
    for r in results:
        if r.get("status") == "ok":
            summary_rows.append({k: v for k, v in r.items()
                                 if k not in ["prob_high_index", "prob_high_series"]})
    df_summary = pd.DataFrame(summary_rows)
    if not df_summary.empty:
        df_summary.to_csv(DATA / "crossmarket_summary.csv", index=False)

    # ─── PLOTS ──────────────────────────────────────────────────────
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        print("\nno markets fit successfully; no plot produced")
        return

    n = len(ok_results)
    fig, ax = plt.subplots(3, 1, figsize=(13, 4 * min(n, 3)))
    if n == 1:
        ax = [ax]

    # Panel 1: regime R² by market
    ax_a = plt.subplot(3, 1, 1)
    x = np.arange(len(ok_results))
    width = 0.28
    r0_r2 = [r["low_R2"] for r in ok_results]
    r1_r2 = [r["high_R2"] for r in ok_results]
    baseline_r2 = [r["baseline_R2"] for r in ok_results]
    labels = [r["market"] for r in ok_results]
    ax_a.bar(x - width, baseline_r2, width, color="#95a5a6", label="pooled OLS", alpha=0.85)
    ax_a.bar(x, r0_r2, width, color="#3498db", label="low-vol regime", alpha=0.85)
    ax_a.bar(x + width, r1_r2, width, color="#e74c3c", label="high-vol regime", alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_a.set_ylabel("Within-regime R²")
    ax_a.set_title("Cross-market R² comparison: does regime split improve fit universally?")
    ax_a.legend(fontsize=9); ax_a.grid(alpha=0.3, axis="y")

    # Panel 2: β_eq by regime, by market
    ax_b = plt.subplot(3, 1, 2)
    r0_b = [r["low_beta_eq"] for r in ok_results]
    r1_b = [r["high_beta_eq"] for r in ok_results]
    ax_b.bar(x - width/2, r0_b, width, color="#3498db",
             label="low-vol regime β", alpha=0.85)
    ax_b.bar(x + width/2, r1_b, width, color="#e74c3c",
             label="high-vol regime β", alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_b.axhline(0, color="k", lw=0.5)
    ax_b.set_ylabel("β on equity momentum")
    ax_b.set_title("Cross-market β_eq: is equity sensitivity concentrated in high-vol regimes?")
    # significance stars
    for i, r in enumerate(ok_results):
        for offset, key in [(-width/2, "low_t_eq"), (width/2, "high_t_eq")]:
            t = r.get(key, np.nan)
            if pd.notna(t):
                stars = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
                y = r.get("low_beta_eq" if offset < 0 else "high_beta_eq", 0)
                ax_b.text(i + offset, y + (0.005 if y >= 0 else -0.005),
                          stars, ha="center", fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=9); ax_b.grid(alpha=0.3, axis="y")

    # Panel 3: regime probability paths overlaid
    ax_c = plt.subplot(3, 1, 3)
    for r in ok_results:
        idx = r.get("prob_high_index")
        ser = r.get("prob_high_series")
        if idx is None or ser is None:
            continue
        ax_c.plot(idx, ser, lw=1.5, label=r["market"], alpha=0.85)
    ax_c.axhline(0.5, ls="--", color="k", alpha=0.4)
    ax_c.set_ylabel("P(high-vol regime)")
    ax_c.set_title("Cross-market regime paths — do carbon markets switch together?")
    ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "crossmarket_regime_comparison.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/crossmarket_regime_comparison.png and data/crossmarket_summary.csv")


if __name__ == "__main__":
    main()
