"""
16_regime_switching.py — Markov-switching regression model of EUA returns.

The hypothesis this test formalizes:
    "The demand-side causal chain has time-varying explanatory power.
    In some periods (2020-2022), fundamentals dominate and IP + Stoxx
    momentum explain a substantial fraction of EUA monthly returns.
    In other periods (2014-2019, 2023+), policy dynamics dominate and
    fundamentals are essentially noise."

We fit a two-state Markov regression:
    r_eua_t = α(s_t) + β_ip(s_t) × z_ip_t + β_stx(s_t) × z_stx_t + ε_t(s_t)
where s_t ∈ {0, 1} is a latent regime state following a first-order
Markov chain with transition matrix P.

The model *endogenously* decides which observations belong to which
regime, based on which coefficient set fits better. This is the crucial
difference from ad-hoc rolling regression: no arbitrary window choice,
no look-ahead in the regime classification, and the transition matrix
tells us how persistent each regime is.

Three tests, in order:
    1. LR test: MS-2 vs single-regime OLS. Is the switching significant?
    2. Regime coefficients: does one regime show β_ip, β_stx > 0 with
       high significance while the other shows β ≈ 0?
    3. Regime path: do the estimated "fundamentals-active" periods
       coincide with the R² peaks we observed in Plot 01?

Output:
    data/msr_regime_probs.csv     monthly smoothed regime probabilities
    data/msr_regime_params.csv    regime-specific coefficients
    plots/regime_switching.png    4-panel diagnostic
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def build_monthly_panel(f: pd.DataFrame) -> pd.DataFrame:
    """Same construction as 11_report_plots.py — compute demand features at
    monthly frequency, robust to sparsity in daily EUA prints."""
    eua_m = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last()
    m = pd.DataFrame(index=eua_m.index)
    m["ret"] = np.log(eua_m / eua_m.shift(1))
    ip_m = f["ip_ea19"].resample("ME").last().ffill(limit=1)
    m["ip_yoy"] = ip_m.pct_change(12)
    m["stx_mom"] = np.log(f["stoxx50"] / f["stoxx50"].shift(20)).resample("ME").last()

    def z(s, w=24):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    m["z_ip"] = z(m["ip_yoy"])
    m["z_stx"] = z(m["stx_mom"])
    return m[["ret", "z_ip", "z_stx"]].dropna()


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet")
    m = build_monthly_panel(f)
    print(f"monthly panel: {len(m)} observations "
          f"{m.index.min().date()} → {m.index.max().date()}")

    y = m["ret"].values
    X = m[["z_ip", "z_stx"]].values

    # ─── Baseline: single-regime OLS ─────────────────────────────────
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    print(f"\nSingle-regime OLS (baseline):")
    print(f"  β_ip  = {ols.params[1]:+.4f}  t = {ols.tvalues[1]:+.2f}")
    print(f"  β_stx = {ols.params[2]:+.4f}  t = {ols.tvalues[2]:+.2f}")
    print(f"  R²    = {ols.rsquared:.3f}")
    print(f"  log-likelihood = {ols.llf:.2f}")

    # ─── Two-regime Markov switching ─────────────────────────────────
    # Switching intercept, slopes, AND variance — full state-dependence
    ms_model = MarkovRegression(
        y, k_regimes=2, exog=X,
        switching_variance=True,
    )
    try:
        ms_result = ms_model.fit(disp=False, maxiter=200)
    except Exception as e:
        print(f"\nMS-2 fit failed: {e}")
        return

    print(f"\nMarkov-switching 2-regime model:")
    print(f"  log-likelihood = {ms_result.llf:.2f}")

    # regime-specific params
    p = ms_result.params
    # statsmodels stores param names on the model, not the result wrapper
    try:
        param_names = ms_result.model.param_names
    except AttributeError:
        param_names = [f"param_{i}" for i in range(len(p))]
    print("\n  parameters:")
    for name, val in zip(param_names, p):
        print(f"    {name:40s} {val:+.4f}")

    # LR test: 2 × (LL_MS − LL_OLS) ~ χ² with df = added params
    lr_stat = 2 * (ms_result.llf - ols.llf)
    df = len(p) - len(ols.params)
    print(f"\nLR test MS-2 vs OLS:")
    print(f"  LR statistic = {lr_stat:.2f}  df = {df}")
    from scipy.stats import chi2
    lr_p = 1 - chi2.cdf(lr_stat, df)
    print(f"  p-value      = {lr_p:.4f}   {'(SIGNIFICANT)' if lr_p < 0.05 else '(not significant)'}")

    # smoothed regime probabilities — statsmodels sometimes drops the
    # first observation depending on version; align by shorter length.
    smoothed = ms_result.smoothed_marginal_probabilities
    if hasattr(smoothed, "columns"):
        p0 = np.asarray(smoothed.iloc[:, 0].values)
        p1 = np.asarray(smoothed.iloc[:, 1].values)
    else:
        arr = np.asarray(smoothed)
        # shape is either (k, T) or (T, k)
        if arr.shape[0] == 2:
            p0, p1 = arr[0], arr[1]
        else:
            p0, p1 = arr[:, 0], arr[:, 1]

    n_prob = len(p0)
    if n_prob != len(m):
        # trim m to match probability length (usually from the end)
        idx = m.index[-n_prob:]
        y_trim = y[-n_prob:]
        z_ip_trim = m["z_ip"].values[-n_prob:]
        z_stx_trim = m["z_stx"].values[-n_prob:]
    else:
        idx = m.index
        y_trim = y
        z_ip_trim = m["z_ip"].values
        z_stx_trim = m["z_stx"].values

    probs = pd.DataFrame({
        "prob_regime_0": p0,
        "prob_regime_1": p1,
        "ret": y_trim,
        "z_ip": z_ip_trim,
        "z_stx": z_stx_trim,
    }, index=idx)
    probs.index.name = "date"
    probs.to_csv(DATA / "msr_regime_probs.csv")

    # regime dominant assignment
    regime = (probs["prob_regime_1"] > 0.5).astype(int)

    # regime-specific descriptive stats
    for r in [0, 1]:
        mask = regime == r
        if mask.sum() < 20:
            continue
        sub_ret = probs.loc[mask, "ret"]
        sub_ip  = probs.loc[mask, "z_ip"]
        sub_stx = probs.loc[mask, "z_stx"]
        sub_frame = pd.DataFrame({"ret": sub_ret, "z_ip": sub_ip, "z_stx": sub_stx}).dropna()
        try:
            r_ols = sm.OLS(sub_frame["ret"],
                           sm.add_constant(sub_frame[["z_ip", "z_stx"]])).fit()
            print(f"\n  Regime {r} in-sample OLS (n={len(sub_frame)}):")
            print(f"    β_ip  = {r_ols.params['z_ip']:+.4f}  t = {r_ols.tvalues['z_ip']:+.2f}")
            print(f"    β_stx = {r_ols.params['z_stx']:+.4f}  t = {r_ols.tvalues['z_stx']:+.2f}")
            print(f"    R²    = {r_ols.rsquared:.3f}")
            print(f"    mean r_eua = {sub_ret.mean() * 12 * 100:+.1f}% ann.  "
                  f"vol = {sub_ret.std() * np.sqrt(12) * 100:.1f}%")
            # dates in this regime
            print(f"    period sample: {sub_ret.index[0].strftime('%Y-%m')} ... "
                  f"{sub_ret.index[-1].strftime('%Y-%m')}")
        except Exception as e:
            print(f"  Regime {r} OLS failed: {e}")

    # transition matrix
    print("\n  Transition matrix (row = from, col = to):")
    tp = ms_result.regime_transition[:, :, 0]
    print(f"    P(stay in 0) = {tp[0, 0]:.3f}   P(0→1) = {tp[0, 1]:.3f}")
    print(f"    P(1→0)       = {tp[1, 0]:.3f}   P(stay in 1) = {tp[1, 1]:.3f}")
    print(f"    Expected duration in regime 0: {1/(1-tp[0, 0]):.1f} months")
    print(f"    Expected duration in regime 1: {1/(1-tp[1, 1]):.1f} months")

    # ─── PLOTS ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # (1) EUA price
    eua_price = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last().reindex(probs.index)
    ax[0].plot(eua_price.index, eua_price.values, color="#1a1a1a", lw=1.5)
    ax[0].set_title("EUA auction clearing price (EUR/tCO₂)")
    ax[0].grid(alpha=0.3)

    # (2) Smoothed probability of the "high-vol / crisis-driven" regime.
    # We label regime 1 as the high-vol one based on the fitted variances.
    var0 = ms_result.params[[i for i, n in enumerate(param_names) if "sigma2[0]" in n][0]] if any("sigma2[0]" in n for n in param_names) else 0.001
    var1 = ms_result.params[[i for i, n in enumerate(param_names) if "sigma2[1]" in n][0]] if any("sigma2[1]" in n for n in param_names) else 0.001
    high_vol_regime = 1 if var1 > var0 else 0
    p_high_vol = probs[f"prob_regime_{high_vol_regime}"]
    ax[1].fill_between(p_high_vol.index, 0, p_high_vol, color="#e74c3c", alpha=0.55)
    ax[1].set_ylim(0, 1)
    ax[1].set_title(f"Smoothed probability of high-vol / crisis-driven regime (regime {high_vol_regime})")
    ax[1].axhline(0.5, ls="--", color="k", alpha=0.4)
    ax[1].grid(alpha=0.3)

    # highlight known events
    ax[1].axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2019-07-01"),
                  color="#e74c3c", alpha=0.10, label="MSR reform")
    ax[1].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"),
                  color="#f39c12", alpha=0.15, label="COVID crash")
    ax[1].axvspan(pd.Timestamp("2021-08-01"), pd.Timestamp("2022-12-01"),
                  color="#27ae60", alpha=0.10, label="gas crisis")
    ax[1].legend(loc="upper left", fontsize=9)

    # (3) IP YoY
    ax[2].plot(probs.index, probs["z_ip"], color="#c0392b", lw=1.3, label="z(IP YoY)")
    ax[2].plot(probs.index, probs["z_stx"], color="#2980b9", lw=1.0, alpha=0.7, label="z(Stoxx mom)")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("Demand-side features (z-scores)")
    ax[2].legend(loc="best", fontsize=9); ax[2].grid(alpha=0.3)

    # (4) EUA monthly returns colored by regime
    colors = ["#e74c3c" if r == high_vol_regime else "#3498db" for r in regime]
    ax[3].bar(probs.index, probs["ret"] * 100, color=colors, width=25, alpha=0.85)
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set_title(f"Monthly EUA returns coloured by regime "
                    f"(red = high-vol/crisis, blue = low-vol/quiet)")
    ax[3].set_ylabel("% monthly return")
    ax[3].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS / "regime_switching.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/regime_switching.png")

    # ─── SECOND FIGURE: intuition-building diagnostics ────────────────
    # Six panels answering:
    #   (1) WHEN does each regime happen? — timeline with shading
    #   (2) HOW BIG are returns in each regime? — vol comparison
    #   (3) DO IP fundamentals matter in each regime? — scatter + fit
    #   (4) DO EQUITIES matter in each regime? — scatter + fit
    #   (5) HOW LONG do regimes last? — persistence bar
    #   (6) WHICH REGIME AM I IN NOW? — recent probability path

    fig2, ax = plt.subplots(3, 2, figsize=(14, 12))

    low_vol_regime = 1 - high_vol_regime
    is_high = regime == high_vol_regime
    is_low  = regime == low_vol_regime

    low_returns  = probs.loc[is_low, "ret"]
    high_returns = probs.loc[is_high, "ret"]

    RED = "#e74c3c"; BLUE = "#3498db"

    # (1) EUA price with high-vol regime shading
    eua_price = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last().reindex(probs.index)
    ax[0,0].plot(eua_price.index, eua_price.values, color="#1a1a1a", lw=1.4, zorder=5)
    # shade high-vol periods
    in_high = False; start = None
    for d, h in zip(probs.index, is_high):
        if h and not in_high:
            start = d; in_high = True
        elif not h and in_high:
            ax[0,0].axvspan(start, d, color=RED, alpha=0.15, zorder=1)
            in_high = False
    if in_high:
        ax[0,0].axvspan(start, probs.index[-1], color=RED, alpha=0.15, zorder=1)
    ax[0,0].set_title("When is EUA in each regime?\n(red shading = high-vol / crisis regime)")
    ax[0,0].set_ylabel("EUA (EUR/tCO₂)")
    ax[0,0].grid(alpha=0.3)

    # (2) Return distributions by regime
    bins = np.linspace(-0.30, 0.35, 26)
    ax[0,1].hist(low_returns * 100, bins=bins*100, color=BLUE, alpha=0.65, label=f"low-vol regime (n={is_low.sum()})", edgecolor="white")
    ax[0,1].hist(high_returns * 100, bins=bins*100, color=RED, alpha=0.65, label=f"high-vol regime (n={is_high.sum()})", edgecolor="white")
    ax[0,1].axvline(low_returns.mean() * 100, color=BLUE, ls="--", lw=1.5, alpha=0.8)
    ax[0,1].axvline(high_returns.mean() * 100, color=RED, ls="--", lw=1.5, alpha=0.8)
    ax[0,1].set_title(f"How big are monthly returns in each regime?\nlow-vol σ={low_returns.std()*100:.1f}%  |  high-vol σ={high_returns.std()*100:.1f}%")
    ax[0,1].set_xlabel("Monthly EUA return (%)")
    ax[0,1].legend(); ax[0,1].grid(alpha=0.3, axis="y")

    # (3) IP YoY vs EUA return, coloured by regime, with within-regime regression
    _scatter_with_fit(ax[1,0], probs["z_ip"], probs["ret"], is_low, is_high,
                      "IP YoY (z-score)", "Monthly EUA return",
                      "Does industrial production predict EUA returns?", BLUE, RED)

    # (4) Stoxx momentum vs EUA return
    _scatter_with_fit(ax[1,1], probs["z_stx"], probs["ret"], is_low, is_high,
                      "Stoxx 50 momentum (z-score)", "Monthly EUA return",
                      "Does European equity sentiment predict EUA returns?", BLUE, RED)

    # (5) Persistence / transition
    tp = ms_result.regime_transition[:, :, 0]
    dur_low = 1 / (1 - tp[low_vol_regime, low_vol_regime]) if tp[low_vol_regime, low_vol_regime] < 1 else 100
    dur_high = 1 / (1 - tp[high_vol_regime, high_vol_regime]) if tp[high_vol_regime, high_vol_regime] < 1 else 100
    labels = ["low-vol\nregime", "high-vol\nregime"]
    durations = [dur_low, dur_high]
    colors = [BLUE, RED]
    ax[2,0].bar(labels, durations, color=colors, alpha=0.75)
    for i, d in enumerate(durations):
        ax[2,0].text(i, d + 0.3, f"{d:.1f} months", ha="center", fontsize=11, fontweight="bold")
    ax[2,0].set_ylabel("Expected duration (months)")
    ax[2,0].set_title("How long does each regime typically last?")
    ax[2,0].grid(alpha=0.3, axis="y")

    # (6) Recent regime probability
    recent = probs.iloc[-60:]  # last 5 years
    ax[2,1].fill_between(recent.index, 0, recent[f"prob_regime_{high_vol_regime}"],
                         color=RED, alpha=0.55, label="high-vol regime")
    ax[2,1].fill_between(recent.index, recent[f"prob_regime_{high_vol_regime}"], 1,
                         color=BLUE, alpha=0.35, label="low-vol regime")
    ax[2,1].axhline(0.5, ls="--", color="k", alpha=0.4)
    ax[2,1].set_ylim(0, 1)
    ax[2,1].set_title("Which regime is EUA in? (last 5 years)")
    ax[2,1].set_ylabel("smoothed probability")
    ax[2,1].legend(loc="best", fontsize=9); ax[2,1].grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(PLOTS / "regime_intuition.png", dpi=140)
    plt.close(fig2)
    print(f"wrote plots/regime_intuition.png and data/msr_regime_probs.csv")


def _scatter_with_fit(ax, x, y, mask_low, mask_high, xlabel, ylabel, title, blue, red):
    """Scatter of y on x, coloured by regime, with within-regime OLS lines."""
    ax.scatter(x[mask_low], y[mask_low] * 100, color=blue, alpha=0.6, s=30, label="low-vol")
    ax.scatter(x[mask_high], y[mask_high] * 100, color=red, alpha=0.6, s=30, label="high-vol")

    # regression lines per regime
    for mask, colour, name in [(mask_low, blue, "low"), (mask_high, red, "high")]:
        xs = x[mask].dropna()
        ys = y[mask].reindex(xs.index).dropna()
        common = xs.index.intersection(ys.index)
        if len(common) < 10:
            continue
        x_arr = xs.loc[common].values
        y_arr = ys.loc[common].values * 100
        try:
            b, a = np.polyfit(x_arr, y_arr, 1)
            xline = np.linspace(x_arr.min(), x_arr.max(), 20)
            ax.plot(xline, a + b * xline, color=colour, lw=2, ls="-",
                    label=f"{name} slope={b:+.2f}%/z")
        except Exception:
            pass

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel + " (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    main()
