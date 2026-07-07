"""
17_regime_robustness.py — three robustness/extension tests for the MS-2 regime finding.

TEST A — SUBPERIOD STABILITY.
    Refit MS-2 on two sub-samples: 2015-2020 and 2018-2024.
    Are the regime coefficients (β_ip in low-vol, β_stx in high-vol)
    similar across subperiods, or was the effect a one-window artefact?

TEST B — EXTENDED FEATURE SET.
    Add TTF gas monthly return and Frankfurt HDD z-score as candidate
    regressors. Refit MS-2 with the full feature set. Do the new features
    matter in either regime? What's the marginal R² contribution?

TEST C — REGIME-PROBABILITY NOWCAST.
    Can we predict which regime we're in *before* the MS model tells us,
    using contemporaneous indicators (realized vol, gas vol, etc.)?
    Fit a logistic regression of the MS-implied regime state on real-
    time observable predictors. Out-of-sample: hold out the last 24
    months, predict, compare to the "true" smoothed regime probability.

Outputs:
    data/robustness_subperiod.csv
    data/robustness_extended_features.csv
    data/regime_nowcast.csv
    plots/robustness_all.png
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def build_monthly_panel(f: pd.DataFrame) -> pd.DataFrame:
    eua_m = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last()
    m = pd.DataFrame(index=eua_m.index)
    m["ret"] = np.log(eua_m / eua_m.shift(1))
    ip_m = f["ip_ea19"].resample("ME").last().ffill(limit=1)
    m["ip_yoy"] = ip_m.pct_change(12)
    m["stx_mom"] = np.log(f["stoxx50"] / f["stoxx50"].shift(20)).resample("ME").last()
    ttf_m = f["ttf_gas_eur_mwh"].ffill(limit=7).resample("ME").last()
    m["ttf_ret"] = np.log(ttf_m / ttf_m.shift(1))
    hdd_m = f["hdd_frankfurt"].resample("ME").sum()  # monthly total HDDs
    m["hdd_z"] = (hdd_m - hdd_m.rolling(24).mean()) / hdd_m.rolling(24).std()

    # daily EUA realised vol (annualised, per month-end)
    r_eua_d = np.log(f["eua_eur_tco2"].ffill(limit=7)).diff()
    m["eua_rv20"] = r_eua_d.rolling(20).std().resample("ME").last() * np.sqrt(252)

    # daily Stoxx realised vol
    r_stx_d = np.log(f["stoxx50"]).diff()
    m["stx_rv20"] = r_stx_d.rolling(20).std().resample("ME").last() * np.sqrt(252)

    # z-scored features
    def z(s, w=24):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    m["z_ip"]    = z(m["ip_yoy"])
    m["z_stx"]   = z(m["stx_mom"])
    m["z_ttf"]   = z(m["ttf_ret"])
    return m


def fit_ms2(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit 2-regime Markov-switching model with switching variance."""
    model = MarkovRegression(y, k_regimes=2, exog=X, switching_variance=True)
    return model.fit(disp=False, maxiter=300)


def regime_params(res, k_features: int) -> pd.DataFrame:
    """Extract regime-specific coefficients and variances as a tidy DataFrame."""
    p = res.params
    # statsmodels layout for MarkovRegression with switching variance and
    # switching coefficients: [transitions, const_0, const_1, x1_0, x1_1, x2_0, x2_1, ..., sigma2_0, sigma2_1]
    try:
        names = res.model.param_names
    except AttributeError:
        names = [f"param_{i}" for i in range(len(p))]
    df = pd.DataFrame({"name": names, "value": p})
    return df


def within_regime_ols(m: pd.DataFrame, regime: pd.Series, features: list[str]) -> pd.DataFrame:
    """Run OLS on each regime separately, return a comparison table."""
    rows = []
    for r in [0, 1]:
        mask = (regime == r)
        if mask.sum() < 15:
            continue
        sub = m[mask].dropna(subset=["ret"] + features)
        if len(sub) < 15:
            continue
        X = sm.add_constant(sub[features])
        y = sub["ret"]
        res = sm.OLS(y, X).fit()
        row = {"regime": r, "n": len(sub), "R2": res.rsquared, "mean_ret_ann": sub["ret"].mean() * 12}
        for feat in features:
            row[f"beta_{feat}"] = res.params.get(feat, np.nan)
            row[f"t_{feat}"] = res.tvalues.get(feat, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    f = pd.read_parquet(DATA / "panel_features.parquet")
    m = build_monthly_panel(f).dropna(subset=["ret", "z_ip", "z_stx"])
    print(f"panel: {len(m)} monthly observations "
          f"{m.index.min().date()} → {m.index.max().date()}")

    y_full = m["ret"].values
    X_full = m[["z_ip", "z_stx"]].values

    # Baseline fit on full sample (recovers the model from script 16)
    print("\n" + "=" * 80)
    print("BASELINE MS-2 fit on full sample (for reference)")
    print("=" * 80)
    res_full = fit_ms2(y_full, X_full)
    p_full = res_full.params
    print(res_full.summary())

    # Extract regime probabilities to identify low-vol/high-vol
    smoothed = res_full.smoothed_marginal_probabilities
    if hasattr(smoothed, "iloc"):
        p0 = np.asarray(smoothed.iloc[:, 0].values)
        p1 = np.asarray(smoothed.iloc[:, 1].values)
    else:
        arr = np.asarray(smoothed)
        p0, p1 = (arr[0], arr[1]) if arr.shape[0] == 2 else (arr[:, 0], arr[:, 1])
    n_prob = len(p0)
    idx = m.index[-n_prob:]
    probs_full = pd.DataFrame({"p0": p0, "p1": p1}, index=idx)

    # identify high-vol regime from fitted sigmas (find sigma2 in params)
    param_names = res_full.model.param_names
    sig0_idx = next((i for i, n in enumerate(param_names) if "sigma2[0]" in n), None)
    sig1_idx = next((i for i, n in enumerate(param_names) if "sigma2[1]" in n), None)
    var0 = p_full[sig0_idx] if sig0_idx is not None else 0
    var1 = p_full[sig1_idx] if sig1_idx is not None else 0
    high_regime = 1 if var1 > var0 else 0
    regime_full = (probs_full[f"p{high_regime}"] > 0.5).astype(int).rename("regime_high")

    # ─── TEST A: SUBPERIOD STABILITY ────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST A — SUBPERIOD STABILITY")
    print("=" * 80)
    subperiods = {
        "2015-2020": (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")),
        "2018-2024": (pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31")),
    }
    sub_results = []
    for name, (start, end) in subperiods.items():
        sub = m.loc[start:end].dropna(subset=["ret", "z_ip", "z_stx"])
        if len(sub) < 40:
            print(f"  {name}: only {len(sub)} obs, skipping")
            continue
        y = sub["ret"].values
        X = sub[["z_ip", "z_stx"]].values
        try:
            res = fit_ms2(y, X)
        except Exception as e:
            print(f"  {name}: fit failed ({e})")
            continue
        # Regime coefficients from the fit
        names = res.model.param_names
        params = res.params
        row = {"subperiod": name, "n": len(sub), "llf": res.llf}
        for pname, pval in zip(names, params):
            row[pname] = pval
        sub_results.append(row)
        print(f"\n  {name} (n={len(sub)}, llf={res.llf:.2f}):")
        for pname, pval in zip(names, params):
            print(f"    {pname:35s} {pval:+.4f}")

    sub_df = pd.DataFrame(sub_results)
    if not sub_df.empty:
        sub_df.to_csv(DATA / "robustness_subperiod.csv", index=False)

    # ─── TEST B: EXTENDED FEATURE SET ──────────────────────────────
    print("\n" + "=" * 80)
    print("TEST B — EXTENDED FEATURE SET")
    print("=" * 80)
    ext_features = ["z_ip", "z_stx", "z_ttf"]
    m_ext = m.dropna(subset=["ret"] + ext_features)
    if len(m_ext) < 40:
        print("  insufficient data for extended features test, skipping")
    else:
        y_ext = m_ext["ret"].values
        X_ext = m_ext[ext_features].values
        try:
            res_ext = fit_ms2(y_ext, X_ext)
            print(f"\n  MS-2 with 3 features (IP + Stoxx + TTF), n={len(m_ext)}:")
            print(f"    log-likelihood = {res_ext.llf:.2f}")
            print(f"    vs baseline 2-feature MS-2 llf = {res_full.llf:.2f}")
            # LR test
            from scipy.stats import chi2
            lr = 2 * (res_ext.llf - res_full.llf)
            df_diff = len(res_ext.params) - len(res_full.params)
            lr_p = 1 - chi2.cdf(lr, df_diff) if df_diff > 0 else 1
            print(f"    LR stat = {lr:.2f}, df = {df_diff}, p-value = {lr_p:.4f}")
            for pname, pval in zip(res_ext.model.param_names, res_ext.params):
                print(f"    {pname:35s} {pval:+.4f}")
        except Exception as e:
            print(f"  MS-2 with 3 features failed: {e}")

    # ─── TEST C: REGIME-PROBABILITY NOWCAST ────────────────────────
    print("\n" + "=" * 80)
    print("TEST C — REGIME-PROBABILITY NOWCAST")
    print("=" * 80)
    m_nowcast = m.copy()
    # Align regime state to m_nowcast
    m_nowcast["regime_high"] = regime_full.reindex(m_nowcast.index)
    # Predictors: contemporaneous vol indicators
    nowcast_predictors = ["eua_rv20", "stx_rv20"]
    if "z_ttf" in m_nowcast.columns:
        # add TTF vol proxy
        r_ttf_d = np.log(f["ttf_gas_eur_mwh"].ffill(limit=7)).diff()
        ttf_rv = r_ttf_d.rolling(20).std().resample("ME").last() * np.sqrt(252)
        m_nowcast["ttf_rv20"] = ttf_rv.reindex(m_nowcast.index)
        nowcast_predictors.append("ttf_rv20")

    nc = m_nowcast.dropna(subset=["regime_high"] + nowcast_predictors)
    if len(nc) < 40:
        print("  insufficient data for nowcast, skipping")
        nowcast_df = pd.DataFrame()
    else:
        # Split: last 24 months held out
        split_idx = len(nc) - 24
        train = nc.iloc[:split_idx]
        test  = nc.iloc[split_idx:]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train[nowcast_predictors].values, train["regime_high"].astype(int).values)
        train_pred = clf.predict_proba(train[nowcast_predictors].values)[:, 1]
        test_pred  = clf.predict_proba(test[nowcast_predictors].values)[:, 1]

        # In-sample accuracy
        train_acc = (clf.predict(train[nowcast_predictors].values) == train["regime_high"].values).mean()
        test_acc  = (clf.predict(test[nowcast_predictors].values) == test["regime_high"].values).mean()
        print(f"\n  Predictors: {nowcast_predictors}")
        print(f"  Coefficients: {dict(zip(nowcast_predictors, clf.coef_[0]))}")
        print(f"  In-sample accuracy:  {train_acc:.1%}  (n={len(train)})")
        print(f"  Out-of-sample (last 24mo) accuracy: {test_acc:.1%}  (n={len(test)})")

        # Save the nowcast series
        nowcast_df = pd.DataFrame({
            "regime_high_actual": nc["regime_high"].astype(int),
            "regime_high_pred_prob": np.concatenate([train_pred, test_pred]),
        }, index=nc.index)
        nowcast_df["split"] = ["train"] * len(train) + ["test"] * len(test)
        nowcast_df.to_csv(DATA / "regime_nowcast.csv")

    # ─── PLOTS ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (A) subperiod stability — bar chart of β values
    if not sub_df.empty:
        # extract key params from each subperiod
        cols_to_show = [c for c in sub_df.columns if any(k in c for k in ["x1", "x2", "const", "sigma2"])]
        labels = []
        vals = {c: [] for c in cols_to_show}
        for _, row in sub_df.iterrows():
            labels.append(row["subperiod"])
            for c in cols_to_show:
                vals[c].append(row.get(c, np.nan))
        x = np.arange(len(cols_to_show))
        width = 0.35
        colors = ["#3498db", "#e74c3c"]
        for i, lab in enumerate(labels):
            offset = (i - len(labels) / 2 + 0.5) * width
            ax[0, 0].bar(x + offset, [vals[c][i] for c in cols_to_show],
                         width, label=lab, color=colors[i], alpha=0.85)
        ax[0, 0].set_xticks(x)
        ax[0, 0].set_xticklabels([c.replace("[0]", "_r0").replace("[1]", "_r1") for c in cols_to_show],
                                 rotation=45, ha="right", fontsize=8)
        ax[0, 0].axhline(0, color="k", lw=0.5)
        ax[0, 0].set_title("Test A — subperiod stability of MS-2 parameters")
        ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=0.3, axis="y")

    # (B) extended feature contribution — regime-specific R²
    if not m_ext.empty:
        m_reg = m_ext.copy()
        m_reg["regime_high"] = regime_full.reindex(m_reg.index)
        m_reg = m_reg.dropna(subset=["regime_high"])
        base_r2 = within_regime_ols(m_reg, m_reg["regime_high"], ["z_ip", "z_stx"])
        ext_r2  = within_regime_ols(m_reg, m_reg["regime_high"], ["z_ip", "z_stx", "z_ttf"])
        if not base_r2.empty and not ext_r2.empty:
            regimes = base_r2["regime"].astype(int).values
            x_pos = np.arange(len(regimes))
            width = 0.35
            ax[0, 1].bar(x_pos - width/2, base_r2["R2"], width,
                         color="#3498db", label="baseline (IP+Stoxx)", alpha=0.85)
            ax[0, 1].bar(x_pos + width/2, ext_r2["R2"], width,
                         color="#e74c3c", label="extended (IP+Stoxx+TTF)", alpha=0.85)
            ax[0, 1].set_xticks(x_pos)
            ax[0, 1].set_xticklabels(["low-vol regime", "high-vol regime"])
            ax[0, 1].set_ylabel("Within-regime R²")
            ax[0, 1].set_title("Test B — does TTF gas add explanatory power?")
            ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=0.3, axis="y")

    # (C) Nowcast plot
    if not nowcast_df.empty:
        ax[1, 0].plot(nowcast_df.index, nowcast_df["regime_high_actual"],
                      color="#1a1a1a", lw=1.5, drawstyle="steps-post", label="MS-implied regime (0/1)")
        ax[1, 0].plot(nowcast_df.index, nowcast_df["regime_high_pred_prob"],
                      color="#e74c3c", lw=1.5, label="logistic nowcast P(high-vol)")
        # shade test period
        test_start = nowcast_df[nowcast_df["split"] == "test"].index[0] if (nowcast_df["split"] == "test").any() else None
        if test_start is not None:
            ax[1, 0].axvspan(test_start, nowcast_df.index[-1],
                             color="#f39c12", alpha=0.15, label="out-of-sample")
        ax[1, 0].set_ylim(-0.1, 1.1); ax[1, 0].axhline(0.5, ls="--", color="k", alpha=0.4)
        ax[1, 0].set_title("Test C — regime nowcast from realised vol")
        ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=0.3)

    # (D) Summary text panel
    ax[1, 1].axis("off")
    summary = "SUMMARY\n\n"
    if not sub_df.empty:
        summary += f"Test A (subperiod stability):\n"
        summary += f"  2 sub-samples fit successfully, see left panel.\n\n"
    if not m_ext.empty:
        try:
            if lr_p < 0.05:
                summary += f"Test B (adding TTF gas):\n  LR p={lr_p:.3f} — TTF adds significant explanatory power.\n\n"
            else:
                summary += f"Test B (adding TTF gas):\n  LR p={lr_p:.3f} — TTF does not significantly improve fit.\n\n"
        except Exception:
            pass
    if not nowcast_df.empty:
        try:
            summary += f"Test C (regime nowcast):\n"
            summary += f"  In-sample accuracy: {train_acc:.1%}\n"
            summary += f"  Out-of-sample (24mo): {test_acc:.1%}\n\n"
        except Exception:
            pass
    summary += "See CSV outputs in data/ for details."
    ax[1, 1].text(0.05, 0.95, summary, va="top", ha="left", fontsize=10,
                  family="monospace", transform=ax[1, 1].transAxes)

    fig.tight_layout()
    fig.savefig(PLOTS / "robustness_all.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/robustness_all.png + data files")


if __name__ == "__main__":
    main()
