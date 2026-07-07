"""
19_regime_diagnosis.py — What does the MS-2 regime state actually capture?

The problem from script 17 test C: the smoothed regime probability from
the MS-2 model can't be predicted from realised vol alone (52% in-sample,
50% out-of-sample). But the finding of two-regime structure with clean
within-regime R² is real. So what is the regime state actually clustering on?

Three competing hypotheses:
  (H1) Vol clustering. Regime state = high-vol vs low-vol months, full
       stop. Then the "explanation" is trivial: vol clustering + higher
       vol months just have less signal-to-noise from any feature.
  (H2) Macro co-movement. Regime state = periods when EUA co-moves
       heavily with broad equity/macro variables vs periods when it
       trades on its own dynamics. That's a "macro correlation cluster"
       story — meaningful, but not policy-related.
  (H3) Policy pressure. Regime state = periods dominated by pending or
       recent MSR/CBAM/ETS2 policy news vs periods where fundamentals
       carry weight. This would validate the boundary-condition story
       we've been telling.

This script tests all three by regressing the smoothed regime probability
on candidate predictors from each hypothesis family. The best-explaining
variables tell us what the regime actually is.

Method:
  1. Refit MS-2 (baseline model with IP + Stoxx z-scores)
  2. Extract smoothed regime probability
  3. Build predictor panel with three families:
        VOL: EUA_rv20, EUA_rv60, Stoxx_rv20, TTF_rv20
        MACRO: |Stoxx return|, |EUA return|, EUA-Stoxx 60d correlation,
               EUA-TTF 60d correlation
        POLICY: TNAC z-score, months-to-next-May, months-since-last-May,
                MSR intake ratio z-score
  4. Univariate logistic regressions — which single predictor best
     separates high-vol from low-vol regime?
  5. Multivariate logistic regression — which combination wins?
  6. Family comparison — do we need all three families or does one dominate?
  7. AUC comparison — reports which family/predictor separates cleanest

Output:
    data/regime_diagnosis.csv
    plots/regime_diagnosis.png
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# TNAC publication dates (from 02c_fetch_msr_tnac.py)
TNAC_DATES = [
    "2017-05-12", "2018-05-15", "2019-05-14", "2020-05-08",
    "2021-05-14", "2022-05-13", "2023-05-15", "2024-06-06",
    "2025-05-28", "2026-06-01",
]


def build_monthly_panel(f: pd.DataFrame) -> pd.DataFrame:
    """Rich monthly panel with predictors from all three hypothesis families."""
    eua_m = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last()
    m = pd.DataFrame(index=eua_m.index)
    m["ret"]     = np.log(eua_m / eua_m.shift(1))
    m["abs_ret_eua"] = m["ret"].abs()

    # equity + gas monthly
    stx_m = f["stoxx50"].resample("ME").last()
    ttf_m = f["ttf_gas_eur_mwh"].ffill(limit=7).resample("ME").last()
    m["stx_ret"]     = np.log(stx_m / stx_m.shift(1))
    m["abs_ret_stx"] = m["stx_ret"].abs()
    m["ttf_ret"]     = np.log(ttf_m / ttf_m.shift(1))

    # ip + stoxx momentum for MS-2 fit
    ip_m = f["ip_ea19"].resample("ME").last().ffill(limit=1)
    m["ip_yoy"] = ip_m.pct_change(12)
    m["stx_mom20"] = np.log(f["stoxx50"] / f["stoxx50"].shift(20)).resample("ME").last()

    def z(s, w=24):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    m["z_ip"]  = z(m["ip_yoy"])
    m["z_stx"] = z(m["stx_mom20"])

    # ─── VOL family ────────────────────────────────────────────────
    r_eua_d = np.log(f["eua_eur_tco2"].ffill(limit=7)).diff()
    r_stx_d = np.log(f["stoxx50"]).diff()
    r_ttf_d = np.log(f["ttf_gas_eur_mwh"].ffill(limit=7)).diff()

    m["eua_rv20"] = r_eua_d.rolling(20).std().resample("ME").last() * np.sqrt(252)
    m["eua_rv60"] = r_eua_d.rolling(60).std().resample("ME").last() * np.sqrt(252)
    m["stx_rv20"] = r_stx_d.rolling(20).std().resample("ME").last() * np.sqrt(252)
    m["ttf_rv20"] = r_ttf_d.rolling(20).std().resample("ME").last() * np.sqrt(252)

    # ─── MACRO family (co-movement / correlation) ───────────────────
    m["corr_eua_stx_60d"] = r_eua_d.rolling(60).corr(r_stx_d).resample("ME").last()
    m["corr_eua_ttf_60d"] = r_eua_d.rolling(60).corr(r_ttf_d).resample("ME").last()

    # ─── POLICY family ────────────────────────────────────────────
    dates = pd.to_datetime(TNAC_DATES)
    def months_to_next(idx):
        return np.array([
            min([(d - t).days / 30 for d in dates if d >= t], default=np.nan)
            for t in idx
        ])
    def months_since_prev(idx):
        return np.array([
            min([(t - d).days / 30 for d in dates if d <= t], default=np.nan)
            for t in idx
        ])
    m["months_to_next_tnac"] = months_to_next(m.index)
    m["months_since_last_tnac"] = months_since_prev(m.index)

    # MSR tightness (from data/tnac_annual.parquet if present)
    tnac_path = DATA / "tnac_annual.parquet"
    if tnac_path.exists():
        tnac = pd.read_parquet(tnac_path)
        # published each May 15 of year N+1 for year N
        tnac.index = tnac.index + pd.DateOffset(months=4, days=15)
        for col in ["tnac", "msr_tightness"]:
            if col in tnac.columns:
                m[col] = tnac[col].reindex(m.index).ffill()
    return m


def fit_ms2_and_extract_regime(m: pd.DataFrame) -> pd.Series:
    """Fit the baseline MS-2 model and return the smoothed high-vol regime probability."""
    sub = m.dropna(subset=["ret", "z_ip", "z_stx"])
    y = sub["ret"].values
    X = sub[["z_ip", "z_stx"]].values
    model = MarkovRegression(y, k_regimes=2, exog=X, switching_variance=True)
    res = model.fit(disp=False, maxiter=300)

    smoothed = res.smoothed_marginal_probabilities
    if hasattr(smoothed, "iloc"):
        p0 = np.asarray(smoothed.iloc[:, 0].values)
        p1 = np.asarray(smoothed.iloc[:, 1].values)
    else:
        arr = np.asarray(smoothed)
        p0, p1 = (arr[0], arr[1]) if arr.shape[0] == 2 else (arr[:, 0], arr[:, 1])

    param_names = res.model.param_names
    sig0_idx = next((i for i, n in enumerate(param_names) if "sigma2[0]" in n), None)
    sig1_idx = next((i for i, n in enumerate(param_names) if "sigma2[1]" in n), None)
    var0 = res.params[sig0_idx] if sig0_idx is not None else 0
    var1 = res.params[sig1_idx] if sig1_idx is not None else 0
    high_regime = 1 if var1 > var0 else 0
    prob_high = p1 if high_regime == 1 else p0

    idx = sub.index[-len(prob_high):]
    return pd.Series(prob_high, index=idx, name="prob_high"), (var0, var1)


def univariate_logit(y_bin: pd.Series, x: pd.Series) -> dict:
    df = pd.concat([y_bin, x.rename("x")], axis=1).dropna()
    if len(df) < 30 or df["x"].std() == 0:
        return {"n": len(df), "coef": np.nan, "t": np.nan, "auc": np.nan}
    X = sm.add_constant(df["x"])
    try:
        m = sm.Logit(df.iloc[:, 0], X).fit(disp=False)
        pred = m.predict(X)
        auc = roc_auc_score(df.iloc[:, 0], pred)
        return {
            "n":    len(df),
            "coef": m.params["x"],
            "t":    m.tvalues["x"],
            "auc":  auc,
        }
    except Exception:
        return {"n": len(df), "coef": np.nan, "t": np.nan, "auc": np.nan}


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet")
    m = build_monthly_panel(f)
    print(f"panel: {len(m)} monthly obs "
          f"{m.index.min().date()} → {m.index.max().date()}")

    # Fit MS-2 and get regime probability
    print("\nfitting MS-2 to extract regime probability path…")
    prob_high, (var0, var1) = fit_ms2_and_extract_regime(m)
    y_bin = (prob_high > 0.5).astype(int).rename("high")
    print(f"  high-vol regime (post-labelled): "
          f"{y_bin.sum()} of {len(y_bin)} months ({y_bin.mean():.0%})")
    print(f"  fitted variances: low={min(var0, var1):.5f}  high={max(var0, var1):.5f}")

    # ─── Univariate diagnostics ─────────────────────────────────────
    predictor_families = {
        "VOL": [
            "eua_rv20", "eua_rv60", "stx_rv20", "ttf_rv20",
            "abs_ret_eua", "abs_ret_stx",
        ],
        "MACRO": [
            "corr_eua_stx_60d", "corr_eua_ttf_60d", "stx_ret",
        ],
        "POLICY": [
            "months_to_next_tnac", "months_since_last_tnac",
            "tnac", "msr_tightness",
        ],
    }

    print("\n" + "=" * 90)
    print("UNIVARIATE LOGISTIC REGRESSIONS  —  P(high-vol regime) on each predictor")
    print("=" * 90)
    print(f"{'family':<8s} {'predictor':<25s} {'n':>4s}  {'coef':>10s}  {'t-stat':>7s}  {'AUC':>6s}")
    print("─" * 90)

    rows = []
    for family, preds in predictor_families.items():
        for p in preds:
            if p not in m.columns:
                continue
            r = univariate_logit(y_bin, m[p])
            r["family"] = family
            r["predictor"] = p
            rows.append(r)
            sig = "***" if abs(r["t"]) > 2.58 else "**" if abs(r["t"]) > 1.96 else "*" if abs(r["t"]) > 1.64 else " "
            auc_marker = " ←" if r["auc"] and r["auc"] > 0.7 else ""
            print(f"{family:<8s} {p:<25s} {r['n']:>4d}  {r['coef']:>+10.4f}  "
                  f"{r['t']:>+7.2f} {sig:<3s}  {r['auc']:>6.3f}{auc_marker}")

    df_uni = pd.DataFrame(rows)
    df_uni.to_csv(DATA / "regime_diagnosis.csv", index=False)

    # ─── Multivariate: best-of-each-family ────────────────────────
    print("\n" + "=" * 90)
    print("MULTIVARIATE LOGISTIC —  can we predict regime from all three families?")
    print("=" * 90)

    # Pick top predictor per family (by AUC)
    tops = df_uni.sort_values("auc", ascending=False).groupby("family").head(1)
    top_preds = tops.set_index("family")["predictor"].to_dict()
    print(f"\nTop predictor per family (by AUC):")
    for fam, pred in top_preds.items():
        aucv = tops[tops["family"] == fam]["auc"].iloc[0]
        print(f"  {fam:<8s} → {pred:<25s} (AUC = {aucv:.3f})")

    combined = m[[p for p in top_preds.values() if p in m.columns]].join(y_bin, how="inner").dropna()
    if len(combined) < 30:
        print("  insufficient joint sample for multivariate fit")
    else:
        X = sm.add_constant(combined.drop(columns=["high"]))
        y = combined["high"]
        m_mult = sm.Logit(y, X).fit(disp=False)
        print(f"\n  n = {len(combined)}")
        print(f"  Multivariate coefficients:")
        for name, coef, t in zip(m_mult.params.index, m_mult.params.values, m_mult.tvalues.values):
            if name == "const": continue
            sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else " "
            print(f"    {name:<25s} coef={coef:+.4f}  t={t:+.2f} {sig}")
        pred_prob = m_mult.predict(X)
        auc_mv = roc_auc_score(y, pred_prob)
        print(f"  Multivariate AUC = {auc_mv:.3f}")
        print(f"  Pseudo R² = {m_mult.prsquared:.3f}")

    # ─── Family-only regressions ──────────────────────────────────
    print("\n" + "=" * 90)
    print("FAMILY-ONLY LOGISTIC — each family alone: which hypothesis wins?")
    print("=" * 90)
    family_aucs = {}
    for family, preds in predictor_families.items():
        avail = [p for p in preds if p in m.columns]
        joined = m[avail].join(y_bin, how="inner").dropna()
        if len(joined) < 30:
            continue
        X = sm.add_constant(joined[avail])
        y = joined["high"]
        try:
            m_fam = sm.Logit(y, X).fit(disp=False)
            pred_prob = m_fam.predict(X)
            auc = roc_auc_score(y, pred_prob)
            family_aucs[family] = auc
            print(f"  {family:<8s}  n={len(joined):3d}  AUC = {auc:.3f}  "
                  f"pseudo-R² = {m_fam.prsquared:.3f}")
        except Exception as e:
            print(f"  {family:<8s} fit failed: {e}")

    # ─── PLOTS ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(3, 1, figsize=(13, 11))

    # (1) Regime probability + top predictors
    ax[0].plot(prob_high.index, prob_high.values, color="#1a1a1a",
               lw=1.5, label="P(high-vol regime)")
    ax[0].axhline(0.5, ls="--", color="k", alpha=0.4)
    ax[0].set_ylabel("Regime probability")
    ax[0].set_title("Smoothed high-vol regime probability")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # (2) Bar chart: univariate AUC by predictor, coloured by family
    df_plot = df_uni.dropna(subset=["auc"]).sort_values("auc", ascending=True)
    palette = {"VOL": "#3498db", "MACRO": "#e74c3c", "POLICY": "#27ae60"}
    colors_bar = [palette[fam] for fam in df_plot["family"]]
    ax[1].barh(df_plot["predictor"], df_plot["auc"] - 0.5, color=colors_bar, alpha=0.85,
               left=0.5)
    ax[1].axvline(0.5, color="k", lw=0.5)
    ax[1].axvline(0.7, color="k", ls="--", alpha=0.5)
    ax[1].set_xlim(0.4, 1.0)
    ax[1].set_xlabel("AUC (0.5 = random, ≥0.7 = meaningful separation)")
    ax[1].set_title("Which predictors distinguish the regime state?")
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=c, label=f) for f, c in palette.items()],
                 fontsize=9, loc="lower right")
    ax[1].grid(alpha=0.3, axis="x")

    # (3) Family AUC comparison
    if family_aucs:
        fams = list(family_aucs.keys())
        aucs = [family_aucs[f] for f in fams]
        bar_colors = [palette[f] for f in fams]
        ax[2].bar(fams, [a - 0.5 for a in aucs], color=bar_colors, alpha=0.85, bottom=0.5)
        ax[2].set_ylim(0.4, 1.0)
        ax[2].axhline(0.5, color="k", lw=0.5)
        ax[2].axhline(0.7, color="k", ls="--", alpha=0.5)
        ax[2].set_ylabel("Family AUC")
        ax[2].set_title("Which hypothesis family best explains the regime state?")
        for i, a in enumerate(aucs):
            ax[2].text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS / "regime_diagnosis.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote plots/regime_diagnosis.png and data/regime_diagnosis.csv")

    # ─── FINAL VERDICT ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("VERDICT — which hypothesis best explains the regime state?")
    print("=" * 90)
    if family_aucs:
        winner = max(family_aucs, key=family_aucs.get)
        winner_auc = family_aucs[winner]
        others = {k: v for k, v in family_aucs.items() if k != winner}
        margin = winner_auc - max(others.values()) if others else 0
        print(f"\n  Winning family: {winner}  (AUC = {winner_auc:.3f})")
        print(f"  Margin over next best: {margin:+.3f}")
        if winner == "VOL":
            print(f"\n  → H1 (vol clustering) wins. The 'regime' the model finds is essentially")
            print(f"    'high-vol vs low-vol months'. The macro/policy interpretation is post-hoc.")
        elif winner == "MACRO":
            print(f"\n  → H2 (macro co-movement) wins. Regime state = periods of high vs low")
            print(f"    equity-carbon correlation. Not the same as 'fundamentals vs policy'.")
        elif winner == "POLICY":
            print(f"\n  → H3 (policy pressure) wins. Regime state does align with proximity to")
            print(f"    MSR events and TNAC dynamics. Boundary-condition claim survives.")


if __name__ == "__main__":
    main()
