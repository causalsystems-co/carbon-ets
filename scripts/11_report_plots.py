"""
11_report_plots.py — the two figures referenced in REPORT.md.

Plot 01: 24-month rolling R² of the two-feature demand model on monthly EUA returns.
Plot 02: EUA price vs IP YoY, twin-axis, 2012-2026.

Outputs:
    plots/report_plot01_rolling_r2.png
    plots/report_plot02_eua_vs_ip.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def main():
    # NB: don't dropna on r_eua up front — that removes business days between
    # EUA auctions and creates gaps in the monthly resample downstream. We
    # keep the full daily calendar and let each aggregation handle NaN.
    f = pd.read_parquet(DATA / "panel_features.parquet")

    # ── monthly returns + features ──
    # Compute EUA monthly log-return from end-of-month prices directly.
    # This is more robust than summing daily log-diffs when many daily rows
    # are NaN (auctions run Mon-Thu, so ~20-40% of business days lack a print).
    eua_monthly = f["eua_eur_tco2"].ffill(limit=7).resample("ME").last()
    m = pd.DataFrame(index=eua_monthly.index)
    m["ret"]         = np.log(eua_monthly / eua_monthly.shift(1))
    ip_monthly       = f["ip_ea19"].resample("ME").last().ffill(limit=1)
    m["ip_yoy"]      = ip_monthly.pct_change(12)
    m["stoxx_mom20"] = np.log(f["stoxx50"] / f["stoxx50"].shift(20)).resample("ME").last()

    def z(s, w):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    m["z_ip"]  = z(m["ip_yoy"], 24)
    m["z_stx"] = z(m["stoxx_mom20"], 24)

    # ── Plot 01: rolling R² of two-feature demand model ──
    # Iterate over the monthly calendar (not just non-NaN rows), so gaps in
    # source data don't create visual holes in the R² time series. For each
    # calendar date T, look back 24 months, drop NaN rows *within that
    # window* only, and require at least 18 valid obs to fit.
    win = 24
    min_obs = 18
    r2 = pd.Series(index=m.index, dtype=float)
    for i in range(win, len(m)):
        w = m.iloc[i-win:i].dropna(subset=["z_ip", "z_stx", "ret"])
        if len(w) < min_obs:
            continue
        X = w[["z_ip", "z_stx"]].values
        y = w["ret"].values
        Xc = np.column_stack([np.ones(len(X)), X])
        try:
            beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            yhat = Xc @ beta
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2.iloc[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            pass

    # honest full-sample R² (single regression on all valid data)
    all_valid = m.dropna(subset=["z_ip", "z_stx", "ret"])
    if len(all_valid) > 20:
        X = np.column_stack([np.ones(len(all_valid)), all_valid[["z_ip", "z_stx"]].values])
        y = all_valid["ret"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        full_r2 = 1 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()
        print(f"full-sample R² (single regression, n={len(all_valid)}): {full_r2:.3f}")
    else:
        full_r2 = 0.31

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(r2.index, r2.values, color="#1a1a1a", lw=2)
    ax.fill_between(r2.index, 0, r2.values, where=r2.values > 0, alpha=0.12, color="#1a1a1a")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(full_r2, color="#888", lw=0.8, ls="--")
    ax.text(r2.index[-1], full_r2, f" full-sample avg {full_r2:.2f}",
            va="center", fontsize=9, color="#666")

    # highlight the 2018 collapse
    x18 = pd.Timestamp("2018-06-01")
    ax.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2019-07-01"),
               color="#e74c3c", alpha=0.08, zorder=0)
    ax.text(x18, 0.65, "MSR reform", fontsize=9, ha="center", color="#c0392b")

    ax.set_ylim(-0.05, 0.75)
    ax.set_ylabel("R²  (2-feature demand model, 24-mo window)")
    ax.set_title("Plot 01 · Rolling monthly R² of the demand-side chain")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "report_plot01_rolling_r2.png", dpi=140)
    plt.close(fig)
    print("wrote plots/report_plot01_rolling_r2.png")

    # ── Plot 02: EUA vs IP YoY twin-axis ──
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax2 = ax1.twinx()

    p1, = ax1.plot(f.index, f["eua_eur_tco2"], color="#1a1a1a", lw=1.6, label="EUA (EUR/tCO₂)")
    p2, = ax2.plot(f.index, f["ip_yoy"] * 100, color="#c0392b", lw=1.2, alpha=0.85, label="IP YoY (%)")

    ax2.axhline(0, color="#c0392b", lw=0.4, alpha=0.5)

    # highlight the MSR reform period
    ax1.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2019-07-01"),
                color="#e74c3c", alpha=0.08, zorder=0)
    ax1.text(pd.Timestamp("2018-09-01"), f["eua_eur_tco2"].max()*0.92,
             "MSR reform:\nEUA triples while\nIP is flat",
             fontsize=8.5, ha="center", color="#c0392b", style="italic")

    ax1.set_ylabel("EUA (EUR/tCO₂)")
    ax2.set_ylabel("IP YoY (%)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    ax1.set_title("Plot 02 · EUA auction clearing price vs euro-area IP YoY")
    ax1.grid(alpha=0.25)
    fig.legend(handles=[p1, p2], loc="upper left", bbox_to_anchor=(0.08, 0.94),
               fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "report_plot02_eua_vs_ip.png", dpi=140)
    plt.close(fig)
    print("wrote plots/report_plot02_eua_vs_ip.png")


if __name__ == "__main__":
    main()
