"""
21_showcase_plots.py — generate publication-quality showcase plots for
the carbon-ets toolkit.

Produces 5 plots designed for:
    - README.md hero images
    - CS/RES/05 report figures
    - LinkedIn / Twitter distribution
    - Journalist screenshots

All plots use a consistent visual language (typography, palette, gridlines)
and are designed to be immediately readable without a caption.

Outputs to plots/showcase/:
    01_tnac_history.png       — TNAC series with MSR thresholds annotated
    02_tnac_yoy_change.png    — Annual TNAC deltas showing MSR tightening
    03_eua_prices.png         — EUA daily auction prices with phase annotations
    04_tnac_vs_eua.png        — Twin-axis overlay of TNAC and EUA price
    05_msr_intake.png         — Annual MSR intake schedule
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

from carbon_ets.tnac import (
    get_reference_series,
    TNAC_ANNUAL_REFERENCE,
    MSR_UPPER_THRESHOLD,
    MSR_LOWER_THRESHOLD,
    MSR_RELEASE_THRESHOLD,
    INVALIDATIONS,
)


# ─── consistent visual language ────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titleweight":  "semibold",
    "axes.titlesize":    13,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

NAVY   = "#1a1a2e"
CARBON = "#16213e"
RED    = "#c0392b"
ORANGE = "#e67e22"
GREEN  = "#27ae60"
BLUE   = "#2980b9"
GREY   = "#7f8c8d"

OUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "showcase"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_source(ax, text: str = "Source: European Commission MSR Communications · carbon-ets") -> None:
    """Add a source citation to the bottom-right of a chart."""
    ax.text(0.99, -0.14, text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=GREY, style="italic")


def add_watermark(fig, text: str = "carbonsystems.co/research") -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=GREY, alpha=0.7)


# ═══════════════════ Plot 1: TNAC history ═══════════════════

def plot_01_tnac_history():
    ref = get_reference_series()
    fig, ax = plt.subplots(figsize=(11, 5))

    tnac_b = ref["tnac"] / 1e9
    ax.plot(ref.index, tnac_b, color=NAVY, lw=2.4, marker="o",
            markersize=7, markerfacecolor=NAVY, markeredgecolor="white",
            markeredgewidth=1.5, zorder=5)

    # MSR threshold zones as shaded regions
    ax.axhspan(MSR_UPPER_THRESHOLD/1e9, 2.5, color=RED, alpha=0.06, zorder=1)
    ax.axhspan(MSR_LOWER_THRESHOLD/1e9, MSR_UPPER_THRESHOLD/1e9,
               color=ORANGE, alpha=0.08, zorder=1)
    ax.axhspan(MSR_RELEASE_THRESHOLD/1e9, MSR_LOWER_THRESHOLD/1e9,
               color=BLUE, alpha=0.05, zorder=1)
    ax.axhspan(0, MSR_RELEASE_THRESHOLD/1e9,
               color=GREEN, alpha=0.06, zorder=1)

    # threshold lines
    ax.axhline(MSR_UPPER_THRESHOLD/1e9, color=RED, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(MSR_LOWER_THRESHOLD/1e9, color=ORANGE, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(MSR_RELEASE_THRESHOLD/1e9, color=GREEN, lw=1.2, ls="--", alpha=0.6)

    # threshold labels on the right
    ax.text(ref.index[-1] + pd.Timedelta(days=180), MSR_UPPER_THRESHOLD/1e9,
            "  Full intake\n  (24%)", va="center", fontsize=9, color=RED)
    ax.text(ref.index[-1] + pd.Timedelta(days=180), MSR_LOWER_THRESHOLD/1e9,
            "  Partial intake\n  band starts", va="center", fontsize=9, color=ORANGE)
    ax.text(ref.index[-1] + pd.Timedelta(days=180), MSR_RELEASE_THRESHOLD/1e9,
            "  Release", va="center", fontsize=9, color=GREEN)

    # highlight 2025 partial-intake activation
    ax.annotate(
        "2025 — first partial\nintake activation",
        xy=(ref.index[-1], ref["tnac"].iloc[-1] / 1e9),
        xytext=(pd.Timestamp("2022-06-01"), 1.35),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1, alpha=0.7),
        fontsize=10, fontweight="bold", color=NAVY,
        ha="center",
    )

    ax.set_ylim(0, 2.0)
    ax.set_ylabel("TNAC (billion allowances)")
    ax.set_title("EU ETS Total Number of Allowances in Circulation, 2016–2025",
                 loc="left")
    ax.text(0.0, 1.02, "The surplus has fallen 40% since 2016, crossing below the "
            "1.096B threshold for the first time in 2025.",
            transform=ax.transAxes, fontsize=10, color=GREY, style="italic",
            va="bottom")
    add_source(ax)
    add_watermark(fig)
    plt.subplots_adjust(right=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_tnac_history.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote 01_tnac_history.png")


# ═══════════════════ Plot 2: TNAC YoY change ═══════════════════

def plot_02_tnac_yoy_change():
    ref = get_reference_series()
    changes = ref["tnac"].diff().dropna() / 1e6   # in million allowances

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [RED if c > 0 else GREEN for c in changes]
    bars = ax.bar(changes.index.year, changes.values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.7)

    for bar, val in zip(bars, changes.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (10 if val > 0 else -10),
                f"{val:+.0f}", ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=9, fontweight="bold",
                color=RED if val > 0 else GREEN)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(changes.index.year)
    ax.set_ylabel("Change in TNAC (million allowances)")
    ax.set_title("Annual change in TNAC — MSR mechanism tightening the surplus",
                 loc="left")
    ax.text(0.0, 1.02,
            "Reduction bars (green) show years when MSR intake + emissions exceeded new supply. "
            "Rebound bars (red) are COVID and post-COVID.",
            transform=ax.transAxes, fontsize=10, color=GREY, style="italic",
            va="bottom")
    add_source(ax)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_tnac_yoy_change.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote 02_tnac_yoy_change.png")


# ═══════════════════ Plot 3: MSR intake schedule ═══════════════════

def plot_03_msr_intake():
    ref = get_reference_series()
    intake = ref["msr_intake_next_12mo"] / 1e6
    intake = intake[intake > 0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(intake.index.year, intake.values,
           color=CARBON, alpha=0.85,
           edgecolor="white", linewidth=1.5, width=0.7)

    for x, y in zip(intake.index.year, intake.values):
        ax.text(x, y + 8, f"{y:.0f}M", ha="center", fontsize=9, fontweight="bold",
                color=CARBON)

    ax.set_xticks(intake.index.year)
    ax.set_ylabel("MSR intake following 12 months (million allowances)")
    ax.set_title("MSR intake schedule from Commission announcements",
                 loc="left")
    ax.text(0.0, 1.02,
            "Amount diverted from auctions to the reserve during Sept(year)–Aug(year+1). "
            "2025's ~190M is the first partial-intake reading since the mechanism began.",
            transform=ax.transAxes, fontsize=10, color=GREY, style="italic",
            va="bottom")
    add_source(ax)
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_msr_intake.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote 03_msr_intake.png")


# ═══════════════════ Plot 4: EUA prices with phase annotations ═══════════════════

def plot_04_eua_prices():
    """EUA daily prices from the local parquet (no network needed)."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if (data_dir / "eua_daily.parquet").exists():
        eua = pd.read_parquet(data_dir / "eua_daily.parquet")
        price = eua["eua_eur_tco2"]
    else:
        print("skipping 04 — no local EUA parquet found; run 01b_fetch_eua_auctions first")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(price.index, price.values, color=NAVY, lw=1.4)

    # phase annotations
    phases = [
        ("2012-01-01", "2018-01-01", "Phase III oversupply",         "#95a5a6"),
        ("2018-01-01", "2019-07-01", "MSR reform rally",             "#e74c3c"),
        ("2019-07-01", "2020-03-01", "MSR operating",                "#f39c12"),
        ("2020-03-01", "2021-01-01", "COVID crash + recovery",       "#27ae60"),
        ("2021-01-01", "2023-01-01", "Gas crisis bull",              "#8e44ad"),
        ("2023-01-01", "2024-06-01", "Demand destruction",           "#16a085"),
        ("2024-06-01", str(price.index.max().date()), "Partial intake era", "#c0392b"),
    ]
    for start, end, label, color in phases:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(start_ts, end_ts, color=color, alpha=0.08, zorder=0)
        mid = start_ts + (end_ts - start_ts) / 2
        y_pos = price.max() * 0.95 if "reform" in label or "Gas" in label else price.max() * 0.85
        ax.text(mid, y_pos, label, ha="center", fontsize=8.5,
                color=color, alpha=0.85, style="italic")

    ax.set_ylabel("EUA clearing price (EUR / tCO₂)")
    ax.set_title("EU ETS daily primary-auction clearing price, 2012–2026",
                 loc="left")
    ax.text(0.0, 1.02,
            "From <€5 during the Phase III oversupply through €100 at the 2023 peak — "
            "the market that shaped European climate policy.",
            transform=ax.transAxes, fontsize=10, color=GREY, style="italic",
            va="bottom")
    add_source(ax, "Source: EEX primary-auction reports · carbon-ets")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_eua_prices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote 04_eua_prices.png")


# ═══════════════════ Plot 5: TNAC vs EUA overlay ═══════════════════

def plot_05_tnac_vs_eua():
    """The full causal-chain story — TNAC declines drive EUA rises structurally."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not (data_dir / "eua_daily.parquet").exists():
        print("skipping 05 — no local EUA parquet found")
        return

    eua = pd.read_parquet(data_dir / "eua_daily.parquet")
    price = eua["eua_eur_tco2"].resample("ME").mean()
    ref = get_reference_series()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # EUA on primary axis
    ax1.plot(price.index, price.values, color=NAVY, lw=1.6, label="EUA monthly avg (€/tCO₂)")
    ax1.set_ylabel("EUA (EUR / tCO₂)", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)

    # TNAC on secondary axis
    tnac_b = ref["tnac"] / 1e9
    ax2.plot(ref.index, tnac_b, color=RED, lw=2.2, marker="o",
             markersize=7, label="TNAC (billion, right)", zorder=5)
    ax2.set_ylabel("TNAC (billion allowances)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)

    # inverse-relationship annotation
    ax1.text(pd.Timestamp("2015-06-01"), 65,
             "As TNAC surplus shrinks 40%,\nEUA rises from €5 to €90",
             fontsize=11, fontweight="bold", color=NAVY,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.5", fc="white",
                       ec=NAVY, alpha=0.9))

    ax1.set_title("Structural EUA rally aligns with MSR-driven TNAC tightening",
                  loc="left")
    ax1.text(0.0, 1.02,
             "Left axis: EUA monthly average.  Right axis (inverted): TNAC.",
             transform=ax1.transAxes, fontsize=10, color=GREY, style="italic",
             va="bottom")
    add_source(ax1, "Source: EEX + Commission MSR · carbon-ets")
    add_watermark(fig)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_tnac_vs_eua.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote 05_tnac_vs_eua.png")


# ═══════════════════ main ═══════════════════

def main():
    print(f"generating showcase plots into {OUT_DIR}")
    plot_01_tnac_history()
    plot_02_tnac_yoy_change()
    plot_03_msr_intake()
    plot_04_eua_prices()
    plot_05_tnac_vs_eua()
    print()
    print("done. See plots/showcase/ for images.")


if __name__ == "__main__":
    main()
