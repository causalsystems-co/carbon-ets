"""
carbon_ets.plots — standard visualisations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(eq: pd.Series, benchmark: pd.Series | None = None,
                     title: str = "Equity curve", savepath=None):
    """Standard equity curve chart with optional benchmark overlay."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq.index, eq.values, color="#1a1a1a", lw=1.8, label="strategy")
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, color="grey", alpha=0.7,
                lw=1.2, label="benchmark")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    if savepath:
        fig.tight_layout(); fig.savefig(savepath, dpi=140); plt.close(fig)
    return fig


def plot_tnac_nowcast(reference: pd.DataFrame, current_estimate: dict | None = None,
                     savepath=None):
    """Plot historical TNAC series + optional current-year nowcast marker."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(reference.index, reference["tnac"] / 1e9,
            color="#1a1a1a", lw=1.8, marker="o", label="Commission-published TNAC")
    if current_estimate is not None and "as_of" in current_estimate and "tnac_estimate" in current_estimate:
        ax.scatter([current_estimate["as_of"]], [current_estimate["tnac_estimate"] / 1e9],
                   color="#e74c3c", s=100, zorder=5, marker="D",
                   label=f"Nowcast (confidence: {current_estimate.get('confidence', 'n/a')})")
    ax.axhline(1.096, ls="--", color="#c0392b", alpha=0.5, label="MSR upper threshold (1.096B)")
    ax.axhline(0.833, ls="--", color="#e67e22", alpha=0.5, label="MSR partial-intake band (833M)")
    ax.axhline(0.400, ls="--", color="#27ae60", alpha=0.5, label="MSR release threshold (400M)")
    ax.set_ylabel("TNAC (billion allowances)")
    ax.set_title("EU ETS Total Number of Allowances in Circulation — history + nowcast")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    if savepath:
        fig.tight_layout(); fig.savefig(savepath, dpi=140); plt.close(fig)
    return fig
