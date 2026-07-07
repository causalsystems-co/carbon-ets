"""
20_validate_nowcast.py — honest quality check on the TNAC nowcast.

If the nowcast systematically over- or under-estimates published TNAC values,
we need to fix it or discard it. This script runs the nowcast for every
historical year we have ground truth for, plots error trajectory, and
prints an honest verdict.

Success criteria (needed for shipping):
    - Median absolute error < 5% of actual TNAC across years 2017-2024
    - No systematic bias (mean error close to zero)
    - Errors don't grow year over year

If ANY of these fail, the nowcast is not ready. Options:
    a) Improve the annual constants with real EUTL / EEX data
    b) Add missing components (invalidations, credits exhaustion timing)
    c) Ship without the nowcast, focus on data infrastructure only
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from carbon_ets.tnac import TNAC_ANNUAL_REFERENCE, nowcast_monthly, validate_reference


def main():
    print("=" * 78)
    print("TNAC nowcast validation")
    print("=" * 78)
    print()
    print(f"Reference years available: {sorted(TNAC_ANNUAL_REFERENCE.keys())}")
    print()

    val_df = validate_reference()
    print(val_df.to_string())
    print()

    # honest metrics
    errors_pct = val_df["error_pct"]
    n = len(errors_pct)
    mean_err = errors_pct.mean()
    median_abs_err = errors_pct.abs().median()
    max_abs_err = errors_pct.abs().max()

    print(f"n = {n} year-end validation points")
    print(f"mean error:            {mean_err:+.2f}%   (systematic bias)")
    print(f"median |error|:        {median_abs_err:.2f}%")
    print(f"max |error|:           {max_abs_err:.2f}%")
    print()

    # verdict
    print("VERDICT")
    print("-" * 78)
    if median_abs_err < 3:
        print("EXCELLENT — median error under 3%. Ship the nowcast.")
    elif median_abs_err < 5:
        print("ACCEPTABLE — median error under 5%. Ship with caveats.")
    elif median_abs_err < 10:
        print("MARGINAL — median error 5-10%. Only ship if improved.")
    else:
        print("UNACCEPTABLE — median error above 10%. Nowcast NOT ready.")
        print("            Do NOT ship. Either improve annual constants or")
        print("            remove the nowcast feature entirely.")

    if abs(mean_err) > 5:
        print(f"WARNING: systematic bias of {mean_err:+.1f}% detected.")
        print("         The nowcast is consistently over- or under-estimating.")
        print("         This is more concerning than random error.")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = val_df.index.values
    actual = val_df["actual_tnac"] / 1e9
    nowcast = val_df["nowcast_tnac"] / 1e9

    ax1.plot(x, actual, marker="o", color="#1a1a1a", lw=2, label="Published TNAC")
    ax1.plot(x, nowcast, marker="s", color="#e74c3c", lw=1.5,
             ls="--", label="Nowcast")
    ax1.set_ylabel("TNAC (billion allowances)")
    ax1.set_title("Nowcast vs published TNAC")
    ax1.grid(alpha=0.3); ax1.legend()

    colors = ["#e74c3c" if abs(e) > 5 else "#f39c12" if abs(e) > 3 else "#27ae60"
              for e in errors_pct]
    ax2.bar(x, errors_pct, color=colors, alpha=0.85)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(5, color="k", ls="--", lw=0.5, alpha=0.5)
    ax2.axhline(-5, color="k", ls="--", lw=0.5, alpha=0.5)
    ax2.set_ylabel("Nowcast error (%)")
    ax2.set_xlabel("Year")
    ax2.set_title("Nowcast error — green ≤ 3%, orange ≤ 5%, red > 5%")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("plots/nowcast_validation.png", dpi=140)
    plt.close(fig)
    print()
    print("wrote plots/nowcast_validation.png")


if __name__ == "__main__":
    main()
