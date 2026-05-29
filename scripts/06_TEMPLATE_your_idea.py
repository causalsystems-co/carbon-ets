"""
06_TEMPLATE_your_idea.py — copy this file when you add a strategy variant.

Conventions:
- Numbered 06_, 07_, ... — don't mutate 05_backtest.py (it's the baseline).
- Read panel_features.parquet, write your output to data/<your_name>_*.parquet
  and plots/<your_name>_*.png.
- Print the same stats block as 05 so we can compare apples-to-apples.

Suggested first experiment: replace the IP YoY feature with a weekly-
resampled version, or add a third feature (gas momentum / weather /
calendar dummy), or change the short floor.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

# ----------------------------- knobs ---------------------------------
MY_NAME = "maurizio_v1"                   # used for output filenames
ZSCORE_WIN = 252
SHORT_FLOOR = -0.3
# ---------------------------------------------------------------------


def zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def my_features(f: pd.DataFrame) -> pd.DataFrame:
    """REPLACE THIS — return a DataFrame of z-scored signals to combine."""
    out = pd.DataFrame(index=f.index)
    out["z_ip_yoy"] = zscore(f["ip_yoy"])
    sx = f["stoxx50"]
    out["z_stoxx_mom"] = zscore(np.log(sx / sx.shift(20)))
    # >>> YOUR FEATURE HERE <<<
    # e.g. weekly-resampled IP, gas momentum, weather, calendar dummies, etc.
    return out


def stats(eq, rets):
    mu = rets.mean() * 252
    sd = rets.std() * np.sqrt(252)
    return {
        "ann_return": mu, "ann_vol": sd,
        "sharpe": mu / sd if sd else np.nan,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "hit_rate": (rets > 0).mean(),
        "n_days": int(rets.notna().sum()),
    }


def main():
    f = pd.read_parquet(DATA / "panel_features.parquet")
    f = f.dropna(subset=["r_carbon_proxy_krbn"])

    feats = my_features(f).dropna(how="any")
    score = feats.mean(axis=1).reindex(f.index)
    pos = score.clip(lower=SHORT_FLOOR, upper=1.0).shift(1)
    pnl = (pos * f["r_carbon_proxy_krbn"]).dropna()
    eq = (1 + pnl).cumprod()

    s = stats(eq, pnl)
    print(f"\n=== {MY_NAME} ===")
    for k, v in s.items():
        print(f"  {k:12s} {v:8.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    eq.plot(ax=ax, label=MY_NAME)
    (1 + f["r_carbon_proxy_krbn"].reindex(eq.index).fillna(0)).cumprod().plot(
        ax=ax, color="grey", alpha=0.6, label="buy & hold")
    ax.set_title(f"{MY_NAME}   Sharpe={s['sharpe']:.2f}   MaxDD={s['max_dd']:.0%}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"equity_{MY_NAME}.png", dpi=120)
    pd.DataFrame({"pos": pos, "ret": f["r_carbon_proxy_krbn"], "pnl": pnl, "eq": eq}).to_parquet(
        DATA / f"backtest_{MY_NAME}.parquet")


if __name__ == "__main__":
    main()
