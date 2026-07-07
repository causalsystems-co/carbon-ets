"""
15_msr_event_study.py — Event study around MSR/TNAC publication dates.

The identification strategy:
    Every May (moved to June from 2024), the European Commission publishes
    the Total Number of Allowances in Circulation (TNAC) for the previous
    calendar year. This is a hard, calendar-locked policy shock that
    market participants know is coming but do not know the exact number
    of. If the causal chain has a supply-side channel, we should observe:
      1. Statistically detectable abnormal returns on/near the announcement
      2. A cross-sectional relationship between the *surprise* component
         of the TNAC number and the return magnitude

The test is small-N (10 events, one per year 2017-2026) so we shouldn't
expect stellar t-stats, but the effect size should be interpretable.

Method:
  1. Compute the "market model" for EUA daily returns using non-event days
     (r_eua = α + β × r_stoxx50 + ε), estimated on the 250 trading days
     before each event.
  2. For each event date, compute:
     - Raw cumulative log-return in the [-5, +5] trading-day window
     - Abnormal cumulative return (raw − market-model expectation)
     - CAR standard error from the estimation-window residual variance
     - t-stat on the CAR
  3. Compute TNAC "surprise" as:
     surprise = actual_TNAC − naive_expectation
     where naive_expectation = prior_TNAC × (1 + median_prior_growth)
  4. Cross-sectional regression: CAR on standardized surprise. Report
     the coefficient, t-stat, and R² of that relationship.
  5. Plot event-window returns aligned by day-relative-to-event, and the
     CAR–vs–surprise scatter.

Output:
    data/msr_event_study.csv
    plots/msr_event_study.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# Event dates: Commission Communication publication dates for each annual TNAC.
# Verified from the citations in 02c_fetch_msr_tnac.py.
EVENTS = [
    # (announcement_date, TNAC_year, TNAC_value)
    ("2017-05-12", 2016, 1_693_904_897),
    ("2018-05-15", 2017, 1_654_574_598),
    ("2019-05-14", 2018, 1_654_909_824),
    ("2020-05-08", 2019, 1_385_496_166),
    ("2021-05-14", 2020, 1_578_772_426),
    ("2022-05-13", 2021, 1_449_214_182),
    ("2023-05-15", 2022, 1_134_794_738),
    ("2024-06-06", 2023, 1_111_736_535),  # moved to June in 2024 revision
    ("2025-05-28", 2024, 1_148_049_585),
    ("2026-06-01", 2025, 1_023_494_202),  # (approx if not yet published)
]

EVENT_WINDOW  = (-5, 5)      # trading days around event
EST_WINDOW    = (-260, -10)  # 250 trading-day estimation window ending 10d before event
TARGET_PRICE  = "eua_eur_tco2"


def compute_surprise() -> pd.DataFrame:
    """Naïve expectation = prior TNAC × (1 + median growth of prior 3 years).
    Surprise = actual − expected, then z-score across the 10 events."""
    df = pd.DataFrame(EVENTS, columns=["date", "tnac_year", "tnac"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["prior_tnac"] = df["tnac"].shift(1)
    df["prior_growth"] = df["tnac"].pct_change()
    df["expected"] = df["prior_tnac"] * (
        1 + df["prior_growth"].shift(1).rolling(3, min_periods=1).median()
    )
    df["surprise_raw"] = df["tnac"] - df["expected"]
    valid = df["surprise_raw"].notna()
    df.loc[valid, "surprise_z"] = (
        df.loc[valid, "surprise_raw"] - df.loc[valid, "surprise_raw"].mean()
    ) / df.loc[valid, "surprise_raw"].std()
    return df


def market_model_car(f: pd.DataFrame, event_date: pd.Timestamp) -> dict:
    """Compute cumulative abnormal return for one event using a market model
    estimated on the 250 pre-event trading days."""
    # log returns on business-day scale
    eua = f[TARGET_PRICE].ffill(limit=7)
    stx = f["stoxx50"].ffill(limit=7)
    r_eua = np.log(eua / eua.shift(1))
    r_stx = np.log(stx / stx.shift(1))
    df = pd.DataFrame({"r_eua": r_eua, "r_stx": r_stx}).dropna()

    # locate event in the return series (find nearest business day)
    if event_date not in df.index:
        # snap to next available trading day
        future = df.index[df.index >= event_date]
        if len(future) == 0:
            return {}
        event_idx = df.index.get_loc(future[0])
    else:
        event_idx = df.index.get_loc(event_date)

    # estimation window
    est_lo = event_idx + EST_WINDOW[0]
    est_hi = event_idx + EST_WINDOW[1]
    if est_lo < 0 or est_hi < 0:
        return {}
    est = df.iloc[est_lo:est_hi].dropna()
    if len(est) < 100:
        return {}

    X = sm.add_constant(est["r_stx"])
    m = sm.OLS(est["r_eua"], X).fit()
    alpha, beta = m.params["const"], m.params["r_stx"]
    resid_var = m.resid.var()

    # event window
    evt_lo = event_idx + EVENT_WINDOW[0]
    evt_hi = event_idx + EVENT_WINDOW[1] + 1
    evt = df.iloc[evt_lo:evt_hi].copy()
    evt["expected"]   = alpha + beta * evt["r_stx"]
    evt["abnormal"]   = evt["r_eua"] - evt["expected"]
    car               = evt["abnormal"].sum()
    car_var           = resid_var * len(evt)  # assuming iid residuals
    car_t             = car / np.sqrt(car_var)

    return {
        "event_date":  df.index[event_idx],
        "n_est":       len(est),
        "beta_market": beta,
        "raw_ret":     evt["r_eua"].sum(),
        "abn_ret_car": car,
        "car_tstat":   car_t,
        "abn_by_day":  evt["abnormal"].reset_index(drop=True).values,
        "day_labels":  list(range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1)),
    }


def main() -> None:
    f = pd.read_parquet(DATA / "panel_features.parquet")
    print(f"panel: {len(f)} rows, {f.index.min().date()} → {f.index.max().date()}")

    surprise = compute_surprise()

    results = []
    abn_series = {}
    for _, r in surprise.iterrows():
        car = market_model_car(f, r["date"])
        if not car:
            print(f"  event {r['date'].date()}: no data (before panel start?), skipped")
            continue
        row = {
            "event_date":   car["event_date"].date(),
            "tnac_year":    int(r["tnac_year"]),
            "tnac":         int(r["tnac"]),
            "expected":     r["expected"] if pd.notna(r["expected"]) else np.nan,
            "surprise":     r["surprise_raw"] if pd.notna(r["surprise_raw"]) else np.nan,
            "surprise_z":   r["surprise_z"] if pd.notna(r["surprise_z"]) else np.nan,
            "raw_car_pct":  car["raw_ret"] * 100,
            "abn_car_pct":  car["abn_ret_car"] * 100,
            "car_tstat":    car["car_tstat"],
            "n_est_days":   car["n_est"],
        }
        results.append(row)
        abn_series[car["event_date"].date()] = (car["day_labels"], car["abn_by_day"])

    if not results:
        print("no events processed; check date coverage of panel_features")
        return

    df = pd.DataFrame(results).set_index("event_date")
    df.to_csv(DATA / "msr_event_study.csv")

    print()
    print("=" * 90)
    print("MSR event study — abnormal returns around annual TNAC publication")
    print("=" * 90)
    print(df.to_string(float_format=lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x)))
    print()

    # pooled tests
    car_mean = df["abn_car_pct"].mean()
    car_std  = df["abn_car_pct"].std(ddof=1)
    n        = len(df)
    t_pooled = car_mean / (car_std / np.sqrt(n))

    print(f"pooled CAR mean:         {car_mean:+.3f}%  ({n} events)")
    print(f"pooled CAR std:          {car_std:.3f}%")
    print(f"pooled t-stat (H0: CAR=0): {t_pooled:+.2f}")
    print()

    # cross-sectional regression: CAR on surprise
    reg = df.dropna(subset=["surprise_z"])
    if len(reg) >= 5:
        X = sm.add_constant(reg["surprise_z"])
        m = sm.OLS(reg["abn_car_pct"], X).fit()
        print("Cross-sectional regression: abnormal CAR (%) on standardized TNAC surprise")
        print(f"  slope: {m.params['surprise_z']:+.3f}%  (t={m.tvalues['surprise_z']:+.2f})")
        print(f"  intercept: {m.params['const']:+.3f}%")
        print(f"  R² = {m.rsquared:.3f},  n = {len(reg)}")
        print()

    # ─── plots ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # left: event-window returns aligned by day
    day_grid = list(range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1))
    all_arr = []
    for d, (days, abn) in abn_series.items():
        cum = np.cumsum(abn) * 100
        ax[0].plot(days, cum, alpha=0.4, lw=1, label=str(d))
        all_arr.append(cum)
    mean_cum = np.mean(all_arr, axis=0)
    ax[0].plot(day_grid, mean_cum, color="k", lw=2.5, label=f"MEAN (n={n})")
    ax[0].axvline(0, ls="--", color="k", alpha=0.5)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title("Cumulative abnormal EUA return around MSR publication (day 0)")
    ax[0].set_xlabel("Trading days from announcement")
    ax[0].set_ylabel("Cumulative abnormal return (%)")
    ax[0].legend(fontsize=7, loc="best"); ax[0].grid(alpha=0.3)

    # right: CAR vs surprise scatter
    if len(reg) >= 5:
        ax[1].scatter(reg["surprise_z"], reg["abn_car_pct"], s=80, alpha=0.8, color="#1a1a1a")
        for d, row in reg.iterrows():
            ax[1].annotate(str(d)[-4:], (row["surprise_z"], row["abn_car_pct"]),
                           xytext=(4, 4), textcoords="offset points", fontsize=8)
        x_line = np.linspace(reg["surprise_z"].min(), reg["surprise_z"].max(), 20)
        ax[1].plot(x_line, m.params["const"] + m.params["surprise_z"] * x_line,
                   color="#e74c3c", lw=1.5, ls="--",
                   label=f"β={m.params['surprise_z']:+.2f}% (t={m.tvalues['surprise_z']:+.2f})")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].axvline(0, color="k", lw=0.5)
        ax[1].set_title("Abnormal CAR vs standardized TNAC surprise")
        ax[1].set_xlabel("TNAC surprise (z-score)  |  positive = more oversupply than expected")
        ax[1].set_ylabel("Cumulative abnormal return (%)")
        ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "msr_event_study.png", dpi=140)
    plt.close(fig)
    print(f"wrote plots/msr_event_study.png")


if __name__ == "__main__":
    main()
