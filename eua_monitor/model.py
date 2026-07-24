"""The CS/RES/05 demand model: monthly panel + rolling-window R^2 + regime.

EUA monthly log returns are regressed on euro-area industrial-production
YoY growth, Stoxx 50 momentum, and an HDD anomaly, over a rolling
24-month window. The rolling R^2 is the regime indicator: high R^2 means
demand fundamentals are currently doing the pricing, low R^2 means
policy/positioning dominate.
"""

from __future__ import annotations

import math

import numpy as np

from . import config


def _month_end_closes(daily: list[tuple[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for day, close in daily:
        out[day[:7]] = close  # daily is date-ordered; last write wins
    return out


def _next_month(month: str) -> str:
    year, mon = int(month[:4]), int(month[5:7])
    return f"{year + mon // 12}-{mon % 12 + 1:02d}" if mon == 12 else f"{year}-{mon + 1:02d}"


def _months_back(month: str, k: int) -> str:
    year, mon = int(month[:4]), int(month[5:7])
    total = year * 12 + (mon - 1) - k
    return f"{total // 12}-{total % 12 + 1:02d}"


def build_panel(
    eua_daily: list[tuple[str, float]],
    stoxx_daily: list[tuple[str, float]],
    ip_index: dict[str, float],
    hdd_monthly: dict[str, float],
) -> list[dict]:
    """Monthly rows with return + features, only months where all align."""
    eua_m = _month_end_closes(eua_daily)
    stoxx_m = _month_end_closes(stoxx_daily)

    # HDD anomaly vs the calendar-month mean over the available sample.
    by_calmonth: dict[str, list[float]] = {}
    for month, hdd in hdd_monthly.items():
        by_calmonth.setdefault(month[5:7], []).append(hdd)
    calmonth_mean = {cm: sum(v) / len(v) for cm, v in by_calmonth.items()}

    rows = []
    for month in sorted(eua_m):
        prev = _months_back(month, 1)
        ip_prev_year = _months_back(month, 12)
        stoxx_back = _months_back(month, config.STOXX_MOMENTUM_MONTHS)
        if (
            prev not in eua_m
            or month not in ip_index
            or ip_prev_year not in ip_index
            or month not in stoxx_m
            or stoxx_back not in stoxx_m
            or month not in hdd_monthly
        ):
            continue
        rows.append(
            {
                "month": month,
                "eua_close": eua_m[month],
                "eua_ret": math.log(eua_m[month] / eua_m[prev]),
                "ip_yoy": math.log(ip_index[month] / ip_index[ip_prev_year]),
                "stoxx_mom": math.log(stoxx_m[month] / stoxx_m[stoxx_back]),
                "hdd_anom": hdd_monthly[month] - calmonth_mean[month[5:7]],
            }
        )
    return rows


def rolling_r2(panel: list[dict]) -> list[dict]:
    """Rolling OLS R^2 of the demand model, one point per window end."""
    window = config.ROLLING_WINDOW_MONTHS
    out = []
    for end in range(window, len(panel) + 1):
        chunk = panel[end - window : end]
        y = np.array([r["eua_ret"] for r in chunk])
        x = np.column_stack(
            [
                np.ones(window),
                [r["ip_yoy"] for r in chunk],
                [r["stoxx_mom"] for r in chunk],
                [r["hdd_anom"] for r in chunk],
            ]
        )
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid**2).sum()) / sst if sst > 0 else 0.0
        out.append({"month": chunk[-1]["month"], "r2": max(0.0, r2)})
    return out


def classify(r2: float) -> str:
    if r2 >= config.R2_DEMAND_DRIVEN:
        return "demand-driven"
    if r2 <= config.R2_POLICY_DRIVEN:
        return "policy-driven"
    return "transitional"


def regime_series(r2_series: list[dict]) -> list[dict]:
    return [
        {"month": p["month"], "r2": p["r2"], "regime": classify(p["r2"])}
        for p in r2_series
    ]
