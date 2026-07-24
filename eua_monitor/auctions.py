"""Indicative EUA primary-auction calendar.

Generated from the standard EEX weekday pattern (EU27 common auctions
Mon/Tue/Thu, Germany Wed, Poland Fri biweekly). The published EEX calendar
governs; exact rows can be supplied via data/auction_overrides.csv with
columns date,market,note and take precedence over generated rows.
"""

from __future__ import annotations

import csv
import datetime as dt

from . import config

# Days EEX is closed for trading (fixed-date holidays; indicative).
_FIXED_HOLIDAYS = {(1, 1), (5, 1), (12, 24), (12, 25), (12, 26), (12, 31)}


def upcoming(today: dt.date | None = None) -> list[dict]:
    today = today or dt.date.today()
    overrides: dict[str, dict] = {}
    if config.AUCTION_OVERRIDES_CSV.exists():
        with config.AUCTION_OVERRIDES_CSV.open() as fh:
            for row in csv.DictReader(fh):
                overrides[row["date"]] = {
                    "date": row["date"],
                    "market": row["market"],
                    "note": row.get("note", ""),
                    "source": "EEX calendar",
                }
    out = []
    for offset in range(config.AUCTION_DAYS_AHEAD + 1):
        day = today + dt.timedelta(days=offset)
        iso = day.isoformat()
        if iso in overrides:
            out.append(overrides[iso])
            continue
        market = config.AUCTION_WEEKDAYS.get(day.weekday())
        if market is None or (day.month, day.day) in _FIXED_HOLIDAYS:
            continue
        note = "reduced August volumes" if day.month == 8 else ""
        out.append(
            {"date": iso, "market": market, "note": note, "source": "indicative"}
        )
    return out
