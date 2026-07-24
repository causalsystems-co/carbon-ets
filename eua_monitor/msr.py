"""Market Stability Reserve arithmetic on the published TNAC series."""

from __future__ import annotations

from . import config


def msr_action(tnac: int) -> dict:
    """MSR response to a published TNAC, per Decision (EU) 2015/1814
    as amended by Decision (EU) 2023/852."""
    if tnac >= config.MSR_UPPER_FULL:
        return {
            "kind": "full-intake",
            "amount": round(tnac * config.MSR_INTAKE_RATE),
            "label": f"24% intake ({tnac * config.MSR_INTAKE_RATE / 1e6:,.1f}M withheld)",
        }
    if tnac > config.MSR_UPPER:
        return {
            "kind": "partial-intake",
            "amount": tnac - config.MSR_UPPER,
            "label": f"partial intake ({(tnac - config.MSR_UPPER) / 1e6:,.1f}M withheld)",
        }
    if tnac >= config.MSR_LOWER:
        return {"kind": "no-action", "amount": 0, "label": "no MSR action"}
    return {
        "kind": "release",
        "amount": -config.MSR_RELEASE,
        "label": f"{config.MSR_RELEASE / 1e6:,.0f}M released",
    }


def enrich(history: list[dict]) -> list[dict]:
    return [{**row, "action": msr_action(row["tnac"])} for row in history]


def headline(history: list[dict]) -> dict:
    """Latest TNAC with its MSR consequence and distance to thresholds."""
    latest = history[-1]
    action = msr_action(latest["tnac"])
    return {
        "year": latest["year"],
        "tnac": latest["tnac"],
        "source": latest["source"],
        "action": action,
        "to_full_intake": latest["tnac"] - config.MSR_UPPER_FULL,
        "to_no_action": latest["tnac"] - config.MSR_UPPER,
        "intake_period": f"Sep {latest['year'] + 1} – Aug {latest['year'] + 2}",
    }
