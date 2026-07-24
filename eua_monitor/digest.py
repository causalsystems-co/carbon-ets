"""Digest: a short note generated only when the monitored state changes.

Compares the freshly built state against the previous site/state.json.
A regime flip or a TNAC/MSR change writes site/digest.md and returns it;
no change returns None so the caller (CLI or CI) can skip sending email.
"""

from __future__ import annotations

import json

from . import config

SUBSCRIBE_HINT = (
    "You receive this because you asked for the EUA regime digest. "
    "It only sends when the state changes."
)


def current_state(payload: dict) -> dict:
    cur = payload["current"]
    head = payload["tnacHead"]
    return {
        "month": cur["month"],
        "r2": round(cur["r2"], 4),
        "regime": cur["regime"],
        "tnac_year": head["year"],
        "tnac": head["tnac"],
        "msr_kind": head["action"]["kind"],
    }


def previous_state() -> dict | None:
    if config.STATE_JSON.exists():
        return json.loads(config.STATE_JSON.read_text())
    return None


def diff_digest(prev: dict | None, now: dict) -> str | None:
    if prev is None:
        return None  # first build: nothing to compare against
    events = []
    if now["regime"] != prev["regime"]:
        events.append(
            f"**Regime flip: {prev['regime']} → {now['regime']}.** "
            f"The rolling 24-month R² of the demand model is now {now['r2']:.2f} "
            f"(window ending {now['month']}). "
            + (
                "Industrial demand is back in the driver's seat — the mechanism "
                "documented in CS/RES/05 is active."
                if now["regime"] == "demand-driven"
                else "Demand fundamentals have lost their grip on pricing — "
                "policy and positioning dominate for now."
                if now["regime"] == "policy-driven"
                else "The model sits between its regime cutoffs — a decisive "
                "move either way is worth watching."
            )
        )
    if now["tnac_year"] != prev["tnac_year"] or now["msr_kind"] != prev["msr_kind"]:
        events.append(
            f"**New surplus print: TNAC {now['tnac_year']} = {now['tnac'] / 1e6:,.1f}M** "
            f"→ MSR response: {now['msr_kind'].replace('-', ' ')}. "
            f"Thresholds: 24% intake ≥ 1,096M, partial intake > 833M, release < 400M."
        )
    if not events:
        return None
    body = "\n\n".join(events)
    return (
        f"# EUA regime digest — window ending {now['month']}\n\n"
        f"{body}\n\n"
        f"Live monitor: see the dashboard for charts, the auction calendar and "
        f"methodology. Research: {config.RESEARCH_URL}\n\n"
        f"---\n{SUBSCRIBE_HINT}\n"
    )


def write_state(now: dict) -> None:
    config.STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.STATE_JSON.write_text(json.dumps(now, indent=2) + "\n")
