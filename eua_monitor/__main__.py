"""Build the EUA regime monitor: `python -m eua_monitor build`.

Fetches public data (with cache fallback), fits the rolling demand model,
renders site/index.html, refreshes site/state.json, and — when the regime
or the MSR state changed since the last build — writes site/digest.md.
Exit code is always 0 on a successful build; the presence of a fresh
digest.md is the signal CI uses to send email.
"""

from __future__ import annotations

import sys

from . import auctions, config, dashboard, digest, fetch, model, msr


def build() -> int:
    eua = fetch.eua_prices()
    stoxx = fetch.stoxx_prices()
    ip = fetch.industrial_production()
    hdd = fetch.heating_degree_days()

    panel = model.build_panel(eua, stoxx, ip, hdd)
    if len(panel) < config.ROLLING_WINDOW_MONTHS + 1:
        print(
            f"error: only {len(panel)} aligned months; "
            f"need > {config.ROLLING_WINDOW_MONTHS}",
            file=sys.stderr,
        )
        return 1
    regimes = model.regime_series(model.rolling_r2(panel))

    tnac_rows = msr.enrich(fetch.tnac_history())
    try:
        eex_rows = fetch.eex_auctions()
    except fetch.DataUnavailable:
        eex_rows = None
    payload = dashboard.build_payload(
        regimes=regimes,
        prices=eua,
        tnac_rows=tnac_rows,
        tnac_head=msr.headline(tnac_rows),
        auction_rows=auctions.upcoming(),
        panel=panel,
        eex_rows=eex_rows,
    )
    payload["_tnac_full"] = tnac_rows

    prev = digest.previous_state()
    now = digest.current_state(payload)
    note = digest.diff_digest(prev, now)

    config.SITE_DIR.mkdir(parents=True, exist_ok=True)
    config.DASHBOARD_HTML.write_text("<!doctype html>\n" + dashboard.render(payload))
    digest.write_state(now)
    if note:
        config.DIGEST_MD.write_text(note)
        print(f"digest written: {config.DIGEST_MD}")
    elif config.DIGEST_MD.exists():
        config.DIGEST_MD.unlink()  # stale digest from an earlier state change

    latest = regimes[-1]
    print(
        f"built {config.DASHBOARD_HTML} — regime {latest['regime']} "
        f"(R²={latest['r2']:.2f}, window ending {latest['month']})"
    )
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        return build()
    print(f"unknown command: {cmd} (expected: build)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
