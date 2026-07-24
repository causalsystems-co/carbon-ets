"""Static-HTML dashboard assembly.

Python computes everything and bakes a JSON payload plus pre-rendered
table views into a single self-contained page: no CDN, no external
requests, works as a file, on GitHub Pages, or as a claude.ai artifact.
Charts are drawn client-side into SVG from the baked payload so the
range filter and hover layer stay live.
"""

from __future__ import annotations

import datetime as dt
import html
import json

from . import config

REGIME_META = {
    "demand-driven": {
        "icon": "●",
        "blurb": "Industrial demand is pricing carbon: the two-feature demand model currently explains a large share of monthly EUA returns.",
    },
    "transitional": {
        "icon": "◐",
        "blurb": "Mixed regime: demand fundamentals explain some, but not most, of monthly EUA returns. Watch for a decisive move through either cutoff.",
    },
    "policy-driven": {
        "icon": "○",
        "blurb": "Policy and positioning are pricing carbon: the demand model currently explains little of monthly EUA returns.",
    },
}


def _fmt_m(n: float) -> str:
    return f"{n / 1e6:,.1f}M"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_payload(
    regimes: list[dict],
    prices: list[tuple[str, float]],
    tnac_rows: list[dict],
    tnac_head: dict,
    auction_rows: list[dict],
    panel: list[dict],
    eex_rows: list[dict] | None = None,
) -> dict:
    current = regimes[-1]
    prev = regimes[-2] if len(regimes) > 1 else current
    cover = None
    if eex_rows:
        with_cover = [r for r in eex_rows if r.get("cover")]
        if with_cover:
            latest = with_cover[-1]
            horizon = (
                dt.date.fromisoformat(latest["date"]) - dt.timedelta(days=182)
            ).isoformat()
            window = [r["cover"] for r in with_cover if r["date"] >= horizon]
            cover = {
                "date": latest["date"],
                "ratio": latest["cover"],
                "avg6m": sum(window) / len(window),
            }
    return {
        "cover": cover,
        "auctionEnd": eex_rows[-1]["date"] if eex_rows else None,
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "current": {
            "month": current["month"],
            "r2": current["r2"],
            "r2_prev": prev["r2"],
            "regime": current["regime"],
            "regime_prev": prev["regime"],
        },
        "cutoffs": {
            "demand": config.R2_DEMAND_DRIVEN,
            "policy": config.R2_POLICY_DRIVEN,
        },
        "window": config.ROLLING_WINDOW_MONTHS,
        "regimes": regimes,
        "prices": [[d, round(p, 2)] for d, p in prices],
        "tnac": [
            {
                "year": r["year"],
                "tnac": r["tnac"],
                "kind": r["action"]["kind"],
                "label": r["action"]["label"],
            }
            for r in tnac_rows
        ],
        "tnacHead": tnac_head,
        "msr": {
            "full": config.MSR_UPPER_FULL,
            "upper": config.MSR_UPPER,
            "lower": config.MSR_LOWER,
        },
        "auctions": auction_rows,
        "panelEnd": panel[-1]["month"],
    }



def _subscribe_block() -> str:
    """Fetch-based double-opt-in form when SUBSCRIBE_URL is set, mailto otherwise."""
    if not getattr(config, "SUBSCRIBE_URL", ""):
        mailto = html.escape(config.CONTACT_MAILTO, quote=True)
        return f'      <a class="cta" href="{mailto}">Subscribe by email</a>'
    url = config.SUBSCRIBE_URL.rstrip("/")
    return f"""      <div class="subrow">
        <input id="subemail" type="email" placeholder="you@organisation.org" aria-label="Your email address">
        <button class="cta" type="button" id="subbtn">Subscribe</button>
      </div>
      <p class="submsg" id="submsg"></p>
      <script>
      document.getElementById('subbtn').addEventListener('click', async () => {{
        const box = document.getElementById('submsg');
        const email = document.getElementById('subemail').value.trim();
        if (!email) {{ box.textContent = 'Enter an email address first.'; box.className = 'submsg err'; return; }}
        box.textContent = 'Sending confirmation email…'; box.className = 'submsg';
        try {{
          const r = await fetch('{url}/subscribe', {{
            method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{email}})
          }});
          const d = await r.json();
          if (d.ok) {{
            box.textContent = 'Check your inbox and click the confirmation link (double opt-in).';
            box.className = 'submsg ok';
          }} else {{
            box.textContent = d.error === 'invalid email' ? 'That does not look like a valid address.' : 'Something went wrong, try again later.';
            box.className = 'submsg err';
          }}
        }} catch (e) {{ box.textContent = 'Network error, try again later.'; box.className = 'submsg err'; }}
      }});
      </script>"""


def render(payload: dict) -> str:
    cur = payload["current"]
    meta = REGIME_META[cur["regime"]]
    price_date, price = payload["prices"][-1]
    on_auctions = payload.get("auctionEnd") and price_date <= payload["auctionEnd"]
    price_src = (
        "EEX auction, volume-weighted"
        if on_auctions
        else "CO2.L proxy (EEX feed stale)"
    )
    cover = payload.get("cover")
    head = payload["tnacHead"]
    r2_delta = cur["r2"] - cur["r2_prev"]

    regime_table = _table(
        ["Month", "Rolling 24m R²", "Regime"],
        [[r["month"], f"{r['r2']:.3f}", r["regime"]] for r in payload["regimes"]],
    )
    tnac_table = _table(
        ["Year", "TNAC (allowances)", "MSR response", "Source"],
        [
            [str(r["year"]), f"{r['tnac']:,}", r["action"]["label"], r["source"]]
            for r in payload["_tnac_full"]
        ],
    )
    auction_table_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"muted\">{}</td></tr>".format(
            html.escape(a["date"]),
            html.escape(a["market"]),
            html.escape(a["note"] or "—"),
            html.escape(a["source"]),
        )
        for a in payload["auctions"]
    )

    flip = cur["regime"] != cur["regime_prev"]
    flip_note = (
        f"Regime changed from {cur['regime_prev']} in the latest window — "
        "digest subscribers were notified."
        if flip
        else f"Unchanged from the prior window ({cur['r2_prev']:.2f})."
    )

    slim = {k: v for k, v in payload.items() if not k.startswith("_")}
    page = (
        HTML_TEMPLATE.replace("__DATA__", json.dumps(slim, separators=(",", ":")))
        .replace("__REGIME__", html.escape(cur["regime"]))
        .replace("__REGIME_ICON__", meta["icon"])
        .replace("__REGIME_BLURB__", html.escape(meta["blurb"]))
        .replace("__FLIP_NOTE__", html.escape(flip_note))
        .replace("__R2__", f"{cur['r2']:.2f}")
        .replace("__R2_PCT__", f"{cur['r2'] * 100:.0f}%")
        .replace("__R2_DELTA__", f"{r2_delta:+.2f}")
        .replace("__ASOF__", cur["month"])
        .replace("__PRICE__", f"€{price:,.2f}")
        .replace("__PRICE_DATE__", price_date)
        .replace("__PRICE_SRC__", price_src)
        .replace("__COVER__", f"{cover['ratio']:.2f}" if cover else "—")
        .replace("__COVER_DATE__", cover["date"] if cover else "n/a")
        .replace("__COVER_AVG__", f"{cover['avg6m']:.2f}" if cover else "—")
        .replace("__TNAC_M__", _fmt_m(head["tnac"]))
        .replace("__TNAC_YEAR__", str(head["year"]))
        .replace("__TNAC_DELTA__", _fmt_m(head["tnac"] - payload["tnac"][-2]["tnac"]))
        .replace("__INTAKE_M__", _fmt_m(abs(head["action"]["amount"])))
        .replace("__INTAKE_KIND__", html.escape(head["action"]["kind"].replace("-", " ")))
        .replace("__INTAKE_PERIOD__", head["intake_period"])
        .replace(
            "__NEXT_AUCTION__",
            dt.date.fromisoformat(payload["auctions"][0]["date"]).strftime("%-d %b")
            if payload["auctions"]
            else "—",
        )
        .replace("__NEXT_AUCTION_MKT__", html.escape(payload["auctions"][0]["market"]) if payload["auctions"] else "")
        .replace("__AUCTION_ROWS__", auction_table_rows)
        .replace("__REGIME_TABLE__", regime_table)
        .replace("__TNAC_TABLE__", tnac_table)
        .replace("__BUILT__", payload["built"])
        .replace("__PANEL_END__", payload["panelEnd"])
        .replace("__WINDOW__", str(payload["window"]))
        .replace("__CUT_HI__", f"{config.R2_DEMAND_DRIVEN:.2f}")
        .replace("__CUT_LO__", f"{config.R2_POLICY_DRIVEN:.2f}")
        .replace("__RESEARCH_URL__", config.RESEARCH_URL)
        .replace("__TOOLKIT_URL__", config.TOOLKIT_URL)
        .replace("__MAILTO__", html.escape(config.CONTACT_MAILTO, quote=True))
        .replace("__SUBSCRIBE_BLOCK__", _subscribe_block())
    )
    return page


HTML_TEMPLATE = r"""<title>EUA Regime Monitor — Causal Systems</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --hairline: rgba(11,11,11,0.10);
  --series: #2a78d6; --series-wash: rgba(42,120,214,0.10);
  --good: #0ca30c; --warn: #fab219; --serious: #ec835a;
  --good-wash: rgba(12,163,12,0.11); --warn-wash: rgba(250,178,25,0.13);
  --serious-wash: rgba(236,131,90,0.13);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --hairline: rgba(255,255,255,0.10);
    --series: #3987e5; --series-wash: rgba(57,135,229,0.14);
    --good-wash: rgba(12,163,12,0.16); --warn-wash: rgba(250,178,25,0.14);
    --serious-wash: rgba(236,131,90,0.14);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --hairline: rgba(255,255,255,0.10);
  --series: #3987e5; --series-wash: rgba(57,135,229,0.14);
  --good-wash: rgba(12,163,12,0.16); --warn-wash: rgba(250,178,25,0.14);
  --serious-wash: rgba(236,131,90,0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
a { color: var(--series); text-decoration: none; }
a:hover, a:focus-visible { text-decoration: underline; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 20px 56px; }
header.top {
  display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: baseline;
  padding-bottom: 18px;
}
.wordmark { font-weight: 650; font-size: 17px; letter-spacing: 0.01em; }
.wordmark span { color: var(--muted); font-weight: 450; }
.doccode {
  font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--ink-2); border: 1px solid var(--hairline); border-radius: 999px;
  padding: 1px 10px;
}
.top .asof { margin-left: auto; color: var(--muted); font-size: 13px; }
.eyebrow {
  font-size: 11px; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; margin: 0 0 6px;
}
.card {
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 10px; padding: 20px 22px; margin-top: 16px;
}
.hero { display: flex; flex-wrap: wrap; gap: 18px 36px; align-items: center; }
.hero .light { display: flex; align-items: center; gap: 14px; }
.hero .dot { font-size: 30px; line-height: 1; }
.hero .dot.demand-driven { color: var(--good); }
.hero .dot.transitional { color: var(--warn); }
.hero .dot.policy-driven { color: var(--serious); }
.hero h1 {
  margin: 0; font-size: 42px; font-weight: 650; letter-spacing: -0.015em;
  text-transform: capitalize; text-wrap: balance;
}
.hero .sub { max-width: 46ch; color: var(--ink-2); font-size: 14px; }
.hero .sub strong { color: var(--ink); }
.tiles {
  display: grid; gap: 12px; margin-top: 16px;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
}
.tile {
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: 10px; padding: 14px 16px;
}
.tile .label { font-size: 12px; color: var(--muted); }
.tile .value { font-size: 26px; font-weight: 620; margin-top: 2px; white-space: nowrap; }
.tile .delta { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.tile .delta.up { color: var(--good); }
.section-head { margin: 30px 0 0; }
.section-head h2 { margin: 0; font-size: 19px; font-weight: 650; }
.section-head p { margin: 2px 0 0; color: var(--ink-2); font-size: 13.5px; }
.filters { display: flex; gap: 8px; align-items: center; margin-top: 14px; }
.filters .flabel { font-size: 12px; color: var(--muted); margin-right: 4px; }
.filters button {
  font: 600 13px/1 system-ui, sans-serif; color: var(--ink-2);
  background: none; border: 1px solid var(--hairline); border-radius: 999px;
  padding: 6px 14px; cursor: pointer;
}
.filters button[aria-pressed="true"] { color: var(--ink); border-color: var(--series); background: var(--series-wash); }
.filters button:focus-visible { outline: 2px solid var(--series); outline-offset: 2px; }
.chart-title { margin: 0; font-size: 15px; font-weight: 650; }
.chart-sub { margin: 1px 0 10px; color: var(--muted); font-size: 12.5px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 10px 0 0; font-size: 12.5px; color: var(--ink-2); }
.chips .chip { display: inline-flex; gap: 6px; align-items: center; }
.chips .chip .g { font-size: 13px; }
.chip .g.demand-driven { color: var(--good); }
.chip .g.transitional { color: var(--warn); }
.chip .g.policy-driven { color: var(--serious); }
.chartbox { position: relative; margin-top: 6px; }
.chartbox svg { display: block; width: 100%; height: auto; }
.chartbox:focus-visible { outline: 2px solid var(--series); outline-offset: 4px; border-radius: 6px; }
.tip {
  position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--hairline); border-radius: 8px; padding: 7px 10px;
  font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,0.10);
  display: none; min-width: 130px; z-index: 3;
}
.tip .tv { font-size: 15px; font-weight: 650; color: var(--ink); }
.tip .tl { color: var(--ink-2); }
.tip .tk { display: inline-block; width: 12px; height: 0; border-top: 3px solid var(--series); border-radius: 2px; vertical-align: middle; margin-right: 6px; }
details.tblview { margin-top: 12px; }
details.tblview summary { cursor: pointer; font-size: 12.5px; color: var(--muted); }
details.tblview[open] summary { margin-bottom: 8px; }
.tscroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; }
th, td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid); }
td { font-variant-numeric: tabular-nums; color: var(--ink-2); }
td:first-child { color: var(--ink); }
.muted { color: var(--muted); }
.grid2 { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 860px) { .grid2 { grid-template-columns: 3fr 2fr; } }
.digest p { color: var(--ink-2); font-size: 14px; max-width: 60ch; }
.cta {
  display: inline-block; background: var(--series); color: #fff;
  font-weight: 600; font-size: 14px; border-radius: 8px; padding: 9px 18px;
}
.cta:hover, .cta:focus-visible { text-decoration: none; filter: brightness(1.08); }
footer.method { margin-top: 34px; color: var(--ink-2); font-size: 13px; }
footer.method h2 { font-size: 14px; color: var(--ink); margin: 0 0 6px; }
footer.method p { max-width: 76ch; margin: 6px 0; }
.legalese { color: var(--muted); font-size: 12px; }
.subrow { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 10px; }
.subrow input { flex: 1 1 240px; max-width: 340px; background: var(--page); color: var(--ink);
  border: 1px solid var(--hairline); border-radius: 8px; padding: 10px 12px; font: 14px/1 inherit; }
.subrow input:focus { outline: 2px solid var(--series); outline-offset: 1px; }
.subrow .cta { cursor: pointer; border: 0; }
.submsg { font-size: 13px; margin-top: 6px; }
.submsg.ok { color: var(--good); } .submsg.err { color: var(--serious); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header class="top">
    <div class="wordmark">Causal Systems <span>/ EUA Regime Monitor</span></div>
    <span class="doccode">CS/RES/05 &middot; live companion</span>
    <div class="asof">Model through __PANEL_END__ &middot; built __BUILT__</div>
  </header>

  <div class="card hero">
    <div>
      <p class="eyebrow">Current regime &middot; rolling __WINDOW__-month window</p>
      <div class="light">
        <span class="dot __REGIME__" aria-hidden="true">__REGIME_ICON__</span>
        <h1>__REGIME__</h1>
      </div>
    </div>
    <div class="sub">
      <p><strong>R&sup2; = __R2__</strong> &mdash; the demand model currently explains
      <strong>__R2_PCT__</strong> of monthly EUA return variance. __REGIME_BLURB__</p>
      <p>__FLIP_NOTE__</p>
    </div>
  </div>

  <div class="tiles">
    <div class="tile"><div class="label">Rolling R&sup2; (__ASOF__)</div>
      <div class="value">__R2__</div><div class="delta">__R2_DELTA__ vs prior window</div></div>
    <div class="tile"><div class="label">EUA clear (__PRICE_DATE__)</div>
      <div class="value">__PRICE__</div><div class="delta">__PRICE_SRC__</div></div>
    <div class="tile"><div class="label">Cover ratio (__COVER_DATE__)</div>
      <div class="value">__COVER__&times;</div><div class="delta">6m avg __COVER_AVG__&times;</div></div>
    <div class="tile"><div class="label">TNAC __TNAC_YEAR__</div>
      <div class="value">__TNAC_M__</div><div class="delta">__TNAC_DELTA__ vs prior year</div></div>
    <div class="tile"><div class="label">MSR &middot; __INTAKE_PERIOD__</div>
      <div class="value">__INTAKE_M__</div><div class="delta">__INTAKE_KIND__</div></div>
    <div class="tile"><div class="label">Next auction (indicative)</div>
      <div class="value">__NEXT_AUCTION__</div><div class="delta">__NEXT_AUCTION_MKT__</div></div>
  </div>

  <div class="section-head">
    <h2>Market signal</h2>
    <p>The regime indicator and the price history it classifies.</p>
  </div>

  <div class="filters" role="group" aria-label="Date range for the two time charts">
    <span class="flabel">Range</span>
    <button data-range="12" aria-pressed="false">1Y</button>
    <button data-range="36" aria-pressed="false">3Y</button>
    <button data-range="0" aria-pressed="true">Max</button>
  </div>

  <div class="card">
    <h3 class="chart-title">When does industrial demand price carbon?</h3>
    <p class="chart-sub">Rolling __WINDOW__-month R&sup2; of monthly EUA returns on euro-area IP growth, Stoxx&nbsp;50 momentum and HDD anomaly. Cutoffs: __CUT_HI__ / __CUT_LO__.</p>
    <div class="chartbox" id="r2chart" tabindex="0" role="img"
         aria-label="Line chart of rolling R squared by month with regime bands. Data in the table below."></div>
    <div class="chips" id="regimechips"></div>
    <details class="tblview"><summary>Table view</summary><div class="tscroll">__REGIME_TABLE__</div></details>
  </div>

  <div class="card">
    <h3 class="chart-title">EUA price, shaded by regime</h3>
    <p class="chart-sub">EEX primary-auction clearing prices, volume-weighted per auction day, EUR/tCO2. Shading applies from the first complete __WINDOW__-month model window.</p>
    <div class="chartbox" id="pricechart" tabindex="0" role="img"
         aria-label="Line chart of the EUA price with regime shading. Monthly data in the table above."></div>
  </div>

  <div class="section-head">
    <h2>Supply mechanics</h2>
    <p>Annual surplus indicator and the Market Stability Reserve response.</p>
  </div>

  <div class="card">
    <h3 class="chart-title">TNAC vs the MSR thresholds</h3>
    <p class="chart-sub">Total number of allowances in circulation, per the Commission&rsquo;s annual May communications.</p>
    <div class="chartbox" id="tnacchart" tabindex="0" role="img"
         aria-label="Bar chart of TNAC by year against MSR thresholds. Data in the table below."></div>
    <details class="tblview"><summary>Table view &amp; sources</summary><div class="tscroll">__TNAC_TABLE__</div></details>
  </div>

  <div class="grid2">
    <div class="card">
      <h3 class="chart-title">Auction calendar</h3>
      <p class="chart-sub">Next EUA primary auctions. Indicative weekday pattern &mdash; the published EEX calendar governs.</p>
      <div class="tscroll"><table>
        <thead><tr><th>Date</th><th>Market</th><th>Note</th><th>Source</th></tr></thead>
        <tbody>__AUCTION_ROWS__</tbody>
      </table></div>
    </div>
    <div class="card digest">
      <h3 class="chart-title">The regime digest</h3>
      <p>A short email when the regime flips or TNAC crosses an MSR threshold
      &mdash; the two events this monitor exists to catch. No schedule, no noise:
      it only sends when the state changes.</p>
__SUBSCRIBE_BLOCK__
      <p class="legalese">Or watch this page &mdash; it rebuilds daily from public data.</p>
    </div>
  </div>

  <footer class="method">
    <h2>Method &amp; sources</h2>
    <p>Companion monitor to <a href="__RESEARCH_URL__">CS/RES/05 &mdash; Factories to
    Carbon</a>. Monthly EUA log returns are regressed on euro-area industrial
    production YoY growth (Eurostat sts_inpr_m, EA20, B&ndash;D, SCA), Stoxx&nbsp;50
    3-month momentum, and a population-weighted heating-degree-day anomaly
    (Open-Meteo, six-city basket), over a rolling __WINDOW__-month window. The
    window R&sup2; is the regime indicator; cutoffs at __CUT_HI__ and __CUT_LO__ mirror the
    regime boundaries documented in the paper.</p>
    <p>The price series is the same one the paper uses: EEX primary-auction
    clearing prices from 2012 to present (EU27, DE and PL common auctions,
    EUA spot contracts, volume-weighted per auction day), from the public EEX
    auction reports &mdash; topped up live from the current-year report, with the
    WisdomTree Carbon ETC as a recency fallback only. The cover-ratio tile is
    total bids over auctioned volume from the same reports. TNAC figures are
    the European Commission&rsquo;s annual MSR communications; intake arithmetic
    follows Decision (EU)&nbsp;2015/1814 as amended by
    Decision&nbsp;(EU)&nbsp;2023/852. Open-source pipeline:
    <a href="__TOOLKIT_URL__">carbon-ets</a>.</p>
    <p class="legalese">Research demonstration on public data. Not investment
    advice. &copy; Causal Systems, a models brand of Yarim Trade UG.</p>
  </footer>
</div>

<script>
"use strict";
const DATA = __DATA__;
const CSS = getComputedStyle(document.documentElement);
const V = name => CSS.getPropertyValue(name).trim();
const NS = "http://www.w3.org/2000/svg";
const REGIME_WASH = { "demand-driven": "--good-wash", "transitional": "--warn-wash", "policy-driven": "--serious-wash" };
const REGIME_ICON = { "demand-driven": "●", "transitional": "◐", "policy-driven": "○" };

function el(name, attrs, parent) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}
function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + 0.5) || 10 * mag;
  const t0 = Math.ceil(lo / step) * step, out = [];
  for (let t = t0; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
function monthNum(m) { return +m.slice(0, 4) * 12 + (+m.slice(5, 7) - 1); }
function dateNum(d) { const t = new Date(d + "T00:00:00Z"); return t.getTime() / 86400000; }

/* Generic time-series line chart with optional background bands,
   dashed reference lines, endpoint label, crosshair + tooltip. */
function lineChart(box, opts) {
  box.textContent = "";
  const W = 960, H = 300, m = { t: 16, r: opts.mr || 74, b: 30, l: 46 };
  const pts = opts.points; // [{x(number), label, y, extra}]
  if (!pts.length) return;
  const xs = pts.map(p => p.x);
  const xlo = xs[0], xhi = xs[xs.length - 1];
  let ylo = opts.ymin !== undefined ? opts.ymin : Math.min(...pts.map(p => p.y));
  let yhi = Math.max(...pts.map(p => p.y), ...(opts.refs || []).map(r => r.y));
  const pad = (yhi - ylo) * 0.08 || 1; yhi += pad; if (opts.ymin === undefined) ylo -= pad;
  const X = x => m.l + (x - xlo) / (xhi - xlo || 1) * (W - m.l - m.r);
  const Y = y => H - m.b - (y - ylo) / (yhi - ylo || 1) * (H - m.t - m.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, box);

  for (const band of opts.bands || []) {
    const bx = Math.max(m.l, X(band.x0));
    const bx1 = Math.min(W - m.r, X(band.x1));
    el("rect", {
      x: bx, y: m.t, width: Math.max(0, bx1 - bx),
      height: H - m.t - m.b, fill: `var(${band.wash})`,
    }, svg);
  }
  const yticks = niceTicks(ylo, yhi, 4);
  for (const t of yticks) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    const lab = el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11, style: "font-variant-numeric:tabular-nums" }, svg);
    lab.textContent = opts.yFmt(t);
  }
  for (const tick of opts.xTicks) {
    el("line", { x1: X(tick.x), x2: X(tick.x), y1: H - m.b, y2: H - m.b + 4, stroke: "var(--baseline)", "stroke-width": 1 }, svg);
    const lab = el("text", { x: X(tick.x), y: H - m.b + 17, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }, svg);
    lab.textContent = tick.label;
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--baseline)", "stroke-width": 1 }, svg);

  for (const ref of opts.refs || []) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(ref.y), y2: Y(ref.y), stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
    const lab = el("text", { x: W - m.r + 6, y: Y(ref.y) + 4, fill: "var(--muted)", "font-size": 11 }, svg);
    lab.textContent = ref.label;
  }
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
  el("path", { d, fill: "none", stroke: "var(--series)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  const last = pts[pts.length - 1];
  el("circle", { cx: X(last.x), cy: Y(last.y), r: 4.5, fill: "var(--series)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
  const endLab = el("text", { x: X(last.x) + 9, y: Y(last.y) + 4, fill: "var(--ink)", "font-size": 12, "font-weight": 650 }, svg);
  endLab.textContent = opts.yFmt(last.y);

  // Hover layer: crosshair snapping to the nearest point, one tooltip.
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--baseline)", "stroke-width": 1, visibility: "hidden" }, svg);
  const dot = el("circle", { r: 4.5, fill: "var(--series)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg);
  const tip = document.createElement("div");
  tip.className = "tip"; box.appendChild(tip);
  const tv = document.createElement("div"); tv.className = "tv";
  const tl = document.createElement("div"); tl.className = "tl";
  const key = document.createElement("span"); key.className = "tk";
  tip.append(tv, tl);
  let idx = -1;
  function show(i) {
    idx = Math.max(0, Math.min(pts.length - 1, i));
    const p = pts[idx];
    cross.setAttribute("x1", X(p.x)); cross.setAttribute("x2", X(p.x));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", X(p.x)); dot.setAttribute("cy", Y(p.y));
    dot.setAttribute("visibility", "visible");
    tv.textContent = ""; tv.append(key.cloneNode(), document.createTextNode(opts.yFmt(p.y)));
    tl.textContent = p.extra ? `${p.label} · ${p.extra}` : p.label;
    tip.style.display = "block";
    const rect = box.getBoundingClientRect(), fx = X(p.x) / W * rect.width;
    tip.style.left = Math.min(rect.width - tip.offsetWidth - 4, Math.max(0, fx + 12)) + "px";
    tip.style.top = Math.max(0, Y(p.y) / H * rect.height - tip.offsetHeight - 10) + "px";
  }
  function hide() { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.style.display = "none"; idx = -1; }
  svg.addEventListener("pointermove", ev => {
    const rect = svg.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width * W;
    let best = 0, bd = Infinity;
    for (let i = 0; i < pts.length; i++) { const dd = Math.abs(X(pts[i].x) - x); if (dd < bd) { bd = dd; best = i; } }
    show(best);
  });
  svg.addEventListener("pointerleave", hide);
  box.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      ev.preventDefault();
      show((idx < 0 ? pts.length - 1 : idx) + (ev.key === "ArrowRight" ? 1 : -1));
    } else if (ev.key === "Escape") hide();
  });
  box.addEventListener("focus", () => show(pts.length - 1));
  box.addEventListener("blur", hide);
}

function yearTicks(points, xOf) {
  const out = [], seen = new Set();
  for (const p of points) {
    const yr = p.iso.slice(0, 4);
    if (!seen.has(yr)) { seen.add(yr); out.push({ x: xOf(p), label: yr }); }
  }
  return out.length > 1 ? out.slice(1) : out; // first partial year crowds the axis
}

function regimeBands(regimes, xOfMonth) {
  const bands = [];
  let cur = null;
  for (const r of regimes) {
    const x0 = xOfMonth(r.month), x1 = xOfMonth(r.month) + 1;
    if (cur && cur.regime === r.regime) { cur.x1 = x1; continue; }
    if (cur) bands.push(cur);
    cur = { regime: r.regime, x0, x1, wash: REGIME_WASH[r.regime] };
  }
  if (cur) bands.push(cur);
  return bands;
}

function drawR2(rangeMonths) {
  let rs = DATA.regimes;
  if (rangeMonths) rs = rs.slice(-rangeMonths);
  const pts = rs.map(r => ({ x: monthNum(r.month), y: r.r2, label: r.month, extra: r.regime, iso: r.month }));
  lineChart(document.getElementById("r2chart"), {
    points: pts,
    ymin: 0,
    mr: 116,
    yFmt: v => (v * 100).toFixed(0) + "%",
    refs: [
      { y: DATA.cutoffs.demand, label: "demand ≥ " + DATA.cutoffs.demand.toFixed(2) },
      { y: DATA.cutoffs.policy, label: "policy ≤ " + DATA.cutoffs.policy.toFixed(2) },
    ],
    bands: regimeBands(rs, m => monthNum(m)).map(b => ({ x0: b.x0, x1: b.x1, wash: b.wash })),
    xTicks: yearTicks(pts, p => p.x),
  });
}

function drawPrice(rangeMonths) {
  let ps = DATA.prices;
  if (rangeMonths) {
    const cutoff = new Date(); cutoff.setUTCMonth(cutoff.getUTCMonth() - rangeMonths);
    const iso = cutoff.toISOString().slice(0, 10);
    ps = ps.filter(p => p[0] >= iso);
  }
  const pts = ps.map(p => ({ x: dateNum(p[0]), y: p[1], label: p[0], iso: p[0] }));
  const monthDays = m => dateNum(m + "-01");
  const visible = DATA.regimes.filter(r => pts.length && monthDays(r.month) + 31 >= pts[0].x);
  let cur = null; const dayBands = [];
  for (const r of visible) {
    const x0 = Math.max(pts[0].x, monthDays(r.month)), x1 = monthDays(r.month) + 31;
    if (cur && cur.regime === r.regime) { cur.x1 = x1; continue; }
    if (cur) dayBands.push(cur);
    cur = { regime: r.regime, x0, x1, wash: REGIME_WASH[r.regime] };
  }
  if (cur) { cur.x1 = Math.min(cur.x1, pts[pts.length - 1].x); dayBands.push(cur); }
  lineChart(document.getElementById("pricechart"), {
    points: pts,
    yFmt: v => "€" + (v >= 100 ? v.toFixed(0) : v.toFixed(1)),
    bands: dayBands,
    xTicks: yearTicks(pts, p => p.x),
  });
}

function drawTnac() {
  const box = document.getElementById("tnacchart");
  box.textContent = "";
  const W = 960, H = 300, m = { t: 18, r: 150, b: 30, l: 56 };
  const rows = DATA.tnac;
  const yhi = 1800, ylo = 0; // millions; comfortably above the 1.69B peak
  const Y = vM => H - m.b - (vM - ylo) / (yhi - ylo) * (H - m.t - m.b);
  const band = (W - m.l - m.r) / rows.length;
  const bw = Math.min(24, band - 2);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, box);
  for (const t of niceTicks(0, yhi, 4)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "var(--grid)", "stroke-width": 1 }, svg);
    const lab = el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11, style: "font-variant-numeric:tabular-nums" }, svg);
    lab.textContent = t.toLocaleString("en-US") + "M";
  }
  const refs = [
    { v: DATA.msr.full, label: "24% intake ≥ 1,096M" },
    { v: DATA.msr.upper, label: "partial intake > 833M" },
    { v: DATA.msr.lower, label: "release < 400M" },
  ];
  for (const ref of refs) {
    const ry = Y(ref.v / 1e6);
    el("line", { x1: m.l, x2: W - m.r, y1: ry, y2: ry, stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
    const lab = el("text", { x: W - m.r + 6, y: ry + 4, fill: "var(--muted)", "font-size": 11 }, svg);
    lab.textContent = ref.label;
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--baseline)", "stroke-width": 1 }, svg);

  const tip = document.createElement("div");
  tip.className = "tip"; box.appendChild(tip);
  const tv = document.createElement("div"); tv.className = "tv";
  const tl = document.createElement("div"); tl.className = "tl";
  tip.append(tv, tl);
  rows.forEach((row, i) => {
    const cx = m.l + band * (i + 0.5);
    const y = Y(row.tnac / 1e6);
    const barH = H - m.b - y;
    const bar = el("path", {
      d: `M${cx - bw / 2},${H - m.b} L${cx - bw / 2},${y + 4} Q${cx - bw / 2},${y} ${cx - bw / 2 + 4},${y} L${cx + bw / 2 - 4},${y} Q${cx + bw / 2},${y} ${cx + bw / 2},${y + 4} L${cx + bw / 2},${H - m.b} Z`,
      fill: "var(--series)",
    }, svg);
    const xl = el("text", { x: cx, y: H - m.b + 17, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }, svg);
    xl.textContent = row.year;
    if (i === rows.length - 1) {
      const cap = el("text", { x: cx, y: y - 8, "text-anchor": "middle", fill: "var(--ink)", "font-size": 12, "font-weight": 650 }, svg);
      cap.textContent = (row.tnac / 1e6).toFixed(0) + "M";
    }
    const hit = el("rect", { x: m.l + band * i, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent" }, svg);
    function show() {
      bar.setAttribute("fill-opacity", "0.82");
      tv.textContent = (row.tnac / 1e6).toLocaleString("en-US", { maximumFractionDigits: 1 }) + "M";
      tl.textContent = `TNAC ${row.year} · ${row.label}`;
      tip.style.display = "block";
      const rect = box.getBoundingClientRect();
      tip.style.left = Math.min(rect.width - tip.offsetWidth - 4, cx / W * rect.width + 10) + "px";
      tip.style.top = Math.max(0, y / H * rect.height - 8) + "px";
    }
    function hide() { bar.removeAttribute("fill-opacity"); tip.style.display = "none"; }
    hit.addEventListener("pointerenter", show);
    hit.addEventListener("pointerleave", hide);
  });
}

function drawChips() {
  const box = document.getElementById("regimechips");
  box.textContent = "";
  for (const regime of ["demand-driven", "transitional", "policy-driven"]) {
    const chip = document.createElement("span"); chip.className = "chip";
    const g = document.createElement("span"); g.className = "g " + regime;
    g.textContent = REGIME_ICON[regime];
    chip.append(g, document.createTextNode(regime));
    box.appendChild(chip);
  }
}

let currentRange = 0;
function redraw() { drawR2(currentRange); drawPrice(currentRange); }
document.querySelectorAll(".filters button").forEach(btn => {
  btn.addEventListener("click", () => {
    currentRange = +btn.dataset.range;
    document.querySelectorAll(".filters button").forEach(b => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
    redraw();
  });
});
drawChips(); redraw(); drawTnac();
let resizeTimer = 0;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { redraw(); drawTnac(); }, 150); });
</script>
"""
