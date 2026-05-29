"""
01_fetch_prices.py — daily price inputs for the EU ETS causal chain.

What we get for free
--------------------
KRBN  (KraneShares Global Carbon ETF, USD)     — proxy for EUA front-month
GRN   (iPath Series B Carbon ETN, USD)         — second EUA proxy (lower AUM)
TTF=F (Dutch TTF gas front-month, EUR/MWh)     — real product
CL=F  (WTI front-month, USD/bbl)               — global energy proxy
^STOXX50E (Euro Stoxx 50)                      — Eurozone risk-on/off
^VSTOXX                                        — Eurozone vol gauge
EURUSD=X                                       — FX, EUA is EUR-denominated

What we'd want with a paid feed
-------------------------------
ICE EUA front-month settlement (EUR/tonne)     — the actual EUA price
EEX primary-auction clearing (EUR/tonne, daily)— supply-side anchor
S&P Global / Argus EUA assessments             — secondary-market depth

KRBN is a *weighted basket* of EUA + RGGI + CCA. For a first-cut chain
analysis the correlation with pure EUA is ~0.85 — directionally fine,
quantitatively lossy. Maurizio: ICE provides EUA settlement free with
a 24-hr delay via the ICE public reports page; scraping that into a
daily parquet is a high-value upgrade.

Output: data/prices_daily.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

TICKERS = {
    "KRBN":      "carbon_proxy_krbn",
    "GRN":       "carbon_proxy_grn",
    "TTF=F":     "ttf_gas_eur_mwh",
    "CL=F":      "wti_usd_bbl",
    "^STOXX50E": "stoxx50",
    "^VSTOXX":   "vstoxx",
    "EURUSD=X":  "eurusd",
}


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        print(f"  WARN: {ticker} returned no rows")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else pd.to_datetime(s.index)
    out = s.to_frame(name=TICKERS[ticker])
    out.index.name = "Date"
    return out.reset_index()


def main(start: str, end: str) -> None:
    frames = []
    for tkr in TICKERS:
        print(f"fetch {tkr}")
        df = fetch_one(tkr, start, end)
        if not df.empty:
            frames.append(df.set_index("Date"))
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    out = out.ffill(limit=3)
    path = DATA / "prices_daily.parquet"
    out.to_parquet(path)
    print(f"wrote {path}  rows={len(out)}  cols={list(out.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    main(args.start, args.end)
