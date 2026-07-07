"""
02_fetch_emissions.py — fundamental drivers of EUA demand.

Hardened for first-clone usability: every external call has a timeout,
year-chunked where applicable, and falls back to a sensible proxy
if the source is down. Maurizio should always get a working
emissions_drivers.parquet, even if one of the three sources is unreachable.

Sources (all free, no API keys):

1. Eurostat short-term IP index (EA19, monthly)        — fast, reliable
2. Open-Meteo ERA5 daily mean temp (Frankfurt)          — chunked by year
3. Fraunhofer Energy-Charts EU5 load (DE+FR+IT+ES+PL)   — optional, may
   time out from some networks; if so we fall back to a temp-based
   load proxy that captures seasonality but not real demand shocks.

The proxy is honest about its limits: it's tagged `load_eu5_mw_proxy`.
When you wire real ENTSO-E (via the entsoe-py library + free API token
from transparency.entsoe.eu), rename it to `load_eu5_mw` and the rest
of the pipeline picks it up.

Output: data/emissions_drivers.parquet
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FRAUNHOFER_LOAD = "https://api.energy-charts.info/total_power"
# Eurostat: try SDMX-CSV (needs an Accept header), fall back to legacy JSON.
EUROSTAT_SDMX = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "sts_inpr_m/M.SCA.I21.B-D.EA19"
)
EUROSTAT_JSON = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "sts_inpr_m?format=JSON&geo=EA19&s_adj=SCA&unit=I21&nace_r2=B-D"
)
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
# Fraunhofer renamed/removed several country endpoints — try uppercase first.
COUNTRIES = ["de", "fr", "it", "es", "pl"]
TIMEOUT = 20


def fetch_eurostat_ip() -> pd.DataFrame:
    """EA19 industrial production (NACE B-D, seasonally adjusted, 2021=100).

    Tries the SDMX-CSV endpoint first (live vintage, freshest data),
    then falls back to the legacy JSON-stat endpoint if SDMX 406s.
    """
    # 1) SDMX-CSV — requires explicit Accept header.
    try:
        r = requests.get(
            EUROSTAT_SDMX,
            headers={"Accept": "text/csv"},
            params={"format": "SDMX_CSV"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
            df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "ip_ea19"})
            df = df[["date", "ip_ea19"]].copy()
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            return df.resample("D").ffill()
    except Exception as e:
        print(f"  SDMX failed ({e}); falling back to JSON-stat")

    # 2) Fallback: legacy JSON-stat endpoint.
    r = requests.get(EUROSTAT_JSON, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    idx = j["dimension"]["time"]["category"]["index"]
    vals = j["value"]
    rows = [
        {"date": pd.Timestamp(label + "-01"), "ip_ea19": vals.get(str(k))}
        for label, k in idx.items() if vals.get(str(k)) is not None
    ]
    ip = pd.DataFrame(rows).set_index("date").sort_index()
    return ip.resample("D").ffill()


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Open-Meteo, chunked by year so a 5+ year window doesn't time out."""
    start_y = pd.Timestamp(start).year
    end_y = pd.Timestamp(end).year
    chunks = []
    for yr in range(start_y, end_y + 1):
        params = {
            "latitude": 50.11, "longitude": 8.68,
            "start_date": f"{yr}-01-01",
            "end_date":   min(f"{yr}-12-31", end),
            "daily": "temperature_2m_mean", "timezone": "UTC",
        }
        try:
            r = requests.get(OPEN_METEO, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()
            c = pd.DataFrame({
                "date": pd.to_datetime(j["daily"]["time"]),
                "tmean_c_frankfurt": j["daily"]["temperature_2m_mean"],
            })
            chunks.append(c)
        except Exception as e:
            print(f"  WARN open-meteo {yr}: {e}")
    if not chunks:
        return pd.DataFrame()
    w = pd.concat(chunks).set_index("date").sort_index()
    w["hdd_frankfurt"] = (18 - w["tmean_c_frankfurt"]).clip(lower=0)
    w["cdd_frankfurt"] = (w["tmean_c_frankfurt"] - 22).clip(lower=0)
    return w


def fetch_load(country: str, start: str, end: str) -> pd.DataFrame:
    """Daily mean MW from Fraunhofer. Tries lowercase, then uppercase
    country code — Fraunhofer's API changed casing on some markets."""
    params_lo = {"country": country.lower(), "start": start, "end": end}
    r = requests.get(FRAUNHOFER_LOAD, params=params_lo, timeout=15,
                     headers={"User-Agent": "causal-trading/0.1"})
    if r.status_code == 404:
        # retry uppercase
        params_up = {"country": country.upper(), "start": start, "end": end}
        r = requests.get(FRAUNHOFER_LOAD, params=params_up, timeout=15,
                         headers={"User-Agent": "causal-trading/0.1"})
    r.raise_for_status()
    j = r.json()
    if "xAxisValues" not in j or not j.get("series"):
        return pd.DataFrame()
    ts = pd.to_datetime(j["xAxisValues"], unit="s")
    series = j["series"][0]["data"]
    df = pd.DataFrame({"ts": ts, f"load_{country}_mw": series})
    df["date"] = df["ts"].dt.tz_localize(None).dt.normalize()
    return df.groupby("date")[f"load_{country}_mw"].mean().to_frame()


def proxy_load_from_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Synthetic EU5 load: base + seasonal driver from temperature.

    This is a *proxy*, not a measurement. It captures winter peaks
    and summer cooling load but cannot show industrial demand shocks.
    Maurizio: replace with real ENTSO-E load when you can.
    """
    if weather.empty:
        return pd.DataFrame()
    base = 250_000  # MW
    seasonal = 60_000 * ((18 - weather["tmean_c_frankfurt"]).clip(lower=-5) / 18)
    cooling  = 30_000 * weather["cdd_frankfurt"] / 10
    proxy = (base + seasonal + cooling).rename("load_eu5_mw_proxy").to_frame()
    return proxy


def main(start: str, end: str) -> None:
    pieces = []

    print("fetching Eurostat IP …")
    try:
        pieces.append(fetch_eurostat_ip())
    except Exception as e:
        print(f"  WARN eurostat: {e}")

    print("fetching Open-Meteo weather …")
    weather = fetch_weather(start, end)
    if not weather.empty:
        pieces.append(weather)

    print("fetching Fraunhofer Energy-Charts load …")
    load_pieces = []
    for c in COUNTRIES:
        try:
            load_pieces.append(fetch_load(c, start, end))
            print(f"  {c}: ok")
        except Exception as e:
            print(f"  WARN load {c}: {e}")
    if load_pieces:
        load = pd.concat(load_pieces, axis=1)
        load["load_eu5_mw"] = load[[c for c in load.columns
                                    if c.startswith("load_")]].sum(axis=1, min_count=1)
        pieces.append(load)
    else:
        print("  Fraunhofer unreachable → using temperature-derived load proxy")
        pieces.append(proxy_load_from_weather(weather))

    out = pd.concat(pieces, axis=1).sort_index()
    out.index.name = "date"
    out = out.loc[start:end]
    path = DATA / "emissions_drivers.parquet"
    out.to_parquet(path)
    print(f"wrote {path}  rows={len(out)}  cols={list(out.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end",   default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    main(args.start, args.end)
