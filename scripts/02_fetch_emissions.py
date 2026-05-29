"""
02_fetch_emissions.py — fundamental drivers of EUA demand.

Three free public sources:

1. ENTSO-E day-ahead total load for DE/FR/IT/ES/PL          (daily MWh)
   via Fraunhofer Energy-Charts JSON (no API key needed).
2. Eurostat short-term industrial production index, EA19    (monthly)
   via Eurostat REST API.
3. Open-Meteo ERA5 heating-degree-days for Frankfurt        (daily)
   via open-meteo.com (no key needed) — proxy for power demand.

What we'd want with a paid feed
-------------------------------
- Verified annual EUA emissions by installation (EUTL XML, free but slow)
- Daily power-sector CO2 intensity (gCO2/kWh) from Electricity Maps API
- Monthly cement & steel production from CEMBUREAU / Eurofer

The point of this script is to give Maurizio a *working* daily panel of
demand-side drivers. He can swap in better sources as he goes.

Output: data/emissions_drivers.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FRAUNHOFER_LOAD = "https://api.energy-charts.info/total_power"   # MW, hourly
EUROSTAT_IP = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "sts_inpr_m?format=JSON&geo=EA19&s_adj=SCA&unit=I21&nace_r2=B-D"
)
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"

COUNTRIES = ["de", "fr", "it", "es", "pl"]


def fetch_load(country: str, start: str, end: str) -> pd.DataFrame:
    """Daily mean MW load from Fraunhofer."""
    params = {"country": country, "start": start, "end": end}
    r = requests.get(FRAUNHOFER_LOAD, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    if "xAxisValues" not in j:
        return pd.DataFrame()
    ts = pd.to_datetime(j["xAxisValues"], unit="s")
    series = j["series"][0]["data"]
    df = pd.DataFrame({"ts": ts, f"load_{country}_mw": series})
    df["date"] = df["ts"].dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date")[f"load_{country}_mw"].mean().to_frame()
    return daily


def fetch_eurostat_ip() -> pd.DataFrame:
    """Eurostat monthly industrial production, EA19, NACE B-D."""
    r = requests.get(EUROSTAT_IP, timeout=60)
    r.raise_for_status()
    j = r.json()
    time_idx = j["dimension"]["time"]["category"]["index"]
    values = j["value"]
    rows = []
    for label, k in time_idx.items():
        v = values.get(str(k))
        if v is None:
            continue
        rows.append({"date": pd.Timestamp(label + "-01"), "ip_ea19": v})
    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_hdd(start: str, end: str) -> pd.DataFrame:
    """Frankfurt heating-degree-days (base 18C)."""
    params = {
        "latitude": 50.11, "longitude": 8.68,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean", "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(j["daily"]["time"]),
        "tmean_c_frankfurt": j["daily"]["temperature_2m_mean"],
    }).set_index("date")
    df["hdd_frankfurt"] = (18 - df["tmean_c_frankfurt"]).clip(lower=0)
    df["cdd_frankfurt"] = (df["tmean_c_frankfurt"] - 22).clip(lower=0)
    return df


def main(start: str, end: str) -> None:
    pieces = []

    print("fetch ENTSO-E load via Fraunhofer …")
    for c in COUNTRIES:
        try:
            pieces.append(fetch_load(c, start, end))
        except Exception as e:
            print(f"  WARN load {c}: {e}")

    print("fetch Eurostat IP …")
    try:
        ip = fetch_eurostat_ip()
        # forward-fill monthly value across daily index
        ip = ip.resample("D").ffill()
        pieces.append(ip)
    except Exception as e:
        print(f"  WARN eurostat: {e}")

    print("fetch Open-Meteo HDD …")
    try:
        pieces.append(fetch_hdd(start, end))
    except Exception as e:
        print(f"  WARN open-meteo: {e}")

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
