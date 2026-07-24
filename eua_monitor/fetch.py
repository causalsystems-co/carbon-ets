"""Data fetchers with a cache-or-fail-soft layer.

Every fetcher tries the live source, snapshots the raw payload into
data/cache/ on success, and falls back to the last snapshot when the source
is unreachable. Only when there is neither live data nor a snapshot does it
raise DataUnavailable — the dashboard then renders that panel as unavailable
instead of showing invented numbers.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import urllib.parse
import urllib.request

from . import config


class DataUnavailable(Exception):
    """No live data and no cached snapshot for a required source."""


def _http_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (eua-monitor)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_json(url: str, timeout: int = 30) -> dict:
    return json.loads(_http_bytes(url, timeout).decode("utf-8"))


def _cached_bytes(url: str, cache_name: str) -> bytes:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.CACHE_DIR / f"{cache_name}.bin"
    try:
        payload = _http_bytes(url)
        cache_path.write_bytes(payload)
        return payload
    except Exception as exc:  # noqa: BLE001 - any network failure falls back
        if cache_path.exists():
            return cache_path.read_bytes()
        raise DataUnavailable(f"{cache_name}: {exc}") from exc


def _cached_json(url: str, cache_name: str) -> dict:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.CACHE_DIR / f"{cache_name}.json"
    try:
        payload = _http_json(url)
        cache_path.write_text(json.dumps(payload))
        return payload
    except Exception as exc:  # noqa: BLE001 - any network failure falls back
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        raise DataUnavailable(f"{cache_name}: {exc}") from exc


# --------------------------------------------------------------------- prices

def _yahoo_daily(
    symbol: str, range_: str, cache_name: str, interval: str = "1d"
) -> list[tuple[str, float]]:
    url = config.YAHOO_CHART_URL.format(symbol=urllib.parse.quote(symbol))
    url += f"?range={range_}&interval={interval}&events=history"
    payload = _cached_json(url, cache_name)
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0].get("close") or []
    out = []
    for ts, close in zip(stamps, closes):
        if close is None:
            continue
        day = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
        out.append((day, float(close)))
    if not out:
        raise DataUnavailable(f"{cache_name}: empty series")
    return out


def eex_auctions() -> list[dict]:
    """Daily EUA auction rows {date, close, volume, cover}: the imported
    2012+ history CSV topped up with the live current-year EEX report."""
    from . import eex_import

    rows: list[dict] = []
    if config.EUA_HISTORY_CSV.exists():
        with config.EUA_HISTORY_CSV.open() as fh:
            for r in csv.DictReader(fh):
                rows.append(
                    {
                        "date": r["date"],
                        "close": float(r["close"]),
                        "volume": int(r["volume"]) if r.get("volume") else None,
                        "cover": float(r["cover_ratio"]) if r.get("cover_ratio") else None,
                    }
                )
    last = rows[-1]["date"] if rows else "1900-01-01"
    this_year = dt.date.today().year
    for year in range(max(int(last[:4]), this_year - 1), this_year + 1):
        try:
            data = _cached_bytes(
                config.EEX_CURRENT_URL_TMPL.format(year=year), f"eex_{year}"
            )
        except DataUnavailable:
            continue  # e.g. new year's report not published yet
        fresh = eex_import.aggregate_daily(
            eex_import.parse_xlsx_bytes(data)
        )
        rows += [
            {
                "date": r["date"],
                "close": float(r["close"]),
                "volume": r["volume"],
                "cover": r["cover_ratio"] if r["cover_ratio"] != "" else None,
            }
            for r in fresh
            if r["date"] > last
        ]
        if rows:
            last = rows[-1]["date"]
    return sorted(rows, key=lambda r: r["date"])


def eua_prices() -> list[tuple[str, float]]:
    """Daily EUA series (EUR/tCO2): EEX auction clearing prices, with the
    CO2.L ETC appended only when the auction feed is stale (report gap)."""
    auctions = eex_auctions()
    series = [(r["date"], r["close"]) for r in auctions]
    today = dt.date.today()
    stale = (
        not series
        or (today - dt.date.fromisoformat(series[-1][0])).days > config.EEX_STALE_DAYS
    )
    if stale:
        try:
            proxy = _yahoo_daily(config.EUA_TICKER, "10y", "eua_price")
            last = series[-1][0] if series else "1900-01-01"
            series += [(d, p) for d, p in proxy if d > last]
        except DataUnavailable:
            if not series:
                raise
    return series


def stoxx_prices() -> list[tuple[str, float]]:
    # Monthly bars back to 2007; month-close mapping downstream is unaffected.
    return _yahoo_daily(config.STOXX_TICKER, "max", "stoxx", interval="1mo")


# -------------------------------------------------------------------- eurostat

def industrial_production() -> dict[str, float]:
    """Euro-area industrial production index by month ('YYYY-MM' -> index)."""
    payload = _cached_json(config.EUROSTAT_URL, "eurostat_ip")
    time_index = payload["dimension"]["time"]["category"]["index"]
    values = payload["value"]
    out = {}
    for period, idx in time_index.items():
        val = values.get(str(idx))
        if val is not None:
            out[period] = float(val)
    if not out:
        raise DataUnavailable("eurostat_ip: empty series")
    return out


# ------------------------------------------------------------------------ hdd

def heating_degree_days() -> dict[str, float]:
    """Population-weighted monthly HDD sum ('YYYY-MM' -> degree-days)."""
    end = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    monthly_weighted: dict[str, float] = {}
    monthly_weight_seen: dict[str, float] = {}
    for name, lat, lon, weight in config.HDD_CITIES:
        query = urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": config.HDD_START,
                "end_date": end,
                "daily": "temperature_2m_mean",
                "timezone": "UTC",
            }
        )
        payload = _cached_json(
            f"{config.OPEN_METEO_URL}?{query}", f"hdd_{name.lower()}"
        )
        days = payload["daily"]["time"]
        temps = payload["daily"]["temperature_2m_mean"]
        city_monthly: dict[str, float] = {}
        for day, temp in zip(days, temps):
            if temp is None:
                continue
            month = day[:7]
            city_monthly[month] = city_monthly.get(month, 0.0) + max(
                0.0, config.HDD_BASE_C - float(temp)
            )
        for month, hdd in city_monthly.items():
            monthly_weighted[month] = monthly_weighted.get(month, 0.0) + weight * hdd
            monthly_weight_seen[month] = monthly_weight_seen.get(month, 0.0) + weight
    # Normalise in case a city series is missing for a month.
    out = {
        m: monthly_weighted[m] / monthly_weight_seen[m]
        for m in monthly_weighted
        if monthly_weight_seen[m] > 0
    }
    if not out:
        raise DataUnavailable("hdd: empty series")
    return out


# ----------------------------------------------------------------------- tnac

def tnac_history() -> list[dict]:
    """TNAC by year from the EC's annual MSR communications (data/tnac.csv)."""
    with config.TNAC_CSV.open() as fh:
        rows = [
            {
                "year": int(r["year"]),
                "tnac": int(r["tnac"]),
                "source": r["source"],
            }
            for r in csv.DictReader(fh)
        ]
    return sorted(rows, key=lambda r: r["year"])
