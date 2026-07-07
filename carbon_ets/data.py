"""
carbon_ets.data — data fetchers for EU ETS analysis.

All fetchers work from free public sources. No API keys required.
Fetchers return pandas DataFrames indexed by date.
"""

from __future__ import annotations

import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

_HEADERS = {"User-Agent": "Mozilla/5.0 (carbon-ets)"}

EEX_ARCHIVE_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/"
    "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
    "emission-spot-primary-market-auction-report-2012-2025-data.zip"
)
EEX_CURRENT_URL_TMPL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)
EUROSTAT_SDMX = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "sts_inpr_m/M.SCA.I21.B-D.EA19"
)
EUROSTAT_JSON = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    "sts_inpr_m?format=JSON&geo=EA19&s_adj=SCA&unit=I21&nace_r2=B-D"
)


# ────────────────────────────────  EUA prices  ──────────────────────

def fetch_eua_prices(
    cache_dir: Path | str | None = None,
    include_history: bool = True,
) -> pd.DataFrame:
    """Fetch EEX daily primary-auction clearing prices, 2012-present.

    Parameters
    ----------
    cache_dir : path
        If provided, read local ZIP/XLSX files first before fetching from EEX.
    include_history : bool
        If True (default), include the pre-2020 XLS files. Requires python-calamine.

    Returns
    -------
    DataFrame with DatetimeIndex and 'eua_eur_tco2' column.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    # Load archive
    archive_bytes = _load_or_fetch(
        EEX_ARCHIVE_URL,
        "emission-spot-primary-market-auction-report-2012-2025-data.zip",
        cache_dir,
    )
    if archive_bytes is None:
        raise RuntimeError("Could not fetch EEX archive")

    df = _parse_eex_zip(archive_bytes, include_history)

    # Current year
    year = pd.Timestamp.today().year
    cur_url = EEX_CURRENT_URL_TMPL.format(year=year)
    cur_bytes = _load_or_fetch(
        cur_url,
        f"emission-spot-primary-market-auction-report-{year}-data.xlsx",
        cache_dir,
    )
    if cur_bytes is not None:
        try:
            cur = _parse_eex_excel(io.BytesIO(cur_bytes), "current.xlsx")
            df = pd.concat([df, cur], ignore_index=True)
        except Exception:
            pass

    df = (
        df.drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
        .sort_index()
    )
    return df


def _load_or_fetch(url: str, filename: str, cache_dir: Path | None) -> bytes | None:
    """Try cache first, then network."""
    if cache_dir is not None:
        p = cache_dir / filename
        if p.exists():
            return p.read_bytes()
    try:
        r = requests.get(url, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


def _find_header_row(raw: pd.DataFrame) -> int | None:
    kw_sets = [
        ["auction", "price"], ["clearing", "price"],
        ["preis"], ["eur/eua"], ["eur/tco2"], ["eur/t"],
    ]
    for idx, row in raw.iterrows():
        cells = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
        for kw_set in kw_sets:
            if all(k in cells for k in kw_set):
                if "date" in cells or "datum" in cells:
                    return int(idx)
    return None


def _parse_eex_excel(buf: io.BytesIO, fname: str) -> pd.DataFrame:
    ext = Path(fname).suffix.lower()
    engines = ["calamine", "openpyxl"] if ext == ".xlsx" else ["calamine"]
    raw = None
    for engine in engines:
        try:
            buf.seek(0)
            raw = pd.read_excel(buf, sheet_name=0, header=None, engine=engine)
            break
        except Exception:
            continue
    if raw is None:
        return pd.DataFrame(columns=["date", "eua_eur_tco2"])

    hdr = _find_header_row(raw)
    if hdr is None:
        return pd.DataFrame(columns=["date", "eua_eur_tco2"])

    df = raw.iloc[hdr + 1:].copy()
    df.columns = raw.iloc[hdr].astype(str).tolist()

    def col_match(*needles):
        for c in df.columns:
            s = str(c).lower()
            if all(n.lower() in s for n in needles):
                return c
        return None

    pc = col_match("auction", "price") or col_match("eur", "tco2") or col_match("preis")
    dc = col_match("date") or col_match("datum")
    if pc is None or dc is None:
        return pd.DataFrame(columns=["date", "eua_eur_tco2"])

    out = df[[dc, pc]].copy()
    out.columns = ["date", "eua_eur_tco2"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["eua_eur_tco2"] = pd.to_numeric(out["eua_eur_tco2"], errors="coerce")
    return out.dropna()


def _parse_eex_zip(blob: bytes, include_history: bool) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if not name.lower().endswith((".xlsx", ".xls")):
                continue
            if not include_history and name.lower().endswith(".xls"):
                continue
            try:
                df = _parse_eex_excel(io.BytesIO(z.read(name)), name)
                if not df.empty:
                    pieces.append(df)
            except Exception:
                pass
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["date", "eua_eur_tco2"])


# ────────────────────────────────  Eurostat IP  ──────────────────────

def fetch_eurostat_ip() -> pd.DataFrame:
    """Eurostat monthly industrial production, EA19, NACE B-D, seasonally adjusted.

    Tries multiple SDMX URL variants for freshness, then falls back to
    legacy JSON-stat (which historically stops updating around Aug 2024
    but stays available for older data). Emits a warning if only stale
    data can be obtained.

    Returns
    -------
    DataFrame indexed by month-start date with 'ip_ea19' column.
    """
    import warnings as _warnings

    # SDMX endpoint variants — Eurostat has multiple format parameters.
    # Try each in order until we get fresh data.
    sdmx_variants = [
        {"format": "SDMX-CSV"},
        {"format": "SDMX_CSV"},
        {"format": "csvdata"},
        {},   # plain — let Accept header drive
    ]
    for params in sdmx_variants:
        try:
            r = requests.get(
                EUROSTAT_SDMX,
                headers={"Accept": "text/csv, application/vnd.sdmx.data+csv;version=1.0.0"},
                params=params,
                timeout=20,
            )
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
                df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "ip_ea19"})
                df = df[["date", "ip_ea19"]].copy()
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date").sort_index()
                if len(df) > 100:
                    return df
        except Exception:
            continue

    # Fallback: legacy JSON-stat.  This endpoint is known to stop updating
    # around mid-2024 in current Eurostat retirement schedules — data
    # returned may be stale.  Warn the user so they can react.
    r = requests.get(EUROSTAT_JSON, timeout=20)
    r.raise_for_status()
    j = r.json()
    idx = j["dimension"]["time"]["category"]["index"]
    vals = j["value"]
    rows = [
        {"date": pd.Timestamp(label + "-01"), "ip_ea19": vals.get(str(k))}
        for label, k in idx.items() if vals.get(str(k)) is not None
    ]
    df = pd.DataFrame(rows).set_index("date").sort_index()

    if not df.empty:
        gap_days = (pd.Timestamp.today() - df.index.max()).days
        if gap_days > 180:
            _warnings.warn(
                f"Eurostat IP latest observation is {gap_days} days old. "
                f"SDMX endpoint appears down; using stale legacy JSON-stat data. "
                f"Consider filing an issue at github.com/causalsystems/carbon-ets.",
                stacklevel=2,
            )
    return df


# ────────────────────────────────  Panel builder  ────────────────────

def build_full_panel(
    start: str = "2012-01-01",
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build the full daily EU ETS panel.

    Combines EUA auction prices, Eurostat IP, and yfinance-fetched
    equity/gas/oil proxies into a single DataFrame indexed by business day.

    Parameters
    ----------
    start : str
        First date to fetch, YYYY-MM-DD.
    cache_dir : path or None
        Directory containing pre-downloaded EEX ZIP/XLSX files.

    Returns
    -------
    DataFrame with columns:
        eua_eur_tco2, ip_ea19, stoxx50, ttf_gas_eur_mwh, wti_usd_bbl
        plus derived features via `carbon_ets.features.engineer`.
    """
    # Fetch EUA + IP (independent of yfinance)
    eua = fetch_eua_prices(cache_dir=cache_dir)
    ip  = fetch_eurostat_ip()

    # Build business-day index from EUA range
    idx = pd.bdate_range(start=start, end=eua.index.max())
    panel = pd.DataFrame(index=idx)
    panel["eua_eur_tco2"] = eua["eua_eur_tco2"].reindex(panel.index).ffill(limit=7)
    panel["ip_ea19"] = ip["ip_ea19"].reindex(panel.index).ffill(limit=45)

    # yfinance is optional (may not be installed)
    try:
        import yfinance as yf  # noqa: F401
        panel = _add_yfinance_prices(panel, start)
    except ImportError:
        pass

    return panel


def _add_yfinance_prices(panel: pd.DataFrame, start: str) -> pd.DataFrame:
    """Add Stoxx 50, TTF, WTI via yfinance. Requires optional yfinance dependency."""
    import yfinance as yf
    tickers = {"^STOXX50E": "stoxx50", "TTF=F": "ttf_gas_eur_mwh", "CL=F": "wti_usd_bbl"}
    for ticker, colname in tickers.items():
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None) if s.index.tz is not None else pd.to_datetime(s.index)
            panel[colname] = s.reindex(panel.index).ffill(limit=3)
        except Exception:
            pass
    return panel
