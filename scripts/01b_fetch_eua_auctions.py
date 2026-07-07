"""
01b_fetch_eua_auctions.py — daily EUA primary auction clearing prices.

Reads from local cache first (../proj_Carbon_ETS/ or data/raw/), then
falls back to EEX network downloads. Cache-first keeps the pipeline fast
and offline-capable.

Sources concatenated (longest to most-recent history):
  - Germany pre-2013 country auctions  (de-history-...zip)
  - Netherlands pre-2013                (nl-history-...zip)
  - Lithuania pre-2013                  (lt-history-...zip)
  - Main consolidated archive 2020-2025 (emission-spot-...2012-2025-data.zip)
  - Current-year XLSX                   (refreshed daily)

Output: data/eua_daily.parquet  (date → eua_eur_tco2)
"""

from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LOCAL_DIRS = [
    ROOT.parent / "proj_Carbon_ETS",
    DATA / "raw",
]

BASE_ARCHIVE = (
    "https://www.eex.com/fileadmin/EEX/Downloads/"
    "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
)
CURRENT_URL_TMPL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)

SOURCES = [
    # Main archive covers 2012-2025 (xls for 2012-2019, xlsx for 2020-2025).
    # Germany/NL/LT historical ZIPs hold pre-EEX country auctions: Germany
    # has nested quarterly ZIPs (skipped — main archive covers same dates);
    # NL/LT are PDFs (skipped).
    ("main_archive",        "emission-spot-primary-market-auction-report-2012-2025-data.zip"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (causal-trading)"}

# Header-row keywords (lowercase substring match, ALL must appear in the row)
HEADER_KEYWORDS = [
    ["auction", "price"],
    ["clearing", "price"],
    ["preis"],             # German
    ["eur/eua"],
    ["eur/tco2"],
    ["eur/t"],
]


def _load_bytes(filename: str) -> bytes | None:
    for d in LOCAL_DIRS:
        p = d / filename
        if p.exists():
            print(f"  CACHE {p.name}")
            return p.read_bytes()
    url = BASE_ARCHIVE + filename
    print(f"  GET   {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  FAIL  {filename}: {e}")
        return None


def _find_header_row(raw: pd.DataFrame) -> int | None:
    """Find the row whose cells together contain auction-price keywords.
    Defensive: cells can be NaN/float/etc., not all strings."""
    for idx, row in raw.iterrows():
        cells = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
        for kw_set in HEADER_KEYWORDS:
            if all(k in cells for k in kw_set):
                if "date" in cells or "datum" in cells:
                    return int(idx)
    return None


def _find_col(df: pd.DataFrame, *needles_groups) -> str | None:
    """Find a column whose name contains any of the needle groups (case-insensitive).
    Each needles_group is a tuple — all needles in the group must match."""
    for c in df.columns:
        s = str(c).lower()
        for group in needles_groups:
            if all(n.lower() in s for n in group):
                return c
    return None


def _parse_excel(buf: io.BytesIO, fname: str) -> pd.DataFrame:
    """Read an EEX auction file (xlsx or xls); return date/eua_eur_tco2.
    Uses python-calamine which handles both .xls and .xlsx in pandas 2+
    (xlrd ≥2.0 dropped .xls support, so calamine is the modern path)."""
    ext = Path(fname).suffix.lower()
    # Engine order: calamine handles everything; openpyxl as fallback for .xlsx.
    engines = ["calamine", "openpyxl"] if ext == ".xlsx" else ["calamine"]
    raw = None
    last_err = None
    for engine in engines:
        try:
            buf.seek(0)
            raw = pd.read_excel(buf, sheet_name=0, header=None, engine=engine)
            break
        except Exception as e:
            last_err = e
    if raw is None:
        print(f"      cannot read excel: {last_err}")
        return pd.DataFrame()

    hdr = _find_header_row(raw)
    if hdr is None:
        print(f"      no header row found in {fname} ({raw.shape[0]} rows scanned)")
        return pd.DataFrame()

    # promote that row to header
    df = raw.iloc[hdr + 1 :].copy()
    df.columns = raw.iloc[hdr].astype(str).tolist()

    dc = _find_col(df, ("date",), ("datum",), ("auction", "date"))
    pc = _find_col(
        df,
        ("auction", "price"),
        ("clearing", "price"),
        ("non-disclosure", "price"),
        ("eur/eua",),
        ("eur/tco2",),
        ("eur/t",),
        ("preis",),
    )
    if dc is None or pc is None:
        cols = list(df.columns)[:8]
        print(f"      can't find date/price cols in {fname}; headers={cols}")
        return pd.DataFrame()

    out = df[[dc, pc]].copy()
    out.columns = ["date", "eua_eur_tco2"]
    # Normalize to midnight — some years (e.g. 2016) have 09:00 timestamps
    # which silently drop out of the daily panel join otherwise.
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=False).dt.normalize()
    out["eua_eur_tco2"] = pd.to_numeric(out["eua_eur_tco2"], errors="coerce")
    out = out.dropna()
    return out


def parse_zip(blob: bytes, label: str) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        excel_names = [n for n in names if n.lower().endswith((".xlsx", ".xls"))]
        print(f"    [{label}] {len(names)} entries, {len(excel_names)} excel")
        if not excel_names:
            # show what's actually inside for debugging
            print(f"    [{label}] contents: {names[:6]}{' ...' if len(names) > 6 else ''}")
        for name in excel_names:
            try:
                df = _parse_excel(io.BytesIO(z.read(name)), name)
                if not df.empty:
                    pieces.append(df)
                    print(f"      {Path(name).name}: rows={len(df)}  "
                          f"{df['date'].min().date()} → {df['date'].max().date()}")
                else:
                    print(f"      {Path(name).name}: empty after parse")
            except Exception as e:
                print(f"      {Path(name).name}: ERR {e}")
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def fetch_current_year() -> pd.DataFrame:
    fname = f"emission-spot-primary-market-auction-report-{pd.Timestamp.today().year}-data.xlsx"
    for d in LOCAL_DIRS:
        p = d / fname
        if p.exists():
            print(f"  CACHE {p.name}")
            return _parse_excel(io.BytesIO(p.read_bytes()), fname)
    url = CURRENT_URL_TMPL.format(year=pd.Timestamp.today().year)
    print(f"  GET   {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return _parse_excel(io.BytesIO(r.content), fname)
    except Exception as e:
        print(f"  FAIL  current year: {e}")
        return pd.DataFrame()


def main() -> None:
    all_pieces = []
    print("loading EEX sources …")
    for label, fname in SOURCES:
        blob = _load_bytes(fname)
        if blob is None:
            continue
        df = parse_zip(blob, label)
        if not df.empty:
            df["source"] = label
            all_pieces.append(df)

    print(f"fetching EEX current year {pd.Timestamp.today().year} …")
    cur = fetch_current_year()
    if not cur.empty:
        cur["source"] = "current_year"
        all_pieces.append(cur)

    if not all_pieces:
        raise RuntimeError("no EUA data loaded — both cache and network empty")

    df = pd.concat(all_pieces, ignore_index=True)
    priority = {
        "current_year": 0, "main_archive": 1,
        "germany_history": 2, "netherlands_history": 3, "lithuania_history": 4,
    }
    df["prio"] = df["source"].map(priority).fillna(99)
    df = (df.sort_values(["date", "prio"])
            .drop_duplicates(subset=["date"], keep="first")
            .drop(columns="prio")
            .set_index("date")
            .sort_index())

    out = DATA / "eua_daily.parquet"
    df.to_parquet(out)
    print(f"\nwrote {out}")
    print(f"  rows:  {len(df)}")
    print(f"  range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  last:  {df['eua_eur_tco2'].iloc[-1]:.2f} €/tCO2")
    print("\n  by source:")
    print(df.groupby("source").size().to_string())


if __name__ == "__main__":
    main()
