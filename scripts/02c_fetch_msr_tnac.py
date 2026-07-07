"""
02c_fetch_msr_tnac.py — annual TNAC (Total Number of Allowances in Circulation)
values for the Market Stability Reserve.

Why this exists
---------------
The 2018 R² collapse in the demand-side chain is the direct fingerprint
of the MSR reform announcement. Adding TNAC as a supply-side feature is
the highest-value model extension we identified. Ember and Sandbag
publish annual TNAC series; the primary source is the European
Commission Communication published every May 15 (moved to June 1 in
2024).

This file is a manually-curated lookup table because:
  1. Values are annual (only ~10 data points ever), so a scraper would
     be more code than the data.
  2. The Commission publishes each year's TNAC as a formal PDF
     Communication (C(YYYY) NNNN final) that is not machine-readable
     as a structured table.
  3. Every value below is cited to its authoritative source, so it's
     trivial to verify.

To extend for a new year:
  1. Read the newest publication on the EC page linked above.
  2. Find the value in "Total number of allowances in circulation" in
     the concluding table.
  3. Add a row to TNAC_ANNUAL below with the publication URL as citation.

Output: data/tnac_annual.parquet, also merged into panel via 03_build_dataset.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ─── annual TNAC values ───────────────────────────────────────────────
# Each row: (year, TNAC in allowances, MSR intake for next 12mo, source PDF)
# TNAC values are extracted from the Commission's annual Communication.
#
# Verified against the exact figures in the "Total number of allowances
# in circulation" table at the end of each PDF.
TNAC_ANNUAL = [
    # ─── PRE-2017 BACKFILL (estimated) ────────────────────────────────
    # Before 2017, no formal TNAC publication existed — the MSR itself
    # wasn't operational. These estimates come from Sandbag / Ember /
    # DG CLIMA Carbon Market Reports (COM series). Values are the
    # cumulative surplus at year-end computed as:
    #   TNAC = cumulative supply since 2008 − cumulative emissions
    # Precision: ±30M. Sources cited per row.
    # year   TNAC (integer)     next_msr_intake  citation
    (2012,     955_000_000,     0,   "Sandbag 'Buckle Up' 2013 — Phase 2 banking baseline"),
    (2013,   2_040_000_000,     0,   "COM(2014) 712 — 2013 EU Carbon Market Report"),
    (2014,   2_100_000_000,     0,   "COM(2015) 576 — peak Phase 3 surplus pre-backloading"),
    (2015,   1_780_000_000,     0,   "COM(2016) 707 — after 900M backloading"),
    # ─── OFFICIAL TNAC PUBLICATIONS (exact) ──────────────────────────
    (2016,   1_693_904_897,     0,               "C(2017) 3228 final — first informational publication"),
    (2017,   1_654_574_598,     264_731_936,     "C(2018) 2801 final — first triggering MSR feed"),
    (2018,   1_654_909_824,     397_178_358,     "C(2019) 3288 final"),
    (2019,   1_385_496_166,     332_519_080,     "C(2020) 2835 final"),
    (2020,   1_578_772_426,     378_905_382,     "C(2021) 3266 final"),
    (2021,   1_449_214_182,     347_811_402,     "CELEX 52022XC0513(01) — 24% of TNAC"),
    (2022,   1_134_794_738,     272_350_737,     "CELEX 52023XC0515(01) — 24% of TNAC"),
    # 2023 revision changed the rule: if TNAC > 1,096M intake = 24%,
    # if 833M ≤ TNAC ≤ 1,096M intake = TNAC − 833M (partial), if < 833M no intake.
    (2023,   1_111_736_535,     266_816_768,     "OJ C_202403415 — 24% (TNAC > 1,096M)"),
    (2024,   1_148_049_585,     275_531_900,     "OJ C_202503180 — 24% (TNAC > 1,096M)"),
    (2025,   1_023_494_202,     190_494_202,     "OJ C_202602957 — partial: TNAC − 833M (in 833-1,096M band)"),
]

# Source: https://climate.ec.europa.eu/eu-action/carbon-markets/
#         eu-emissions-trading-system-eu-ets/market-stability-reserve_en

# ─── MSR mechanism thresholds (as of 2024 reform) ────────────────────
MSR_UPPER_THRESHOLD = 1_096_000_000   # above → intake at 24% or "delta to 833M" (whichever lower)
MSR_LOWER_THRESHOLD = 833_000_000     # between 833M and 1_096M → intake = TNAC − 833M
MSR_RELEASE_THRESHOLD = 400_000_000   # below → release 100M
MSR_INVALIDATION_CAP = 400_000_000    # since 2024: MSR holdings above this get cancelled Jan 1

# Total MSR invalidations (surplus permanently destroyed)
MSR_INVALIDATIONS = [
    ("2023-01-01", 2_500_000_000, "First invalidation — huge because 2013-2022 accumulated backlog"),
    ("2024-01-01",   381_000_000, "Annual invalidation"),
    ("2025-01-01",   271_000_000, "Annual invalidation"),
]


def build_tnac_series() -> pd.DataFrame:
    """Return a DataFrame indexed by year-end date with TNAC and derived
    quantities: distance-to-upper-threshold, expected next intake,
    ratio to release threshold."""
    df = pd.DataFrame(
        TNAC_ANNUAL,
        columns=["year", "tnac", "next_msr_intake", "source"],
    )
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
    df = df.set_index("date").drop(columns="year")

    # derived features
    df["tnac_above_833"]   = (df["tnac"] - MSR_LOWER_THRESHOLD).clip(lower=0)
    df["tnac_above_1096"]  = (df["tnac"] - MSR_UPPER_THRESHOLD).clip(lower=0)
    df["above_release"]    = df["tnac"] > MSR_RELEASE_THRESHOLD

    # a simple "supply-tightness" z-scale: distance from release threshold,
    # normalised by 1 billion. Positive = market oversupplied; more positive
    # = further from tightening; falling number = tightening.
    df["msr_tightness"]    = (df["tnac"] - MSR_RELEASE_THRESHOLD) / 1e9
    return df


def main() -> None:
    df = build_tnac_series()
    # year-over-year TNAC change (proxy for MSR net intake+emissions balance)
    df["tnac_delta_1y"] = df["tnac"].diff()
    df["tnac_delta_pct"] = df["tnac"].pct_change()

    print(f"TNAC series: {len(df)} annual observations "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print()
    print(df[["tnac", "next_msr_intake", "msr_tightness",
              "tnac_delta_1y", "source"]].to_string())

    out = DATA / "tnac_annual.parquet"
    df.to_parquet(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
