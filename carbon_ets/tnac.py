"""
carbon_ets.tnac — TNAC (Total Number of Allowances in Circulation) reference data.

The European Commission publishes TNAC annually each May (moved to June in 2024).
This module provides:

    1. HARDCODED HISTORICAL TNAC REFERENCE SERIES.
       Every value is copied directly from the corresponding Commission
       Communication PDF. This is the single-source-of-truth reference
       for anyone analysing EU ETS supply dynamics — replaces the need
       to open ten separate PDFs to build the series yourself.

    2. MSR MECHANISM CONSTANTS.
       Upper (1,096M) and lower (833M) thresholds, release threshold (400M),
       intake rate (24% from 2019), invalidation amounts by year.

    3. HELPER FUNCTIONS.
       get_reference_series() returns the series as a pandas DataFrame.

WHAT THIS MODULE DOES NOT DO — HONESTY NOTE
-------------------------------------------
It does NOT provide a monthly TNAC nowcast. An accurate nowcast requires:
    - Real per-year free-allocation figures from EUTL (not stylised averages)
    - Real per-year auction volumes from EEX totals
    - Correct accounting for MSR intake vs auction reduction (avoiding
      double-counting)
    - Correct treatment of invalidations (they reduce MSR holdings, NOT TNAC)
    - Real monthly emissions nowcast from ENTSO-E + Eurostat

The stylised nowcast we prototyped had ~20% median error and systematic
bias. Rather than ship a broken forecasting feature, this v0.1 provides
only the verified reference series. A validated nowcast is planned for
v0.2 once the correct data pipeline is built.

Confirmed TNAC values (from Commission Communications):
    2016: 1,693,904,897   2017: 1,654,574,598   2018: 1,654,909,824
    2019: 1,385,496,166   2020: 1,578,772,426   2021: 1,449,214,182
    2022: 1,134,794,738   2023: 1,111,736,535   2024: 1,148,049,585
    2025: 1,023,494,202
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ─────────────────────  Reference series  ────────────────────

TNAC_ANNUAL_REFERENCE = {
    # year_end : (TNAC value, MSR_intake_next_12mo, source citation)
    2016: (1_693_904_897,           0, "C(2017) 3228 final"),
    2017: (1_654_574_598,   264_731_936, "C(2018) 2801 final"),
    2018: (1_654_909_824,   397_178_358, "C(2019) 3288 final"),
    2019: (1_385_496_166,   332_519_080, "C(2020) 2835 final"),
    2020: (1_578_772_426,   378_905_382, "C(2021) 3266 final"),
    2021: (1_449_214_182,   347_811_402, "CELEX 52022XC0513(01)"),
    2022: (1_134_794_738,   272_350_737, "CELEX 52023XC0515(01)"),
    2023: (1_111_736_535,   266_816_768, "OJ C_202403415"),
    2024: (1_148_049_585,   275_531_900, "OJ C_202503180"),
    2025: (1_023_494_202,   190_494_202, "OJ C_202602957"),
}

# MSR thresholds (as of 2024 revision)
MSR_UPPER_THRESHOLD   = 1_096_000_000  # above → 24% intake
MSR_LOWER_THRESHOLD   = 833_000_000    # between 833M and 1.096B → partial intake
MSR_RELEASE_THRESHOLD = 400_000_000    # below → 100M release

# Cumulative invalidations (permanent supply reductions since 2023 revision)
INVALIDATIONS = {
    2023: 2_500_000_000,
    2024:   381_000_000,
    2025:   271_000_000,
}

# EU ETS aggregate cap trajectory (approx annual free-allocation for reference)
# Actual annual figures are published in EUTL; these are stylised averages.
ANNUAL_FREE_ALLOCATION = {
    2013: 1_088_000_000, 2014:   903_000_000, 2015:   878_000_000,
    2016:   861_000_000, 2017:   822_000_000, 2018:   805_000_000,
    2019:   790_000_000, 2020:   776_000_000, 2021:   760_000_000,
    2022:   745_000_000, 2023:   730_000_000, 2024:   615_000_000,  # cap step-up 4.3%
    2025:   585_000_000, 2026:   558_000_000,
}

# Annual auction volumes (approx, published on EEX)
ANNUAL_AUCTION_VOLUME = {
    2013:   808_000_000, 2014:   528_000_000, 2015:   632_000_000,
    2016:   715_000_000, 2017:   940_000_000, 2018:   889_000_000,
    2019:   615_000_000, 2020:   756_000_000, 2021:   555_000_000,
    2022:   639_000_000, 2023:   716_000_000, 2024:   693_000_000,
    2025:   595_000_000,
}

# Annual verified emissions (from EUTL, aggregate, in tonnes)
ANNUAL_EMISSIONS = {
    2013: 1_902_000_000, 2014: 1_812_000_000, 2015: 1_802_000_000,
    2016: 1_753_000_000, 2017: 1_756_000_000, 2018: 1_684_000_000,
    2019: 1_527_000_000, 2020: 1_331_000_000, 2021: 1_405_000_000,
    2022: 1_419_000_000, 2023: 1_222_000_000, 2024: 1_150_000_000,  # approx
    2025: 1_100_000_000,  # nowcast
}

INT_CREDITS_TOTAL   = 450_221_816   # exhausted by 2019
NER300_MONETIZED    = 300_000_000   # one-time 2011-2014
PHASE2_BANKING      = 1_749_540_826  # opening balance Jan 1 2013


def get_reference_series() -> pd.DataFrame:
    """Return the confirmed historical TNAC series indexed by year-end date."""
    rows = []
    for year, (tnac, intake, source) in TNAC_ANNUAL_REFERENCE.items():
        rows.append({
            "date": pd.Timestamp(year, 12, 31),
            "year": year,
            "tnac": tnac,
            "msr_intake_next_12mo": intake,
            "source": source,
        })
    return pd.DataFrame(rows).set_index("date")


def get_regime_context(tnac_value: float) -> dict:
    """Given a TNAC value, describe the MSR regime it triggers.

    Useful when reading the reference series: instantly tells you whether
    each year's TNAC triggered auction reduction, release, or partial
    intake.

    Returns a dict with:
        regime: 'partial_intake' | 'full_intake' | 'release' | 'neutral'
        distance_to_upper_threshold: allowances until 24% intake triggers
        distance_to_lower_threshold: allowances until release triggers
    """
    if tnac_value > MSR_UPPER_THRESHOLD:
        return {
            "regime": "full_intake",
            "description": "24% of TNAC diverted from auctions to MSR",
            "distance_to_upper_threshold": 0,
            "distance_to_release_threshold": tnac_value - MSR_RELEASE_THRESHOLD,
        }
    elif tnac_value >= MSR_LOWER_THRESHOLD:
        return {
            "regime": "partial_intake",
            "description": f"Partial intake: TNAC-833M = {tnac_value - MSR_LOWER_THRESHOLD:,.0f} diverted",
            "distance_to_upper_threshold": MSR_UPPER_THRESHOLD - tnac_value,
            "distance_to_release_threshold": tnac_value - MSR_RELEASE_THRESHOLD,
        }
    elif tnac_value >= MSR_RELEASE_THRESHOLD:
        return {
            "regime": "neutral",
            "description": "Between release threshold and lower band — no MSR action",
            "distance_to_upper_threshold": MSR_UPPER_THRESHOLD - tnac_value,
            "distance_to_release_threshold": tnac_value - MSR_RELEASE_THRESHOLD,
        }
    else:
        return {
            "regime": "release",
            "description": "100M allowances released from MSR to auctions",
            "distance_to_upper_threshold": MSR_UPPER_THRESHOLD - tnac_value,
            "distance_to_release_threshold": 0,
        }


# ─── Deprecated / removed for v0.1 ─────────────────────────────────

def nowcast_monthly(*_, **__):
    """Removed in v0.1. See module docstring for the reason.

    A validated monthly nowcast is planned for v0.2 once the required
    data pipeline is built (real per-year EUTL numbers, correct MSR
    accounting, monthly emissions nowcast from ENTSO-E).
    """
    raise NotImplementedError(
        "TNAC nowcast was removed from v0.1 after validation revealed "
        "systematic accounting errors (median 20% error, max 227%). "
        "The nowcast will return in v0.2 with correct EUTL data and "
        "validated accuracy. See carbon_ets.tnac module docstring."
    )


def cli() -> None:
    """Command-line entry point: print the reference TNAC series."""
    df = get_reference_series()
    print("EU ETS TNAC reference series (from Commission Communications)")
    print("=" * 70)
    print(df.to_string())
    print()
    print("Current MSR regime for most recent TNAC:")
    latest_tnac = df["tnac"].iloc[-1]
    ctx = get_regime_context(latest_tnac)
    print(f"  TNAC {df.index[-1].year}: {latest_tnac:,}")
    print(f"  Regime: {ctx['regime']}")
    print(f"  {ctx['description']}")


if __name__ == "__main__":
    cli()
