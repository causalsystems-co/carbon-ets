"""Tests for carbon_ets.tnac reference data.

Every hardcoded TNAC value must exactly match the Commission Communication.
These tests are the correctness gate — if any TNAC value is wrong,
the entire toolkit's reference data is unreliable.
"""

import pytest
import pandas as pd

from carbon_ets.tnac import (
    TNAC_ANNUAL_REFERENCE,
    get_reference_series,
    get_regime_context,
    nowcast_monthly,
    MSR_UPPER_THRESHOLD,
    MSR_LOWER_THRESHOLD,
    MSR_RELEASE_THRESHOLD,
)


# Every one of these is quoted directly from a Commission PDF.
# Change ONLY if the Commission publishes a correction.
KNOWN_TNAC_VALUES = {
    2016: 1_693_904_897,   # C(2017) 3228 final
    2017: 1_654_574_598,   # C(2018) 2801 final
    2018: 1_654_909_824,   # C(2019) 3288 final
    2019: 1_385_496_166,   # C(2020) 2835 final
    2020: 1_578_772_426,   # C(2021) 3266 final
    2021: 1_449_214_182,   # CELEX 52022XC0513(01)
    2022: 1_134_794_738,   # CELEX 52023XC0515(01)
    2023: 1_111_736_535,   # OJ C_202403415
    2024: 1_148_049_585,   # OJ C_202503180
    2025: 1_023_494_202,   # OJ C_202602957
}


@pytest.mark.parametrize("year,expected_tnac", list(KNOWN_TNAC_VALUES.items()))
def test_reference_tnac_matches_commission_publication(year, expected_tnac):
    """Every year's TNAC must match the Commission Communication exactly."""
    actual = TNAC_ANNUAL_REFERENCE[year][0]
    assert actual == expected_tnac, (
        f"TNAC for {year} is {actual:,} but Commission published {expected_tnac:,}"
    )


def test_reference_series_returns_dataframe():
    df = get_reference_series()
    assert isinstance(df, pd.DataFrame)
    assert "tnac" in df.columns
    assert "source" in df.columns
    assert len(df) >= 10


def test_reference_series_indexed_by_year_end():
    df = get_reference_series()
    for date in df.index:
        assert date.month == 12
        assert date.day == 31


def test_reference_series_covers_2016_through_2025():
    df = get_reference_series()
    years = sorted(df["year"].unique())
    assert 2016 in years, "Reference series must start in 2016 (first informational publication)"
    assert 2025 in years, "Reference series must include the most recent Commission publication"


# ─── MSR regime context ─────────────────────────────────────

def test_regime_context_full_intake_above_upper_threshold():
    ctx = get_regime_context(MSR_UPPER_THRESHOLD + 1)
    assert ctx["regime"] == "full_intake"


def test_regime_context_partial_intake_in_band():
    ctx = get_regime_context((MSR_UPPER_THRESHOLD + MSR_LOWER_THRESHOLD) // 2)
    assert ctx["regime"] == "partial_intake"


def test_regime_context_neutral_between_lower_and_release():
    ctx = get_regime_context((MSR_LOWER_THRESHOLD + MSR_RELEASE_THRESHOLD) // 2)
    assert ctx["regime"] == "neutral"


def test_regime_context_release_below_release_threshold():
    ctx = get_regime_context(MSR_RELEASE_THRESHOLD - 1)
    assert ctx["regime"] == "release"


def test_2025_is_partial_intake_regime():
    """The 2025 TNAC of 1,023,494,202 sits in the partial-intake band."""
    ctx = get_regime_context(KNOWN_TNAC_VALUES[2025])
    assert ctx["regime"] == "partial_intake", (
        "TNAC 2025 (1.023B) is between 833M and 1.096B — must trigger partial intake"
    )


# ─── Nowcast is explicitly removed ─────────────────────────

def test_nowcast_monthly_is_removed_in_v01():
    """The nowcast was validated at 20% median error; removed from v0.1."""
    with pytest.raises(NotImplementedError):
        nowcast_monthly()


# ─── Structural properties ─────────────────────────────────

def test_msr_thresholds_are_ordered():
    assert MSR_RELEASE_THRESHOLD < MSR_LOWER_THRESHOLD < MSR_UPPER_THRESHOLD
    assert MSR_RELEASE_THRESHOLD == 400_000_000
    assert MSR_LOWER_THRESHOLD == 833_000_000
    assert MSR_UPPER_THRESHOLD == 1_096_000_000
