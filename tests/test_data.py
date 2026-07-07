"""Smoke tests for carbon_ets.data fetchers.

These require network access. Skipped automatically if the underlying
source is unreachable, so CI without network still passes.
"""

import pytest
import pandas as pd
import requests


def _reachable(url: str, timeout: int = 5) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code < 500
    except Exception:
        return False


@pytest.mark.network
def test_fetch_eurostat_ip_returns_dataframe():
    """Eurostat IP fetcher should return a well-formed DataFrame."""
    if not _reachable("https://ec.europa.eu/eurostat"):
        pytest.skip("Eurostat unreachable")
    from carbon_ets.data import fetch_eurostat_ip
    df = fetch_eurostat_ip()
    assert isinstance(df, pd.DataFrame)
    assert "ip_ea19" in df.columns
    assert len(df) > 100, "Eurostat IP should have >100 monthly observations"
    assert isinstance(df.index, pd.DatetimeIndex)


@pytest.mark.network
def test_fetch_eurostat_ip_returns_recent_or_warns():
    """Eurostat should either return fresh data (SDMX working) or
    fall back to stale JSON-stat with an explicit warning.

    We do NOT hard-fail on staleness — Eurostat's SDMX API has known
    reliability issues. What we require is that the fetcher succeeds
    and returns *some* usable data.
    """
    if not _reachable("https://ec.europa.eu/eurostat"):
        pytest.skip("Eurostat unreachable")
    from carbon_ets.data import fetch_eurostat_ip
    df = fetch_eurostat_ip()
    # Must return SOME data
    assert len(df) > 100, "Eurostat IP fetcher returned insufficient data"
    latest = df.index.max()
    gap_days = (pd.Timestamp.today() - latest).days
    # Data should be less than 3 years old (soft check — real staleness
    # will emit a UserWarning at fetch time)
    assert gap_days < 1100, (
        f"Eurostat IP data is {gap_days} days old — both SDMX and JSON-stat "
        f"endpoints appear broken. This is an infrastructure issue at Eurostat."
    )


@pytest.mark.network
@pytest.mark.slow
def test_fetch_eua_prices_returns_dataframe():
    """EEX EUA price fetcher smoke test.

    Marked slow because the EEX archive download can take 30-60 seconds.
    Requires openpyxl to be installed.
    """
    if not _reachable("https://www.eex.com"):
        pytest.skip("EEX unreachable")
    from carbon_ets.data import fetch_eua_prices
    df = fetch_eua_prices(include_history=False)   # skip xls files for speed
    assert isinstance(df, pd.DataFrame)
    assert "eua_eur_tco2" in df.columns
    assert len(df) > 100
    assert isinstance(df.index, pd.DatetimeIndex)
    # Sanity check: prices should be in a plausible range 2020-2026
    latest_price = df["eua_eur_tco2"].iloc[-1]
    assert 20 < latest_price < 200, (
        f"Latest EUA price {latest_price} is outside plausible €20-200 range"
    )
