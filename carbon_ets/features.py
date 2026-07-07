"""
carbon_ets.features — feature engineering for the EU ETS panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer(panel: pd.DataFrame) -> pd.DataFrame:
    """Add log-returns, z-scores, and derived features to a panel.

    Parameters
    ----------
    panel : DataFrame
        Output of `carbon_ets.data.build_full_panel`.

    Returns
    -------
    DataFrame with additional columns:
        r_eua_eur_tco2, r_stoxx50, r_ttf_gas_eur_mwh, r_wti_usd_bbl
        ip_yoy, z_ip, z_stx, z_ttf
        eua_rv20, eua_rv60
    """
    f = panel.copy()

    # log-returns
    for c in ["eua_eur_tco2", "stoxx50", "ttf_gas_eur_mwh", "wti_usd_bbl"]:
        if c in f:
            f[f"r_{c}"] = np.log(f[c]).diff()

    # IP YoY (computed at monthly level, propagated to daily)
    if "ip_ea19" in f:
        ip_monthly = f["ip_ea19"].resample("ME").last().ffill(limit=1)
        ip_yoy = ip_monthly.pct_change(12)
        f["ip_yoy"] = ip_yoy.reindex(f.index, method="ffill")

    # Stoxx momentum (20-day log return)
    if "stoxx50" in f:
        f["stoxx_mom20"] = np.log(f["stoxx50"] / f["stoxx50"].shift(20))

    # z-scores on rolling window
    def z(s, w):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    for col, win in [
        ("ip_yoy", 252),
        ("stoxx_mom20", 252),
        ("r_ttf_gas_eur_mwh", 252),
    ]:
        if col in f:
            f[f"z_{col}"] = z(f[col], win)

    # realised vol
    if "r_eua_eur_tco2" in f:
        f["eua_rv20"] = f["r_eua_eur_tco2"].rolling(20).std() * np.sqrt(252)
        f["eua_rv60"] = f["r_eua_eur_tco2"].rolling(60).std() * np.sqrt(252)

    return f
