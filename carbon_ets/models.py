"""
carbon_ets.models — baseline OLS + Markov-switching + backtest utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_v1(
    panel: pd.DataFrame,
    target_ret: str = "r_eua_eur_tco2",
    features: tuple[str, ...] = ("z_ip_yoy", "z_stoxx_mom20"),
    short_floor: float = 0.0,
    upper: float = 1.0,
) -> tuple[dict, pd.Series]:
    """Baseline V1 causal-chain strategy: mean of z-scored features, long-only.

    Parameters
    ----------
    panel : DataFrame
        Feature-engineered panel from `carbon_ets.features.engineer`.
    target_ret : str
        Column of daily log-returns for the traded asset.
    features : tuple
        Feature columns to average (equal-weighted).
    short_floor : float
        Lower bound on position (0.0 = long-only).
    upper : float
        Upper bound on position.

    Returns
    -------
    stats : dict
        {'cagr', 'sharpe', 'vol', 'max_dd', 'hit_rate', 'n_days'}
    equity : Series
        Cumulative equity curve.
    """
    f = panel.dropna(subset=[target_ret]).copy()

    # Compose signal
    feat_cols = [c for c in features if c in f.columns]
    if not feat_cols:
        raise ValueError(f"None of {features} in panel columns")
    feats = f[feat_cols].dropna(how="any")
    score = feats.mean(axis=1).reindex(f.index)
    pos = score.clip(lower=short_floor, upper=upper).shift(1)

    pnl = (pos * f[target_ret]).dropna()
    eq = (1 + pnl).cumprod()

    stats = compute_stats(eq, pnl)
    return stats, eq


def compute_stats(eq: pd.Series, pnl: pd.Series) -> dict:
    """Standard performance stats."""
    if len(eq) == 0:
        return dict(cagr=np.nan, sharpe=np.nan, vol=np.nan, max_dd=np.nan, hit_rate=np.nan, n_days=0)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    mu = pnl.mean() * 252
    sd = pnl.std() * np.sqrt(252)
    return {
        "cagr":     (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
        "sharpe":   mu / sd if sd else np.nan,
        "vol":      sd,
        "max_dd":   (eq / eq.cummax() - 1).min(),
        "hit_rate": (pnl > 0).mean(),
        "n_days":   int(pnl.notna().sum()),
    }


def fit_markov_switching(panel: pd.DataFrame, target_ret: str = "r_eua_eur_tco2",
                        features: tuple[str, ...] = ("z_ip_yoy", "z_stoxx_mom20")):
    """Fit a two-regime Markov-switching regression on monthly EUA returns.

    Returns
    -------
    result : statsmodels MarkovRegressionResults
    monthly_panel : DataFrame used for fitting
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    # Aggregate to monthly
    monthly = panel[target_ret].resample("ME").sum().to_frame(name=target_ret)
    for c in features:
        if c in panel.columns:
            monthly[c] = panel[c].resample("ME").last()
    monthly = monthly.dropna(subset=[target_ret] + list(features))
    if len(monthly) < 30:
        raise ValueError(f"Too few observations for MS fit: {len(monthly)}")

    y = monthly[target_ret].values
    X = monthly[list(features)].values
    model = MarkovRegression(y, k_regimes=2, exog=X, switching_variance=True)
    res = model.fit(disp=False, maxiter=300)
    return res, monthly
