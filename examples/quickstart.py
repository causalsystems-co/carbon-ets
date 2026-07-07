"""
carbon-ets quickstart — end-to-end example.

Reproduces the full CS/RES/05 backtest in ~20 lines of user code.

Run:
    pip install carbon-ets[yfinance]
    python examples/quickstart.py
"""

from carbon_ets.data import build_full_panel
from carbon_ets.features import engineer
from carbon_ets.models import backtest_v1
from carbon_ets.tnac import get_reference_series, nowcast_monthly, validate_reference
from carbon_ets.plots import plot_equity_curve, plot_tnac_nowcast


def main():
    print("Fetching EEX + Eurostat + yfinance data (may take 30-60 seconds)...")
    panel = build_full_panel(start="2012-01-01")
    print(f"  panel shape: {panel.shape}")
    print(f"  range: {panel.index.min().date()} → {panel.index.max().date()}")

    print("\nEngineering features...")
    feats = engineer(panel)
    print(f"  columns: {list(feats.columns)}")

    print("\nRunning V1 backtest (IP YoY + Stoxx 50 momentum, long-only)...")
    stats, eq = backtest_v1(feats)
    for k, v in stats.items():
        print(f"  {k:10s}: {v:+.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")

    plot_equity_curve(eq, title=f"EUA V1 strategy — Sharpe {stats['sharpe']:.2f}",
                      savepath="equity_curve.png")
    print("  wrote equity_curve.png")

    print("\nTNAC nowcast for today:")
    est = nowcast_monthly()
    print(f"  Estimated TNAC: {est.tnac_estimate:,.0f}")
    print(f"  Confidence:     {est.confidence}")

    print("\nHistorical TNAC nowcast accuracy:")
    val = validate_reference()
    print(val.to_string())

    plot_tnac_nowcast(
        get_reference_series(),
        current_estimate={
            "as_of": est.as_of, "tnac_estimate": est.tnac_estimate,
            "confidence": est.confidence,
        },
        savepath="tnac_nowcast.png",
    )
    print("  wrote tnac_nowcast.png")


if __name__ == "__main__":
    main()
