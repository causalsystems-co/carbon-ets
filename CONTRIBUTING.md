# Contributing to carbon-ets

Thank you for considering a contribution. This is a small, quality-first project maintained by [Causal Systems](https://causalsystems.co).

## Values

- **Correctness before features.** If a value cannot be traced to a primary source, it doesn't ship.
- **Reproducibility over convenience.** Every claim in the docs must be reproducible from the code.
- **Honest limitations.** If a feature fails internal validation, we document the failure rather than hide it.

## Kinds of contributions especially welcome

- **Data gap fills.** Missing pre-2013 EEX data, UK ETS integration, California CCA integration.
- **Correctness fixes.** Any discrepancy between a hardcoded value and its primary source.
- **Test coverage.** New tests, especially against publicly-verifiable numbers.
- **Documentation.** Clearer explanations, worked examples, tutorial notebooks.

## Kinds of contributions we will not accept

- **Trading strategies without out-of-sample validation.**
- **Data fetched from paid feeds.** All data must come from freely accessible sources.
- **Features that fail internal accuracy checks.** See the v0.2 roadmap in CHANGELOG.

## Development setup

```bash
git clone https://github.com/causalsystems-co/carbon-ets
cd carbon-ets
pip install -e ".[dev,all]"
pytest tests/
```

Tests must pass before a PR will be reviewed.

## Contact

hello@causalsystems.co
