# Carbon_ETS — one-command targets.
# Usage:   make setup       (first time only)
#          make run         (fetch + analyse + backtest end-to-end)
#          make backtest    (just re-run the strategy)
#          make notebook    (open the explorer)
#          make clean       (wipe data/ and plots/)

PY      := .venv/bin/python
PIP     := .venv/bin/pip
START   := 2018-01-01

.PHONY: setup fetch analyse backtest run notebook clean

setup:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo
	@echo "  ✓ venv ready at .venv/"
	@echo "  next:  make run"

fetch:
	$(PY) scripts/01_fetch_prices.py     --start $(START)
	$(PY) scripts/01b_fetch_eua_auctions.py
	$(PY) scripts/02_fetch_emissions.py  --start $(START)
	$(PY) scripts/03_build_dataset.py

# fetch only the long-history EUA series (~13 years, EEX auctions)
fetch-eua:
	$(PY) scripts/01b_fetch_eua_auctions.py

analyse:
	$(PY) scripts/04_analyze_chain.py

backtest:
	$(PY) scripts/05_backtest.py

run: fetch analyse backtest
	@echo
	@echo "  ✓ pipeline done.  see plots/equity_curve.png and plots/leadlag.png"

notebook:
	$(PY) -m jupyter lab notebooks/

clean:
	rm -f data/*.parquet data/*.csv plots/*.png
	@echo "  ✓ data/ and plots/ cleaned"
