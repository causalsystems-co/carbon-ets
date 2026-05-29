# Contributing

## Branches
- `main` is always green — baseline runs end-to-end without errors.
- Work on `feature/<short-name>` branches.
- Open a PR when ready, request review from Halis.

## Don't touch
- `scripts/05_backtest.py` is the reference baseline. Copy it to a new
  numbered file rather than editing in place.
- `data/*.parquet` and `data/*.csv` are gitignored — never commit them.

## Do touch
- New scripts: `scripts/NN_verb_object.py` where NN > 05.
- New notebooks: `notebooks/NN_<topic>.ipynb`.
- README or HANDOFF if you change the workflow.

## Commit style
- Imperative, present tense, ≤ 72 chars on first line.
- "add gas momentum feature" not "Added gas momentum feature."
- No emoji.

## Before opening a PR
- `make clean && make run` finishes without errors.
- New script prints a stats block compatible with `05_backtest.py`'s.
- If it adds a dependency, update `requirements.txt`.
