# Contributing

This repository accompanies a research paper. Bug reports, reproduction
issues, and focused pull requests are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

Please run the same checks as continuous integration before opening a pull
request:

```bash
ruff check src scripts tests
ruff format --check src scripts tests
pytest
```

## Scope

- Keep the one-module-per-concern layout under `src/pidjepa/`.
- Add a regression test for behaviour changes.
- Preserve the `report.json` schema unless a change is explicitly a
  breaking release; figures and saved results depend on it.
