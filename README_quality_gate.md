# Quality Gate

This project uses a single quality gate command:

```bash
make qa
```

`qa` runs:
- lint (`ruff`)
- typecheck (`mypy`)
- tests (`unittest`, including the golden E2E snapshot test)

## Local setup

Install dev dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

If `make` is unavailable, run commands directly:

```bash
python3 -m ruff check . --select E9,F63,F7,F82
python3 -m mypy --config-file mypy.ini
python3 -m unittest discover -s tests -q
```

## Golden snapshot workflow

Run only the golden test:

```bash
make test-golden
```

Update snapshot intentionally:

```bash
make golden-update
```

## CI

GitHub Actions workflow: `.github/workflows/quality-gate.yml`

Current triggers:

- `pull_request` targeting `main`
- `push` to `main`
- `workflow_dispatch` (manual run)

Required repository check name: `qa`

Operational rule:

- Do not edit workflow/documents directly on `main`.
- Use a branch -> PR -> `qa` pass -> merge.
