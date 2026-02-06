# Contributing Guide

## Workflow

1. Create a working branch from `main`.
2. Implement the change in small commits.
3. Run local quality gate before push.
4. Push branch and open a pull request to `main`.
5. Wait for GitHub Actions `Quality Gate / qa` success.
6. Merge only after all required checks pass.

Direct push to `main` is blocked by repository rules.
This also applies to workflow/config/docs changes.

## Branch Naming

Use short and explicit branch names:

- `feature/<topic>`
- `fix/<topic>`
- `chore/<topic>`
- `test/<topic>`

Examples:

- `feature/l7-prompt-audit`
- `fix/l6-duration-score`

## Local Quality Gate

Use the project virtual environment and run:

```bash
source .venv/bin/activate
make qa
```

`make qa` runs lint, typecheck, and unit tests.

## Golden Snapshot Rule

If a change intentionally affects pipeline output, update the golden snapshot:

```bash
make golden-update
```

Then review snapshot diff carefully and mention the reason in PR description.

## Pull Request Rules

- Keep PR scope focused.
- Include behavior impact and risk in the PR body.
- If schema or contract changes, update related docs/samples/tests in the same PR.
- If CI fails, push fixes to the same PR branch and rerun checks.
- For CI workflow edits, verify both local `make qa` and PR `Quality Gate / qa`.

## Suggested Commit Message Style

- `feat: add L7 continuity scoring guard`
- `fix: resolve mypy type narrowing in L6`
- `test: add golden snapshot for L1-L8 regression`
- `docs: update QA workflow`
