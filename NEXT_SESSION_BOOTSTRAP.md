# Next Session Bootstrap (2026-02-06)

## 1) What Was Finalized
- L5/L6/L7 now enforce strict session/objective alignment by default.
- Recovery rerun exception exists via `--allow-cross-session`.
- `session_objective.score_spec_ref` is now the default single source of truth for score spec.
- CLI score spec override is allowed only when all conditions are met:
  - `--allow-score-spec-override`
  - non-empty `--override-reason`
  - `--pipeline-env dev` (blocked in `prod`)
- Full test suite currently passes:
  - `python3 -m unittest discover -s "project i3/tests" -p "test_*.py"`
  - result: `Ran 59 tests ... OK`
- L8 downstream integration baseline is implemented:
  - `l8_downstream_asset_runner.py`
  - outputs: `clip_render_plan.json`, `dubbing_plan.json`, `sound_design_plan.json`, `asset_package_manifest.json`

## 2) Next Discussion Topics (Priority Order)
1. Prompt regeneration cache key/hash policy (scope + invalidation rules).
2. Runtime observability policy for per-scene token/cost metrics.
3. L7 regression expansion across multiple style presets.
4. L8 provider-adapter strategy for clip/dubbing/sound/package execution.

## 3) Discussion Checklist (Decide Before Coding)
1. Downstream compatibility
- In-scope consumers: clip, dubbing, sound, packaging (confirm all).
- Contract style: legacy fields must remain stable; additional fields must be non-breaking.
- Failure policy: downstream should fail fast on missing required legacy fields only.

2. Cache key/hash
- Reuse scope: same session only vs cross-session reuse.
- Key inputs (recommended baseline):
  - `scene_id`, normalized prompt inputs, `style_preset`
  - score spec file hash
  - model map file hash
  - prompt builder version string
- Invalidation trigger: any key input hash change.

3. Observability
- Required runtime log fields:
  - `tokens_in`, `tokens_out`, `cost_estimate_usd`
  - per-scene and total aggregates
- Sampling policy: always-on in dev and prod (recommended).

4. Regression coverage
- Minimum new dataset: at least 3 style presets with fixed fixtures.
- Assertions:
  - contract shape stability
  - hard-constraint pass behavior
  - deterministic tie-break behavior

## 4) Recommended Immediate Execution Plan
1. Lock discussion decisions above (10-15 min).
2. Implement cache key module and wire into L7 prompt generation.
3. Add observability fields to `runtime_log_l7.json`.
4. Expand style-based regression fixtures/tests.
5. Define L8 provider-adapter interface and minimal stub wiring.
6. Run full test suite and update `PROGRESS_HANDOFF.md`.

## 5) Files To Touch First
- `l7_scene_prompt_map_runner.py`
- `tests/test_l7_prompt_internal_regression.py`
- `tests/test_l7_scene_prompt_map_runner.py`
- `l8_downstream_asset_runner.py`
- `PROGRESS_HANDOFF.md`

## 6) Quick Start Commands
```bash
python3 -m unittest discover -s "project i3/tests" -p "test_*.py"
python3 -m unittest "project i3/tests/test_l8_downstream_asset_runner.py"
python3 -m unittest "project i3/tests/test_l7_prompt_internal_regression.py"
```

## 7) If No Further Discussion Is Needed
- Start with Topic 1 (cache key/hash policy) and keep defaults conservative:
  - deterministic key inputs only
  - invalidate on any key input hash change
  - keep schema strictness unchanged
