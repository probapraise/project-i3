# Project i3 Progress Handoff

Last updated: 2026-02-06
Workspace root: `/home/ljhljh/project i3`

## 1) Current Status (One-line)
L1~L8 are implemented with Gate-In/Out + schema validation + tests passing; downstream integration runner baseline is added and next priority is cache/observability hardening.

## 2) Non-negotiable Rules (Keep As-Is)
- Single model per agent, manual switch only (`config/agent_model_map.yaml` policy).
- Gate-In/Out validation is mandatory for every runner.
- Fixed schema contracts are enforced (fail closed on schema failure).
- Lock policy is fixed: `REAL > INFERRED > NO_INFO`.
- L5/L6/L7 score spec source is `session_objective.score_spec_ref` by default; CLI override is recovery-only.

## 3) Implemented Scope Summary

### Foundation
- `gate_validator.py`
- sample manifests:
  - `examples/gate_in_manifest.sample.json`
  - `examples/gate_out_manifest.sample.json`
- tests:
  - `tests/test_gate_validator.py`

### L1 Topic Research
- runner: `l1_topic_research_runner.py`
- readme: `README_l1_runner.md`
- tests: `tests/test_l1_topic_research_runner.py`
- outputs include `topic_research_bundle.json`, selection/evaluation traces, gate reports

### L2/L3 Evidence + Inference
- runner: `l2_l3_profile_runner.py`
- lock rules: `profile_lock_rules.py`
- readme: `README_l2_l3_runner.md`
- tests:
  - `tests/test_l2_l3_profile_runner.py`
  - `tests/test_profile_lock_rules.py`
- lock rule and inferred merge behavior validated

### L4 Hook
- runner: `l4_hook_runner.py`
- readme: `README_l4_runner.md`
- tests: `tests/test_l4_hook_runner.py`
- output contract: `script_seed_pack.json`

### L5 Script Session (newly implemented)
- runner: `l5_script_session_runner.py`
- readme: `README_l5_runner.md`
- tests: `tests/test_l5_script_session_runner.py`
- new contracts:
  - `schemas/script_session_pack.schema.json`
  - `score_specs/score_spec_script.json`
- sample artifacts:
  - `examples/artifacts/sample_session_input_l5.json`
  - `examples/artifacts/sample_session_objective_l5_script.json`
  - `examples/artifacts/sample_script_seed_pack.json`

### L6 Scene Planning (newly implemented)
- runner: `l6_scene_planning_runner.py`
- readme: `README_l6_runner.md`
- tests: `tests/test_l6_scene_planning_runner.py`
- new config:
  - `config/shorts.yaml`
- sample artifacts:
  - `examples/artifacts/sample_script_session_pack.json`
  - `examples/artifacts/sample_session_input_l6.json`
  - `examples/artifacts/sample_session_objective_l6_scene_plan.json`
- outputs include:
  - `story_bible.json`
  - `beats.json`
  - `candidates_scene_split.json`
  - `evaluation_report_scene_split.json`
  - `selection_trace_scene_split.json`
  - `scene_skeletons.json`
  - `scenes_v2.json`

### L7 Scene Prompt Map (newly implemented)
- runner: `l7_scene_prompt_map_runner.py`
- readme: `README_l7_runner.md`
- tests:
  - `tests/test_l7_scene_prompt_map_runner.py`
  - `tests/test_l5_l6_l7_integration.py`
  - `tests/test_l7_downstream_compatibility.py`
  - `tests/test_l7_prompt_internal_regression.py`
- hardening updates:
  - strict one-to-one scene/select mapping and prompt-shape checks in prompt-application stage
  - downstream compatibility matrix coverage for clip/dubbing/sound/packaging consumers
- new contracts:
  - `schemas/prompt_map_pack.schema.json`
- sample artifacts:
  - `examples/artifacts/sample_session_input_l7.json`
  - `examples/artifacts/sample_session_objective_l7_prompt_map.json`
- outputs include:
  - `candidates_prompt.json`
  - `evaluation_report_prompt.json`
  - `selection_trace_prompt.json`
  - `continuity_audit_prompt.json`
  - `prompt_map_pack.json`
  - updated `scenes_v2.json`

### L8 Downstream Asset Integration (newly implemented)
- runner: `l8_downstream_asset_runner.py`
- readme: `README_l8_runner.md`
- tests:
  - `tests/test_l8_downstream_asset_runner.py`
- new contracts:
  - `schemas/clip_render_plan.schema.json`
  - `schemas/dubbing_plan.schema.json`
  - `schemas/sound_design_plan.schema.json`
  - `schemas/asset_package_manifest.schema.json`
- outputs include:
  - `clip_render_plan.json`
  - `dubbing_plan.json`
  - `sound_design_plan.json`
  - `asset_package_manifest.json`

## 4) Key Config / Schema Additions
- `config/agent_model_map.yaml`
  - L5/L6/L7 agent definitions added
  - policy checks expected by runners (`manual_only`, single-model policy)
- `config/shorts.yaml`
  - scene constraints and style defaults for L6
- `schemas/script_session_pack.schema.json`
  - L5 handoff contract used by L6 Gate-In
- `schemas/prompt_map_pack.schema.json`
  - L7 prompt map output contract used by L7 Gate-Out
- `schemas/clip_render_plan.schema.json`
  - L8 clip downstream handoff contract
- `schemas/dubbing_plan.schema.json`
  - L8 dubbing downstream handoff contract
- `schemas/sound_design_plan.schema.json`
  - L8 sound downstream handoff contract
- `schemas/asset_package_manifest.schema.json`
  - L8 packaging manifest contract
- `score_specs/score_spec_script.json`
  - L5 script selection scoring
- `score_specs/score_spec_prompt.json`
  - L7 prompt candidate scoring/selection

## 5) Artifact Handoff Chain (Current)
- L1 output: `topic_research_bundle.json`
- L2/L3 output: `entity_profile_pack.json`
- L4 output: `script_seed_pack.json`
- L5 output: `script_session_pack.json`
- L6 output: `scenes_v2.json`
- L7 output: `prompt_map_pack.json` + prompt-enriched `scenes_v2.json`
- L8 output: `clip_render_plan.json` + `dubbing_plan.json` + `sound_design_plan.json` + `asset_package_manifest.json`

## 6) Latest Verification Snapshot
Executed on 2026-02-06:
- `python3 -m unittest discover -s "project i3/tests" -p "test_*.py"`
- Result: `Ran 59 tests ... OK`

Also executed successfully:
- `python3 "project i3/l6_scene_planning_runner.py" --session-input "project i3/examples/artifacts/sample_session_input_l6.json" --session-objective "project i3/examples/artifacts/sample_session_objective_l6_scene_plan.json" --script-session-pack "project i3/examples/artifacts/sample_script_session_pack.json" --out-dir "project i3/examples/l6_run_output"`
- `python3 "project i3/l7_scene_prompt_map_runner.py" --session-input "project i3/examples/artifacts/sample_session_input_l7.json" --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" --scenes-v2 "project i3/examples/l6_run_output/scenes_v2.json" --out-dir "project i3/examples/l7_run_output"`

## 7) Open Work (Next)
1. Add optional cache keys for scene prompt regeneration in L7 (hash policy + invalidation rules).
2. Add cost/token observability fields for scene-level map/reduce loops (L7 runtime log).
3. Expand golden/regression coverage for L7 prompt scoring on multiple script styles.
4. Extend L8 output from contract-level plans to provider-specific execution adapters.

## 8) Recommended First Steps in New Session
1. Read first: `NEXT_SESSION_BOOTSTRAP.md` (single-file quick start).
2. Then read: `PROGRESS_HANDOFF.md`.
3. Reconfirm frozen contract sections: `에이전트_맵_리팩토링_v1.md` sections 12/13/14.
4. Run full tests:
   - `python3 -m unittest discover -s "project i3/tests" -p "test_*.py"`
5. Start downstream compatibility verification with:
   - `examples/l7_run_output/scenes_v2.json`
   - `examples/l7_run_output/prompt_map_pack.json`
6. Execute L8 downstream integration baseline:
   - `python3 "project i3/l8_downstream_asset_runner.py" --session-input "project i3/examples/artifacts/sample_session_input_l7.json" --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" --scenes-v2 "project i3/examples/l7_run_output/scenes_v2.json" --prompt-map-pack "project i3/examples/l7_run_output/prompt_map_pack.json" --out-dir "project i3/examples/l8_run_output"`

## 9) Operational Notes
- This workspace is not a git repository (`git status` unavailable).
- Keep ASCII output style in generated JSON/MD unless existing file requires otherwise.
- Do not relax schema strictness (`additionalProperties: false` contracts are intentional).
- L5/L6/L7 enforce strict cross-artifact session/objective matching by default.
- Use `--allow-cross-session` only for recovery reruns that intentionally reuse older upstream artifacts.
- L5/L6/L7 enforce `session_objective.score_spec_ref` and reject mismatched `--score-spec` by default.
- `--allow-score-spec-override` requires non-empty `--override-reason` and is blocked in `--pipeline-env prod`.
