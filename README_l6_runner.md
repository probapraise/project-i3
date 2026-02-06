# L6 Scene Planning Runner

`l6_scene_planning_runner.py` runs:
- story bible construction
- beat splitting
- scene split candidate generation
- score-based selection (`score_spec_scene_split.json`)
- `scenes_v2.json` export
- Gate-In/Out validation

## Run

```bash
python3 "project i3/l6_scene_planning_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l6.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l6_scene_plan.json" \
  --script-session-pack "project i3/examples/artifacts/sample_script_session_pack.json" \
  --out-dir "project i3/examples/l6_run_output"
```

Recovery rerun (reuse cross-session upstream artifact):

```bash
python3 "project i3/l6_scene_planning_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l6.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l6_scene_plan.json" \
  --script-session-pack "project i3/examples/artifacts/sample_script_session_pack.json" \
  --allow-cross-session \
  --out-dir "project i3/examples/l6_run_output"
```

Score spec override (allowed only in non-prod recovery reruns with explicit reason):

```bash
python3 "project i3/l6_scene_planning_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l6.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l6_scene_plan.json" \
  --script-session-pack "project i3/examples/artifacts/sample_script_session_pack.json" \
  --score-spec "project i3/score_specs/score_spec_scene_split.json" \
  --allow-score-spec-override \
  --override-reason "recovery rerun after upstream failure" \
  --pipeline-env dev \
  --out-dir "project i3/examples/l6_run_output"
```

Policy notes:
- 기본값은 `session_objective.score_spec_ref`와 `--score-spec` strict 일치입니다.
- 경로가 다르면 `--allow-score-spec-override` + `--override-reason`이 필요합니다.
- `--pipeline-env prod`에서는 score spec override가 금지됩니다.
- `session_input`과 `session_objective`의 `session_id/objective_id`는 항상 strict 일치해야 합니다.
- `--allow-cross-session`은 `script_session_pack` 재사용(복구 rerun) 용도로만 적용됩니다.

## Outputs
- `gate_in_report.json`
- `story_bible.json`
- `beats.json`
- `candidates_scene_split.json`
- `evaluation_report_scene_split.json`
- `selection_trace_scene_split.json`
- `scene_skeletons.json`
- `scenes_v2.json`
- `runtime_log_l6.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: config/runtime/contract error
