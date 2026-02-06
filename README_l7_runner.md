# L7 Scene Prompt Map Runner

`l7_scene_prompt_map_runner.py` runs:
- scene context packaging
- per-scene prompt candidate generation
- score-based selection (`score_spec_prompt.json`)
- continuity audit and selective prompt revision
- `prompt_map_pack.json` + updated `scenes_v2.json` export
- Gate-In/Out validation

## Run

```bash
python3 "project i3/l7_scene_prompt_map_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l6_run_output/scenes_v2.json" \
  --out-dir "project i3/examples/l7_run_output"
```

Recovery rerun (reuse cross-session upstream artifact):

```bash
python3 "project i3/l7_scene_prompt_map_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l6_run_output/scenes_v2.json" \
  --allow-cross-session \
  --out-dir "project i3/examples/l7_run_output"
```

Score spec override (allowed only in non-prod recovery reruns with explicit reason):

```bash
python3 "project i3/l7_scene_prompt_map_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l6_run_output/scenes_v2.json" \
  --score-spec "project i3/score_specs/score_spec_prompt.json" \
  --allow-score-spec-override \
  --override-reason "recovery rerun after upstream failure" \
  --pipeline-env dev \
  --out-dir "project i3/examples/l7_run_output"
```

Policy notes:
- 기본값은 `session_objective.score_spec_ref`와 `--score-spec` strict 일치입니다.
- 경로가 다르면 `--allow-score-spec-override` + `--override-reason`이 필요합니다.
- `--pipeline-env prod`에서는 score spec override가 금지됩니다.
- `session_input`과 `session_objective`의 `session_id/objective_id`는 항상 strict 일치해야 합니다.
- `--allow-cross-session`은 `scenes_v2` 재사용(복구 rerun) 용도로만 적용됩니다.

Scene-level partial rerun (regenerate only specific scenes):

```bash
python3 "project i3/l7_scene_prompt_map_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l7_run_output/scenes_v2.json" \
  --scene-id "scene_003" \
  --scene-id "scene_005" \
  --base-prompt-map-pack "project i3/examples/l7_run_output/prompt_map_pack.json" \
  --out-dir "project i3/examples/l7_rerun_output"
```

Partial rerun notes:
- `--scene-id`를 주면 해당 씬만 재생성/재채점합니다.
- 나머지 씬의 선택 결과는 `--base-prompt-map-pack`에서 그대로 병합합니다.
- `--scene-id` 사용 시 `--base-prompt-map-pack`는 필수입니다.
- `--base-prompt-map-pack`의 `session_id/objective_id/style_preset`은 현재 입력과 일치해야 합니다.

## Outputs
- `gate_in_report.json`
- `candidates_prompt.json`
- `evaluation_report_prompt.json`
- `selection_trace_prompt.json`
- `continuity_audit_prompt.json`
- `prompt_map_pack.json`
- `scenes_v2.json`
- `runtime_log_l7.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: config/runtime/contract error
