# L5 Script Session Runner

`l5_script_session_runner.py` runs:
- script candidate generation
- score-based evaluation (`score_spec_script.json`)
- script selection + translation/reconstruction outputs
- narration extraction
- Gate-In/Out validation

## Run

```bash
python3 "project i3/l5_script_session_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l5.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l5_script.json" \
  --topic-research-bundle "project i3/examples/artifacts/sample_topic_research_bundle.json" \
  --entity-profile-pack "project i3/examples/artifacts/sample_entity_profile_pack.json" \
  --script-seed-pack "project i3/examples/artifacts/sample_script_seed_pack.json" \
  --out-dir "project i3/examples/l5_run_output"
```

Recovery rerun (reuse cross-session upstream artifact):

```bash
python3 "project i3/l5_script_session_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l5.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l5_script.json" \
  --topic-research-bundle "project i3/examples/artifacts/sample_topic_research_bundle.json" \
  --entity-profile-pack "project i3/examples/artifacts/sample_entity_profile_pack.json" \
  --script-seed-pack "project i3/examples/artifacts/sample_script_seed_pack.json" \
  --allow-cross-session \
  --out-dir "project i3/examples/l5_run_output"
```

Score spec override (allowed only in non-prod recovery reruns with explicit reason):

```bash
python3 "project i3/l5_script_session_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l5.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l5_script.json" \
  --topic-research-bundle "project i3/examples/artifacts/sample_topic_research_bundle.json" \
  --entity-profile-pack "project i3/examples/artifacts/sample_entity_profile_pack.json" \
  --script-seed-pack "project i3/examples/artifacts/sample_script_seed_pack.json" \
  --score-spec "project i3/score_specs/score_spec_script.json" \
  --allow-score-spec-override \
  --override-reason "recovery rerun after upstream failure" \
  --pipeline-env dev \
  --out-dir "project i3/examples/l5_run_output"
```

Policy notes:
- 기본값은 `session_objective.score_spec_ref`와 `--score-spec` strict 일치입니다.
- 경로가 다르면 `--allow-score-spec-override` + `--override-reason`이 필요합니다.
- `--pipeline-env prod`에서는 score spec override가 금지됩니다.

## Outputs
- `gate_in_report.json`
- `candidates_script.json`
- `evaluation_report_script.json`
- `selection_trace_script.json`
- `script_session_pack.json`
- `runtime_log_l5.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: config/runtime error
