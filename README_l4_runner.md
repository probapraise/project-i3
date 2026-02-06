# L4 Hook Runner

`l4_hook_runner.py` runs:
- hook candidate generation
- score-based evaluation (`score_spec_hook.json`)
- selector output + `script_seed_pack.json`
- Gate-In/Out validation

## Run

```bash
python3 "project i3/l4_hook_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l4.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l4_hook.json" \
  --topic-research-bundle "project i3/examples/artifacts/sample_topic_research_bundle.json" \
  --entity-profile-pack "project i3/examples/artifacts/sample_entity_profile_pack.json" \
  --out-dir "project i3/examples/l4_run_output"
```

## Outputs
- `gate_in_report.json`
- `candidates_hook.json`
- `evaluation_report_hook.json`
- `selection_trace_hook.json`
- `script_seed_pack.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: runtime error
