# L1 Topic Research Runner

`l1_topic_research_runner.py` runs L1 with:
- config-driven model map load (`agent_model_map.yaml`)
- Gate-In validation
- L1 artifact generation
- Gate-Out validation

## Run

```bash
python3 "project i3/l1_topic_research_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective.json" \
  --topic-intake "project i3/examples/artifacts/sample_topic_intake_input.json" \
  --out-dir "project i3/examples/l1_run_output"
```

## Inputs
- `session_input.json`
- `session_objective.json`
- `topic_intake_input.json`
- `config/agent_model_map.yaml`
- `score_specs/score_spec_topic_intake.json`

## Outputs
- `gate_in_report.json`
- `candidates_topic_intake.json`
- `evaluation_report_topic_intake.json`
- `selection_trace_topic_intake.json`
- `topic_research_bundle.json`
- `claim_verification_report.json`
- `session_output.json`
- `runtime_log_l1.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: config/runtime error
