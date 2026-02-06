# L2/L3 Profile Runner

`l2_l3_profile_runner.py` runs:
- L2: evidence structuring (`REAL`, `NO_INFO`)
- L3: inference completion (`INFERRED`) with lock rule
- lock rule: `REAL > INFERRED > NO_INFO`

## Run

```bash
python3 "project i3/l2_l3_profile_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l2.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l2_l3.json" \
  --topic-research-bundle "project i3/examples/artifacts/sample_topic_research_bundle.json" \
  --inference-rules "project i3/config/inference_rules.yaml" \
  --out-dir "project i3/examples/l2_l3_run_output"
```

## Outputs
- `gate_in_report.json`
- `entity_evidence_pack.json`
- `inference_candidates.json`
- `inference_merge_report.json`
- `inference_rules_snapshot.json`
- `entity_profile_pack.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: runtime error
