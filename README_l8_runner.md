# L8 Downstream Asset Runner

`l8_downstream_asset_runner.py` runs:
- downstream contract validation against prompt-enriched `scenes_v2.json` + `prompt_map_pack.json`
- clip render plan export
- dubbing plan export
- sound design plan export
- packaging manifest export
- Gate-In/Out validation

## Run

```bash
python3 "project i3/l8_downstream_asset_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l7_run_output/scenes_v2.json" \
  --prompt-map-pack "project i3/examples/l7_run_output/prompt_map_pack.json" \
  --out-dir "project i3/examples/l8_run_output"
```

Recovery rerun (cross-session upstream artifact reuse):

```bash
python3 "project i3/l8_downstream_asset_runner.py" \
  --session-input "project i3/examples/artifacts/sample_session_input_l7.json" \
  --session-objective "project i3/examples/artifacts/sample_session_objective_l7_prompt_map.json" \
  --scenes-v2 "project i3/examples/l7_run_output/scenes_v2.json" \
  --prompt-map-pack "project i3/examples/l7_run_output/prompt_map_pack.json" \
  --allow-cross-session \
  --out-dir "project i3/examples/l8_run_output"
```

## Outputs
- `gate_in_report.json`
- `clip_render_plan.json`
- `dubbing_plan.json`
- `sound_design_plan.json`
- `asset_package_manifest.json`
- `runtime_log_l8.json`
- `session_output.json`
- `gate_out_report.json`

## Exit codes
- `0`: success
- `1`: gate validation failed
- `2`: contract/runtime error
# probe
# probe
# probe again
# probe
# visibility probe
