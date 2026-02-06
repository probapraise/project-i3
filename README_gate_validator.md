# Gate Validator

`gate_validator.py` validates pipeline contracts at Gate-In and Gate-Out with JSON Schema.

## CLI

```bash
python3 "project i3/gate_validator.py" \
  --gate in \
  --manifest "project i3/examples/gate_in_manifest.sample.json" \
  --report-out "project i3/examples/reports/gate_in_report.json" \
  --pretty
```

```bash
python3 "project i3/gate_validator.py" \
  --gate out \
  --manifest "project i3/examples/gate_out_manifest.sample.json" \
  --report-out "project i3/examples/reports/gate_out_report.json" \
  --pretty
```

## Manifest format

```json
{
  "schema_version": "1.0.0",
  "gate": "in",
  "artifacts": [
    {
      "name": "session_input",
      "artifact_path": "artifacts/sample_session_input.json",
      "schema_path": "../schemas/session_input.schema.json"
    }
  ]
}
```

Rules:
- `gate` must match `--gate`.
- `artifact_path` and `schema_path` are resolved relative to manifest file location.
- Exit code `0`: PASS, `1`: validation FAIL, `2`: runtime/manifest error.

## Tests

```bash
python3 -m unittest discover -s "project i3/tests" -p "test_*.py"
```
