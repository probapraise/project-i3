import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate_validator import ManifestError, run_gate_validation  # noqa: E402


class GateValidatorTests(unittest.TestCase):
    def test_gate_in_sample_manifest_passes(self) -> None:
        report = run_gate_validation(
            manifest_path=ROOT / "examples" / "gate_in_manifest.sample.json",
            requested_gate="in",
        )
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["summary"]["failed"], 0)

    def test_gate_out_sample_manifest_passes(self) -> None:
        report = run_gate_validation(
            manifest_path=ROOT / "examples" / "gate_out_manifest.sample.json",
            requested_gate="out",
        )
        self.assertEqual(report["overall_status"], "PASS")
        self.assertEqual(report["summary"]["failed"], 0)

    def test_schema_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_artifact = tmpdir / "bad_session_input.json"
            bad_artifact.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "objective_id": "obj_missing_session_id",
                        "created_at": "2026-02-06T12:00:00Z",
                        "input_artifact_refs": [
                            {"name": "topic_intake_input", "path": "x.json"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = tmpdir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "gate": "in",
                        "artifacts": [
                            {
                                "name": "session_input",
                                "artifact_path": "bad_session_input.json",
                                "schema_path": str(
                                    ROOT / "schemas" / "session_input.schema.json"
                                ),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_gate_validation(manifest_path=manifest, requested_gate="in")
            self.assertEqual(report["overall_status"], "FAIL")
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertGreater(report["checks"][0]["error_count"], 0)

    def test_gate_mismatch_raises(self) -> None:
        with self.assertRaises(ManifestError):
            run_gate_validation(
                manifest_path=ROOT / "examples" / "gate_in_manifest.sample.json",
                requested_gate="out",
            )

    def test_cli_exit_code_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            artifact = tmpdir / "bad.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = tmpdir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "gate": "out",
                        "artifacts": [
                            {
                                "name": "session_output",
                                "artifact_path": "bad.json",
                                "schema_path": str(
                                    ROOT / "schemas" / "session_output.schema.json"
                                ),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "gate_validator.py"),
                    "--gate",
                    "out",
                    "--manifest",
                    str(manifest),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
