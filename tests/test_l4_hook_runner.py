import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class L4HookRunnerTests(unittest.TestCase):
    def _write_aligned_topic_research_bundle(self, tmpdir: Path) -> Path:
        session_input = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_session_input_l4.json").read_text(
                encoding="utf-8"
            )
        )
        bundle = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        bundle["session_id"] = session_input["session_id"]
        bundle["objective_id"] = session_input["objective_id"]

        bundle_path = tmpdir / "aligned_topic_research_bundle.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        return bundle_path

    def test_runner_completes_and_outputs_script_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            aligned_bundle = self._write_aligned_topic_research_bundle(tmpdir)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l4_hook_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l4.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l4_hook.json"),
                    "--topic-research-bundle",
                    str(aligned_bundle),
                    "--entity-profile-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            required_files = [
                "gate_in_report.json",
                "candidates_hook.json",
                "evaluation_report_hook.json",
                "selection_trace_hook.json",
                "script_seed_pack.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            seed_pack = json.loads((out_dir / "script_seed_pack.json").read_text(encoding="utf-8"))
            self.assertIn(seed_pack["quality_status"], {"READY_FOR_SCRIPT", "NEEDS_HOOK_REWORK"})
            self.assertIn("hook_text", seed_pack["selected_hook"])
            self.assertGreaterEqual(len(seed_pack["trigger_words"]), 1)

    def test_selected_hook_exists_in_candidate_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            aligned_bundle = self._write_aligned_topic_research_bundle(tmpdir)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l4_hook_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l4.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l4_hook.json"),
                    "--topic-research-bundle",
                    str(aligned_bundle),
                    "--entity-profile-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            seed_pack = json.loads((out_dir / "script_seed_pack.json").read_text(encoding="utf-8"))
            selected_id = seed_pack["selected_hook"]["hook_id"]
            hook_ids = {row["hook_id"] for row in seed_pack.get("hook_bank", [])}
            self.assertIn(selected_id, hook_ids)

    def test_runner_fails_on_invalid_gate_in_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_entity_profile = tmpdir / "bad_entity_profile.json"
            bad_entity_profile.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "session_id": "sess_l4_0001",
                        "objective_id": "obj_l4_hook",
                        "created_at": "2026-02-06T12:30:00Z",
                        "case_id": "case_sess_l4_0001",
                        "entities": [],
                        "context_fields": [],
                        "profile_status": "NEEDS_HITL_REVIEW",
                    }
                ),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l4_hook_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l4.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l4_hook.json"),
                    "--topic-research-bundle",
                    str(ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json"),
                    "--entity-profile-pack",
                    str(bad_entity_profile),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            gate_in = json.loads((out_dir / "gate_in_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_in["overall_status"], "FAIL")

    def test_runner_fails_on_topic_research_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l4_hook_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l4.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l4_hook.json"),
                    "--topic-research-bundle",
                    str(ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json"),
                    "--entity-profile-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("topic_research_bundle.session_id must match", proc.stderr)


if __name__ == "__main__":
    unittest.main()
