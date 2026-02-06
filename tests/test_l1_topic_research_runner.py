import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l1_topic_research_runner import build_candidate  # noqa: E402


class L1TopicResearchRunnerTests(unittest.TestCase):
    def test_build_candidate_merges_sources_for_duplicate_claim_text(self) -> None:
        candidate = build_candidate(
            candidate_id="cand_test",
            topic_query="duplicate claim source merge",
            materials=[
                {
                    "material_id": "mat_001",
                    "material_type": "URL",
                    "content": "John was arrested in 2020.",
                    "url": "https://example.com/a",
                    "source_name": "SourceA",
                },
                {
                    "material_id": "mat_002",
                    "material_type": "URL",
                    "content": "John was arrested in 2020.",
                    "url": "https://example.com/b",
                    "source_name": "SourceB",
                },
            ],
        )

        self.assertEqual(len(candidate["claims"]), 1)
        claim = candidate["claims"][0]
        self.assertEqual(claim["verification_status"], "VERIFIED")
        self.assertEqual(set(claim["supporting_source_ids"]), {"src_001", "src_002"})

    def test_build_candidate_merges_sources_for_punctuation_variants(self) -> None:
        candidate = build_candidate(
            candidate_id="cand_test_punct",
            topic_query="punctuation variant merge",
            materials=[
                {
                    "material_id": "mat_001",
                    "material_type": "URL",
                    "content": "John was arrested in 2020!",
                    "url": "https://example.com/a",
                    "source_name": "SourceA",
                },
                {
                    "material_id": "mat_002",
                    "material_type": "URL",
                    "content": " John  was arrested in 2020 . ",
                    "url": "https://example.com/b",
                    "source_name": "SourceB",
                },
            ],
        )

        self.assertEqual(len(candidate["claims"]), 1)
        claim = candidate["claims"][0]
        self.assertEqual(set(claim["supporting_source_ids"]), {"src_001", "src_002"})

    def test_runner_completes_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l1_topic_research_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective.json"),
                    "--topic-intake",
                    str(ROOT / "examples" / "artifacts" / "sample_topic_intake_input.json"),
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
                "topic_research_bundle.json",
                "claim_verification_report.json",
                "evaluation_report_topic_intake.json",
                "selection_trace_topic_intake.json",
                "candidates_topic_intake.json",
                "session_output.json",
                "runtime_log_l1.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            bundle = json.loads(
                (out_dir / "topic_research_bundle.json").read_text(encoding="utf-8")
            )
            self.assertIn(bundle["research_status"], {"READY_FOR_L2", "NEEDS_MORE_RESEARCH"})
            self.assertGreaterEqual(len(bundle["normalized_sources"]), 1)
            self.assertGreaterEqual(len(bundle["claims"]), 1)

    def test_runner_fails_when_gate_in_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_topic = tmpdir / "bad_topic.json"
            bad_topic.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "session_id": "sess_l1_0001",
                        "objective_id": "obj_l1_research",
                        "created_at": "2026-02-06T12:00:00Z",
                        "user_collected_materials": [
                            {
                                "material_id": "mat_001",
                                "material_type": "NOTE",
                                "content": "missing topic_query",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l1_topic_research_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective.json"),
                    "--topic-intake",
                    str(bad_topic),
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

    def test_runner_fails_on_session_objective_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_objective = tmpdir / "bad_session_objective.json"
            objective_payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_session_objective.json").read_text(
                    encoding="utf-8"
                )
            )
            objective_payload["objective_id"] = "obj_mismatch_l1"
            bad_objective.write_text(json.dumps(objective_payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l1_topic_research_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input.json"),
                    "--session-objective",
                    str(bad_objective),
                    "--topic-intake",
                    str(ROOT / "examples" / "artifacts" / "sample_topic_intake_input.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("session_input.objective_id must match", proc.stderr)


if __name__ == "__main__":
    unittest.main()
