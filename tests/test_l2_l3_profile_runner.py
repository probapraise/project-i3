import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2_l3_profile_runner import extract_age  # noqa: E402


class L2L3ProfileRunnerTests(unittest.TestCase):
    def test_extract_age_returns_exact_age_for_year_old_pattern(self) -> None:
        age, sources = extract_age(
            [
                {
                    "claim_text": "A 37-year-old suspect was arrested after investigation.",
                    "supporting_source_ids": ["src_001"],
                }
            ]
        )
        self.assertEqual(age, "37")
        self.assertEqual(sources, ["src_001"])

    def test_extract_age_returns_decade_for_decade_pattern(self) -> None:
        age, sources = extract_age(
            [
                {
                    "claim_text": "Witnesses said he was in his 30s.",
                    "supporting_source_ids": ["src_010"],
                }
            ]
        )
        self.assertEqual(age, "30s")
        self.assertEqual(sources, ["src_010"])

    def test_extract_age_supports_single_digit_exact_age(self) -> None:
        age, sources = extract_age(
            [
                {
                    "claim_text": "A 9-year-old child was identified in witness reports.",
                    "supporting_source_ids": ["src_020"],
                }
            ]
        )
        self.assertEqual(age, "9")
        self.assertEqual(sources, ["src_020"])

    def test_extract_age_ignores_out_of_range_age(self) -> None:
        age, sources = extract_age(
            [
                {
                    "claim_text": "A 130-year-old witness appeared in the rumor.",
                    "supporting_source_ids": ["src_030"],
                }
            ]
        )
        self.assertIsNone(age)
        self.assertEqual(sources, [])

    def _write_aligned_research_bundle(self, tmpdir: Path) -> Path:
        session_input = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_session_input_l2.json").read_text(
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

    def test_runner_completes_and_outputs_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            aligned_bundle = self._write_aligned_research_bundle(tmpdir)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l2_l3_profile_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l2.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l2_l3.json"),
                    "--topic-research-bundle",
                    str(aligned_bundle),
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
                "entity_evidence_pack.json",
                "inference_candidates.json",
                "inference_merge_report.json",
                "inference_rules_snapshot.json",
                "entity_profile_pack.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

    def test_real_lock_is_preserved_after_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            aligned_bundle = self._write_aligned_research_bundle(tmpdir)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l2_l3_profile_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l2.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l2_l3.json"),
                    "--topic-research-bundle",
                    str(aligned_bundle),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            evidence_pack = json.loads(
                (out_dir / "entity_evidence_pack.json").read_text(encoding="utf-8")
            )
            profile_pack = json.loads(
                (out_dir / "entity_profile_pack.json").read_text(encoding="utf-8")
            )

            def find_field(pack: dict, entity_id: str, field_name: str) -> dict:
                entity = next(e for e in pack["entities"] if e["entity_id"] == entity_id)
                return next(f for f in entity["fields"] if f["field"] == field_name)

            evidence_occupation = find_field(evidence_pack, "person_victim", "occupation")
            profile_occupation = find_field(profile_pack, "person_victim", "occupation")
            if evidence_occupation["source_tag"] == "REAL":
                self.assertEqual(profile_occupation["source_tag"], "REAL")
                self.assertEqual(profile_occupation["value"], evidence_occupation["value"])

    def test_runner_applies_custom_inference_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            rules_path = tmpdir / "custom_rules.yaml"
            rules_path.write_text(
                "\n".join(
                    [
                        'version: "1.0.0"',
                        "defaults:",
                        '  source_tag: "INFERRED"',
                        "rules:",
                        "  PERSON:",
                        "    hair_style:",
                        '      - value: "buzz cut"',
                        "        confidence: 0.91",
                        '        reason: "custom test rule"',
                        "  LOCATION: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            aligned_bundle = self._write_aligned_research_bundle(tmpdir)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l2_l3_profile_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l2.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l2_l3.json"),
                    "--topic-research-bundle",
                    str(aligned_bundle),
                    "--inference-rules",
                    str(rules_path),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            profile_pack = json.loads(
                (out_dir / "entity_profile_pack.json").read_text(encoding="utf-8")
            )
            victim = next(e for e in profile_pack["entities"] if e["entity_id"] == "person_victim")
            hair_style = next(f for f in victim["fields"] if f["field"] == "hair_style")
            self.assertEqual(hair_style["source_tag"], "INFERRED")
            self.assertEqual(hair_style["value"], "buzz cut")

    def test_runner_fails_on_research_bundle_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l2_l3_profile_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l2.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l2_l3.json"),
                    "--topic-research-bundle",
                    str(ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json"),
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
