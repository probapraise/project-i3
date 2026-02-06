import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class L6ScenePlanningRunnerTests(unittest.TestCase):
    def test_runner_completes_and_outputs_scenes_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
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
                "story_bible.json",
                "beats.json",
                "candidates_scene_split.json",
                "evaluation_report_scene_split.json",
                "selection_trace_scene_split.json",
                "scene_skeletons.json",
                "scenes_v2.json",
                "runtime_log_l6.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            scenes_v2 = json.loads((out_dir / "scenes_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(scenes_v2["schema_version"], "2.0.0")
            self.assertGreaterEqual(len(scenes_v2["scenes"]), 1)

    def test_selected_candidate_exists_in_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            selection = json.loads(
                (out_dir / "selection_trace_scene_split.json").read_text(encoding="utf-8")
            )
            selected_id = selection["selected_candidate_id"]
            candidates_doc = json.loads(
                (out_dir / "candidates_scene_split.json").read_text(encoding="utf-8")
            )
            candidate_ids = {row["candidate_id"] for row in candidates_doc["candidates"]}
            self.assertIn(selected_id, candidate_ids)

    def test_hook_scene_meets_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            skeletons = json.loads((out_dir / "scene_skeletons.json").read_text(encoding="utf-8"))
            first_scene = skeletons["scenes"][0]
            self.assertEqual(first_scene["purpose"], "hook")
            self.assertLessEqual(first_scene["estimated_duration_sec"], 3.0)

    def test_runner_fails_on_cross_session_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_session_pack = tmpdir / "cross_script_session_pack.json"
            payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_script_session_pack.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["session_id"] = "sess_other_9999"
            cross_session_pack.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(cross_session_pack),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)

    def test_runner_allows_cross_session_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_session_pack = tmpdir / "cross_script_session_pack.json"
            payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_script_session_pack.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["session_id"] = "sess_other_9999"
            cross_session_pack.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(cross_session_pack),
                    "--allow-cross-session",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

    def test_runner_fails_on_cross_objective_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_objective_pack = tmpdir / "cross_objective_script_session_pack.json"
            payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_script_session_pack.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["objective_id"] = "obj_other_9999"
            cross_objective_pack.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(cross_objective_pack),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("script_session_pack.objective_id must match", proc.stderr)

    def test_runner_fails_on_session_objective_contract_mismatch_even_with_cross_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_session_objective = tmpdir / "bad_session_objective_l6.json"
            payload = json.loads(
                (
                    ROOT
                    / "examples"
                    / "artifacts"
                    / "sample_session_objective_l6_scene_plan.json"
                ).read_text(encoding="utf-8")
            )
            payload["objective_id"] = "obj_mismatch_l6"
            bad_session_objective.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(bad_session_objective),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--allow-cross-session",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("session_input.objective_id must match", proc.stderr)

    def test_runner_fails_on_score_spec_mismatch_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_scene_split_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_scene_split.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--score-spec",
                    str(alt_score_spec),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CLI --score-spec does not match", proc.stderr)

    def test_runner_allows_score_spec_override_in_dev_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_scene_split_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_scene_split.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--score-spec",
                    str(alt_score_spec),
                    "--allow-score-spec-override",
                    "--override-reason",
                    "recovery rerun after upstream failure",
                    "--pipeline-env",
                    "dev",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runtime_log = json.loads((out_dir / "runtime_log_l6.json").read_text(encoding="utf-8"))
            self.assertTrue(runtime_log["policy"]["score_spec_override_used"])
            self.assertEqual(
                runtime_log["policy"]["score_spec_override_reason"],
                "recovery rerun after upstream failure",
            )

    def test_runner_rejects_score_spec_override_in_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_scene_split_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_scene_split.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--score-spec",
                    str(alt_score_spec),
                    "--allow-score-spec-override",
                    "--override-reason",
                    "recovery rerun after upstream failure",
                    "--pipeline-env",
                    "prod",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--pipeline-env=prod", proc.stderr)

    def test_runner_rejects_score_spec_override_without_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_scene_split_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_scene_split.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(ROOT / "examples" / "artifacts" / "sample_script_session_pack.json"),
                    "--score-spec",
                    str(alt_score_spec),
                    "--allow-score-spec-override",
                    "--pipeline-env",
                    "dev",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--override-reason must be non-empty", proc.stderr)

    def test_runner_allows_needs_script_rework_quality_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            script_session_pack = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_script_session_pack.json").read_text(
                    encoding="utf-8"
                )
            )
            script_session_pack["quality_status"] = "NEEDS_SCRIPT_REWORK"
            custom_script_session = tmpdir / "script_session_pack_needs_rework.json"
            custom_script_session.write_text(
                json.dumps(script_session_pack),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(custom_script_session),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runtime_log = json.loads((out_dir / "runtime_log_l6.json").read_text(encoding="utf-8"))
            self.assertEqual(
                runtime_log["policy"]["upstream_quality_status"],
                "NEEDS_SCRIPT_REWORK",
            )
            self.assertTrue(runtime_log["policy"]["manual_rework_recommended"])

    def test_runner_fails_on_invalid_gate_in_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_script_session = tmpdir / "bad_script_session_pack.json"
            bad_script_session.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "session_id": "sess_l6_0001",
                        "objective_id": "obj_l6_scene_plan",
                        "created_at": "2026-02-06T14:00:00Z",
                        "quality_status": "READY_FOR_SCENE_PLANNING"
                    }
                ),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l6.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l6_scene_plan.json"
                    ),
                    "--script-session-pack",
                    str(bad_script_session),
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


if __name__ == "__main__":
    unittest.main()
