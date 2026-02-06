import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class L7ScenePromptMapRunnerTests(unittest.TestCase):
    def test_runner_completes_and_outputs_prompt_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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
                "candidates_prompt.json",
                "evaluation_report_prompt.json",
                "selection_trace_prompt.json",
                "continuity_audit_prompt.json",
                "prompt_map_pack.json",
                "scenes_v2.json",
                "runtime_log_l7.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            prompt_pack = json.loads((out_dir / "prompt_map_pack.json").read_text(encoding="utf-8"))
            self.assertIn(
                prompt_pack["quality_status"],
                {"READY_FOR_ASSET_PIPELINE", "NEEDS_PROMPT_REVIEW"},
            )
            self.assertGreaterEqual(len(prompt_pack["selected_prompts"]), 1)

    def test_selected_prompt_exists_in_candidates_by_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            prompt_pack = json.loads((out_dir / "prompt_map_pack.json").read_text(encoding="utf-8"))
            candidates_doc = json.loads((out_dir / "candidates_prompt.json").read_text(encoding="utf-8"))
            candidates_by_scene = {
                row["scene_id"]: {cand["candidate_id"] for cand in row["candidates"]}
                for row in candidates_doc["scenes"]
            }

            for selected in prompt_pack["selected_prompts"]:
                scene_id = selected["scene_id"]
                candidate_id = selected["candidate_id"]
                self.assertIn(candidate_id, candidates_by_scene[scene_id])

    def test_partial_rerun_requires_base_prompt_map_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                    "--scene-id",
                    "scene_001",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--base-prompt-map-pack is required", proc.stderr)

    def test_partial_rerun_updates_subset_and_preserves_other_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_out_dir = tmpdir / "base_out"
            partial_out_dir = tmpdir / "partial_out"

            full_run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                    "--out-dir",
                    str(base_out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(full_run.returncode, 0, full_run.stderr)

            base_prompt_pack = json.loads(
                (base_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            selected_prompts = base_prompt_pack["selected_prompts"]
            self.assertGreaterEqual(len(selected_prompts), 2)
            target_scene_id = selected_prompts[0]["scene_id"]

            partial_run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(base_out_dir / "scenes_v2.json"),
                    "--scene-id",
                    target_scene_id,
                    "--base-prompt-map-pack",
                    str(base_out_dir / "prompt_map_pack.json"),
                    "--out-dir",
                    str(partial_out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(partial_run.returncode, 0, partial_run.stderr)

            partial_prompt_pack = json.loads(
                (partial_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            partial_scenes_v2 = json.loads(
                (partial_out_dir / "scenes_v2.json").read_text(encoding="utf-8")
            )
            partial_by_scene = {
                row["scene_id"]: row for row in partial_prompt_pack["selected_prompts"]
            }
            base_by_scene = {row["scene_id"]: row for row in base_prompt_pack["selected_prompts"]}
            partial_scene_prompt_by_scene = {
                row["scene_id"]: row["visual_prompt"] for row in partial_scenes_v2["scenes"]
            }
            self.assertEqual(len(partial_by_scene), len(base_by_scene))

            for scene_id, base_row in base_by_scene.items():
                if scene_id == target_scene_id:
                    continue
                self.assertEqual(partial_by_scene[scene_id], base_row)

            for scene_id, selected_row in partial_by_scene.items():
                self.assertIn(scene_id, partial_scene_prompt_by_scene)
                self.assertEqual(
                    partial_scene_prompt_by_scene[scene_id]["image_prompt"],
                    selected_row["visual_prompt"]["image_prompt"],
                )
                self.assertEqual(
                    partial_scene_prompt_by_scene[scene_id]["video_prompt"],
                    selected_row["visual_prompt"]["video_prompt"],
                )

            candidates_doc = json.loads(
                (partial_out_dir / "candidates_prompt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(candidates_doc["scenes"]), 1)
            self.assertEqual(candidates_doc["scenes"][0]["scene_id"], target_scene_id)

            runtime_log = json.loads(
                (partial_out_dir / "runtime_log_l7.json").read_text(encoding="utf-8")
            )
            self.assertTrue(runtime_log["policy"]["scene_filter_applied"])
            self.assertEqual(runtime_log["policy"]["scene_filter"], [target_scene_id])

            session_output = json.loads(
                (partial_out_dir / "session_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(session_output["metrics"]["processed_scene_count"], 1)

    def test_partial_rerun_rejects_unknown_scene_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_out_dir = tmpdir / "base_out"
            partial_out_dir = tmpdir / "partial_out"

            full_run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                    "--out-dir",
                    str(base_out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(full_run.returncode, 0, full_run.stderr)

            partial_run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(base_out_dir / "scenes_v2.json"),
                    "--scene-id",
                    "scene_999",
                    "--base-prompt-map-pack",
                    str(base_out_dir / "prompt_map_pack.json"),
                    "--out-dir",
                    str(partial_out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(partial_run.returncode, 2)
            self.assertIn("requested --scene-id not found", partial_run.stderr)

    def test_runner_fails_on_cross_session_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_scenes = tmpdir / "cross_scenes_v2.json"
            payload = json.loads(
                (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
            )
            payload["session_id"] = "sess_other_9999"
            cross_scenes.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(cross_scenes),
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
            cross_scenes = tmpdir / "cross_scenes_v2.json"
            payload = json.loads(
                (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
            )
            payload["session_id"] = "sess_other_9999"
            cross_scenes.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(cross_scenes),
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
            cross_scenes = tmpdir / "cross_objective_scenes_v2.json"
            payload = json.loads(
                (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
            )
            payload["objective_id"] = "obj_other_9999"
            cross_scenes.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(cross_scenes),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("scenes_v2.objective_id must match", proc.stderr)

    def test_runner_fails_on_session_objective_contract_mismatch_even_with_cross_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_session_objective = tmpdir / "bad_session_objective_l7.json"
            payload = json.loads(
                (
                    ROOT
                    / "examples"
                    / "artifacts"
                    / "sample_session_objective_l7_prompt_map.json"
                ).read_text(encoding="utf-8")
            )
            payload["objective_id"] = "obj_mismatch_l7"
            bad_session_objective.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(bad_session_objective),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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
            alt_score_spec = tmpdir / "score_spec_prompt_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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
            alt_score_spec = tmpdir / "score_spec_prompt_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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
            runtime_log = json.loads((out_dir / "runtime_log_l7.json").read_text(encoding="utf-8"))
            self.assertTrue(runtime_log["policy"]["score_spec_override_used"])
            self.assertEqual(
                runtime_log["policy"]["score_spec_override_reason"],
                "recovery rerun after upstream failure",
            )

    def test_runner_rejects_score_spec_override_in_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_prompt_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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
            alt_score_spec = tmpdir / "score_spec_prompt_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
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

    def test_runner_fails_on_invalid_gate_in_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_scenes = tmpdir / "bad_scenes_v2.json"
            bad_scenes.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0.0",
                        "session_id": "sess_l6_0001",
                        "objective_id": "obj_l6_scene_plan",
                        "created_at": "2026-02-06T04:46:04Z"
                    }
                ),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(bad_scenes),
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

    def test_runner_fails_on_contract_error_for_blank_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            blank_style = tmpdir / "blank_style_scenes_v2.json"
            payload = json.loads(
                (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
            )
            payload["style_preset"] = "   "
            blank_style.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l7_prompt_map.json"
                    ),
                    "--scenes-v2",
                    str(blank_style),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
