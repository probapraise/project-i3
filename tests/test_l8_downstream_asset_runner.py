import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class L8DownstreamAssetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls._tmpdir = Path(cls._tmp.name)
        cls._l7_out_dir = cls._tmpdir / "l7_out"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "l7_scene_prompt_map_runner.py"),
                "--session-input",
                str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                "--session-objective",
                str(ROOT / "examples" / "artifacts" / "sample_session_objective_l7_prompt_map.json"),
                "--scenes-v2",
                str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                "--out-dir",
                str(cls._l7_out_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_runner_completes_and_exports_downstream_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
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
                    str(self._l7_out_dir / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(self._l7_out_dir / "prompt_map_pack.json"),
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
                "clip_render_plan.json",
                "dubbing_plan.json",
                "sound_design_plan.json",
                "asset_package_manifest.json",
                "runtime_log_l8.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            clip_plan = json.loads((out_dir / "clip_render_plan.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(clip_plan["clips"]), 1)

            package_manifest = json.loads(
                (out_dir / "asset_package_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                package_manifest["quality_status"],
                {"READY_FOR_ASSET_PIPELINE", "NEEDS_PROMPT_REVIEW"},
            )

    def test_runner_accepts_partial_rerun_outputs_from_l7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_l7_out = tmpdir / "base_l7_out"
            partial_l7_out = tmpdir / "partial_l7_out"
            l8_out = tmpdir / "l8_out"

            full_l7 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l7_prompt_map.json"),
                    "--scenes-v2",
                    str(ROOT / "examples" / "l6_run_output" / "scenes_v2.json"),
                    "--out-dir",
                    str(base_l7_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(full_l7.returncode, 0, full_l7.stderr)

            base_prompt_pack = json.loads((base_l7_out / "prompt_map_pack.json").read_text(encoding="utf-8"))
            target_scene_id = base_prompt_pack["selected_prompts"][0]["scene_id"]

            partial_l7 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l7_prompt_map.json"),
                    "--scenes-v2",
                    str(base_l7_out / "scenes_v2.json"),
                    "--scene-id",
                    target_scene_id,
                    "--base-prompt-map-pack",
                    str(base_l7_out / "prompt_map_pack.json"),
                    "--out-dir",
                    str(partial_l7_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(partial_l7.returncode, 0, partial_l7.stderr)

            l8_proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l7.json"),
                    "--session-objective",
                    str(ROOT / "examples" / "artifacts" / "sample_session_objective_l7_prompt_map.json"),
                    "--scenes-v2",
                    str(partial_l7_out / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(partial_l7_out / "prompt_map_pack.json"),
                    "--out-dir",
                    str(l8_out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l8_proc.returncode, 0, l8_proc.stderr)

            prompt_pack = json.loads((partial_l7_out / "prompt_map_pack.json").read_text(encoding="utf-8"))
            clip_plan = json.loads((l8_out / "clip_render_plan.json").read_text(encoding="utf-8"))

            selected_by_scene = {row["scene_id"]: row for row in prompt_pack["selected_prompts"]}
            clip_by_scene = {row["scene_id"]: row for row in clip_plan["clips"]}
            self.assertEqual(set(clip_by_scene.keys()), set(selected_by_scene.keys()))
            for scene_id, selected in selected_by_scene.items():
                self.assertEqual(
                    clip_by_scene[scene_id]["image_prompt"],
                    selected["visual_prompt"]["image_prompt"],
                )
                self.assertEqual(
                    clip_by_scene[scene_id]["video_prompt"],
                    selected["visual_prompt"]["video_prompt"],
                )

    def test_runner_fails_when_prompt_map_missing_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_prompt_map = tmpdir / "bad_prompt_map_pack.json"
            payload = json.loads(
                (self._l7_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            payload["selected_prompts"] = payload["selected_prompts"][1:]
            bad_prompt_map.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
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
                    str(self._l7_out_dir / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(bad_prompt_map),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("missing selected prompt", proc.stderr)

    def test_runner_fails_when_prompt_map_visual_prompt_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            mismatched_prompt_map = tmpdir / "mismatched_prompt_map_pack.json"
            payload = json.loads(
                (self._l7_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            payload["selected_prompts"][0]["visual_prompt"]["image_prompt"] = "mismatched prompt image"
            mismatched_prompt_map.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
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
                    str(self._l7_out_dir / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(mismatched_prompt_map),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("visual_prompt must match", proc.stderr)

    def test_runner_fails_on_cross_session_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_prompt_map = tmpdir / "cross_prompt_map_pack.json"
            payload = json.loads(
                (self._l7_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            payload["session_id"] = "sess_other_9999"
            cross_prompt_map.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
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
                    str(self._l7_out_dir / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(cross_prompt_map),
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("prompt_map_pack.session_id", proc.stderr)

    def test_runner_allows_cross_session_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cross_prompt_map = tmpdir / "cross_prompt_map_pack.json"
            payload = json.loads(
                (self._l7_out_dir / "prompt_map_pack.json").read_text(encoding="utf-8")
            )
            payload["session_id"] = "sess_other_9999"
            cross_prompt_map.write_text(json.dumps(payload), encoding="utf-8")

            out_dir = tmpdir / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l8_downstream_asset_runner.py"),
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
                    str(self._l7_out_dir / "scenes_v2.json"),
                    "--prompt-map-pack",
                    str(cross_prompt_map),
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


if __name__ == "__main__":
    unittest.main()
