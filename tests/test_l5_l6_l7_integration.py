import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class L5L6L7IntegrationTests(unittest.TestCase):
    def _write_aligned_l5_upstream_artifacts(self, tmpdir: Path) -> tuple[Path, Path, Path]:
        session_input_l5 = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_session_input_l5.json").read_text(
                encoding="utf-8"
            )
        )
        session_id = session_input_l5["session_id"]
        objective_id = session_input_l5["objective_id"]

        topic_bundle = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        topic_bundle["session_id"] = session_id
        topic_bundle["objective_id"] = objective_id
        topic_path = tmpdir / "aligned_topic_research_bundle_l5.json"
        topic_path.write_text(json.dumps(topic_bundle), encoding="utf-8")

        profile_pack = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json").read_text(
                encoding="utf-8"
            )
        )
        profile_pack["session_id"] = session_id
        profile_pack["objective_id"] = objective_id
        profile_path = tmpdir / "aligned_entity_profile_pack_l5.json"
        profile_path.write_text(json.dumps(profile_pack), encoding="utf-8")

        seed_pack = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_script_seed_pack.json").read_text(
                encoding="utf-8"
            )
        )
        seed_pack["session_id"] = session_id
        seed_pack["objective_id"] = objective_id
        seed_path = tmpdir / "aligned_script_seed_pack_l5.json"
        seed_path.write_text(json.dumps(seed_pack), encoding="utf-8")

        return topic_path, profile_path, seed_path

    def test_l5_to_l6_to_l7_handoff_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_l5 = tmpdir / "out_l5"
            out_l6 = tmpdir / "out_l6"
            out_l7 = tmpdir / "out_l7"
            topic_path, profile_path, seed_path = self._write_aligned_l5_upstream_artifacts(
                tmpdir
            )

            l5 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l5_script_session_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l5.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l5_script.json"
                    ),
                    "--topic-research-bundle",
                    str(topic_path),
                    "--entity-profile-pack",
                    str(profile_path),
                    "--script-seed-pack",
                    str(seed_path),
                    "--out-dir",
                    str(out_l5),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l5.returncode, 0, l5.stderr)

            l5_pack = json.loads((out_l5 / "script_session_pack.json").read_text(encoding="utf-8"))
            aligned_l6_input = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_session_input_l6.json").read_text(
                    encoding="utf-8"
                )
            )
            aligned_l6_obj = json.loads(
                (
                    ROOT
                    / "examples"
                    / "artifacts"
                    / "sample_session_objective_l6_scene_plan.json"
                ).read_text(encoding="utf-8")
            )
            aligned_l6_input["session_id"] = l5_pack["session_id"]
            aligned_l6_input["objective_id"] = l5_pack["objective_id"]
            aligned_l6_obj["session_id"] = l5_pack["session_id"]
            aligned_l6_obj["objective_id"] = l5_pack["objective_id"]
            l6_input_path = tmpdir / "aligned_session_input_l6.json"
            l6_obj_path = tmpdir / "aligned_session_objective_l6.json"
            l6_input_path.write_text(json.dumps(aligned_l6_input), encoding="utf-8")
            l6_obj_path.write_text(json.dumps(aligned_l6_obj), encoding="utf-8")

            l6 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l6_scene_planning_runner.py"),
                    "--session-input",
                    str(l6_input_path),
                    "--session-objective",
                    str(l6_obj_path),
                    "--script-session-pack",
                    str(out_l5 / "script_session_pack.json"),
                    "--out-dir",
                    str(out_l6),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l6.returncode, 0, l6.stderr)

            l6_scenes = json.loads((out_l6 / "scenes_v2.json").read_text(encoding="utf-8"))
            aligned_l7_input = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_session_input_l7.json").read_text(
                    encoding="utf-8"
                )
            )
            aligned_l7_obj = json.loads(
                (
                    ROOT
                    / "examples"
                    / "artifacts"
                    / "sample_session_objective_l7_prompt_map.json"
                ).read_text(encoding="utf-8")
            )
            aligned_l7_input["session_id"] = l6_scenes["session_id"]
            aligned_l7_input["objective_id"] = l6_scenes["objective_id"]
            aligned_l7_obj["session_id"] = l6_scenes["session_id"]
            aligned_l7_obj["objective_id"] = l6_scenes["objective_id"]
            l7_input_path = tmpdir / "aligned_session_input_l7.json"
            l7_obj_path = tmpdir / "aligned_session_objective_l7.json"
            l7_input_path.write_text(json.dumps(aligned_l7_input), encoding="utf-8")
            l7_obj_path.write_text(json.dumps(aligned_l7_obj), encoding="utf-8")

            l7 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l7_scene_prompt_map_runner.py"),
                    "--session-input",
                    str(l7_input_path),
                    "--session-objective",
                    str(l7_obj_path),
                    "--scenes-v2",
                    str(out_l6 / "scenes_v2.json"),
                    "--out-dir",
                    str(out_l7),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l7.returncode, 0, l7.stderr)

            gate_out = json.loads((out_l7 / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            prompt_pack = json.loads((out_l7 / "prompt_map_pack.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(prompt_pack["selected_prompts"]), 1)

            scenes_v2 = json.loads((out_l7 / "scenes_v2.json").read_text(encoding="utf-8"))
            for scene in scenes_v2["scenes"]:
                continuity_state = scene.get("continuity_state", {})
                self.assertIn("prompt_candidate_id", continuity_state)
                self.assertIn("prompt_strategy", continuity_state)

    def test_l5_to_l6_to_l7_handoff_recovery_mode_allow_cross_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_l5 = tmpdir / "out_l5"
            out_l6 = tmpdir / "out_l6"
            out_l7 = tmpdir / "out_l7"
            topic_path, profile_path, seed_path = self._write_aligned_l5_upstream_artifacts(
                tmpdir
            )

            l5 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "l5_script_session_runner.py"),
                    "--session-input",
                    str(ROOT / "examples" / "artifacts" / "sample_session_input_l5.json"),
                    "--session-objective",
                    str(
                        ROOT
                        / "examples"
                        / "artifacts"
                        / "sample_session_objective_l5_script.json"
                    ),
                    "--topic-research-bundle",
                    str(topic_path),
                    "--entity-profile-pack",
                    str(profile_path),
                    "--script-seed-pack",
                    str(seed_path),
                    "--out-dir",
                    str(out_l5),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l5.returncode, 0, l5.stderr)

            l6 = subprocess.run(
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
                    str(out_l5 / "script_session_pack.json"),
                    "--allow-cross-session",
                    "--out-dir",
                    str(out_l6),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l6.returncode, 0, l6.stderr)

            l7 = subprocess.run(
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
                    str(out_l6 / "scenes_v2.json"),
                    "--allow-cross-session",
                    "--out-dir",
                    str(out_l7),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(l7.returncode, 0, l7.stderr)


if __name__ == "__main__":
    unittest.main()
