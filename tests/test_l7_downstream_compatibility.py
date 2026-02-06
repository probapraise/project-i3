import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _required_value(payload: dict[str, Any], key: str, path: str) -> Any:
    if key not in payload:
        raise ValueError(f"missing required field: {path}.{key}")
    return payload[key]


def _required_text(payload: dict[str, Any], key: str, path: str) -> str:
    value = _required_value(payload, key, path)
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty required field: {path}.{key}")
    return text


def _required_number(payload: dict[str, Any], key: str, path: str) -> float:
    value = _required_value(payload, key, path)
    if not isinstance(value, (int, float)):
        raise ValueError(f"invalid number field: {path}.{key}")
    return float(value)


def _required_object(payload: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = _required_value(payload, key, path)
    if not isinstance(value, dict):
        raise ValueError(f"invalid object field: {path}.{key}")
    return value


def _required_scenes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = _required_value(payload, "scenes", "scenes_v2")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("invalid required field: scenes_v2.scenes")
    for idx, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"invalid scene object: scenes_v2.scenes[{idx}]")
    return scenes


def _consume_clip_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    clips = []
    for idx, scene in enumerate(_required_scenes(payload)):
        path = f"scenes_v2.scenes[{idx}]"
        visual_prompt = _required_object(scene, "visual_prompt", path)
        clips.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "duration_sec": _required_number(scene, "duration_sec", path),
                "image_prompt": _required_text(
                    visual_prompt,
                    "image_prompt",
                    f"{path}.visual_prompt",
                ),
                "video_prompt": _required_text(
                    visual_prompt,
                    "video_prompt",
                    f"{path}.visual_prompt",
                ),
            }
        )
    return clips


def _consume_dubbing_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dubbing_lines = []
    for idx, scene in enumerate(_required_scenes(payload)):
        path = f"scenes_v2.scenes[{idx}]"
        dubbing_lines.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "narration_text": _required_text(scene, "narration_text", path),
                "duration_sec": _required_number(scene, "duration_sec", path),
            }
        )
    return dubbing_lines


def _consume_sound_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sound_cues = []
    for idx, scene in enumerate(_required_scenes(payload)):
        path = f"scenes_v2.scenes[{idx}]"
        action_emotion = _required_object(scene, "action_emotion", path)
        time_weather = _required_object(scene, "time_weather", path)
        sound_cues.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "action": _required_text(action_emotion, "action", f"{path}.action_emotion"),
                "emotion": _required_text(action_emotion, "emotion", f"{path}.action_emotion"),
                "time_of_day": _required_text(time_weather, "time_of_day", f"{path}.time_weather"),
                "weather": _required_text(time_weather, "weather", f"{path}.time_weather"),
            }
        )
    return sound_cues


def _consume_packaging_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session_id = _required_text(payload, "session_id", "scenes_v2")
    objective_id = _required_text(payload, "objective_id", "scenes_v2")
    style_preset = _required_text(payload, "style_preset", "scenes_v2")
    scene_ids = []
    for idx, scene in enumerate(_required_scenes(payload)):
        path = f"scenes_v2.scenes[{idx}]"
        scene_ids.append(_required_text(scene, "scene_id", path))
    return {
        "package_id": f"{session_id}:{objective_id}",
        "style_preset": style_preset,
        "scene_ids": scene_ids,
    }


class L7DownstreamCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        out_dir = Path(cls._tmp.name) / "out"
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
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr)

        cls.before = json.loads(
            (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
        )
        cls.after = json.loads((out_dir / "scenes_v2.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_l7_output_preserves_legacy_scenes_v2_contract_shape(self) -> None:
        expected_top_keys = {
            "schema_version",
            "session_id",
            "objective_id",
            "created_at",
            "case_id",
            "style_preset",
            "global_negative_constraints",
            "scenes",
        }
        self.assertEqual(set(self.after.keys()), expected_top_keys)

        self.assertEqual(self.before["schema_version"], self.after["schema_version"])
        self.assertEqual(self.before["session_id"], self.after["session_id"])
        self.assertEqual(self.before["objective_id"], self.after["objective_id"])
        self.assertEqual(self.before["case_id"], self.after["case_id"])
        self.assertEqual(self.before["style_preset"], self.after["style_preset"])

        before_scenes = self.before["scenes"]
        after_scenes = self.after["scenes"]
        self.assertEqual(len(before_scenes), len(after_scenes))

        required_scene_keys = {
            "scene_id",
            "beat_id",
            "duration_sec",
            "narration_text",
            "visual_goal",
            "character_refs",
            "location_ref",
            "continuity_state",
            "shot_spec",
            "action_emotion",
            "time_weather",
            "negative_constraints",
            "visual_prompt",
        }

        for prev_scene, next_scene in zip(before_scenes, after_scenes):
            self.assertTrue(required_scene_keys.issubset(set(next_scene.keys())))
            self.assertEqual(prev_scene["scene_id"], next_scene["scene_id"])
            self.assertEqual(prev_scene["beat_id"], next_scene["beat_id"])
            self.assertEqual(prev_scene["duration_sec"], next_scene["duration_sec"])
            self.assertEqual(prev_scene["location_ref"], next_scene["location_ref"])

            prev_state = prev_scene.get("continuity_state", {})
            next_state = next_scene.get("continuity_state", {})
            self.assertEqual(prev_state.get("scene_index"), next_state.get("scene_index"))
            self.assertEqual(prev_state.get("purpose"), next_state.get("purpose"))
            self.assertIn("prompt_candidate_id", next_state)
            self.assertIn("prompt_strategy", next_state)
            self.assertIn("prompt_revised", next_state)

    def test_downstream_compatibility_matrix_for_clip_dubbing_sound_packaging(self) -> None:
        clip_rows = _consume_clip_payload(self.after)
        dubbing_rows = _consume_dubbing_payload(self.after)
        sound_rows = _consume_sound_payload(self.after)
        package_row = _consume_packaging_payload(self.after)

        scene_count = len(self.after["scenes"])
        self.assertEqual(len(clip_rows), scene_count)
        self.assertEqual(len(dubbing_rows), scene_count)
        self.assertEqual(len(sound_rows), scene_count)
        self.assertEqual(len(package_row["scene_ids"]), scene_count)

    def test_downstream_is_tolerant_to_additive_enrichment_fields(self) -> None:
        enriched = copy.deepcopy(self.after)
        enriched["future_packaging_hint"] = "non_blocking_extension"
        for scene in enriched["scenes"]:
            continuity_state = scene.get("continuity_state", {})
            if not isinstance(continuity_state, dict):
                continuity_state = {}
            continuity_state["future_continuity_hint"] = "safe_extension"
            scene["continuity_state"] = continuity_state
            scene["future_scene_hint"] = {"v": 1}

            continuity_state.pop("prompt_candidate_id", None)
            continuity_state.pop("prompt_strategy", None)
            continuity_state.pop("prompt_revised", None)

        self.assertEqual(len(_consume_clip_payload(enriched)), len(self.after["scenes"]))
        self.assertEqual(len(_consume_dubbing_payload(enriched)), len(self.after["scenes"]))
        self.assertEqual(len(_consume_sound_payload(enriched)), len(self.after["scenes"]))
        self.assertEqual(len(_consume_packaging_payload(enriched)["scene_ids"]), len(self.after["scenes"]))

    def test_downstream_fails_fast_only_for_missing_required_legacy_fields(self) -> None:
        clip_missing = copy.deepcopy(self.after)
        clip_missing["scenes"][0]["visual_prompt"].pop("image_prompt", None)
        with self.assertRaisesRegex(ValueError, "image_prompt"):
            _consume_clip_payload(clip_missing)

        dubbing_missing = copy.deepcopy(self.after)
        dubbing_missing["scenes"][0].pop("narration_text", None)
        with self.assertRaisesRegex(ValueError, "narration_text"):
            _consume_dubbing_payload(dubbing_missing)

        sound_missing = copy.deepcopy(self.after)
        sound_missing["scenes"][0]["action_emotion"].pop("emotion", None)
        with self.assertRaisesRegex(ValueError, "emotion"):
            _consume_sound_payload(sound_missing)

        packaging_missing = copy.deepcopy(self.after)
        packaging_missing.pop("session_id", None)
        with self.assertRaisesRegex(ValueError, "session_id"):
            _consume_packaging_payload(packaging_missing)


if __name__ == "__main__":
    unittest.main()
