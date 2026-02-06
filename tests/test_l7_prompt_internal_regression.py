import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import l7_scene_prompt_map_runner as l7


class L7PromptInternalRegressionTests(unittest.TestCase):
    def test_score_prompt_candidate_blocks_unsafe_token(self) -> None:
        score_spec = json.loads(
            (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8")
        )
        scenes_v2 = json.loads(
            (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
        )
        scene = scenes_v2["scenes"][0]

        candidate = {
            "visual_prompt": {
                "image_prompt": "cinematic frame with dismemberment close-up",
                "video_prompt": "CameraMotion: HANDHELD. Action: reveal evidence. Emotion: shock.",
            }
        }
        total, criteria, hard = l7.score_prompt_candidate(
            candidate=candidate,
            scene=scene,
            style_preset=scenes_v2["style_preset"],
            global_negative_constraints=[],
            score_spec=score_spec,
        )

        self.assertIsInstance(total, float)
        self.assertIn("K3", criteria)
        self.assertFalse(hard["K5"])

    def test_continuity_audit_revises_only_flagged_scene(self) -> None:
        score_spec = json.loads(
            (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8")
        )
        scenes_v2 = json.loads(
            (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
        )
        scene1 = copy.deepcopy(scenes_v2["scenes"][0])
        scene2 = copy.deepcopy(scenes_v2["scenes"][1])

        selected_rows = [
            {
                "scene_id": scene1["scene_id"],
                "candidate": {
                    "candidate_id": f"{scene1['scene_id']}_prompt_001",
                    "strategy": "continuity_focus",
                    "visual_prompt": {
                        "image_prompt": (
                            f"{scenes_v2['style_preset']}, scene frame. "
                            f"Location: {scene1['location_ref']}. Character anchor: {scene1['character_refs'][0]}."
                        ),
                        "video_prompt": "CameraMotion: HANDHELD. Action: reveal. Emotion: shock.",
                    },
                },
                "evaluation": {
                    "total_score": 78.0,
                    "criteria": {"K1": 75.0, "K2": 80.0, "K3": 79.0, "K4": 78.0},
                    "hard_constraints": {"K5": True},
                    "hard_pass": True,
                },
                "pass_threshold_met": True,
                "revised_by_continuity_auditor": False,
                "revision_notes": [],
            },
            {
                "scene_id": scene2["scene_id"],
                "candidate": {
                    "candidate_id": f"{scene2['scene_id']}_prompt_002",
                    "strategy": "impact_focus",
                    "visual_prompt": {
                        "image_prompt": "minimal frame with tension",
                        "video_prompt": "CameraMotion: PAN. Action: move.",
                    },
                },
                "evaluation": {
                    "total_score": 70.0,
                    "criteria": {"K1": 68.0, "K2": 69.0, "K3": 72.0, "K4": 71.0},
                    "hard_constraints": {"K5": True},
                    "hard_pass": True,
                },
                "pass_threshold_met": False,
                "revised_by_continuity_auditor": False,
                "revision_notes": [],
            },
        ]

        audit_report = l7.run_continuity_audit(
            selected_rows=selected_rows,
            scenes_v2=scenes_v2,
            score_spec=score_spec,
        )

        self.assertEqual(audit_report["flagged_scene_ids"], [scene2["scene_id"]])
        self.assertEqual(audit_report["revised_scene_ids"], [scene2["scene_id"]])

        self.assertFalse(selected_rows[0]["revised_by_continuity_auditor"])
        self.assertTrue(selected_rows[1]["revised_by_continuity_auditor"])

        revised_prompt = selected_rows[1]["candidate"]["visual_prompt"]
        self.assertIn(scenes_v2["style_preset"], revised_prompt["image_prompt"])
        self.assertIn(scene2["location_ref"], revised_prompt["image_prompt"])
        self.assertIn(scene2["character_refs"][0], revised_prompt["image_prompt"])
        self.assertIn("continuity", revised_prompt["video_prompt"].lower())

    def test_continuity_audit_recomputes_pass_threshold_and_syncs_reports(self) -> None:
        score_spec = json.loads(
            (ROOT / "score_specs" / "score_spec_prompt.json").read_text(encoding="utf-8")
        )
        scenes_v2 = json.loads(
            (ROOT / "examples" / "l6_run_output" / "scenes_v2.json").read_text(encoding="utf-8")
        )
        scene = copy.deepcopy(scenes_v2["scenes"][1])
        scene_id = scene["scene_id"]
        candidate_id = f"{scene_id}_prompt_002"

        selected_rows = [
            {
                "scene_id": scene_id,
                "candidate": {
                    "candidate_id": candidate_id,
                    "strategy": "impact_focus",
                    "visual_prompt": {
                        "image_prompt": "minimal frame with tension",
                        "video_prompt": "CameraMotion: PAN. Action: move.",
                    },
                },
                "evaluation": {
                    "total_score": 95.0,
                    "criteria": {"K1": 95.0, "K2": 95.0, "K3": 95.0, "K4": 95.0},
                    "hard_constraints": {"K5": True},
                    "hard_pass": True,
                },
                "pass_threshold_met": True,
                "revised_by_continuity_auditor": False,
                "revision_notes": [],
            }
        ]

        evaluation_report = {
            "scene_evaluations": [
                {
                    "scene_id": scene_id,
                    "candidate_evaluations": [
                        {
                            "candidate_id": candidate_id,
                            "strategy": "impact_focus",
                            "total_score": 95.0,
                            "criteria": {"K1": 95.0, "K2": 95.0, "K3": 95.0, "K4": 95.0},
                            "hard_constraints": {"K5": True},
                            "hard_pass": True,
                        }
                    ],
                    "selected_candidate_id": candidate_id,
                    "min_accept_score": score_spec["selection_rule"]["min_accept_score"],
                    "pass_threshold_met": True,
                }
            ],
            "overall": {"scene_count": 1, "pass_scene_count": 1, "failed_scene_count": 0},
        }
        selection_trace = {
            "selected_by_scene": [
                {
                    "scene_id": scene_id,
                    "selected_candidate_id": candidate_id,
                    "selected_score": 95.0,
                    "selection_reason": "Highest weighted score under hard-constraint policy.",
                }
            ]
        }

        l7.run_continuity_audit(
            selected_rows=selected_rows,
            scenes_v2=scenes_v2,
            score_spec=score_spec,
        )
        l7._sync_reports_after_continuity_audit(
            selected_rows=selected_rows,
            evaluation_report=evaluation_report,
            selection_trace=selection_trace,
        )

        min_accept_score = float(score_spec["selection_rule"]["min_accept_score"])
        expected_pass_threshold = (
            selected_rows[0]["evaluation"]["total_score"] >= min_accept_score
            and selected_rows[0]["evaluation"]["hard_pass"]
        )
        self.assertEqual(selected_rows[0]["pass_threshold_met"], expected_pass_threshold)
        self.assertEqual(
            evaluation_report["scene_evaluations"][0]["pass_threshold_met"],
            expected_pass_threshold,
        )
        self.assertEqual(
            selection_trace["selected_by_scene"][0]["selected_score"],
            selected_rows[0]["evaluation"]["total_score"],
        )
        self.assertEqual(
            evaluation_report["overall"]["pass_scene_count"],
            1 if expected_pass_threshold else 0,
        )


if __name__ == "__main__":
    unittest.main()
