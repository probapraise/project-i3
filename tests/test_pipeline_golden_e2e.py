import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "golden" / "pipeline_e2e_snapshot.json"

VOLATILE_KEYS = {
    "created_at",
    "validated_at",
    "manifest_path",
    "input_scenes_path",
}


def _normalize_for_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if key in VOLATILE_KEYS:
                continue
            if key == "path":
                out[key] = "<PATH>"
                continue
            out[key] = _normalize_for_snapshot(value[key])
        return out
    if isinstance(value, list):
        return [_normalize_for_snapshot(item) for item in value]
    return value


def _json_digest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = _normalize_for_snapshot(payload)
    canonical = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cmd(cmd: list[str], stage: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"{stage} failed with exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _run_pipeline_and_collect_snapshot(workdir: Path) -> dict[str, Any]:
    out_l1 = workdir / "l1"
    out_l2 = workdir / "l2"
    out_l4 = workdir / "l4"
    out_l5 = workdir / "l5"
    out_l6 = workdir / "l6"
    out_l7 = workdir / "l7"
    out_l8 = workdir / "l8"

    session_input = ROOT / "examples" / "artifacts" / "sample_session_input.json"
    session_objective = ROOT / "examples" / "artifacts" / "sample_session_objective.json"
    topic_intake = ROOT / "examples" / "artifacts" / "sample_topic_intake_input.json"

    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l1_topic_research_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--topic-intake",
            str(topic_intake),
            "--out-dir",
            str(out_l1),
        ],
        "L1",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l2_l3_profile_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--topic-research-bundle",
            str(out_l1 / "topic_research_bundle.json"),
            "--out-dir",
            str(out_l2),
        ],
        "L2/L3",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l4_hook_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--topic-research-bundle",
            str(out_l1 / "topic_research_bundle.json"),
            "--entity-profile-pack",
            str(out_l2 / "entity_profile_pack.json"),
            "--out-dir",
            str(out_l4),
        ],
        "L4",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l5_script_session_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--topic-research-bundle",
            str(out_l1 / "topic_research_bundle.json"),
            "--entity-profile-pack",
            str(out_l2 / "entity_profile_pack.json"),
            "--script-seed-pack",
            str(out_l4 / "script_seed_pack.json"),
            "--score-spec",
            str(ROOT / "score_specs" / "score_spec_script.json"),
            "--allow-score-spec-override",
            "--override-reason",
            "pipeline golden e2e",
            "--out-dir",
            str(out_l5),
        ],
        "L5",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l6_scene_planning_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--script-session-pack",
            str(out_l5 / "script_session_pack.json"),
            "--score-spec",
            str(ROOT / "score_specs" / "score_spec_scene_split.json"),
            "--allow-score-spec-override",
            "--override-reason",
            "pipeline golden e2e",
            "--out-dir",
            str(out_l6),
        ],
        "L6",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l7_scene_prompt_map_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--scenes-v2",
            str(out_l6 / "scenes_v2.json"),
            "--score-spec",
            str(ROOT / "score_specs" / "score_spec_prompt.json"),
            "--allow-score-spec-override",
            "--override-reason",
            "pipeline golden e2e",
            "--out-dir",
            str(out_l7),
        ],
        "L7",
    )
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "l8_downstream_asset_runner.py"),
            "--session-input",
            str(session_input),
            "--session-objective",
            str(session_objective),
            "--scenes-v2",
            str(out_l7 / "scenes_v2.json"),
            "--prompt-map-pack",
            str(out_l7 / "prompt_map_pack.json"),
            "--out-dir",
            str(out_l8),
        ],
        "L8",
    )

    digested_artifacts = {
        "l1/topic_research_bundle.json": out_l1 / "topic_research_bundle.json",
        "l2/entity_profile_pack.json": out_l2 / "entity_profile_pack.json",
        "l4/script_seed_pack.json": out_l4 / "script_seed_pack.json",
        "l5/script_session_pack.json": out_l5 / "script_session_pack.json",
        "l6/scenes_v2.json": out_l6 / "scenes_v2.json",
        "l7/prompt_map_pack.json": out_l7 / "prompt_map_pack.json",
        "l7/scenes_v2.json": out_l7 / "scenes_v2.json",
        "l8/clip_render_plan.json": out_l8 / "clip_render_plan.json",
        "l8/dubbing_plan.json": out_l8 / "dubbing_plan.json",
        "l8/sound_design_plan.json": out_l8 / "sound_design_plan.json",
        "l8/asset_package_manifest.json": out_l8 / "asset_package_manifest.json",
    }
    artifact_sha256 = {name: _json_digest(path) for name, path in digested_artifacts.items()}

    l1_bundle = _load_json(out_l1 / "topic_research_bundle.json")
    l2_profile = _load_json(out_l2 / "entity_profile_pack.json")
    l4_seed = _load_json(out_l4 / "script_seed_pack.json")
    l5_pack = _load_json(out_l5 / "script_session_pack.json")
    l6_scenes = _load_json(out_l6 / "scenes_v2.json")
    l7_prompt = _load_json(out_l7 / "prompt_map_pack.json")
    l8_manifest = _load_json(out_l8 / "asset_package_manifest.json")
    l8_clip = _load_json(out_l8 / "clip_render_plan.json")

    return {
        "artifact_sha256": artifact_sha256,
        "stage_summary": {
            "l1_research_status": l1_bundle["research_status"],
            "l1_claim_count": len(l1_bundle["claims"]),
            "l2_profile_status": l2_profile["profile_status"],
            "l4_quality_status": l4_seed["quality_status"],
            "l4_selected_hook_id": l4_seed["selected_hook"]["hook_id"],
            "l5_quality_status": l5_pack["quality_status"],
            "l5_selected_script_id": l5_pack["selected_script"]["script_id"],
            "l6_scene_count": len(l6_scenes["scenes"]),
            "l7_quality_status": l7_prompt["quality_status"],
            "l7_selected_prompt_count": len(l7_prompt["selected_prompts"]),
            "l8_quality_status": l8_manifest["quality_status"],
            "l8_clip_count": len(l8_clip["clips"]),
        },
    }


class PipelineGoldenE2ETests(unittest.TestCase):
    def test_l1_to_l8_pipeline_matches_golden_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = _run_pipeline_and_collect_snapshot(Path(tmp))

        if os.environ.get("UPDATE_GOLDEN") == "1":
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            self.skipTest("Golden snapshot updated")

        expected = _load_json(GOLDEN_PATH)
        self.assertEqual(snapshot, expected)


if __name__ == "__main__":
    unittest.main()
