#!/usr/bin/env python3
"""L8 downstream asset runner (clip/dubbing/sound/package integration)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, write_gate_manifest, write_json


class ContractError(Exception):
    """Raised when input contract preconditions are violated."""


def normalize_text(value: Any) -> str:
    return str(value).strip()


def _required_text(payload: dict[str, Any], key: str, path: str) -> str:
    if key not in payload:
        raise ContractError(f"missing required field: {path}.{key}")
    text = normalize_text(payload[key])
    if not text:
        raise ContractError(f"empty required field: {path}.{key}")
    return text


def _required_object(payload: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in payload:
        raise ContractError(f"missing required field: {path}.{key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ContractError(f"invalid object field: {path}.{key}")
    return value


def _required_number(payload: dict[str, Any], key: str, path: str) -> float:
    if key not in payload:
        raise ContractError(f"missing required field: {path}.{key}")
    value = payload[key]
    if not isinstance(value, (int, float)):
        raise ContractError(f"invalid number field: {path}.{key}")
    return float(value)


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    scenes_v2: dict[str, Any],
    prompt_map_pack: dict[str, Any],
    allow_cross_session: bool,
) -> None:
    input_session_id = _required_text(session_input, "session_id", "session_input")
    input_objective_id = _required_text(session_input, "objective_id", "session_input")
    objective_session_id = _required_text(
        session_objective,
        "session_id",
        "session_objective",
    )
    objective_objective_id = _required_text(
        session_objective,
        "objective_id",
        "session_objective",
    )

    if input_session_id != objective_session_id:
        raise ContractError("session_input.session_id must match session_objective.session_id")
    if input_objective_id != objective_objective_id:
        raise ContractError("session_input.objective_id must match session_objective.objective_id")

    if allow_cross_session:
        return

    scenes_session_id = _required_text(scenes_v2, "session_id", "scenes_v2")
    scenes_objective_id = _required_text(scenes_v2, "objective_id", "scenes_v2")
    prompt_session_id = _required_text(prompt_map_pack, "session_id", "prompt_map_pack")
    prompt_objective_id = _required_text(prompt_map_pack, "objective_id", "prompt_map_pack")

    if scenes_session_id != input_session_id:
        raise ContractError(
            "scenes_v2.session_id must match session_input.session_id "
            "(use --allow-cross-session for recovery reruns)"
        )
    if scenes_objective_id != input_objective_id:
        raise ContractError(
            "scenes_v2.objective_id must match session_input.objective_id "
            "(use --allow-cross-session for recovery reruns)"
        )
    if prompt_session_id != input_session_id:
        raise ContractError(
            "prompt_map_pack.session_id must match session_input.session_id "
            "(use --allow-cross-session for recovery reruns)"
        )
    if prompt_objective_id != input_objective_id:
        raise ContractError(
            "prompt_map_pack.objective_id must match session_input.objective_id "
            "(use --allow-cross-session for recovery reruns)"
        )


def _index_selected_prompts(prompt_map_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected_prompts = prompt_map_pack.get("selected_prompts", [])
    if not isinstance(selected_prompts, list) or not selected_prompts:
        raise ContractError("prompt_map_pack.selected_prompts must be non-empty")

    selected_by_scene: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(selected_prompts):
        path = f"prompt_map_pack.selected_prompts[{idx}]"
        if not isinstance(row, dict):
            raise ContractError(f"invalid object: {path}")
        scene_id = _required_text(row, "scene_id", path)
        if scene_id in selected_by_scene:
            raise ContractError(f"duplicate selected prompt for scene_id: {scene_id}")
        visual_prompt = _required_object(row, "visual_prompt", path)
        _required_text(visual_prompt, "image_prompt", f"{path}.visual_prompt")
        _required_text(visual_prompt, "video_prompt", f"{path}.visual_prompt")
        selected_by_scene[scene_id] = row
    return selected_by_scene


def _load_validated_scenes(scenes_v2: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = scenes_v2.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("scenes_v2.scenes must be non-empty")

    seen_scene_ids: set[str] = set()
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        if not isinstance(scene, dict):
            raise ContractError(f"invalid object: {path}")
        scene_id = _required_text(scene, "scene_id", path)
        if scene_id in seen_scene_ids:
            raise ContractError(f"duplicate scene_id in scenes_v2: {scene_id}")
        seen_scene_ids.add(scene_id)
        _required_number(scene, "duration_sec", path)
        _required_text(scene, "narration_text", path)
        _required_object(scene, "action_emotion", path)
        _required_object(scene, "time_weather", path)
        visual_prompt = _required_object(scene, "visual_prompt", path)
        _required_text(visual_prompt, "image_prompt", f"{path}.visual_prompt")
        _required_text(visual_prompt, "video_prompt", f"{path}.visual_prompt")
    return scenes


def _assert_prompt_scene_compatibility(
    *,
    scenes_v2: dict[str, Any],
    prompt_map_pack: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    style_preset = _required_text(scenes_v2, "style_preset", "scenes_v2")
    prompt_style_preset = _required_text(prompt_map_pack, "style_preset", "prompt_map_pack")
    if prompt_style_preset != style_preset:
        raise ContractError("prompt_map_pack.style_preset must match scenes_v2.style_preset")

    scenes = _load_validated_scenes(scenes_v2)
    selected_by_scene = _index_selected_prompts(prompt_map_pack)
    scene_id_set = {str(scene["scene_id"]) for scene in scenes}
    selected_id_set = set(selected_by_scene.keys())

    missing_selected = sorted(scene_id_set.difference(selected_id_set))
    if missing_selected:
        raise ContractError(
            "missing selected prompt for scene_id(s): "
            f"{missing_selected}"
        )
    unknown_selected = sorted(selected_id_set.difference(scene_id_set))
    if unknown_selected:
        raise ContractError(
            "selected prompts contain unknown scene_id(s): "
            f"{unknown_selected}"
        )

    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        scene_id = _required_text(scene, "scene_id", path)
        scene_visual_prompt = _required_object(scene, "visual_prompt", path)
        scene_image_prompt = _required_text(scene_visual_prompt, "image_prompt", f"{path}.visual_prompt")
        scene_video_prompt = _required_text(scene_visual_prompt, "video_prompt", f"{path}.visual_prompt")

        selected_row = selected_by_scene[scene_id]
        selected_path = f"prompt_map_pack.selected_prompts[scene_id={scene_id}]"
        selected_visual_prompt = _required_object(selected_row, "visual_prompt", selected_path)
        selected_image_prompt = _required_text(
            selected_visual_prompt,
            "image_prompt",
            f"{selected_path}.visual_prompt",
        )
        selected_video_prompt = _required_text(
            selected_visual_prompt,
            "video_prompt",
            f"{selected_path}.visual_prompt",
        )

        image_mismatch = normalize_text(scene_image_prompt) != normalize_text(selected_image_prompt)
        video_mismatch = normalize_text(scene_video_prompt) != normalize_text(selected_video_prompt)
        if image_mismatch or video_mismatch:
            raise ContractError(
                "scenes_v2.visual_prompt must match prompt_map_pack.selected_prompts.visual_prompt "
                f"for scene_id: {scene_id}"
            )
    return selected_by_scene


def build_clip_render_plan(
    *,
    scenes_v2: dict[str, Any],
) -> dict[str, Any]:
    clips = []
    scenes = _load_validated_scenes(scenes_v2)
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        visual_prompt = _required_object(scene, "visual_prompt", path)
        clips.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "beat_id": _required_text(scene, "beat_id", path),
                "duration_sec": round(_required_number(scene, "duration_sec", path), 2),
                "image_prompt": _required_text(visual_prompt, "image_prompt", f"{path}.visual_prompt"),
                "video_prompt": _required_text(visual_prompt, "video_prompt", f"{path}.visual_prompt"),
            }
        )
    return {
        "schema_version": "1.0.0",
        "session_id": _required_text(scenes_v2, "session_id", "scenes_v2"),
        "objective_id": _required_text(scenes_v2, "objective_id", "scenes_v2"),
        "created_at": now_utc_iso(),
        "style_preset": _required_text(scenes_v2, "style_preset", "scenes_v2"),
        "clips": clips,
    }


def build_dubbing_plan(
    *,
    scenes_v2: dict[str, Any],
) -> dict[str, Any]:
    lines = []
    scenes = _load_validated_scenes(scenes_v2)
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        lines.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "narration_text": _required_text(scene, "narration_text", path),
                "duration_sec": round(_required_number(scene, "duration_sec", path), 2),
            }
        )
    return {
        "schema_version": "1.0.0",
        "session_id": _required_text(scenes_v2, "session_id", "scenes_v2"),
        "objective_id": _required_text(scenes_v2, "objective_id", "scenes_v2"),
        "created_at": now_utc_iso(),
        "style_preset": _required_text(scenes_v2, "style_preset", "scenes_v2"),
        "lines": lines,
    }


def build_sound_design_plan(
    *,
    scenes_v2: dict[str, Any],
) -> dict[str, Any]:
    cues = []
    scenes = _load_validated_scenes(scenes_v2)
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        action_emotion = _required_object(scene, "action_emotion", path)
        time_weather = _required_object(scene, "time_weather", path)
        cues.append(
            {
                "scene_id": _required_text(scene, "scene_id", path),
                "action": _required_text(action_emotion, "action", f"{path}.action_emotion"),
                "emotion": _required_text(action_emotion, "emotion", f"{path}.action_emotion"),
                "time_of_day": _required_text(time_weather, "time_of_day", f"{path}.time_weather"),
                "weather": _required_text(time_weather, "weather", f"{path}.time_weather"),
            }
        )
    return {
        "schema_version": "1.0.0",
        "session_id": _required_text(scenes_v2, "session_id", "scenes_v2"),
        "objective_id": _required_text(scenes_v2, "objective_id", "scenes_v2"),
        "created_at": now_utc_iso(),
        "style_preset": _required_text(scenes_v2, "style_preset", "scenes_v2"),
        "cues": cues,
    }


def build_asset_package_manifest(
    *,
    scenes_v2: dict[str, Any],
    prompt_map_pack: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    scenes = _load_validated_scenes(scenes_v2)
    quality_status = _required_text(prompt_map_pack, "quality_status", "prompt_map_pack")
    return {
        "schema_version": "1.0.0",
        "session_id": _required_text(scenes_v2, "session_id", "scenes_v2"),
        "objective_id": _required_text(scenes_v2, "objective_id", "scenes_v2"),
        "created_at": now_utc_iso(),
        "style_preset": _required_text(scenes_v2, "style_preset", "scenes_v2"),
        "quality_status": quality_status,
        "scene_ids": [str(scene["scene_id"]) for scene in scenes],
        "artifacts": [
            {
                "name": "clip_render_plan",
                "path": str((out_dir / "clip_render_plan.json").resolve()),
            },
            {
                "name": "dubbing_plan",
                "path": str((out_dir / "dubbing_plan.json").resolve()),
            },
            {
                "name": "sound_design_plan",
                "path": str((out_dir / "sound_design_plan.json").resolve()),
            },
        ],
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.perf_counter()

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    scenes_v2_path = Path(args.scenes_v2).resolve()
    prompt_map_pack_path = Path(args.prompt_map_pack).resolve()

    gate_in_started_at = time.perf_counter()
    gate_in_manifest = out_dir / "gate_in_manifest.runtime.json"
    write_gate_manifest(
        manifest_path=gate_in_manifest,
        gate="in",
        artifacts=[
            ("session_input", session_input_path, root / "schemas" / "session_input.schema.json"),
            (
                "session_objective",
                session_objective_path,
                root / "schemas" / "session_objective.schema.json",
            ),
            ("scenes_v2", scenes_v2_path, root / "schemas" / "scenes_v2.schema.json"),
            ("prompt_map_pack", prompt_map_pack_path, root / "schemas" / "prompt_map_pack.schema.json"),
        ],
    )
    gate_in_report = run_gate_validation(manifest_path=gate_in_manifest, requested_gate="in")
    gate_in_duration_ms = round((time.perf_counter() - gate_in_started_at) * 1000.0, 2)
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l8-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    scenes_v2 = read_json(scenes_v2_path)
    prompt_map_pack = read_json(prompt_map_pack_path)

    _assert_session_alignment(
        session_input=session_input,
        session_objective=session_objective,
        scenes_v2=scenes_v2,
        prompt_map_pack=prompt_map_pack,
        allow_cross_session=bool(args.allow_cross_session),
    )
    selected_by_scene = _assert_prompt_scene_compatibility(
        scenes_v2=scenes_v2,
        prompt_map_pack=prompt_map_pack,
    )

    build_started_at = time.perf_counter()
    clip_render_plan = build_clip_render_plan(scenes_v2=scenes_v2)
    dubbing_plan = build_dubbing_plan(scenes_v2=scenes_v2)
    sound_design_plan = build_sound_design_plan(scenes_v2=scenes_v2)
    asset_package_manifest = build_asset_package_manifest(
        scenes_v2=scenes_v2,
        prompt_map_pack=prompt_map_pack,
        out_dir=out_dir,
    )
    build_duration_ms = round((time.perf_counter() - build_started_at) * 1000.0, 2)

    write_json(out_dir / "clip_render_plan.json", clip_render_plan)
    write_json(out_dir / "dubbing_plan.json", dubbing_plan)
    write_json(out_dir / "sound_design_plan.json", sound_design_plan)
    write_json(out_dir / "asset_package_manifest.json", asset_package_manifest)

    scenes = scenes_v2.get("scenes", [])
    scene_count = len(scenes) if isinstance(scenes, list) else 0
    runtime_log: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "pipeline_stage": "L8 Downstream Asset Integration",
        "inputs": {
            "session_input": str(session_input_path),
            "session_objective": str(session_objective_path),
            "scenes_v2": str(scenes_v2_path),
            "prompt_map_pack": str(prompt_map_pack_path),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "scene_count": scene_count,
            "clip_count": len(clip_render_plan["clips"]),
            "dubbing_line_count": len(dubbing_plan["lines"]),
            "sound_cue_count": len(sound_design_plan["cues"]),
        },
        "policy": {
            "allow_cross_session": bool(args.allow_cross_session),
            "quality_status": asset_package_manifest["quality_status"],
            "failure_mode": "fail_fast_on_missing_required_legacy_fields",
            "additive_enrichment_mode": "tolerant_within_schema",
        },
        "integrity": {
            "scene_ids_with_selected_prompts": len(selected_by_scene),
        },
        "timings_ms": {
            "gate_in_validation": gate_in_duration_ms,
            "downstream_plan_build": build_duration_ms,
        },
    }
    write_json(out_dir / "runtime_log_l8.json", runtime_log)

    session_output = {
        "schema_version": "1.0.0",
        "session_id": _required_text(session_input, "session_id", "session_input"),
        "objective_id": _required_text(session_input, "objective_id", "session_input"),
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "clip_render_plan",
                "path": str((out_dir / "clip_render_plan.json").resolve()),
            },
            {
                "name": "dubbing_plan",
                "path": str((out_dir / "dubbing_plan.json").resolve()),
            },
            {
                "name": "sound_design_plan",
                "path": str((out_dir / "sound_design_plan.json").resolve()),
            },
            {
                "name": "asset_package_manifest",
                "path": str((out_dir / "asset_package_manifest.json").resolve()),
            },
            {
                "name": "runtime_log_l8",
                "path": str((out_dir / "runtime_log_l8.json").resolve()),
            },
        ],
        "metrics": {
            "scene_count": scene_count,
            "clip_count": len(clip_render_plan["clips"]),
            "dubbing_line_count": len(dubbing_plan["lines"]),
            "sound_cue_count": len(sound_design_plan["cues"]),
        },
    }
    write_json(out_dir / "session_output.json", session_output)

    gate_out_started_at = time.perf_counter()
    gate_out_manifest = out_dir / "gate_out_manifest.runtime.json"
    write_gate_manifest(
        manifest_path=gate_out_manifest,
        gate="out",
        artifacts=[
            (
                "session_output",
                out_dir / "session_output.json",
                root / "schemas" / "session_output.schema.json",
            ),
            (
                "clip_render_plan",
                out_dir / "clip_render_plan.json",
                root / "schemas" / "clip_render_plan.schema.json",
            ),
            (
                "dubbing_plan",
                out_dir / "dubbing_plan.json",
                root / "schemas" / "dubbing_plan.schema.json",
            ),
            (
                "sound_design_plan",
                out_dir / "sound_design_plan.json",
                root / "schemas" / "sound_design_plan.schema.json",
            ),
            (
                "asset_package_manifest",
                out_dir / "asset_package_manifest.json",
                root / "schemas" / "asset_package_manifest.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(manifest_path=gate_out_manifest, requested_gate="out")
    gate_out_duration_ms = round((time.perf_counter() - gate_out_started_at) * 1000.0, 2)
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l8-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    total_duration_ms = round((time.perf_counter() - run_started_at) * 1000.0, 2)
    runtime_log["timings_ms"]["gate_out_validation"] = gate_out_duration_ms
    runtime_log["timings_ms"]["total_run"] = total_duration_ms
    write_json(out_dir / "runtime_log_l8.json", runtime_log)

    print(f"[l8-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run downstream clip/dubbing/sound/package integration checks and exports."
    )
    parser.add_argument("--session-input", required=True, help="Path to session_input.json")
    parser.add_argument(
        "--session-objective",
        required=True,
        help="Path to session_objective.json",
    )
    parser.add_argument(
        "--scenes-v2",
        required=True,
        help="Path to prompt-enriched scenes_v2.json",
    )
    parser.add_argument(
        "--prompt-map-pack",
        required=True,
        help="Path to prompt_map_pack.json",
    )
    parser.add_argument(
        "--allow-cross-session",
        action="store_true",
        help=(
            "Allow scenes_v2/prompt_map_pack session-objective mismatch for recovery reruns. "
            "Default is strict match."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ContractError as err:
        print(f"[l8-runner] contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l8-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
