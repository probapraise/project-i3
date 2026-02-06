#!/usr/bin/env python3
"""L7 Scene Prompt Map runner (per-scene prompt map + continuity reduce)."""

from __future__ import annotations

import argparse
import copy
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, read_yaml, write_gate_manifest, write_json


UNSAFE_PROMPT_TOKENS = [
    "dismemberment",
    "torture porn",
    "snuff",
    "beheading",
    "graphic gore",
    "graphic injury",
]


@dataclass
class AgentRuntimeInfo:
    agent_id: str
    provider: str
    model: str
    tools: list[str]
    purpose: str


class ConfigError(Exception):
    """Raised when model map config is invalid."""


class ContractError(Exception):
    """Raised when input contract preconditions are violated."""


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("single_model_per_agent") is not True:
        raise ConfigError("policy.single_model_per_agent must be true")
    if str(policy.get("model_update_mode", "")) != "manual_only":
        raise ConfigError("policy.model_update_mode must be manual_only")
    if policy.get("automatic_fallback") not in {False, None}:
        raise ConfigError("policy.automatic_fallback must be false")


def load_agent_runtime_infos(model_map_path: Path) -> dict[str, AgentRuntimeInfo]:
    payload = read_yaml(model_map_path)
    if not isinstance(payload, dict):
        raise ConfigError("agent_model_map root must be an object")

    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ConfigError("agent_model_map.policy must be an object")
    _validate_policy(policy)

    manual_change_log = payload.get("manual_change_log", [])
    if not isinstance(manual_change_log, list):
        raise ConfigError("agent_model_map.manual_change_log must be an array")

    agents = payload.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise ConfigError("agent_model_map.agents must be a non-empty object")

    infos: dict[str, AgentRuntimeInfo] = {}
    for agent_id, item in agents.items():
        if not isinstance(item, dict):
            raise ConfigError(f"agent config must be object: {agent_id}")
        provider = item.get("provider")
        model = item.get("model")
        tools = item.get("tools", [])
        purpose = item.get("purpose", "")
        if not isinstance(provider, str) or not provider:
            raise ConfigError(f"{agent_id}.provider must be non-empty string")
        if not isinstance(model, str) or not model:
            raise ConfigError(f"{agent_id}.model must be non-empty string")
        if not isinstance(tools, list) or any(not isinstance(x, str) for x in tools):
            raise ConfigError(f"{agent_id}.tools must be a string array")
        if not isinstance(purpose, str):
            raise ConfigError(f"{agent_id}.purpose must be string")
        infos[agent_id] = AgentRuntimeInfo(
            agent_id=agent_id,
            provider=provider,
            model=model,
            tools=tools,
            purpose=purpose,
        )

    required_agents = {
        "SceneContextPackager",
        "SceneBriefAgent",
        "VisualPromptAgent",
        "PromptValidator",
        "PromptSelector",
        "ContinuityAuditor",
    }
    missing = required_agents.difference(infos)
    if missing:
        raise ConfigError(f"missing required agents: {sorted(missing)}")

    return infos


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    scenes_v2: dict[str, Any],
    allow_cross_session: bool,
) -> None:
    input_session_id = str(session_input.get("session_id", "")).strip()
    input_objective_id = str(session_input.get("objective_id", "")).strip()
    objective_session_id = str(session_objective.get("session_id", "")).strip()
    objective_objective_id = str(session_objective.get("objective_id", "")).strip()

    if input_session_id != objective_session_id:
        raise ContractError(
            "session_input.session_id must match session_objective.session_id"
        )
    if input_objective_id != objective_objective_id:
        raise ContractError(
            "session_input.objective_id must match session_objective.objective_id"
        )

    if allow_cross_session:
        return

    scenes_session_id = str(scenes_v2.get("session_id", "")).strip()
    scenes_objective_id = str(scenes_v2.get("objective_id", "")).strip()
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


def _resolve_effective_score_spec_path(
    *,
    root: Path,
    cli_score_spec_path: Path,
    session_objective: dict[str, Any],
    allow_score_spec_override: bool,
    override_reason: str,
    pipeline_env: str,
) -> tuple[Path, Path, str, bool, str]:
    objective_score_spec_ref = str(session_objective.get("score_spec_ref", "")).strip()
    if not objective_score_spec_ref:
        raise ContractError("session_objective.score_spec_ref must be non-empty")

    ref_path = Path(objective_score_spec_ref)
    objective_score_spec_path = (
        ref_path.resolve() if ref_path.is_absolute() else (root / ref_path).resolve()
    )
    if not objective_score_spec_path.exists():
        raise ContractError(
            "session_objective.score_spec_ref path does not exist: "
            f"{objective_score_spec_path}"
        )

    effective_score_spec_path = cli_score_spec_path
    score_spec_override_used = False
    score_spec_override_reason = ""
    if cli_score_spec_path != objective_score_spec_path:
        if not allow_score_spec_override:
            raise ContractError(
                "CLI --score-spec does not match session_objective.score_spec_ref. "
                "Use --allow-score-spec-override with --override-reason for recovery reruns."
            )
        if pipeline_env == "prod":
            raise ContractError(
                "--allow-score-spec-override is not allowed when --pipeline-env=prod"
            )
        score_spec_override_reason = override_reason.strip()
        if not score_spec_override_reason:
            raise ContractError(
                "--override-reason must be non-empty when --allow-score-spec-override is set"
            )
        score_spec_override_used = True
    else:
        effective_score_spec_path = objective_score_spec_path

    return (
        effective_score_spec_path,
        objective_score_spec_path,
        objective_score_spec_ref,
        score_spec_override_used,
        score_spec_override_reason,
    )


def _primary_character(scene: dict[str, Any]) -> str:
    refs = scene.get("character_refs", [])
    if isinstance(refs, list) and refs:
        return str(refs[0])
    return ""


def _normalize_scene_filter(scene_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in scene_ids:
        scene_id = str(raw).strip()
        if not scene_id:
            raise ContractError("--scene-id must be a non-empty string")
        if scene_id in seen:
            continue
        seen.add(scene_id)
        out.append(scene_id)
    return out


def build_scene_contexts(scenes_v2: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = list(scenes_v2.get("scenes", []))
    out = []
    for idx, scene in enumerate(scenes):
        prev_scene = scenes[idx - 1] if idx > 0 else None
        next_scene = scenes[idx + 1] if idx < len(scenes) - 1 else None
        out.append(
            {
                "scene_index": idx + 1,
                "scene_count": len(scenes),
                "scene": scene,
                "prev_scene_id": prev_scene.get("scene_id") if isinstance(prev_scene, dict) else None,
                "next_scene_id": next_scene.get("scene_id") if isinstance(next_scene, dict) else None,
                "prev_summary": normalize_text(str(prev_scene.get("narration_text", "")))
                if isinstance(prev_scene, dict)
                else "",
                "next_summary": normalize_text(str(next_scene.get("narration_text", "")))
                if isinstance(next_scene, dict)
                else "",
            }
        )
    return out


def build_scene_brief(*, scene: dict[str, Any], context: dict[str, Any], strategy: str) -> dict[str, Any]:
    scene_id = str(scene.get("scene_id", "scene_unknown"))
    visual_goal = normalize_text(str(scene.get("visual_goal", "")))
    narration_text = normalize_text(str(scene.get("narration_text", "")))
    shot = scene.get("shot_spec", {})

    must_show = [
        f"visual_goal: {visual_goal}",
        f"narration_cue: {narration_text}",
    ]
    if scene.get("location_ref"):
        must_show.append(f"location_anchor: {scene['location_ref']}")
    primary_char = _primary_character(scene)
    if primary_char:
        must_show.append(f"primary_character: {primary_char}")

    continuity_notes = []
    if context.get("prev_summary"):
        continuity_notes.append(f"prev: {context['prev_summary']}")
    if context.get("next_summary"):
        continuity_notes.append(f"next: {context['next_summary']}")

    avoid = [
        str(item)
        for item in scene.get("negative_constraints", [])
        if isinstance(item, str)
    ][:6]

    camera_intent = normalize_text(
        f"{shot.get('shot_size', 'MEDIUM')} / {shot.get('camera_angle', 'EYE_LEVEL')} / {shot.get('camera_motion', 'STATIC')}"
    )

    return {
        "brief_id": f"{scene_id}_{strategy}",
        "strategy": strategy,
        "must_show": must_show,
        "avoid": avoid,
        "camera_intent": camera_intent,
        "continuity_notes": continuity_notes,
    }


def _candidate_prompts(
    *,
    scene: dict[str, Any],
    context: dict[str, Any],
    style_preset: str,
    strategy: str,
) -> tuple[str, str]:
    scene_id = str(scene.get("scene_id", "scene_unknown"))
    visual_goal = normalize_text(str(scene.get("visual_goal", "")))
    narration = normalize_text(str(scene.get("narration_text", "")))
    location_ref = str(scene.get("location_ref", "location_primary"))
    shot = scene.get("shot_spec", {})
    action = scene.get("action_emotion", {}).get("action", "advance the story")
    emotion = scene.get("action_emotion", {}).get("emotion", "tension")
    camera_motion = str(shot.get("camera_motion", "STATIC"))
    camera_size = str(shot.get("shot_size", "MEDIUM"))
    camera_angle = str(shot.get("camera_angle", "EYE_LEVEL"))
    primary_char = _primary_character(scene) or "character_anchor"

    prev_scene_id = context.get("prev_scene_id") or "none"
    next_scene_id = context.get("next_scene_id") or "none"

    if strategy == "continuity_focus":
        image_prompt = normalize_text(
            f"{style_preset}, continuity-focused frame for {scene_id}. Goal: {visual_goal}. "
            f"Narration cue: {narration}. Character anchor: {primary_char}. "
            f"Location: {location_ref}. Shot: {camera_size} {camera_angle}."
        )
        video_prompt = normalize_text(
            f"CameraMotion: {camera_motion}. Action: {action}. Emotion: {emotion}. "
            f"Maintain continuity with previous scene {prev_scene_id} and prepare transition to {next_scene_id}."
        )
        return image_prompt, video_prompt

    if strategy == "impact_focus":
        image_prompt = normalize_text(
            f"{style_preset}, high-contrast dramatic frame for {scene_id}. "
            f"Create immediate scroll-stop impact around: {visual_goal}. "
            f"Narration phrase: {narration}. Location: {location_ref}. "
            f"Shot: {camera_size} {camera_angle}."
        )
        video_prompt = normalize_text(
            f"CameraMotion: {camera_motion}. Action: {action}. Emotion: {emotion}. "
            f"Start with a sharp visual reveal, then sustain suspense while preserving scene continuity."
        )
        return image_prompt, video_prompt

    image_prompt = normalize_text(
        f"{style_preset}. SceneID: {scene_id}. Goal: {visual_goal}. "
        f"Character: {primary_char}. Location: {location_ref}. "
        f"Shot: {camera_size} {camera_angle}. NarrationCue: {narration}."
    )
    video_prompt = normalize_text(
        f"CameraMotion: {camera_motion}. Action: {action}. Emotion: {emotion}. "
        f"Use deterministic staging and maintain continuity with neighboring scenes."
    )
    return image_prompt, video_prompt


def build_prompt_candidates_for_scene(
    *,
    scene: dict[str, Any],
    context: dict[str, Any],
    style_preset: str,
) -> list[dict[str, Any]]:
    scene_id = str(scene.get("scene_id", "scene_unknown"))
    candidates = []
    strategy_specs = [
        ("continuity_focus", "001"),
        ("impact_focus", "002"),
        ("control_focus", "003"),
    ]

    for strategy, suffix in strategy_specs:
        brief = build_scene_brief(scene=scene, context=context, strategy=strategy)
        image_prompt, video_prompt = _candidate_prompts(
            scene=scene,
            context=context,
            style_preset=style_preset,
            strategy=strategy,
        )
        candidates.append(
            {
                "candidate_id": f"{scene_id}_prompt_{suffix}",
                "scene_id": scene_id,
                "strategy": strategy,
                "scene_brief": brief,
                "visual_prompt": {
                    "image_prompt": image_prompt,
                    "video_prompt": video_prompt,
                },
            }
        )

    return candidates


def score_prompt_candidate(
    *,
    candidate: dict[str, Any],
    scene: dict[str, Any],
    style_preset: str,
    global_negative_constraints: list[str],
    score_spec: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, bool]]:
    image_prompt = normalize_text(str(candidate.get("visual_prompt", {}).get("image_prompt", "")))
    video_prompt = normalize_text(str(candidate.get("visual_prompt", {}).get("video_prompt", "")))
    text = f"{image_prompt} {video_prompt}"
    lower = text.lower()

    purpose = str(scene.get("continuity_state", {}).get("purpose", "build"))
    location_ref = str(scene.get("location_ref", ""))
    character_refs = [str(x) for x in scene.get("character_refs", []) if isinstance(x, str)]
    shot = scene.get("shot_spec", {})

    impact_tokens = ["high-contrast", "dramatic", "reveal", "scroll-stop", "tension"]
    retention_tokens = ["suspense", "transition", "continuity", "replay", "hook"]

    k1 = 42.0
    k1 += min(sum(4 for token in impact_tokens if token in lower), 18)
    if purpose == "hook":
        k1 += 8
    if 110 <= len(image_prompt) <= 320:
        k1 += 10
    elif len(image_prompt) > 360:
        k1 -= 8
    k1 = clamp(k1)

    k3 = 44.0
    k3 += min(sum(5 for token in retention_tokens if token in lower), 20)
    if purpose in {"build", "loop"}:
        k3 += 6
    if "cameramotion:" in video_prompt.lower():
        k3 += 8
    k3 = clamp(k3)

    k2 = 40.0
    if location_ref and location_ref.lower() in lower:
        k2 += 10
    if style_preset and style_preset.lower() in lower:
        k2 += 12
    char_hits = sum(1 for ref in character_refs[:2] if ref.lower() in lower)
    k2 += char_hits * 6
    shot_fields = [
        str(shot.get("shot_size", "")).lower(),
        str(shot.get("camera_angle", "")).lower(),
        str(shot.get("camera_motion", "")).lower(),
    ]
    if all(field and field in lower for field in shot_fields):
        k2 += 10
    if "continuity" in lower:
        k2 += 6
    k2 = clamp(k2)

    k4 = 48.0
    if "shot:" in image_prompt.lower():
        k4 += 8
    if "action:" in video_prompt.lower():
        k4 += 8
    if "emotion:" in video_prompt.lower():
        k4 += 8
    if 60 <= len(video_prompt) <= 280:
        k4 += 8
    elif len(video_prompt) > 360:
        k4 -= 10
    k4 = clamp(k4)

    banned_tokens = [str(x).lower() for x in global_negative_constraints]
    banned_tokens += [str(x).lower() for x in scene.get("negative_constraints", [])]
    banned_tokens += UNSAFE_PROMPT_TOKENS
    hard_k5 = not any(token and token in lower for token in banned_tokens)

    criterion_scores = {
        "K3": round(k3, 2),
        "K1": round(k1, 2),
        "K2": round(k2, 2),
        "K4": round(k4, 2),
    }
    hard_constraints = {"K5": hard_k5}

    weight_by_id = {c["id"]: float(c["weight"]) for c in score_spec.get("criteria", [])}
    total = 0.0
    for criterion_id, score in criterion_scores.items():
        total += weight_by_id.get(criterion_id, 0.0) * score
    total = round(total, 2)

    return total, criterion_scores, hard_constraints


def evaluate_and_select_prompt_candidates(
    *,
    scene_candidates: list[dict[str, Any]],
    scenes_v2: dict[str, Any],
    score_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scene_index = {
        str(scene.get("scene_id")): scene
        for scene in scenes_v2.get("scenes", [])
        if isinstance(scene, dict)
    }
    style_preset = str(scenes_v2.get("style_preset", ""))
    global_negative_constraints = [
        str(x) for x in scenes_v2.get("global_negative_constraints", []) if isinstance(x, str)
    ]

    tie_break_order = score_spec.get("selection_rule", {}).get("tie_break_order", ["K3", "K2", "K1"])
    min_accept_score = float(score_spec.get("selection_rule", {}).get("min_accept_score", 0.0))

    selected_rows = []
    scene_evaluations = []
    trace_rows = []

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple([row["total_score"]] + [row["criteria"].get(key, 0.0) for key in tie_break_order])

    for block in scene_candidates:
        scene_id = str(block.get("scene_id", ""))
        scene = scene_index.get(scene_id)
        if scene is None:
            raise ContractError(f"scene_id not found in scenes_v2: {scene_id}")

        rows = []
        for cand in block.get("candidates", []):
            total, criteria, hard = score_prompt_candidate(
                candidate=cand,
                scene=scene,
                style_preset=style_preset,
                global_negative_constraints=global_negative_constraints,
                score_spec=score_spec,
            )
            rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "strategy": cand.get("strategy"),
                    "total_score": total,
                    "criteria": criteria,
                    "hard_constraints": hard,
                    "hard_pass": all(hard.values()),
                }
            )

        if not rows:
            raise ContractError(f"no prompt candidates for scene_id: {scene_id}")

        passing = [row for row in rows if row["hard_pass"]]
        selected_eval = max(passing if passing else rows, key=rank_key)
        selected_candidate = next(
            cand for cand in block["candidates"] if cand["candidate_id"] == selected_eval["candidate_id"]
        )
        pass_threshold = selected_eval["total_score"] >= min_accept_score and selected_eval["hard_pass"]

        selected_rows.append(
            {
                "scene_id": scene_id,
                "candidate": selected_candidate,
                "evaluation": selected_eval,
                "pass_threshold_met": pass_threshold,
                "revised_by_continuity_auditor": False,
                "revision_notes": [],
            }
        )
        scene_evaluations.append(
            {
                "scene_id": scene_id,
                "candidate_evaluations": rows,
                "selected_candidate_id": selected_eval["candidate_id"],
                "min_accept_score": min_accept_score,
                "pass_threshold_met": pass_threshold,
            }
        )
        trace_rows.append(
            {
                "scene_id": scene_id,
                "selected_candidate_id": selected_eval["candidate_id"],
                "selected_score": selected_eval["total_score"],
                "selection_reason": "Highest weighted score under hard-constraint policy.",
            }
        )

    pass_scene_count = sum(1 for row in selected_rows if row["pass_threshold_met"])
    evaluation_report = {
        "schema_version": "1.0.0",
        "task_id": "prompt_selection",
        "created_at": now_utc_iso(),
        "scene_evaluations": scene_evaluations,
        "overall": {
            "scene_count": len(scene_evaluations),
            "pass_scene_count": pass_scene_count,
            "failed_scene_count": len(scene_evaluations) - pass_scene_count,
        },
    }
    selection_trace = {
        "schema_version": "1.0.0",
        "task_id": "prompt_selection",
        "created_at": now_utc_iso(),
        "tie_break_order": tie_break_order,
        "selected_by_scene": trace_rows,
    }

    return selected_rows, evaluation_report, selection_trace


def _maybe_append_sentence(text: str, sentence: str) -> str:
    normalized = normalize_text(text)
    sentence_norm = normalize_text(sentence)
    if not sentence_norm:
        return normalized
    lower = normalized.lower()
    if sentence_norm.lower() in lower:
        return normalized
    out = normalize_text(f"{normalized} {sentence_norm}")
    return out


def run_continuity_audit(
    *,
    selected_rows: list[dict[str, Any]],
    scenes_v2: dict[str, Any],
    score_spec: dict[str, Any],
) -> dict[str, Any]:
    style_preset = str(scenes_v2.get("style_preset", ""))
    min_accept_score = float(score_spec.get("selection_rule", {}).get("min_accept_score", 0.0))
    scene_by_id = {
        str(scene.get("scene_id")): scene
        for scene in scenes_v2.get("scenes", [])
        if isinstance(scene, dict)
    }
    global_negative_constraints = [
        str(x) for x in scenes_v2.get("global_negative_constraints", []) if isinstance(x, str)
    ]

    flagged_scene_ids = []
    revised_scene_ids = []
    issue_rows = []

    for row in selected_rows:
        scene_id = row["scene_id"]
        scene = scene_by_id.get(scene_id)
        if scene is None:
            continue

        candidate = row["candidate"]
        visual_prompt = copy.deepcopy(candidate.get("visual_prompt", {}))
        image_prompt = normalize_text(str(visual_prompt.get("image_prompt", "")))
        video_prompt = normalize_text(str(visual_prompt.get("video_prompt", "")))

        issues = []
        if style_preset and style_preset.lower() not in image_prompt.lower():
            issues.append("missing_style_preset")
        location_ref = str(scene.get("location_ref", ""))
        if location_ref and location_ref.lower() not in f"{image_prompt} {video_prompt}".lower():
            issues.append("missing_location_anchor")
        primary_char = _primary_character(scene)
        if primary_char and primary_char.lower() not in f"{image_prompt} {video_prompt}".lower():
            issues.append("missing_primary_character")
        if scene.get("continuity_state", {}).get("scene_index", 1) > 1 and "continuity" not in video_prompt.lower():
            issues.append("missing_transition_clause")

        if issues:
            flagged_scene_ids.append(scene_id)
            if "missing_style_preset" in issues and style_preset:
                image_prompt = _maybe_append_sentence(image_prompt, f"Style preset: {style_preset}.")
            if "missing_location_anchor" in issues and location_ref:
                image_prompt = _maybe_append_sentence(image_prompt, f"Location anchor: {location_ref}.")
            if "missing_primary_character" in issues and primary_char:
                image_prompt = _maybe_append_sentence(image_prompt, f"Character anchor: {primary_char}.")
            if "missing_transition_clause" in issues:
                video_prompt = _maybe_append_sentence(
                    video_prompt,
                    "Maintain continuity with previous scene transitions.",
                )

            candidate["visual_prompt"] = {
                "image_prompt": normalize_text(image_prompt),
                "video_prompt": normalize_text(video_prompt),
            }
            row["revised_by_continuity_auditor"] = True
            row["revision_notes"] = issues
            revised_scene_ids.append(scene_id)

            total, criteria, hard = score_prompt_candidate(
                candidate=candidate,
                scene=scene,
                style_preset=style_preset,
                global_negative_constraints=global_negative_constraints,
                score_spec=score_spec,
            )
            row["evaluation"]["total_score"] = total
            row["evaluation"]["criteria"] = criteria
            row["evaluation"]["hard_constraints"] = hard
            row["evaluation"]["hard_pass"] = all(hard.values())
            row["pass_threshold_met"] = (
                total >= min_accept_score and row["evaluation"]["hard_pass"]
            )

        issue_rows.append(
            {
                "scene_id": scene_id,
                "issues": issues,
                "status": "revised" if issues else "ok",
            }
        )

    return {
        "schema_version": "1.0.0",
        "task_id": "prompt_continuity_audit",
        "created_at": now_utc_iso(),
        "issues": issue_rows,
        "flagged_scene_ids": flagged_scene_ids,
        "revised_scene_ids": revised_scene_ids,
        "issue_count": sum(len(row["issues"]) for row in issue_rows),
    }


def _sync_reports_after_continuity_audit(
    *,
    selected_rows: list[dict[str, Any]],
    evaluation_report: dict[str, Any],
    selection_trace: dict[str, Any],
) -> None:
    selected_by_scene = {str(row["scene_id"]): row for row in selected_rows}

    scene_evaluations = evaluation_report.get("scene_evaluations", [])
    if isinstance(scene_evaluations, list):
        for scene_eval in scene_evaluations:
            if not isinstance(scene_eval, dict):
                continue
            scene_id = str(scene_eval.get("scene_id", ""))
            row = selected_by_scene.get(scene_id)
            if row is None:
                continue

            candidate = row.get("candidate", {})
            selected_candidate_id = str(candidate.get("candidate_id", ""))
            scene_eval["selected_candidate_id"] = selected_candidate_id
            scene_eval["pass_threshold_met"] = bool(row.get("pass_threshold_met", False))

            candidate_rows = scene_eval.get("candidate_evaluations", [])
            if not isinstance(candidate_rows, list):
                continue

            selected_eval = row.get("evaluation", {})
            for candidate_row in candidate_rows:
                if not isinstance(candidate_row, dict):
                    continue
                if str(candidate_row.get("candidate_id", "")) != selected_candidate_id:
                    continue
                candidate_row["total_score"] = selected_eval.get("total_score", 0.0)
                candidate_row["criteria"] = dict(selected_eval.get("criteria", {}))
                candidate_row["hard_constraints"] = dict(
                    selected_eval.get("hard_constraints", {})
                )
                candidate_row["hard_pass"] = bool(selected_eval.get("hard_pass", False))
                break

    overall = evaluation_report.get("overall")
    if isinstance(overall, dict):
        pass_scene_count = sum(1 for row in selected_rows if row.get("pass_threshold_met"))
        scene_count = len(selected_rows)
        overall["scene_count"] = scene_count
        overall["pass_scene_count"] = pass_scene_count
        overall["failed_scene_count"] = scene_count - pass_scene_count

    trace_rows = selection_trace.get("selected_by_scene", [])
    if isinstance(trace_rows, list):
        for trace_row in trace_rows:
            if not isinstance(trace_row, dict):
                continue
            scene_id = str(trace_row.get("scene_id", ""))
            row = selected_by_scene.get(scene_id)
            if row is None:
                continue
            candidate = row.get("candidate", {})
            evaluation = row.get("evaluation", {})
            trace_row["selected_candidate_id"] = candidate.get("candidate_id", "")
            trace_row["selected_score"] = evaluation.get("total_score", 0.0)


def apply_selected_prompts_to_scenes(
    *,
    scenes_v2: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    allow_partial: bool = False,
) -> dict[str, Any]:
    out = copy.deepcopy(scenes_v2)
    scenes = out.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("scenes_v2.scenes must be non-empty for prompt application")

    scene_ids = []
    seen_scene_ids: set[str] = set()
    duplicate_scene_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ContractError("each scenes_v2 scene must be an object")
        scene_id = str(scene.get("scene_id", "")).strip()
        if not scene_id:
            raise ContractError("each scene must include non-empty scene_id")
        if scene_id in seen_scene_ids:
            duplicate_scene_ids.add(scene_id)
        seen_scene_ids.add(scene_id)
        scene_ids.append(scene_id)
    if duplicate_scene_ids:
        raise ContractError(f"duplicate scene_id in scenes_v2: {sorted(duplicate_scene_ids)}")

    selected_by_scene: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ContractError("each selected row must be an object")
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            raise ContractError("each selected row must include non-empty scene_id")
        if scene_id in selected_by_scene:
            raise ContractError(f"duplicate selected row for scene_id: {scene_id}")

        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            raise ContractError(f"selected row candidate must be object: {scene_id}")
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ContractError(f"selected candidate_id must be non-empty: {scene_id}")
        visual_prompt = candidate.get("visual_prompt")
        if not isinstance(visual_prompt, dict):
            raise ContractError(f"selected visual_prompt must be object: {scene_id}")
        image_prompt = normalize_text(str(visual_prompt.get("image_prompt", "")))
        video_prompt = normalize_text(str(visual_prompt.get("video_prompt", "")))
        if not image_prompt or not video_prompt:
            raise ContractError(
                f"selected visual_prompt.image_prompt/video_prompt must be non-empty: {scene_id}"
            )

        selected_by_scene[scene_id] = {
            "candidate_id": candidate_id,
            "strategy": candidate.get("strategy"),
            "prompt_revised": bool(row.get("revised_by_continuity_auditor", False)),
            "image_prompt": image_prompt,
            "video_prompt": video_prompt,
        }

    scene_id_set = set(scene_ids)
    selected_scene_id_set = set(selected_by_scene.keys())
    missing_selected = sorted(scene_id_set.difference(selected_scene_id_set))
    if missing_selected and not allow_partial:
        raise ContractError(
            "missing selected prompt rows for scene_id(s): "
            f"{missing_selected}"
        )
    unknown_selected = sorted(selected_scene_id_set.difference(scene_id_set))
    if unknown_selected:
        raise ContractError(
            "selected prompt rows contain unknown scene_id(s): "
            f"{unknown_selected}"
        )

    for scene in scenes:
        scene_id = str(scene.get("scene_id", ""))
        selected = selected_by_scene.get(scene_id)
        if selected is None:
            continue
        scene["visual_prompt"] = {
            "image_prompt": selected["image_prompt"],
            "video_prompt": selected["video_prompt"],
        }
        continuity_state = scene.get("continuity_state")
        if continuity_state is None:
            continuity_state = {}
        elif not isinstance(continuity_state, dict):
            raise ContractError(f"continuity_state must be object when present: {scene_id}")
        else:
            continuity_state = copy.deepcopy(continuity_state)
        continuity_state["prompt_candidate_id"] = selected["candidate_id"]
        continuity_state["prompt_strategy"] = selected["strategy"]
        continuity_state["prompt_revised"] = selected["prompt_revised"]
        scene["continuity_state"] = continuity_state

    out["created_at"] = now_utc_iso()
    return out


def _sync_scenes_with_selected_prompts(
    *,
    scenes_v2: dict[str, Any],
    selected_prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    out = copy.deepcopy(scenes_v2)
    scenes = out.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("scenes_v2.scenes must be non-empty for prompt synchronization")

    selected_by_scene = _index_selected_prompts(
        selected_prompts,
        path="selected_prompts",
    )

    scene_ids: list[str] = []
    seen_scene_ids: set[str] = set()
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        if not isinstance(scene, dict):
            raise ContractError(f"{path} must be an object")
        scene_id = str(scene.get("scene_id", "")).strip()
        if not scene_id:
            raise ContractError(f"{path}.scene_id must be non-empty")
        if scene_id in seen_scene_ids:
            raise ContractError(f"duplicate scene_id in scenes_v2: {scene_id}")
        seen_scene_ids.add(scene_id)
        scene_ids.append(scene_id)

    missing = sorted(set(scene_ids).difference(selected_by_scene.keys()))
    if missing:
        raise ContractError(
            "missing selected prompt payload for scene_id(s): "
            f"{missing}"
        )
    unknown = sorted(set(selected_by_scene.keys()).difference(scene_ids))
    if unknown:
        raise ContractError(
            "selected prompt payload contains unknown scene_id(s): "
            f"{unknown}"
        )

    for scene in scenes:
        scene_id = str(scene.get("scene_id", "")).strip()
        selected = selected_by_scene[scene_id]
        visual_prompt = selected.get("visual_prompt")
        if not isinstance(visual_prompt, dict):
            raise ContractError(
                f"selected_prompts.visual_prompt must be an object: {scene_id}"
            )
        image_prompt = normalize_text(str(visual_prompt.get("image_prompt", "")))
        video_prompt = normalize_text(str(visual_prompt.get("video_prompt", "")))
        if not image_prompt or not video_prompt:
            raise ContractError(
                "selected_prompts.visual_prompt.image_prompt/video_prompt must be non-empty: "
                f"{scene_id}"
            )

        candidate_id = str(selected.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ContractError(f"selected_prompts.candidate_id must be non-empty: {scene_id}")

        scene["visual_prompt"] = {
            "image_prompt": image_prompt,
            "video_prompt": video_prompt,
        }

        continuity_state = scene.get("continuity_state")
        if continuity_state is None:
            continuity_state = {}
        elif not isinstance(continuity_state, dict):
            raise ContractError(f"continuity_state must be object when present: {scene_id}")
        else:
            continuity_state = copy.deepcopy(continuity_state)
        continuity_state["prompt_candidate_id"] = candidate_id
        continuity_state["prompt_strategy"] = selected.get("strategy")
        continuity_state["prompt_revised"] = bool(
            selected.get("revised_by_continuity_auditor", False)
        )
        scene["continuity_state"] = continuity_state

    out["created_at"] = now_utc_iso()
    return out


def _selected_prompt_from_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    evaluation = row["evaluation"]
    return {
        "scene_id": row["scene_id"],
        "candidate_id": candidate["candidate_id"],
        "strategy": candidate.get("strategy"),
        "scene_brief": candidate.get("scene_brief", {}),
        "visual_prompt": candidate.get("visual_prompt", {}),
        "total_score": evaluation["total_score"],
        "criteria": evaluation["criteria"],
        "hard_constraints": evaluation["hard_constraints"],
        "hard_pass": evaluation["hard_pass"],
        "revised_by_continuity_auditor": row["revised_by_continuity_auditor"],
        "revision_notes": list(row.get("revision_notes", [])),
    }


def _index_selected_prompts(
    selected_prompts: list[dict[str, Any]],
    *,
    path: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(selected_prompts, list) or not selected_prompts:
        raise ContractError(f"{path} must be a non-empty array")

    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(selected_prompts):
        row_path = f"{path}[{idx}]"
        if not isinstance(row, dict):
            raise ContractError(f"{row_path} must be an object")
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            raise ContractError(f"{row_path}.scene_id must be non-empty")
        if scene_id in out:
            raise ContractError(f"duplicate selected prompt for scene_id: {scene_id}")
        out[scene_id] = copy.deepcopy(row)
    return out


def _merge_selected_prompts_for_output(
    *,
    scenes_v2: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    base_prompt_map_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    merged_by_scene: dict[str, dict[str, Any]] = {}
    if base_prompt_map_pack is not None:
        base_selected = base_prompt_map_pack.get("selected_prompts", [])
        merged_by_scene = _index_selected_prompts(
            base_selected,
            path="base_prompt_map_pack.selected_prompts",
        )

    for row in selected_rows:
        payload = _selected_prompt_from_row(row)
        scene_id = str(payload["scene_id"]).strip()
        if not scene_id:
            raise ContractError("selected row must include non-empty scene_id")
        merged_by_scene[scene_id] = payload

    scenes = scenes_v2.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("scenes_v2.scenes must be non-empty")

    ordered_scene_ids: list[str] = []
    seen_scene_ids: set[str] = set()
    for idx, scene in enumerate(scenes):
        path = f"scenes_v2.scenes[{idx}]"
        if not isinstance(scene, dict):
            raise ContractError(f"{path} must be an object")
        scene_id = str(scene.get("scene_id", "")).strip()
        if not scene_id:
            raise ContractError(f"{path}.scene_id must be non-empty")
        if scene_id in seen_scene_ids:
            raise ContractError(f"duplicate scene_id in scenes_v2: {scene_id}")
        seen_scene_ids.add(scene_id)
        ordered_scene_ids.append(scene_id)

    missing = [scene_id for scene_id in ordered_scene_ids if scene_id not in merged_by_scene]
    if missing:
        raise ContractError(
            "missing selected prompt payload for scene_id(s): "
            f"{missing}"
        )
    unknown = sorted(set(merged_by_scene.keys()).difference(ordered_scene_ids))
    if unknown:
        raise ContractError(
            "selected prompt payload contains unknown scene_id(s): "
            f"{unknown}"
        )

    return [merged_by_scene[scene_id] for scene_id in ordered_scene_ids]


def build_prompt_map_pack(
    *,
    session_input: dict[str, Any],
    input_scenes_path: Path,
    scenes_v2: dict[str, Any],
    selected_prompts: list[dict[str, Any]],
    continuity_audit_report: dict[str, Any],
) -> dict[str, Any]:
    ready = all(bool(item.get("hard_pass")) for item in selected_prompts)

    return {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "input_scenes_path": str(input_scenes_path),
        "style_preset": str(scenes_v2.get("style_preset", "")),
        "selected_prompts": selected_prompts,
        "continuity_audit": {
            "flagged_scene_ids": list(continuity_audit_report.get("flagged_scene_ids", [])),
            "revised_scene_ids": list(continuity_audit_report.get("revised_scene_ids", [])),
            "issue_count": int(continuity_audit_report.get("issue_count", 0)),
        },
        "quality_status": "READY_FOR_ASSET_PIPELINE" if ready else "NEEDS_PROMPT_REVIEW",
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.perf_counter()

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    scenes_v2_path = Path(args.scenes_v2).resolve()
    model_map_path = Path(args.model_map).resolve()
    score_spec_cli_path = Path(args.score_spec).resolve()

    # Gate-In
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
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    gate_in_duration_ms = round((time.perf_counter() - gate_in_started_at) * 1000.0, 2)
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l7-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    agent_runtime_infos = load_agent_runtime_infos(model_map_path)

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    scenes_v2_input = read_json(scenes_v2_path)
    (
        effective_score_spec_path,
        objective_score_spec_path,
        objective_score_spec_ref,
        score_spec_override_used,
        score_spec_override_reason,
    ) = _resolve_effective_score_spec_path(
        root=root,
        cli_score_spec_path=score_spec_cli_path,
        session_objective=session_objective,
        allow_score_spec_override=bool(args.allow_score_spec_override),
        override_reason=str(args.override_reason),
        pipeline_env=str(args.pipeline_env),
    )
    score_spec = read_json(effective_score_spec_path)

    _assert_session_alignment(
        session_input=session_input,
        session_objective=session_objective,
        scenes_v2=scenes_v2_input,
        allow_cross_session=bool(args.allow_cross_session),
    )

    if str(scenes_v2_input.get("schema_version", "")).strip() != "2.0.0":
        raise ContractError("scenes_v2.schema_version must be 2.0.0")
    scenes = scenes_v2_input.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ContractError("scenes_v2.scenes must be non-empty")

    style_preset = str(scenes_v2_input.get("style_preset", "")).strip()
    if not style_preset:
        raise ContractError("scenes_v2.style_preset must be non-empty")

    requested_scene_ids = _normalize_scene_filter(list(args.scene_id))
    base_prompt_map_pack: dict[str, Any] | None = None
    base_prompt_map_pack_path: Path | None = None
    if requested_scene_ids:
        base_prompt_map_pack_ref = str(args.base_prompt_map_pack).strip()
        if not base_prompt_map_pack_ref:
            raise ContractError("--base-prompt-map-pack is required when --scene-id is set")
        base_prompt_map_pack_path = Path(base_prompt_map_pack_ref).resolve()
        base_prompt_map_pack = read_json(base_prompt_map_pack_path)

        if str(base_prompt_map_pack.get("session_id", "")).strip() != str(
            session_input.get("session_id", "")
        ).strip():
            raise ContractError(
                "base_prompt_map_pack.session_id must match session_input.session_id"
            )
        if str(base_prompt_map_pack.get("objective_id", "")).strip() != str(
            session_input.get("objective_id", "")
        ).strip():
            raise ContractError(
                "base_prompt_map_pack.objective_id must match session_input.objective_id"
            )
        if str(base_prompt_map_pack.get("style_preset", "")).strip() != style_preset:
            raise ContractError(
                "base_prompt_map_pack.style_preset must match scenes_v2.style_preset"
            )
    elif str(args.base_prompt_map_pack).strip():
        raise ContractError("--base-prompt-map-pack can only be used with --scene-id")

    phase_started_at = time.perf_counter()
    scene_contexts = build_scene_contexts(scenes_v2_input)
    context_by_scene_id = {
        str(context["scene"].get("scene_id", "")).strip(): context
        for context in scene_contexts
    }
    if requested_scene_ids:
        missing_scene_ids = sorted(set(requested_scene_ids).difference(context_by_scene_id.keys()))
        if missing_scene_ids:
            raise ContractError(
                "requested --scene-id not found in scenes_v2: "
                f"{missing_scene_ids}"
            )
        scene_contexts_for_run = [context_by_scene_id[scene_id] for scene_id in requested_scene_ids]
    else:
        scene_contexts_for_run = scene_contexts
    context_packaging_ms = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    phase_started_at = time.perf_counter()
    scene_candidates = []
    for context in scene_contexts_for_run:
        scene = context["scene"]
        scene_id = str(scene.get("scene_id", "scene_unknown"))
        candidates = build_prompt_candidates_for_scene(
            scene=scene,
            context=context,
            style_preset=style_preset,
        )
        scene_candidates.append(
            {
                "scene_id": scene_id,
                "candidates": candidates,
            }
        )
    candidate_generation_ms = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    phase_started_at = time.perf_counter()
    selected_rows, evaluation_report, selection_trace = evaluate_and_select_prompt_candidates(
        scene_candidates=scene_candidates,
        scenes_v2=scenes_v2_input,
        score_spec=score_spec,
    )
    scoring_selection_ms = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    phase_started_at = time.perf_counter()
    continuity_audit_report = run_continuity_audit(
        selected_rows=selected_rows,
        scenes_v2=scenes_v2_input,
        score_spec=score_spec,
    )
    _sync_reports_after_continuity_audit(
        selected_rows=selected_rows,
        evaluation_report=evaluation_report,
        selection_trace=selection_trace,
    )
    continuity_audit_ms = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    phase_started_at = time.perf_counter()
    scenes_v2_after_selection = apply_selected_prompts_to_scenes(
        scenes_v2=scenes_v2_input,
        selected_rows=selected_rows,
        allow_partial=bool(requested_scene_ids),
    )
    selected_prompts_for_output = _merge_selected_prompts_for_output(
        scenes_v2=scenes_v2_after_selection,
        selected_rows=selected_rows,
        base_prompt_map_pack=base_prompt_map_pack,
    )
    scenes_v2_output = _sync_scenes_with_selected_prompts(
        scenes_v2=scenes_v2_after_selection,
        selected_prompts=selected_prompts_for_output,
    )
    prompt_map_pack = build_prompt_map_pack(
        session_input=session_input,
        input_scenes_path=scenes_v2_path,
        scenes_v2=scenes_v2_output,
        selected_prompts=selected_prompts_for_output,
        continuity_audit_report=continuity_audit_report,
    )
    output_build_ms = round((time.perf_counter() - phase_started_at) * 1000.0, 2)

    write_json(
        out_dir / "candidates_prompt.json",
        {
            "schema_version": "1.0.0",
            "task_id": "prompt_selection",
            "created_at": now_utc_iso(),
            "scenes": scene_candidates,
        },
    )
    write_json(out_dir / "evaluation_report_prompt.json", evaluation_report)
    write_json(out_dir / "selection_trace_prompt.json", selection_trace)
    write_json(out_dir / "continuity_audit_prompt.json", continuity_audit_report)
    write_json(out_dir / "prompt_map_pack.json", prompt_map_pack)
    write_json(out_dir / "scenes_v2.json", scenes_v2_output)

    candidate_count = sum(len(scene_block["candidates"]) for scene_block in scene_candidates)
    processed_scene_count = len(scene_candidates)
    hard_pass_prompt_count = sum(
        1 for item in selected_prompts_for_output if bool(item.get("hard_pass"))
    )

    runtime_log: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "pipeline_stage": "L7 Scene Prompt Map",
        "provider_model_trace": [
            {
                "agent_id": info.agent_id,
                "provider": info.provider,
                "model": info.model,
                "tools": info.tools,
                "purpose": info.purpose,
            }
            for info in agent_runtime_infos.values()
        ],
        "inputs": {
            "session_input": str(session_input_path),
            "session_objective": str(session_objective_path),
            "scenes_v2": str(scenes_v2_path),
            "score_spec_cli": str(score_spec_cli_path),
            "score_spec": str(effective_score_spec_path),
            "objective_score_spec_ref": objective_score_spec_ref,
            "objective_score_spec_path": str(objective_score_spec_path),
            "model_map": str(model_map_path),
            "base_prompt_map_pack": str(base_prompt_map_pack_path)
            if base_prompt_map_pack_path is not None
            else "",
        },
        "outputs": {
            "out_dir": str(out_dir),
            "scene_count": len(scenes),
            "processed_scene_count": processed_scene_count,
        },
        "timings_ms": {
            "gate_in_validation": gate_in_duration_ms,
            "context_packaging": context_packaging_ms,
            "candidate_generation": candidate_generation_ms,
            "scoring_selection": scoring_selection_ms,
            "continuity_audit": continuity_audit_ms,
            "output_build": output_build_ms,
        },
        "scene_metrics": {
            "scene_count": len(scenes),
            "processed_scene_count": processed_scene_count,
            "candidate_per_processed_scene_avg": round(
                (candidate_count / processed_scene_count) if processed_scene_count else 0.0,
                2,
            ),
            "selected_prompt_count": len(selected_prompts_for_output),
            "hard_pass_prompt_count": hard_pass_prompt_count,
            "flagged_scene_count": len(continuity_audit_report["flagged_scene_ids"]),
            "revised_scene_count": len(continuity_audit_report["revised_scene_ids"]),
        },
        "policy": {
            "model_update_mode": "manual_only",
            "hard_constraint": "K5 pass_only",
            "scene_filter_applied": bool(requested_scene_ids),
            "scene_filter": list(requested_scene_ids),
            "allow_cross_session": bool(args.allow_cross_session),
            "pipeline_env": str(args.pipeline_env),
            "allow_score_spec_override": bool(args.allow_score_spec_override),
            "score_spec_override_used": score_spec_override_used,
            "score_spec_override_reason": score_spec_override_reason,
        },
    }
    write_json(out_dir / "runtime_log_l7.json", runtime_log)

    pass_count = sum(1 for row in selected_rows if row["pass_threshold_met"])
    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "candidates_prompt",
                "path": str((out_dir / "candidates_prompt.json").resolve()),
            },
            {
                "name": "evaluation_report_prompt",
                "path": str((out_dir / "evaluation_report_prompt.json").resolve()),
            },
            {
                "name": "selection_trace_prompt",
                "path": str((out_dir / "selection_trace_prompt.json").resolve()),
            },
            {
                "name": "continuity_audit_prompt",
                "path": str((out_dir / "continuity_audit_prompt.json").resolve()),
            },
            {
                "name": "prompt_map_pack",
                "path": str((out_dir / "prompt_map_pack.json").resolve()),
            },
            {
                "name": "scenes_v2",
                "path": str((out_dir / "scenes_v2.json").resolve()),
            },
            {
                "name": "runtime_log_l7",
                "path": str((out_dir / "runtime_log_l7.json").resolve()),
            },
        ],
        "metrics": {
            "scene_count": len(scenes),
            "processed_scene_count": processed_scene_count,
            "candidate_count": candidate_count,
            "pass_threshold_scene_count": pass_count,
            "selected_prompt_count": len(selected_prompts_for_output),
            "hard_pass_prompt_count": hard_pass_prompt_count,
            "flagged_scene_count": len(continuity_audit_report["flagged_scene_ids"]),
            "revised_scene_count": len(continuity_audit_report["revised_scene_ids"]),
            "scene_filter_applied": bool(requested_scene_ids),
        },
    }
    write_json(out_dir / "session_output.json", session_output)

    # Gate-Out
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
                "scenes_v2",
                out_dir / "scenes_v2.json",
                root / "schemas" / "scenes_v2.schema.json",
            ),
            (
                "prompt_map_pack",
                out_dir / "prompt_map_pack.json",
                root / "schemas" / "prompt_map_pack.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    gate_out_duration_ms = round((time.perf_counter() - gate_out_started_at) * 1000.0, 2)
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l7-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    total_duration_ms = round((time.perf_counter() - run_started_at) * 1000.0, 2)
    runtime_log["timings_ms"]["gate_out_validation"] = gate_out_duration_ms
    runtime_log["timings_ms"]["total_run"] = total_duration_ms
    write_json(out_dir / "runtime_log_l7.json", runtime_log)

    print(f"[l7-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L7 Scene Prompt Map with per-scene scoring and continuity reduce."
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
        help="Path to scenes_v2.json",
    )
    parser.add_argument(
        "--model-map",
        default=str(root / "config" / "agent_model_map.yaml"),
        help="Path to agent_model_map.yaml",
    )
    parser.add_argument(
        "--score-spec",
        default=str(root / "score_specs" / "score_spec_prompt.json"),
        help="Path to score_spec_prompt.json",
    )
    parser.add_argument(
        "--allow-cross-session",
        action="store_true",
        help=(
            "Allow scenes_v2 session/objective mismatch for recovery reruns. "
            "Default is strict match."
        ),
    )
    parser.add_argument(
        "--allow-score-spec-override",
        action="store_true",
        help=(
            "Allow --score-spec to differ from session_objective.score_spec_ref "
            "for recovery reruns. Disabled in prod."
        ),
    )
    parser.add_argument(
        "--override-reason",
        default="",
        help=(
            "Required non-empty reason when --allow-score-spec-override is used."
        ),
    )
    parser.add_argument(
        "--pipeline-env",
        choices=["dev", "prod"],
        default="dev",
        help="Runtime environment used for override policy enforcement.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help=(
            "Optional scene_id to rerun. Repeat for multiple scenes. "
            "When provided, --base-prompt-map-pack is required."
        ),
    )
    parser.add_argument(
        "--base-prompt-map-pack",
        default="",
        help=(
            "Path to existing prompt_map_pack.json used as merge base for "
            "non-target scenes during --scene-id partial reruns."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ConfigError, ContractError) as err:
        print(f"[l7-runner] config/contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l7-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
