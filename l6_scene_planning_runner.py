#!/usr/bin/env python3
"""L6 Scene Planning runner (story bible + beat split + scene split + scenes_v2)."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, read_yaml, write_gate_manifest, write_json


LOCK_POLICY = "REAL > INFERRED > NO_INFO"


@dataclass
class AgentRuntimeInfo:
    agent_id: str
    provider: str
    model: str
    tools: list[str]
    purpose: str


@dataclass
class SceneConfig:
    target_duration_sec: float
    min_scene_sec: float
    max_scene_sec: float
    min_scenes: int
    max_scenes: int
    hook_deadline_sec: float
    style_preset: str
    style_language: str
    style_pacing: str


class ConfigError(Exception):
    """Raised when model map config is invalid."""


class SceneConfigError(Exception):
    """Raised when scene planning config is invalid."""


class ContractError(Exception):
    """Raised when input contract preconditions are violated."""


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _validate_model_policy(policy: dict[str, Any]) -> None:
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
    _validate_model_policy(policy)

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
        "StoryBibleBuilder",
        "StoryBibleEvaluator",
        "StoryBibleSelector",
        "BeatSplitter",
        "BeatEvaluator",
        "BeatSelector",
        "SceneComposer",
        "SceneSplitCritic",
        "SceneSplitSelector",
        "ScenesJSONBuilder",
    }
    missing = required_agents.difference(infos)
    if missing:
        raise ConfigError(f"missing required agents: {sorted(missing)}")

    return infos


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    script_session_pack: dict[str, Any],
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

    pack_session_id = str(script_session_pack.get("session_id", "")).strip()
    pack_objective_id = str(script_session_pack.get("objective_id", "")).strip()
    if pack_session_id != input_session_id:
        raise ContractError(
            "script_session_pack.session_id must match session_input.session_id "
            "(use --allow-cross-session for recovery reruns)"
        )
    if pack_objective_id != input_objective_id:
        raise ContractError(
            "script_session_pack.objective_id must match session_input.objective_id "
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


def load_scene_config(path: Path) -> SceneConfig:
    payload = read_yaml(path)
    if not isinstance(payload, dict):
        raise SceneConfigError("scene config root must be an object")

    constraints = payload.get("scene_constraints")
    if not isinstance(constraints, dict):
        raise SceneConfigError("scene_constraints must be an object")

    style = payload.get("style")
    if not isinstance(style, dict):
        raise SceneConfigError("style must be an object")

    try:
        target_duration_sec = float(constraints["target_duration_sec"])
        min_scene_sec = float(constraints["min_scene_sec"])
        max_scene_sec = float(constraints["max_scene_sec"])
        min_scenes = int(constraints["min_scenes"])
        max_scenes = int(constraints["max_scenes"])
        hook_deadline_sec = float(constraints["hook_deadline_sec"])
    except (KeyError, TypeError, ValueError) as err:
        raise SceneConfigError(f"invalid scene_constraints fields: {err}") from err

    style_preset = str(style.get("style_preset", "")).strip()
    style_language = str(style.get("language", "")).strip()
    style_pacing = str(style.get("pacing", "")).strip()

    if target_duration_sec <= 0:
        raise SceneConfigError("target_duration_sec must be > 0")
    if min_scene_sec <= 0:
        raise SceneConfigError("min_scene_sec must be > 0")
    if max_scene_sec < min_scene_sec:
        raise SceneConfigError("max_scene_sec must be >= min_scene_sec")
    if min_scenes <= 0:
        raise SceneConfigError("min_scenes must be > 0")
    if max_scenes < min_scenes:
        raise SceneConfigError("max_scenes must be >= min_scenes")
    if hook_deadline_sec <= 0:
        raise SceneConfigError("hook_deadline_sec must be > 0")
    if not style_preset:
        raise SceneConfigError("style.style_preset must be non-empty")
    if not style_language:
        raise SceneConfigError("style.language must be non-empty")
    if not style_pacing:
        raise SceneConfigError("style.pacing must be non-empty")

    return SceneConfig(
        target_duration_sec=target_duration_sec,
        min_scene_sec=min_scene_sec,
        max_scene_sec=max_scene_sec,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
        hook_deadline_sec=hook_deadline_sec,
        style_preset=style_preset,
        style_language=style_language,
        style_pacing=style_pacing,
    )


def _phase_to_role(phase: str) -> str:
    normalized = normalize_text(phase).lower()
    if normalized in {"hook", "setup", "build", "payoff", "loop"}:
        return normalized
    return "build"


def _estimate_complexity(text: str) -> str:
    words = [w for w in re.split(r"\W+", text) if w]
    count = len(words)
    if count <= 12:
        return "low"
    if count <= 24:
        return "medium"
    return "high"


def _split_sentences(text: str) -> list[str]:
    chunks = [normalize_text(chunk) for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    return [chunk for chunk in chunks if chunk]


def build_story_bible(
    *,
    session_input: dict[str, Any],
    script_session_pack: dict[str, Any],
    scene_cfg: SceneConfig,
) -> dict[str, Any]:
    characters = []
    for entity in script_session_pack.get("character_fact_registry", []):
        if entity.get("entity_type") != "PERSON":
            continue
        salient_facts = []
        for field in entity.get("fields", []):
            value = field.get("value")
            if value is None:
                continue
            salient_facts.append(f"{field.get('field')}: {value}")
        characters.append(
            {
                "id": entity.get("entity_id", "unknown_person"),
                "name": entity.get("display_name", entity.get("role_label", "person")),
                "role_label": entity.get("role_label", "person"),
                "salient_facts": salient_facts[:6],
            }
        )

    return {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "video_goal": "shorts",
        "target_duration_sec": scene_cfg.target_duration_sec,
        "tone": {
            "language": scene_cfg.style_language,
            "style": scene_cfg.style_pacing,
        },
        "global_visual_rules": {
            "style_keywords": [
                "cinematic",
                "high contrast",
                "investigative",
                "clean typography",
            ],
            "subtitle_rules": {
                "position": "bottom",
                "max_chars_per_line": 18,
            },
        },
        "characters": characters,
        "banned": list(script_session_pack.get("banned_expressions", [])),
        "references": {
            "trigger_words": list(script_session_pack.get("trigger_words", [])),
            "lock_policy": script_session_pack.get("lock_policy", ""),
            "script_id": script_session_pack.get("selected_script", {}).get("script_id", "unknown"),
        },
    }


def build_beats(
    *,
    session_input: dict[str, Any],
    script_session_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    structured = script_session_pack.get("translation_outputs", {}).get("structured_doc", [])
    beats = []

    if isinstance(structured, list) and structured:
        for idx, segment in enumerate(structured, start=1):
            narration = normalize_text(str(segment.get("narration_pt", "")))
            if not narration:
                continue
            role = _phase_to_role(str(segment.get("phase", "build")))
            beats.append(
                {
                    "beat_id": f"beat_{idx:03d}",
                    "text": narration,
                    "role": role,
                    "complexity": _estimate_complexity(narration),
                    "source_claim_ids": list(segment.get("source_claim_ids", [])),
                }
            )
    else:
        narration = normalize_text(str(script_session_pack.get("narration_only_pt", "")))
        for idx, sentence in enumerate(_split_sentences(narration), start=1):
            beats.append(
                {
                    "beat_id": f"beat_{idx:03d}",
                    "text": sentence,
                    "role": "build",
                    "complexity": _estimate_complexity(sentence),
                    "source_claim_ids": [],
                }
            )

    if not beats:
        fallback = normalize_text(
            str(script_session_pack.get("selected_script", {}).get("script_text_pt", ""))
        )
        if not fallback:
            fallback = "Relato investigativo sobre um caso com contradicao de versoes."
        beats = [
            {
                "beat_id": "beat_001",
                "text": fallback,
                "role": "build",
                "complexity": _estimate_complexity(fallback),
                "source_claim_ids": [],
            }
        ]

    return beats


def build_beats_doc(
    *,
    session_input: dict[str, Any],
    beats: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "beats": beats,
    }


def estimate_duration_sec(text: str, scene_cfg: SceneConfig) -> float:
    words = [w for w in re.split(r"\W+", text) if w]
    estimate = len(words) / 2.6 if words else scene_cfg.min_scene_sec
    estimate = max(scene_cfg.min_scene_sec * 0.75, estimate)
    estimate = min(scene_cfg.max_scene_sec * 1.2, estimate)
    return round(estimate, 2)


def _select_scene_purpose(roles: list[str]) -> str:
    for label in ["hook", "setup", "build", "payoff", "loop"]:
        if label in roles:
            return label
    return "build"


def _chunk_evenly(beats: list[dict[str, Any]], chunk_count: int) -> list[list[dict[str, Any]]]:
    if not beats:
        return []
    chunk_count = max(1, min(chunk_count, len(beats)))
    base = len(beats) // chunk_count
    rem = len(beats) % chunk_count
    sizes = [base + (1 if i < rem else 0) for i in range(chunk_count)]

    out = []
    start = 0
    for size in sizes:
        out.append(beats[start:start + size])
        start += size
    return [chunk for chunk in out if chunk]


def compose_scene_skeletons(
    *,
    beats: list[dict[str, Any]],
    scene_cfg: SceneConfig,
    strategy: str,
    candidate_id: str,
) -> dict[str, Any]:
    if strategy == "per_beat":
        groups = [[beat] for beat in beats]
        split_reason = "beat_boundary"
    elif strategy == "hook_pair":
        groups = [[beats[0]]]
        cursor = 1
        while cursor < len(beats):
            groups.append(beats[cursor:cursor + 2])
            cursor += 2
        split_reason = "hook_deadline_plus_rhythm"
    else:
        avg_scene = (scene_cfg.min_scene_sec + scene_cfg.max_scene_sec) / 2.0
        target_count = int(round(scene_cfg.target_duration_sec / avg_scene))
        target_count = max(scene_cfg.min_scenes, min(scene_cfg.max_scenes, target_count))
        groups = _chunk_evenly(beats, target_count)
        split_reason = "duration_balancing"

    scenes: list[dict[str, Any]] = []
    total_duration = 0.0
    for idx, group in enumerate(groups, start=1):
        roles = [str(beat.get("role", "build")) for beat in group]
        purpose = _select_scene_purpose(roles)
        narration_text = normalize_text(" ".join(str(beat.get("text", "")) for beat in group))
        duration = round(sum(estimate_duration_sec(str(beat.get("text", "")), scene_cfg) for beat in group), 2)
        total_duration += duration

        source_claim_ids: list[str] = []
        for beat in group:
            for claim_id in beat.get("source_claim_ids", []):
                if claim_id and claim_id not in source_claim_ids:
                    source_claim_ids.append(claim_id)

        scenes.append(
            {
                "scene_id": f"scene_{idx:03d}",
                "purpose": purpose,
                "narration_text": narration_text,
                "beat_ids": [str(beat.get("beat_id", f"beat_{idx:03d}")) for beat in group],
                "estimated_duration_sec": duration,
                "transition_hint": {
                    "in": "hard_cut" if idx == 1 else "match_cut",
                    "out": "whoosh" if idx < len(groups) else "fade_out",
                },
                "split_reason": split_reason,
                "source_claim_ids": source_claim_ids,
            }
        )

    total_duration = round(total_duration, 2)
    return {
        "candidate_id": candidate_id,
        "strategy": strategy,
        "scene_count": len(scenes),
        "total_estimated_duration_sec": total_duration,
        "scenes": scenes,
    }


def build_scene_candidates(beats: list[dict[str, Any]], scene_cfg: SceneConfig) -> list[dict[str, Any]]:
    strategies = [
        ("per_beat", "scene_cand_001"),
        ("hook_pair", "scene_cand_002"),
        ("balanced", "scene_cand_003"),
    ]
    return [
        compose_scene_skeletons(
            beats=beats,
            scene_cfg=scene_cfg,
            strategy=strategy,
            candidate_id=candidate_id,
        )
        for strategy, candidate_id in strategies
    ]


def score_scene_candidate(
    *,
    candidate: dict[str, Any],
    script_session_pack: dict[str, Any],
    scene_cfg: SceneConfig,
    score_spec: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, bool]]:
    scenes = list(candidate.get("scenes", []))
    scene_count = len(scenes)
    durations = [float(scene.get("estimated_duration_sec", 0.0)) for scene in scenes]
    total_duration = sum(durations)
    avg_duration = (total_duration / scene_count) if scene_count else 0.0
    purposes = [str(scene.get("purpose", "")) for scene in scenes]

    k1 = 45.0
    if scenes:
        first_scene = scenes[0]
        if first_scene.get("purpose") == "hook":
            k1 += 10
        if float(first_scene.get("estimated_duration_sec", scene_cfg.hook_deadline_sec + 1)) <= scene_cfg.hook_deadline_sec:
            k1 += 20
        else:
            k1 -= 6
    if scene_cfg.min_scene_sec <= avg_duration <= scene_cfg.max_scene_sec:
        k1 += 10
    if any(d < scene_cfg.min_scene_sec * 0.7 or d > scene_cfg.max_scene_sec * 1.3 for d in durations):
        k1 -= 12
    if scene_count >= 3:
        k1 += 5
    k1 = clamp(k1)

    k3 = 40.0
    unique_purposes = len(set(purposes))
    k3 += min(unique_purposes, 5) * 8
    if scene_cfg.target_duration_sec * 0.85 <= total_duration <= scene_cfg.target_duration_sec * 1.15:
        k3 += 15
    elif scene_cfg.target_duration_sec * 0.7 <= total_duration <= scene_cfg.target_duration_sec * 1.3:
        k3 += 8
    else:
        k3 -= 8
    if scene_cfg.min_scenes <= scene_count <= scene_cfg.max_scenes:
        k3 += 8
    if purposes and purposes[-1] in {"payoff", "loop"}:
        k3 += 6
    k3 = clamp(k3)

    k2 = 45.0
    expected = ["hook", "setup", "build", "payoff", "loop"]
    progress_idx = 0
    bonus_progress = 0.0
    for purpose in purposes:
        if progress_idx < len(expected) and purpose == expected[progress_idx]:
            progress_idx += 1
        elif progress_idx >= 3 and purpose in {"payoff", "loop"}:
            bonus_progress += 0.5
    progress = float(progress_idx) + bonus_progress
    k2 += progress * 8
    if purposes and purposes[0] == "hook":
        k2 += 5
    if len(set(purposes)) >= 4:
        k2 += 6
    k2 = clamp(k2)

    k4 = 55.0
    if scene_cfg.min_scenes <= scene_count <= scene_cfg.max_scenes:
        k4 += 20
    else:
        k4 -= 10
    if scene_count:
        in_range = sum(1 for d in durations if scene_cfg.min_scene_sec <= d <= scene_cfg.max_scene_sec)
        k4 += (in_range / scene_count) * 15
    if total_duration <= scene_cfg.target_duration_sec * 1.4:
        k4 += 6
    else:
        k4 -= 8
    k4 = clamp(k4)

    text_blob = " ".join(str(scene.get("narration_text", "")) for scene in scenes).lower()
    banned_tokens = [str(x).lower() for x in script_session_pack.get("banned_expressions", [])]
    unsafe_tokens = [
        "dismemberment",
        "torture porn",
        "snuff",
        "beheading",
        "graphic gore",
    ]
    hard_k5 = not any(token in text_blob for token in banned_tokens + unsafe_tokens)

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


def evaluate_and_select_scene_candidates(
    *,
    candidates: list[dict[str, Any]],
    script_session_pack: dict[str, Any],
    scene_cfg: SceneConfig,
    score_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    for candidate in candidates:
        total, criteria, hard = score_scene_candidate(
            candidate=candidate,
            script_session_pack=script_session_pack,
            scene_cfg=scene_cfg,
            score_spec=score_spec,
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "total_score": total,
                "criteria": criteria,
                "hard_constraints": hard,
                "hard_pass": all(hard.values()),
                "scene_count": candidate["scene_count"],
                "total_estimated_duration_sec": candidate["total_estimated_duration_sec"],
            }
        )

    passing = [row for row in rows if row["hard_pass"]]
    tie_break_order = score_spec.get("selection_rule", {}).get("tie_break_order", ["K3", "K1"])
    min_accept_score = float(score_spec.get("selection_rule", {}).get("min_accept_score", 0.0))

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple([row["total_score"]] + [row["criteria"].get(k, 0.0) for k in tie_break_order])

    selected_row = max(passing if passing else rows, key=rank_key)
    selected_candidate = next(
        candidate for candidate in candidates if candidate["candidate_id"] == selected_row["candidate_id"]
    )
    pass_threshold = selected_row["total_score"] >= min_accept_score and selected_row["hard_pass"]

    evaluation_report = {
        "schema_version": "1.0.0",
        "task_id": "scene_split",
        "created_at": now_utc_iso(),
        "candidate_evaluations": rows,
        "selected_candidate_id": selected_row["candidate_id"],
        "min_accept_score": min_accept_score,
        "pass_threshold_met": pass_threshold,
    }

    selection_trace = {
        "schema_version": "1.0.0",
        "task_id": "scene_split",
        "created_at": now_utc_iso(),
        "selected_candidate_id": selected_row["candidate_id"],
        "selected_score": selected_row["total_score"],
        "tie_break_order": tie_break_order,
        "selection_reason": "Highest weighted score under hard-constraint policy.",
    }

    selected = {
        "candidate": selected_candidate,
        "evaluation": selected_row,
        "pass_threshold_met": pass_threshold,
    }
    return selected, evaluation_report, selection_trace


def _location_ref(script_session_pack: dict[str, Any]) -> str:
    for entity in script_session_pack.get("character_fact_registry", []):
        if entity.get("entity_type") == "LOCATION":
            return str(entity.get("entity_id", "location_primary"))
    return "location_primary"


def _character_refs(script_session_pack: dict[str, Any]) -> list[str]:
    refs = []
    for entity in script_session_pack.get("character_fact_registry", []):
        if entity.get("entity_type") == "PERSON":
            refs.append(str(entity.get("entity_id", "person_unknown")))
    return refs[:3]


def _shot_spec_for_purpose(purpose: str) -> dict[str, str]:
    mapping = {
        "hook": {
            "shot_size": "CLOSEUP",
            "camera_angle": "LOW_ANGLE",
            "camera_motion": "HANDHELD",
        },
        "setup": {
            "shot_size": "WIDE",
            "camera_angle": "EYE_LEVEL",
            "camera_motion": "PAN",
        },
        "build": {
            "shot_size": "MEDIUM",
            "camera_angle": "EYE_LEVEL",
            "camera_motion": "TRACKING",
        },
        "payoff": {
            "shot_size": "CLOSEUP",
            "camera_angle": "HIGH_ANGLE",
            "camera_motion": "DOLLY",
        },
        "loop": {
            "shot_size": "EXTREME_CLOSEUP",
            "camera_angle": "DUTCH",
            "camera_motion": "STATIC",
        },
    }
    return mapping.get(purpose, mapping["build"])


def _action_emotion_for_purpose(purpose: str) -> dict[str, str]:
    mapping = {
        "hook": {"action": "abrupt reveal of contradiction", "emotion": "shock"},
        "setup": {"action": "establish people and place", "emotion": "unease"},
        "build": {"action": "connect clues and timeline", "emotion": "suspense"},
        "payoff": {"action": "confirm decisive evidence", "emotion": "grim clarity"},
        "loop": {"action": "echo opening line", "emotion": "curiosity"},
    }
    return mapping.get(purpose, mapping["build"])


def _time_weather_for_purpose(purpose: str) -> dict[str, str]:
    if purpose in {"hook", "payoff", "loop"}:
        return {"time_of_day": "NIGHT", "weather": "CLOUDY"}
    if purpose == "setup":
        return {"time_of_day": "AFTERNOON", "weather": "CLEAR"}
    return {"time_of_day": "SUNSET", "weather": "CLOUDY"}


def _visual_goal_for_purpose(purpose: str) -> str:
    mapping = {
        "hook": "Deliver immediate stop-scroll tension in first seconds.",
        "setup": "Establish the case context and principal characters.",
        "build": "Increase curiosity with evidence-driven progression.",
        "payoff": "Resolve contradiction with clear investigative framing.",
        "loop": "Create replay curiosity by mirroring the opening beat.",
    }
    return mapping.get(purpose, "Maintain narrative continuity.")


def build_scenes_v2(
    *,
    session_input: dict[str, Any],
    selected_candidate: dict[str, Any],
    script_session_pack: dict[str, Any],
    scene_cfg: SceneConfig,
) -> dict[str, Any]:
    location_ref = _location_ref(script_session_pack)
    character_refs = _character_refs(script_session_pack)
    banned = list(script_session_pack.get("banned_expressions", []))

    scenes_out = []
    for idx, scene in enumerate(selected_candidate.get("scenes", []), start=1):
        purpose = str(scene.get("purpose", "build"))
        shot_spec = _shot_spec_for_purpose(purpose)
        action_emotion = _action_emotion_for_purpose(purpose)
        time_weather = _time_weather_for_purpose(purpose)
        visual_goal = _visual_goal_for_purpose(purpose)

        narration_text = normalize_text(str(scene.get("narration_text", "")))
        beat_ids = list(scene.get("beat_ids", []))
        beat_id = beat_ids[0] if beat_ids else f"beat_{idx:03d}"
        source_refs = list(scene.get("source_claim_ids", []))

        image_prompt = normalize_text(
            f"{scene_cfg.style_preset}, {visual_goal} "
            f"Narration cue: {narration_text}. "
            f"Shot: {shot_spec['shot_size']} {shot_spec['camera_angle']}. "
            f"Location: {location_ref}."
        )
        video_prompt = normalize_text(
            f"Animate investigative sequence with {shot_spec['camera_motion']} camera motion. "
            f"Action: {action_emotion['action']}. Emotion: {action_emotion['emotion']}. "
            f"Keep continuity with previous scene and respect policy constraints."
        )

        scenes_out.append(
            {
                "scene_id": f"scene_{idx:03d}",
                "beat_id": beat_id,
                "duration_sec": round(float(scene.get("estimated_duration_sec", scene_cfg.min_scene_sec)), 2),
                "narration_text": narration_text,
                "visual_goal": visual_goal,
                "character_refs": character_refs,
                "location_ref": location_ref,
                "continuity_state": {
                    "scene_index": idx,
                    "purpose": purpose,
                },
                "shot_spec": shot_spec,
                "action_emotion": action_emotion,
                "time_weather": time_weather,
                "negative_constraints": banned + [
                    "no explicit gore visuals",
                    "avoid glorifying perpetrator",
                ],
                "visual_prompt": {
                    "image_prompt": image_prompt,
                    "video_prompt": video_prompt,
                },
                "source_refs": source_refs,
            }
        )

    if not scenes_out:
        raise ContractError("selected scene candidate produced no scenes")

    return {
        "schema_version": "2.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "case_id": str(script_session_pack.get("case_id", f"case_{session_input['session_id']}")),
        "style_preset": scene_cfg.style_preset,
        "global_negative_constraints": banned,
        "scenes": scenes_out,
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    script_session_pack_path = Path(args.script_session_pack).resolve()
    scene_config_path = Path(args.scene_config).resolve()
    model_map_path = Path(args.model_map).resolve()
    score_spec_cli_path = Path(args.score_spec).resolve()

    # Gate-In
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
            (
                "script_session_pack",
                script_session_pack_path,
                root / "schemas" / "script_session_pack.schema.json",
            ),
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l6-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    agent_runtime_infos = load_agent_runtime_infos(model_map_path)
    scene_cfg = load_scene_config(scene_config_path)

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    script_session_pack = read_json(script_session_pack_path)
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
        script_session_pack=script_session_pack,
        allow_cross_session=bool(args.allow_cross_session),
    )

    if str(script_session_pack.get("lock_policy", "")).strip() != LOCK_POLICY:
        raise ContractError("script_session_pack.lock_policy mismatch")
    upstream_quality_status = str(script_session_pack.get("quality_status", "")).strip()
    if upstream_quality_status not in {"READY_FOR_SCENE_PLANNING", "NEEDS_SCRIPT_REWORK"}:
        raise ContractError(
            "script_session_pack.quality_status must be one of: "
            "READY_FOR_SCENE_PLANNING, NEEDS_SCRIPT_REWORK"
        )

    story_bible = build_story_bible(
        session_input=session_input,
        script_session_pack=script_session_pack,
        scene_cfg=scene_cfg,
    )
    beats = build_beats(
        session_input=session_input,
        script_session_pack=script_session_pack,
    )

    candidates = build_scene_candidates(beats, scene_cfg)
    selected, evaluation_report, selection_trace = evaluate_and_select_scene_candidates(
        candidates=candidates,
        script_session_pack=script_session_pack,
        scene_cfg=scene_cfg,
        score_spec=score_spec,
    )

    scene_skeletons = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "selected_candidate_id": selected["candidate"]["candidate_id"],
        "strategy": selected["candidate"]["strategy"],
        "scene_count": selected["candidate"]["scene_count"],
        "total_estimated_duration_sec": selected["candidate"]["total_estimated_duration_sec"],
        "scenes": selected["candidate"]["scenes"],
    }

    scenes_v2 = build_scenes_v2(
        session_input=session_input,
        selected_candidate=selected["candidate"],
        script_session_pack=script_session_pack,
        scene_cfg=scene_cfg,
    )

    write_json(out_dir / "story_bible.json", story_bible)
    write_json(out_dir / "beats.json", build_beats_doc(session_input=session_input, beats=beats))
    write_json(
        out_dir / "candidates_scene_split.json",
        {
            "schema_version": "1.0.0",
            "task_id": "scene_split",
            "created_at": now_utc_iso(),
            "candidates": candidates,
        },
    )
    write_json(out_dir / "evaluation_report_scene_split.json", evaluation_report)
    write_json(out_dir / "selection_trace_scene_split.json", selection_trace)
    write_json(out_dir / "scene_skeletons.json", scene_skeletons)
    write_json(out_dir / "scenes_v2.json", scenes_v2)

    runtime_log = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "pipeline_stage": "L6 Scene Planning",
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
            "script_session_pack": str(script_session_pack_path),
            "score_spec_cli": str(score_spec_cli_path),
            "score_spec": str(effective_score_spec_path),
            "objective_score_spec_ref": objective_score_spec_ref,
            "objective_score_spec_path": str(objective_score_spec_path),
            "scene_config": str(scene_config_path),
            "model_map": str(model_map_path),
        },
        "scene_constraints": {
            "target_duration_sec": scene_cfg.target_duration_sec,
            "min_scene_sec": scene_cfg.min_scene_sec,
            "max_scene_sec": scene_cfg.max_scene_sec,
            "min_scenes": scene_cfg.min_scenes,
            "max_scenes": scene_cfg.max_scenes,
            "hook_deadline_sec": scene_cfg.hook_deadline_sec,
        },
        "policy": {
            "lock_priority": LOCK_POLICY,
            "model_update_mode": "manual_only",
            "allow_cross_session": bool(args.allow_cross_session),
            "pipeline_env": str(args.pipeline_env),
            "allow_score_spec_override": bool(args.allow_score_spec_override),
            "score_spec_override_used": score_spec_override_used,
            "score_spec_override_reason": score_spec_override_reason,
            "upstream_quality_status": upstream_quality_status,
            "manual_rework_recommended": upstream_quality_status == "NEEDS_SCRIPT_REWORK",
        },
        "outputs": {
            "out_dir": str(out_dir),
        },
    }
    write_json(out_dir / "runtime_log_l6.json", runtime_log)

    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "story_bible",
                "path": str((out_dir / "story_bible.json").resolve()),
            },
            {
                "name": "beats",
                "path": str((out_dir / "beats.json").resolve()),
            },
            {
                "name": "scene_skeletons",
                "path": str((out_dir / "scene_skeletons.json").resolve()),
            },
            {
                "name": "scenes_v2",
                "path": str((out_dir / "scenes_v2.json").resolve()),
            },
            {
                "name": "evaluation_report_scene_split",
                "path": str((out_dir / "evaluation_report_scene_split.json").resolve()),
            },
            {
                "name": "selection_trace_scene_split",
                "path": str((out_dir / "selection_trace_scene_split.json").resolve()),
            },
            {
                "name": "runtime_log_l6",
                "path": str((out_dir / "runtime_log_l6.json").resolve()),
            },
        ],
        "metrics": {
            "beat_count": len(beats),
            "scene_candidate_count": len(candidates),
            "selected_candidate_id": selected["candidate"]["candidate_id"],
            "selected_scene_count": selected["candidate"]["scene_count"],
            "selected_total_estimated_duration_sec": selected["candidate"][
                "total_estimated_duration_sec"
            ],
            "pass_threshold_met": selected["pass_threshold_met"],
        },
    }
    write_json(out_dir / "session_output.json", session_output)

    # Gate-Out
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
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l6-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    print(f"[l6-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L6 Scene Planning with score-based scene split selection."
    )
    parser.add_argument("--session-input", required=True, help="Path to session_input.json")
    parser.add_argument(
        "--session-objective",
        required=True,
        help="Path to session_objective.json",
    )
    parser.add_argument(
        "--script-session-pack",
        required=True,
        help="Path to script_session_pack.json",
    )
    parser.add_argument(
        "--scene-config",
        default=str(root / "config" / "shorts.yaml"),
        help="Path to shorts.yaml",
    )
    parser.add_argument(
        "--model-map",
        default=str(root / "config" / "agent_model_map.yaml"),
        help="Path to agent_model_map.yaml",
    )
    parser.add_argument(
        "--score-spec",
        default=str(root / "score_specs" / "score_spec_scene_split.json"),
        help="Path to score_spec_scene_split.json",
    )
    parser.add_argument(
        "--allow-cross-session",
        action="store_true",
        help=(
            "Allow script_session_pack session/objective mismatch for recovery reruns. "
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
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ConfigError, SceneConfigError, ContractError) as err:
        print(f"[l6-runner] config/contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l6-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
