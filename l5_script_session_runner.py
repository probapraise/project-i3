#!/usr/bin/env python3
"""L5 Script Session runner (ideation + selection + translation + narration)."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, read_yaml, write_gate_manifest, write_json


LOCK_POLICY = "REAL > INFERRED > NO_INFO"
SOURCE_PRIORITY = {
    "REAL": 3,
    "INFERRED": 2,
    "NO_INFO": 1,
}


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
    """Raised when cross-artifact session contracts are violated."""


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
        "ScriptIdeator",
        "ScriptEvaluator",
        "ScriptSelector",
        "TranslationWriter",
        "TranslationCritic",
        "TranslationReconstructor",
        "NarrationExtractor",
    }
    missing = required_agents.difference(infos)
    if missing:
        raise ConfigError(f"missing required agents: {sorted(missing)}")

    return infos


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    topic_research_bundle: dict[str, Any],
    entity_profile_pack: dict[str, Any],
    script_seed_pack: dict[str, Any],
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

    bundle_session_id = str(topic_research_bundle.get("session_id", "")).strip()
    bundle_objective_id = str(topic_research_bundle.get("objective_id", "")).strip()
    if bundle_session_id != input_session_id:
        raise ContractError(
            "topic_research_bundle.session_id must match session_input.session_id"
        )
    if bundle_objective_id != input_objective_id:
        raise ContractError(
            "topic_research_bundle.objective_id must match session_input.objective_id"
        )

    profile_session_id = str(entity_profile_pack.get("session_id", "")).strip()
    profile_objective_id = str(entity_profile_pack.get("objective_id", "")).strip()
    if profile_session_id != input_session_id:
        raise ContractError(
            "entity_profile_pack.session_id must match session_input.session_id"
        )
    if profile_objective_id != input_objective_id:
        raise ContractError(
            "entity_profile_pack.objective_id must match session_input.objective_id"
        )

    if allow_cross_session:
        return

    seed_session_id = str(script_seed_pack.get("session_id", "")).strip()
    seed_objective_id = str(script_seed_pack.get("objective_id", "")).strip()
    if seed_session_id != input_session_id:
        raise ContractError(
            "script_seed_pack.session_id must match session_input.session_id "
            "(use --allow-cross-session for recovery reruns)"
        )
    if seed_objective_id != input_objective_id:
        raise ContractError(
            "script_seed_pack.objective_id must match session_input.objective_id "
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


def _field_rank(field: dict[str, Any]) -> tuple[int, float, int, int]:
    source_tag = str(field.get("source_tag", "")).upper()
    if source_tag not in SOURCE_PRIORITY:
        raise ValueError(f"unsupported source_tag: {source_tag}")

    confidence = -1.0
    if source_tag == "INFERRED" and isinstance(field.get("confidence"), (int, float)):
        confidence = float(field["confidence"])

    sources = field.get("sources")
    source_count = len(sources) if isinstance(sources, list) else 0
    has_value = 1 if field.get("value") is not None else 0

    return (SOURCE_PRIORITY[source_tag], confidence, source_count, has_value)


def _prefer_incoming(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    ex_rank = _field_rank(existing)
    in_rank = _field_rank(incoming)
    if in_rank[0] != ex_rank[0]:
        return in_rank[0] > ex_rank[0]
    if in_rank[1] != ex_rank[1]:
        return in_rank[1] > ex_rank[1]
    if in_rank[2] != ex_rank[2]:
        return in_rank[2] > ex_rank[2]
    return in_rank[3] > ex_rank[3]


def select_locked_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for field in fields:
        name = str(field.get("field", ""))
        if not name:
            continue
        current = by_name.get(name)
        if current is None or _prefer_incoming(current, field):
            by_name[name] = dict(field)

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        priority = SOURCE_PRIORITY.get(str(item.get("source_tag", "")), 0)
        return (-priority, str(item.get("field", "")))

    return sorted(by_name.values(), key=sort_key)


def build_character_fact_registry(entity_profile_pack: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for entity in entity_profile_pack.get("entities", []):
        selected_fields = select_locked_fields(list(entity.get("fields", [])))
        if not selected_fields:
            continue

        normalized_fields = []
        for field in selected_fields:
            row = {
                "field": field["field"],
                "value": field.get("value"),
                "source_tag": field.get("source_tag"),
            }
            if field.get("source_tag") == "REAL" and isinstance(field.get("sources"), list):
                row["sources"] = field["sources"]
            if field.get("source_tag") == "INFERRED":
                row["confidence"] = round(float(field.get("confidence", 0.0)), 2)
                row["reason"] = str(field.get("reason", "")).strip() or "inference"
            normalized_fields.append(row)

        out.append(
            {
                "entity_id": entity.get("entity_id", "unknown_entity"),
                "entity_type": entity.get("entity_type", "OBJECT"),
                "role_label": entity.get("role_label", "unknown"),
                "display_name": entity.get("display_name", entity.get("role_label", "unknown")),
                "fields": normalized_fields,
            }
        )

    return out


def _find_fact(
    registry: list[dict[str, Any]],
    *,
    entity_id: str,
    field_name: str,
) -> dict[str, Any] | None:
    for entity in registry:
        if entity.get("entity_id") != entity_id:
            continue
        for field in entity.get("fields", []):
            if field.get("field") == field_name:
                return field
    return None


def extract_story_context(
    *,
    topic_research_bundle: dict[str, Any],
    script_seed_pack: dict[str, Any],
    fact_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    claims = topic_research_bundle.get("claims", [])
    claim_snippets = []
    for claim in claims[:4]:
        text = normalize_text(str(claim.get("claim_text", "")))
        if not text:
            continue
        claim_snippets.append(
            {
                "claim_id": claim.get("claim_id", "unknown_claim"),
                "claim_text": text,
            }
        )

    selected_hook = script_seed_pack.get("selected_hook", {})
    narrative_seed = script_seed_pack.get("narrative_seed", {})

    victim_occupation = _find_fact(
        fact_registry,
        entity_id="person_victim",
        field_name="occupation",
    )
    location_country = _find_fact(
        fact_registry,
        entity_id="location_primary",
        field_name="country",
    )

    return {
        "hook_text": normalize_text(str(selected_hook.get("hook_text", "")))
        or "One overlooked detail changed the entire case.",
        "setup": normalize_text(str(narrative_seed.get("setup", "")))
        or "The first report looked routine.",
        "twist": normalize_text(str(narrative_seed.get("twist", "")))
        or "Then a contradiction appeared.",
        "resolution": normalize_text(str(narrative_seed.get("resolution", "")))
        or "Verified records clarified what happened.",
        "loop_line": normalize_text(str(narrative_seed.get("loop_line", "")))
        or "Watch the first line again with this in mind.",
        "topic_query": normalize_text(str(topic_research_bundle.get("topic_query", ""))),
        "claim_snippets": claim_snippets,
        "victim_occupation": victim_occupation.get("value")
        if isinstance(victim_occupation, dict)
        else None,
        "location_country": location_country.get("value")
        if isinstance(location_country, dict)
        else None,
        "trigger_words": list(script_seed_pack.get("trigger_words", [])),
        "banned_expressions": list(script_seed_pack.get("banned_expressions", [])),
    }


def _segment(
    *,
    segment_id: str,
    phase: str,
    text: str,
    source_claim_ids: list[str],
) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "phase": phase,
        "narration_en": normalize_text(text),
        "source_claim_ids": source_claim_ids,
    }


def _claim_ids(context: dict[str, Any]) -> list[str]:
    return [row["claim_id"] for row in context.get("claim_snippets", []) if row.get("claim_id")]


def build_script_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    claim_ids = _claim_ids(context)
    lead_claim = context.get("claim_snippets", [{}])
    lead_claim_text = (
        lead_claim[0].get("claim_text") if lead_claim else "Investigators started from a simple statement."
    )
    occupation = context.get("victim_occupation") or "worker"
    country = context.get("location_country") or "the city"
    topic = context.get("topic_query") or "the case"

    candidates: list[dict[str, Any]] = []

    chronological_segments = [
        _segment(
            segment_id="seg_001",
            phase="hook",
            text=context["hook_text"],
            source_claim_ids=claim_ids[:1],
        ),
        _segment(
            segment_id="seg_002",
            phase="setup",
            text=f"In {country}, reports said the victim was a {occupation} and the story looked straightforward.",
            source_claim_ids=claim_ids[:2],
        ),
        _segment(
            segment_id="seg_003",
            phase="build",
            text=f"But one record did not match the timeline: {lead_claim_text}",
            source_claim_ids=claim_ids[:2],
        ),
        _segment(
            segment_id="seg_004",
            phase="payoff",
            text=context["resolution"],
            source_claim_ids=claim_ids[:3],
        ),
        _segment(
            segment_id="seg_005",
            phase="loop",
            text=context["loop_line"],
            source_claim_ids=claim_ids[:1],
        ),
    ]
    candidates.append(
        {
            "script_id": "scr_001",
            "variant": "chronological_reveal",
            "segments": chronological_segments,
        }
    )

    question_segments = [
        _segment(
            segment_id="seg_001",
            phase="hook",
            text=f"What if the first version of {topic} was incomplete?",
            source_claim_ids=claim_ids[:1],
        ),
        _segment(
            segment_id="seg_002",
            phase="setup",
            text=context["setup"],
            source_claim_ids=claim_ids[:1],
        ),
        _segment(
            segment_id="seg_003",
            phase="build",
            text=f"Witness notes and source checks in {country} kept pointing to the same contradiction.",
            source_claim_ids=claim_ids[:3],
        ),
        _segment(
            segment_id="seg_004",
            phase="payoff",
            text=context["twist"],
            source_claim_ids=claim_ids[:2],
        ),
        _segment(
            segment_id="seg_005",
            phase="loop",
            text="Now replay the opening question and see what changed.",
            source_claim_ids=claim_ids[:1],
        ),
    ]
    candidates.append(
        {
            "script_id": "scr_002",
            "variant": "question_first",
            "segments": question_segments,
        }
    )

    documentary_segments = [
        _segment(
            segment_id="seg_001",
            phase="hook",
            text="Three documents, one victim, and a timeline that refused to stay still.",
            source_claim_ids=claim_ids[:1],
        ),
        _segment(
            segment_id="seg_002",
            phase="setup",
            text=f"The case centered on a {occupation}, and investigators thought they already knew the ending.",
            source_claim_ids=claim_ids[:2],
        ),
        _segment(
            segment_id="seg_003",
            phase="build",
            text=f"Then source cross-checking exposed gaps between statements and records in {country}.",
            source_claim_ids=claim_ids[:3],
        ),
        _segment(
            segment_id="seg_004",
            phase="payoff",
            text="The final record did not support the first public narrative.",
            source_claim_ids=claim_ids[:3],
        ),
        _segment(
            segment_id="seg_005",
            phase="loop",
            text=context["loop_line"],
            source_claim_ids=claim_ids[:1],
        ),
    ]
    candidates.append(
        {
            "script_id": "scr_003",
            "variant": "documentary_pulse",
            "segments": documentary_segments,
        }
    )

    for candidate in candidates:
        segments = candidate.get("segments", [])
        script_text = " ".join(seg["narration_en"] for seg in segments)
        candidate["script_text_en"] = normalize_text(script_text)
        source_claim_ids: list[str] = []
        for seg in segments:
            for claim_id in seg["source_claim_ids"]:
                if claim_id and claim_id not in source_claim_ids:
                    source_claim_ids.append(claim_id)
        candidate["source_claim_ids"] = source_claim_ids

    return candidates


def score_script_candidate(
    *,
    candidate: dict[str, Any],
    script_seed_pack: dict[str, Any],
    score_spec: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, bool]]:
    text = candidate["script_text_en"]
    lower = text.lower()
    words = [w for w in re.split(r"\W+", lower) if w]
    segment_count = len(candidate.get("segments", []))

    k1_tokens = ["what if", "but", "then", "changed", "record", "timeline", "question"]
    k3_tokens = ["then", "now", "again", "final", "watch", "replay", "loop"]

    k1: float = 48 + sum(7 for token in k1_tokens if token in lower)
    if candidate.get("segments"):
        hook = str(candidate["segments"][0].get("narration_en", ""))
        if "?" in hook:
            k1 += 8
    if 70 <= len(words) <= 140:
        k1 += 10
    elif len(words) > 170:
        k1 -= 8
    k1 = clamp(k1)

    k3: float = 45 + sum(6 for token in k3_tokens if token in lower)
    if 4 <= segment_count <= 6:
        k3 += 10
    if str(candidate.get("variant", "")).startswith("question"):
        k3 += 6
    k3 = clamp(k3)

    phase_set = {str(seg.get("phase", "")) for seg in candidate.get("segments", [])}
    k2: float = 50
    for required_phase in ["hook", "setup", "build", "payoff", "loop"]:
        if required_phase in phase_set:
            k2 += 8
    if "contradiction" in lower or "record" in lower:
        k2 += 6
    k2 = clamp(k2)

    avg_words_per_segment = (len(words) / segment_count) if segment_count else 0
    k4: float = 70
    if 9 <= avg_words_per_segment <= 22:
        k4 += 8
    if "graphic" in lower or "gore" in lower:
        k4 -= 30
    k4 = clamp(k4)

    banned = [str(x).lower() for x in script_seed_pack.get("banned_expressions", [])]
    unsafe = [
        "dismemberment",
        "torture porn",
        "snuff",
        "beheading",
    ]
    hard_k5 = not any(token in lower for token in banned + unsafe)

    criterion_scores = {
        "K1": round(k1, 2),
        "K3": round(k3, 2),
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


def evaluate_and_select_scripts(
    *,
    candidates: list[dict[str, Any]],
    script_seed_pack: dict[str, Any],
    score_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    for cand in candidates:
        total, criteria, hard = score_script_candidate(
            candidate=cand,
            script_seed_pack=script_seed_pack,
            score_spec=score_spec,
        )
        rows.append(
            {
                "script_id": cand["script_id"],
                "total_score": total,
                "criteria": criteria,
                "hard_constraints": hard,
                "hard_pass": all(hard.values()),
            }
        )

    passing = [r for r in rows if r["hard_pass"]]
    tie_break_order = score_spec.get("selection_rule", {}).get(
        "tie_break_order", ["K1", "K3"]
    )
    min_accept_score = float(score_spec.get("selection_rule", {}).get("min_accept_score", 0))

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple([row["total_score"]] + [row["criteria"].get(k, 0.0) for k in tie_break_order])

    selected_row = max(passing if passing else rows, key=rank_key)
    selected_cand = next(c for c in candidates if c["script_id"] == selected_row["script_id"])
    pass_threshold = selected_row["total_score"] >= min_accept_score and selected_row["hard_pass"]

    evaluation_report = {
        "schema_version": "1.0.0",
        "task_id": "script_selection",
        "created_at": now_utc_iso(),
        "candidate_evaluations": rows,
        "selected_script_id": selected_row["script_id"],
        "min_accept_score": min_accept_score,
        "pass_threshold_met": pass_threshold,
    }

    selection_trace = {
        "schema_version": "1.0.0",
        "task_id": "script_selection",
        "created_at": now_utc_iso(),
        "selected_script_id": selected_row["script_id"],
        "selected_score": selected_row["total_score"],
        "tie_break_order": tie_break_order,
        "selection_reason": "Highest weighted score under hard-constraint policy.",
    }

    selected = {
        "candidate": selected_cand,
        "evaluation": selected_row,
        "pass_threshold_met": pass_threshold,
    }
    return selected, evaluation_report, selection_trace


def rough_translate_en_to_pt(text: str) -> str:
    out = " " + normalize_text(text).lower() + " "
    replacements = [
        (" what if ", " e se "),
        (" case ", " caso "),
        (" report ", " relatorio "),
        (" victim ", " vitima "),
        (" timeline ", " linha do tempo "),
        (" record ", " registro "),
        (" investigators ", " investigadores "),
        (" final ", " final "),
        (" watch ", " veja "),
        (" now ", " agora "),
    ]
    for before, after in replacements:
        out = out.replace(before, after)
    out = normalize_text(out)
    if out and out[-1] not in ".!?":
        out += "."
    return out[:1].upper() + out[1:]


def rough_translate_pt_to_ko(text: str) -> str:
    return f"[KO] {normalize_text(text)}"


def build_translation_outputs(
    *,
    selected_script: dict[str, Any],
    pass_threshold_met: bool,
) -> dict[str, Any]:
    structured_doc = []
    for seg in selected_script.get("segments", []):
        narration_pt = rough_translate_en_to_pt(str(seg.get("narration_en", "")))
        narration_ko = rough_translate_pt_to_ko(narration_pt)
        structured_doc.append(
            {
                "segment_id": seg.get("segment_id", "unknown_segment"),
                "phase": seg.get("phase", "unknown"),
                "narration_pt": narration_pt,
                "narration_ko": narration_ko,
                "source_claim_ids": list(seg.get("source_claim_ids", [])),
            }
        )

    script_refined_pt = "\n".join(f"{idx + 1}. {row['narration_pt']}" for idx, row in enumerate(structured_doc))
    pure_narration_pt = " ".join(row["narration_pt"] for row in structured_doc)
    korean_translation = "\n".join(
        f"{idx + 1}. {row['narration_ko']}" for idx, row in enumerate(structured_doc)
    )

    return {
        "script_refined_pt": script_refined_pt,
        "pure_narration_pt": pure_narration_pt,
        "korean_translation": korean_translation,
        "structured_doc": structured_doc,
        "reconstruction_status": "AUTO_ACCEPTED"
        if pass_threshold_met
        else "NEEDS_HITL_REVIEW",
    }


def build_script_session_pack(
    *,
    session_input: dict[str, Any],
    entity_profile_pack: dict[str, Any],
    script_seed_pack: dict[str, Any],
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    translation_outputs: dict[str, Any],
    character_fact_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_script = selected["candidate"]
    selected_eval = selected["evaluation"]

    script_bank = []
    for cand in candidates:
        script_bank.append(
            {
                "script_id": cand["script_id"],
                "variant": cand["variant"],
                "script_text_en": cand["script_text_en"],
                "source_claim_ids": cand["source_claim_ids"],
            }
        )

    case_id = str(entity_profile_pack.get("case_id") or f"case_{session_input['session_id']}")

    return {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "case_id": case_id,
        "selected_script": {
            "script_id": selected_script["script_id"],
            "variant": selected_script["variant"],
            "script_text_en": selected_script["script_text_en"],
            "script_text_pt": translation_outputs["script_refined_pt"],
            "total_score": selected_eval["total_score"],
            "criteria": selected_eval["criteria"],
            "hard_constraints": selected_eval["hard_constraints"],
            "hard_pass": selected_eval["hard_pass"],
        },
        "script_bank": script_bank,
        "translation_outputs": translation_outputs,
        "narration_only_pt": translation_outputs["pure_narration_pt"],
        "trigger_words": list(script_seed_pack.get("trigger_words", []))[:12],
        "banned_expressions": list(script_seed_pack.get("banned_expressions", [])),
        "policy_notes": list(script_seed_pack.get("policy_notes", [])),
        "character_fact_registry": character_fact_registry,
        "lock_policy": LOCK_POLICY,
        "quality_status": "READY_FOR_SCENE_PLANNING"
        if selected["pass_threshold_met"]
        else "NEEDS_SCRIPT_REWORK",
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    topic_research_bundle_path = Path(args.topic_research_bundle).resolve()
    entity_profile_pack_path = Path(args.entity_profile_pack).resolve()
    script_seed_pack_path = Path(args.script_seed_pack).resolve()
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
                "topic_research_bundle",
                topic_research_bundle_path,
                root / "schemas" / "topic_research_bundle.schema.json",
            ),
            (
                "entity_profile_pack",
                entity_profile_pack_path,
                root / "schemas" / "entity_profile_pack.schema.json",
            ),
            (
                "script_seed_pack",
                script_seed_pack_path,
                root / "schemas" / "script_seed_pack.schema.json",
            ),
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l5-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    agent_runtime_infos = load_agent_runtime_infos(model_map_path)

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    topic_research_bundle = read_json(topic_research_bundle_path)
    entity_profile_pack = read_json(entity_profile_pack_path)
    script_seed_pack = read_json(script_seed_pack_path)
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
        topic_research_bundle=topic_research_bundle,
        entity_profile_pack=entity_profile_pack,
        script_seed_pack=script_seed_pack,
        allow_cross_session=bool(args.allow_cross_session),
    )

    character_fact_registry = build_character_fact_registry(entity_profile_pack)
    context = extract_story_context(
        topic_research_bundle=topic_research_bundle,
        script_seed_pack=script_seed_pack,
        fact_registry=character_fact_registry,
    )

    candidates = build_script_candidates(context)
    selected, evaluation_report, selection_trace = evaluate_and_select_scripts(
        candidates=candidates,
        script_seed_pack=script_seed_pack,
        score_spec=score_spec,
    )

    translation_outputs = build_translation_outputs(
        selected_script=selected["candidate"],
        pass_threshold_met=selected["pass_threshold_met"],
    )

    script_session_pack = build_script_session_pack(
        session_input=session_input,
        entity_profile_pack=entity_profile_pack,
        script_seed_pack=script_seed_pack,
        selected=selected,
        candidates=candidates,
        translation_outputs=translation_outputs,
        character_fact_registry=character_fact_registry,
    )

    candidates_doc = {
        "schema_version": "1.0.0",
        "task_id": "script_selection",
        "created_at": now_utc_iso(),
        "candidates": candidates,
    }

    write_json(out_dir / "candidates_script.json", candidates_doc)
    write_json(out_dir / "evaluation_report_script.json", evaluation_report)
    write_json(out_dir / "selection_trace_script.json", selection_trace)
    write_json(out_dir / "script_session_pack.json", script_session_pack)

    runtime_log = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "pipeline_stage": "L5 Script Session",
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
            "topic_research_bundle": str(topic_research_bundle_path),
            "entity_profile_pack": str(entity_profile_pack_path),
            "script_seed_pack": str(script_seed_pack_path),
            "score_spec_cli": str(score_spec_cli_path),
            "score_spec": str(effective_score_spec_path),
            "objective_score_spec_ref": objective_score_spec_ref,
            "objective_score_spec_path": str(objective_score_spec_path),
            "model_map": str(model_map_path),
        },
        "outputs": {
            "out_dir": str(out_dir),
        },
        "policy": {
            "lock_priority": LOCK_POLICY,
            "model_update_mode": "manual_only",
            "allow_cross_session": bool(args.allow_cross_session),
            "pipeline_env": str(args.pipeline_env),
            "allow_score_spec_override": bool(args.allow_score_spec_override),
            "score_spec_override_used": score_spec_override_used,
            "score_spec_override_reason": score_spec_override_reason,
        },
    }
    write_json(out_dir / "runtime_log_l5.json", runtime_log)

    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "candidates_script",
                "path": str((out_dir / "candidates_script.json").resolve()),
            },
            {
                "name": "evaluation_report_script",
                "path": str((out_dir / "evaluation_report_script.json").resolve()),
            },
            {
                "name": "selection_trace_script",
                "path": str((out_dir / "selection_trace_script.json").resolve()),
            },
            {
                "name": "script_session_pack",
                "path": str((out_dir / "script_session_pack.json").resolve()),
            },
            {
                "name": "runtime_log_l5",
                "path": str((out_dir / "runtime_log_l5.json").resolve()),
            },
        ],
        "metrics": {
            "script_candidate_count": len(candidates),
            "selected_script_id": selected["candidate"]["script_id"],
            "selected_script_score": selected["evaluation"]["total_score"],
            "pass_threshold_met": selected["pass_threshold_met"],
            "narration_segment_count": len(translation_outputs["structured_doc"]),
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
                "script_session_pack",
                out_dir / "script_session_pack.json",
                root / "schemas" / "script_session_pack.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l5-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    print(f"[l5-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L5 Script Session with score-based selection and translation outputs."
    )
    parser.add_argument("--session-input", required=True, help="Path to session_input.json")
    parser.add_argument(
        "--session-objective",
        required=True,
        help="Path to session_objective.json",
    )
    parser.add_argument(
        "--topic-research-bundle",
        required=True,
        help="Path to topic_research_bundle.json",
    )
    parser.add_argument(
        "--entity-profile-pack",
        required=True,
        help="Path to entity_profile_pack.json",
    )
    parser.add_argument(
        "--script-seed-pack",
        required=True,
        help="Path to script_seed_pack.json",
    )
    parser.add_argument(
        "--model-map",
        default=str(root / "config" / "agent_model_map.yaml"),
        help="Path to agent_model_map.yaml",
    )
    parser.add_argument(
        "--score-spec",
        default=str(root / "score_specs" / "score_spec_script.json"),
        help="Path to score_spec_script.json",
    )
    parser.add_argument(
        "--allow-cross-session",
        action="store_true",
        help=(
            "Allow script_seed_pack session/objective mismatch for recovery reruns. "
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
    except (ConfigError, ContractError) as err:
        print(f"[l5-runner] config/contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l5-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
