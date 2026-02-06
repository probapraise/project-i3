#!/usr/bin/env python3
"""L2/L3 runner: evidence structuring + inference completion with lock rules."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from profile_lock_rules import merge_field_sets
from runner_common import now_utc_iso, read_json, read_yaml, write_gate_manifest, write_json


class InferenceRulesError(Exception):
    """Raised when inference rules config is invalid."""


class ContractError(Exception):
    """Raised when cross-artifact session contracts are violated."""


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    topic_research_bundle: dict[str, Any],
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def detect_country(topic_query: str, claims: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    blob = normalize_text(topic_query + " " + " ".join(c.get("claim_text", "") for c in claims))
    if "brazil" in blob or "brasil" in blob:
        return "Brazil", ["topic_query"]
    if "mexico" in blob or "mexico city" in blob:
        return "Mexico", ["topic_query"]
    if "argentina" in blob:
        return "Argentina", ["topic_query"]
    return None, []


def extract_occupation(claims: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    patterns = [
        r"worked as (an? )?([a-z][a-z ]{1,60})",
        r"was (an? )?([a-z][a-z ]{1,60})",
        r"is (an? )?([a-z][a-z ]{1,60})",
    ]
    for claim in claims:
        text = normalize_text(str(claim.get("claim_text", "")))
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                occupation = match.group(2).strip(" .")
                if occupation:
                    return occupation, list(claim.get("supporting_source_ids", []))
    return None, []


def _parse_age_value(raw: str) -> int | None:
    text = raw.strip()
    if not text.isdigit():
        return None
    value = int(text)
    if not (1 <= value <= 120):
        return None
    return value


def _parse_decade_value(raw: str) -> int | None:
    text = raw.strip()
    if not text.isdigit():
        return None
    value = int(text)
    if value % 10 != 0:
        return None
    if not (10 <= value <= 90):
        return None
    return value


def extract_age(claims: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    exact_age_patterns = [
        r"\b(\d{1,3})-year-old\b",
        r"\b(\d{1,3}) years old\b",
        r"\baged (\d{1,3})\b",
    ]
    decade_patterns = [
        r"\bin (his|her|their) (\d{2})['’]?s\b",
    ]
    for claim in claims:
        text = normalize_text(str(claim.get("claim_text", "")))
        for pat in exact_age_patterns:
            match = re.search(pat, text)
            if match:
                age_value = _parse_age_value(match.group(1))
                if age_value is None:
                    continue
                return str(age_value), list(claim.get("supporting_source_ids", []))
        for pat in decade_patterns:
            match = re.search(pat, text)
            if match:
                decade_value = _parse_decade_value(match.group(2))
                if decade_value is None:
                    continue
                return f"{decade_value}s", list(claim.get("supporting_source_ids", []))
    return None, []


def extract_gender(claims: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    tokens = [
        (" male ", "male"),
        (" female ", "female"),
        (" man ", "male"),
        (" woman ", "female"),
        (" husband ", "male"),
        (" wife ", "female"),
    ]
    for claim in claims:
        text = " " + normalize_text(str(claim.get("claim_text", ""))) + " "
        for token, value in tokens:
            if token in text:
                return value, list(claim.get("supporting_source_ids", []))
    return None, []


def field_real(name: str, value: Any, sources: list[str]) -> dict[str, Any]:
    return {
        "field": name,
        "value": value,
        "source_tag": "REAL",
        "sources": sources if sources else ["unknown_source"],
    }


def field_no_info(name: str) -> dict[str, Any]:
    return {
        "field": name,
        "value": None,
        "source_tag": "NO_INFO",
    }


def build_evidence_person_entity(
    *,
    entity_id: str,
    role_label: str,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    occupation, occupation_sources = extract_occupation(claims)
    age, age_sources = extract_age(claims)
    gender, gender_sources = extract_gender(claims)

    fields = []
    fields.append(
        field_real("occupation", occupation, occupation_sources)
        if occupation
        else field_no_info("occupation")
    )
    fields.append(
        field_real("estimated_age", age, age_sources) if age else field_no_info("estimated_age")
    )
    fields.append(
        field_real("gender", gender, gender_sources) if gender else field_no_info("gender")
    )
    fields.extend(
        [
            field_no_info("ethnicity"),
            field_no_info("hair_style"),
            field_no_info("hair_color"),
            field_no_info("body_type"),
            field_no_info("outfit_signature"),
            field_no_info("face_signature"),
        ]
    )

    return {
        "entity_id": entity_id,
        "entity_type": "PERSON",
        "role_label": role_label,
        "display_name": role_label,
        "fields": fields,
    }


def build_evidence_location_entity(
    *,
    claims: list[dict[str, Any]],
    topic_query: str,
) -> dict[str, Any]:
    country, sources = detect_country(topic_query, claims)
    fields = [
        field_real("country", country, sources) if country else field_no_info("country"),
        field_no_info("city"),
        field_no_info("place_type"),
        field_no_info("scene_lighting"),
        field_no_info("weather"),
    ]
    return {
        "entity_id": "location_primary",
        "entity_type": "LOCATION",
        "role_label": "primary_scene",
        "display_name": "primary_scene",
        "fields": fields,
    }


def build_entity_evidence_pack(
    topic_research_bundle: dict[str, Any],
) -> dict[str, Any]:
    session_id = topic_research_bundle["session_id"]
    objective_id = topic_research_bundle["objective_id"]
    case_id = f"case_{session_id}"
    claims = list(topic_research_bundle["claims"])

    entities = [
        build_evidence_person_entity(
            entity_id="person_victim",
            role_label="victim",
            claims=claims,
        ),
        build_evidence_person_entity(
            entity_id="person_perpetrator",
            role_label="perpetrator",
            claims=claims,
        ),
        build_evidence_location_entity(
            claims=claims,
            topic_query=topic_research_bundle["topic_query"],
        ),
    ]

    visual_assets = topic_research_bundle.get("visual_assets", [])
    context_fields = []
    if visual_assets:
        context_fields.append(
            {
                "field": "visual_assets_available",
                "value": len(visual_assets),
                "source_tag": "REAL",
                "sources": ["topic_research_bundle.visual_assets"],
            }
        )
    else:
        context_fields.append(field_no_info("visual_assets_available"))

    return {
        "schema_version": "1.0.0",
        "session_id": session_id,
        "objective_id": objective_id,
        "created_at": now_utc_iso(),
        "case_id": case_id,
        "entities": entities,
        "context_fields": context_fields,
    }


def load_inference_rules(rules_path: Path) -> dict[str, Any]:
    payload = read_yaml(rules_path)
    if not isinstance(payload, dict):
        raise InferenceRulesError("inference rules root must be an object")

    defaults = payload.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise InferenceRulesError("defaults must be an object")

    source_tag = str(defaults.get("source_tag", "INFERRED"))
    if source_tag != "INFERRED":
        raise InferenceRulesError("defaults.source_tag must be INFERRED")

    rules = payload.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise InferenceRulesError("rules must be a non-empty object")

    for entity_type, field_rules in rules.items():
        if entity_type not in {"PERSON", "LOCATION", "OBJECT", "EVENT"}:
            raise InferenceRulesError(f"unsupported entity_type in rules: {entity_type}")
        if not isinstance(field_rules, dict):
            raise InferenceRulesError(f"rules.{entity_type} must be an object")
        for field_name, candidates in field_rules.items():
            if not isinstance(field_name, str) or not field_name:
                raise InferenceRulesError(f"invalid field name in rules.{entity_type}")
            if not isinstance(candidates, list) or not candidates:
                raise InferenceRulesError(
                    f"rules.{entity_type}.{field_name} must be a non-empty array"
                )
            for idx, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise InferenceRulesError(
                        f"rules.{entity_type}.{field_name}[{idx}] must be an object"
                    )
                for req in ("value", "confidence", "reason"):
                    if req not in candidate:
                        raise InferenceRulesError(
                            f"rules.{entity_type}.{field_name}[{idx}] missing {req}"
                        )
                conf = candidate["confidence"]
                if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
                    raise InferenceRulesError(
                        f"rules.{entity_type}.{field_name}[{idx}].confidence must be 0..1"
                    )
                reason = candidate["reason"]
                if not isinstance(reason, str) or not reason.strip():
                    raise InferenceRulesError(
                        f"rules.{entity_type}.{field_name}[{idx}].reason must be non-empty string"
                    )
                when = candidate.get("when", {})
                if when and not isinstance(when, dict):
                    raise InferenceRulesError(
                        f"rules.{entity_type}.{field_name}[{idx}].when must be an object"
                    )

    return payload


def _text_or_empty(value: Any) -> str:
    return str(value).strip().lower() if isinstance(value, str) else ""


def _to_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return []


def _rule_matches(
    *,
    when: dict[str, Any],
    existing_fields: dict[str, dict[str, Any]],
    context: dict[str, Any],
) -> bool:
    if not when:
        return True

    for key, expected in when.items():
        if key == "location_country_eq":
            location_country = _text_or_empty(context.get("location_country"))
            if location_country != _text_or_empty(expected):
                return False
            continue

        if key == "location_country_in":
            location_country = _text_or_empty(context.get("location_country"))
            expected_set = {_text_or_empty(v) for v in _to_list(expected)}
            if not expected_set or location_country not in expected_set:
                return False
            continue

        if key == "country_field_eq":
            country_value = _text_or_empty(existing_fields.get("country", {}).get("value"))
            if country_value != _text_or_empty(expected):
                return False
            continue

        if key == "occupation_contains":
            occupation = _text_or_empty(existing_fields.get("occupation", {}).get("value"))
            tokens = [_text_or_empty(v) for v in _to_list(expected)]
            if not tokens or not any(token in occupation for token in tokens):
                return False
            continue

        raise InferenceRulesError(f"unsupported condition key: {key}")

    return True


def infer_field_from_rules(
    *,
    entity_type: str,
    field_name: str,
    existing_fields: dict[str, dict[str, Any]],
    context: dict[str, Any],
    inference_rules: dict[str, Any],
) -> dict[str, Any] | None:
    defaults = inference_rules.get("defaults", {})
    default_tag = str(defaults.get("source_tag", "INFERRED"))
    entity_rules = inference_rules.get("rules", {}).get(entity_type, {})
    candidates = entity_rules.get(field_name, [])
    for candidate in candidates:
        when = candidate.get("when", {})
        if not _rule_matches(when=when, existing_fields=existing_fields, context=context):
            continue
        return {
            "field": field_name,
            "value": candidate["value"],
            "source_tag": default_tag,
            "confidence": round(float(candidate["confidence"]), 2),
            "reason": str(candidate["reason"]),
        }
    return None


def build_inference_candidates(
    entity_evidence_pack: dict[str, Any],
    inference_rules: dict[str, Any],
) -> list[dict[str, Any]]:
    location_entity = next(
        (e for e in entity_evidence_pack["entities"] if e["entity_type"] == "LOCATION"),
        None,
    )
    location_country = None
    if location_entity:
        for f in location_entity["fields"]:
            if f["field"] == "country" and isinstance(f.get("value"), str):
                location_country = f["value"]
                break

    candidates: list[dict[str, Any]] = []
    context = {"location_country": location_country}
    for entity in entity_evidence_pack["entities"]:
        field_map = {f["field"]: f for f in entity["fields"]}
        for field in entity["fields"]:
            if field["source_tag"] != "NO_INFO":
                continue

            inferred = infer_field_from_rules(
                entity_type=entity["entity_type"],
                field_name=field["field"],
                existing_fields=field_map,
                context=context,
                inference_rules=inference_rules,
            )

            if inferred is None:
                continue

            candidates.append(
                {
                    "entity_id": entity["entity_id"],
                    "field": inferred,
                }
            )
    return candidates


def apply_inference_with_lock(
    entity_evidence_pack: dict[str, Any],
    inference_candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    entity_map = {e["entity_id"]: e for e in entity_evidence_pack["entities"]}
    candidate_by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in inference_candidates:
        candidate_by_entity.setdefault(row["entity_id"], []).append(row["field"])

    merged_entities = []
    decision_rows = []

    for entity_id, entity in entity_map.items():
        existing_fields = list(entity["fields"])
        incoming_fields = candidate_by_entity.get(entity_id, [])
        merged_fields, decisions = merge_field_sets(existing_fields, incoming_fields)
        merged_entity = dict(entity)
        merged_entity["fields"] = merged_fields
        merged_entities.append(merged_entity)

        for decision in decisions:
            decision_rows.append(
                {
                    "entity_id": entity_id,
                    "field": decision.field,
                    "action": decision.action,
                    "before_tag": decision.before_tag,
                    "incoming_tag": decision.incoming_tag,
                    "after_tag": decision.after_tag,
                    "reason": decision.reason,
                }
            )

    merged_context_fields, context_decisions = merge_field_sets(
        list(entity_evidence_pack.get("context_fields", [])),
        [],
    )
    for decision in context_decisions:
        decision_rows.append(
            {
                "entity_id": "context",
                "field": decision.field,
                "action": decision.action,
                "before_tag": decision.before_tag,
                "incoming_tag": decision.incoming_tag,
                "after_tag": decision.after_tag,
                "reason": decision.reason,
            }
        )

    remaining_no_info = 0
    for entity in merged_entities:
        remaining_no_info += sum(1 for f in entity["fields"] if f["source_tag"] == "NO_INFO")
    remaining_no_info += sum(1 for f in merged_context_fields if f["source_tag"] == "NO_INFO")

    entity_profile_pack = {
        "schema_version": "1.0.0",
        "session_id": entity_evidence_pack["session_id"],
        "objective_id": entity_evidence_pack["objective_id"],
        "created_at": now_utc_iso(),
        "case_id": entity_evidence_pack["case_id"],
        "entities": merged_entities,
        "context_fields": merged_context_fields,
        "profile_status": "READY_FOR_SCRIPT"
        if remaining_no_info == 0
        else "NEEDS_HITL_REVIEW",
    }

    merge_report = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "decision_count": len(decision_rows),
        "remaining_no_info_count": remaining_no_info,
        "decisions": decision_rows,
    }

    return entity_profile_pack, merge_report


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    research_bundle_path = Path(args.topic_research_bundle).resolve()
    inference_rules_path = Path(args.inference_rules).resolve()

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
                research_bundle_path,
                root / "schemas" / "topic_research_bundle.schema.json",
            ),
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l2-l3-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    topic_research_bundle = read_json(research_bundle_path)
    inference_rules = load_inference_rules(inference_rules_path)
    _assert_session_alignment(
        session_input=session_input,
        session_objective=session_objective,
        topic_research_bundle=topic_research_bundle,
    )

    # L2
    entity_evidence_pack = build_entity_evidence_pack(topic_research_bundle)
    write_json(out_dir / "entity_evidence_pack.json", entity_evidence_pack)

    # L3
    inference_candidates = build_inference_candidates(
        entity_evidence_pack,
        inference_rules,
    )
    write_json(out_dir / "inference_candidates.json", inference_candidates)

    entity_profile_pack, merge_report = apply_inference_with_lock(
        entity_evidence_pack,
        inference_candidates,
    )
    write_json(out_dir / "inference_merge_report.json", merge_report)
    write_json(out_dir / "entity_profile_pack.json", entity_profile_pack)
    write_json(
        out_dir / "inference_rules_snapshot.json",
        {
            "schema_version": "1.0.0",
            "created_at": now_utc_iso(),
            "rules_path": str(inference_rules_path),
            "rules": inference_rules,
        },
    )

    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "entity_evidence_pack",
                "path": str((out_dir / "entity_evidence_pack.json").resolve()),
            },
            {
                "name": "entity_profile_pack",
                "path": str((out_dir / "entity_profile_pack.json").resolve()),
            },
            {
                "name": "inference_merge_report",
                "path": str((out_dir / "inference_merge_report.json").resolve()),
            },
            {
                "name": "inference_rules_snapshot",
                "path": str((out_dir / "inference_rules_snapshot.json").resolve()),
            },
        ],
        "metrics": {
            "inference_candidate_count": len(inference_candidates),
            "remaining_no_info_count": merge_report["remaining_no_info_count"],
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
                "entity_evidence_pack",
                out_dir / "entity_evidence_pack.json",
                root / "schemas" / "entity_evidence_pack.schema.json",
            ),
            (
                "entity_profile_pack",
                out_dir / "entity_profile_pack.json",
                root / "schemas" / "entity_profile_pack.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l2-l3-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    print(f"[l2-l3-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L2/L3 (evidence structuring and inference lock merge)."
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
        "--inference-rules",
        default=str(root / "config" / "inference_rules.yaml"),
        help="Path to inference rules YAML",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (InferenceRulesError, ContractError) as err:
        print(f"[l2-l3-runner] rules/contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l2-l3-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
