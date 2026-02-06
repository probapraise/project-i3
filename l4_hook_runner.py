#!/usr/bin/env python3
"""L4 Hook Session runner (ideation + evaluation + selection)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, write_gate_manifest, write_json


class ContractError(Exception):
    """Raised when cross-artifact session contracts are violated."""


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    topic_research_bundle: dict[str, Any],
    entity_profile_pack: dict[str, Any],
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

    topic_session_id = str(topic_research_bundle.get("session_id", "")).strip()
    topic_objective_id = str(topic_research_bundle.get("objective_id", "")).strip()
    if topic_session_id != input_session_id:
        raise ContractError(
            "topic_research_bundle.session_id must match session_input.session_id"
        )
    if topic_objective_id != input_objective_id:
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


def extract_primary_facts(
    topic_research_bundle: dict[str, Any],
    entity_profile_pack: dict[str, Any],
) -> dict[str, Any]:
    claims = topic_research_bundle.get("claims", [])
    claim_snippets = []
    for claim in claims[:4]:
        text = normalize_text(str(claim.get("claim_text", "")))
        if text:
            claim_snippets.append(
                {
                    "claim_id": claim.get("claim_id", "unknown"),
                    "text": text,
                }
            )

    country = None
    victim_occupation = None
    victim_role = "victim"
    for entity in entity_profile_pack.get("entities", []):
        if entity.get("entity_type") == "LOCATION":
            for field in entity.get("fields", []):
                if field.get("field") == "country" and isinstance(field.get("value"), str):
                    country = field["value"]
                    break
        if entity.get("entity_id") == "person_victim":
            victim_role = str(entity.get("role_label", "victim"))
            for field in entity.get("fields", []):
                if field.get("field") == "occupation" and isinstance(field.get("value"), str):
                    victim_occupation = field["value"]
                    break

    visual_assets = topic_research_bundle.get("visual_assets", [])
    visual_available = any(
        asset.get("availability") in {"CONFIRMED", "PROBABLE"} for asset in visual_assets
    )

    return {
        "claim_snippets": claim_snippets,
        "country": country,
        "victim_occupation": victim_occupation,
        "victim_role": victim_role,
        "visual_available": visual_available,
        "topic_query": topic_research_bundle.get("topic_query", ""),
    }


def build_hook_candidates(primary: dict[str, Any]) -> list[dict[str, Any]]:
    claim_snippets = primary["claim_snippets"]
    first_claim = claim_snippets[0]["text"] if claim_snippets else "a routine case"
    first_claim_id = claim_snippets[0]["claim_id"] if claim_snippets else None
    occupation = primary.get("victim_occupation") or "ordinary worker"
    country = primary.get("country") or "the city"
    topic_query = normalize_text(str(primary.get("topic_query", "")))

    templates = [
        (
            f"He was a {occupation}, until one detail rewrote the entire case.",
            "contradiction",
            [first_claim_id] if first_claim_id else [],
        ),
        (
            f"This started as a normal report in {country}, then the timeline cracked.",
            "timeline_twist",
            [first_claim_id] if first_claim_id else [],
        ),
        (
            "Everyone accepted the first story. Then one clue changed everything.",
            "investigation_shift",
            [first_claim_id] if first_claim_id else [],
        ),
        (
            "What if the most trusted version of the case was wrong?",
            "curiosity_question",
            [first_claim_id] if first_claim_id else [],
        ),
        (
            f"{first_claim}. But that is not what the final record suggests.",
            "record_conflict",
            [first_claim_id] if first_claim_id else [],
        ),
        (
            f"{topic_query} looked simple. The reveal made viewers question every second.",
            "loop_reveal",
            [first_claim_id] if first_claim_id else [],
        ),
    ]

    candidates = []
    for idx, (text, angle, source_claim_ids) in enumerate(templates, start=1):
        hook_text = normalize_text(text)
        if not hook_text:
            continue
        candidates.append(
            {
                "hook_id": f"hook_{idx:03d}",
                "hook_text": hook_text,
                "angle": angle,
                "trigger_words": extract_trigger_words(hook_text),
                "source_claim_ids": [c for c in source_claim_ids if c],
            }
        )

    return candidates


def extract_trigger_words(text: str) -> list[str]:
    token_map = {
        "mystery": "Mistério",
        "secret": "Segredo",
        "revealed": "Revelação",
        "revenge": "Vingança",
        "wrong": "Dúvida",
        "clue": "Pista",
        "timeline": "Linha do tempo",
        "truth": "Verdade",
        "shock": "Choque",
        "record": "Registro",
    }
    lower = text.lower()
    out = []
    for token, keyword in token_map.items():
        if token in lower:
            out.append(keyword)
    if not out:
        out = ["Mistério", "Reviravolta"]
    return out


def score_hook_candidate(
    *,
    hook_text: str,
    angle: str,
    primary: dict[str, Any],
    score_spec: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, bool]]:
    lower = hook_text.lower()

    stop_tokens = [
        "until",
        "then",
        "rewrote",
        "changed everything",
        "wrong",
        "question",
        "reveal",
        "final",
    ]
    curiosity_tokens = [
        "what if",
        "?",
        "not what",
        "final",
        "question",
        "clue",
    ]
    clarity_tokens = [
        "case",
        "report",
        "record",
        "timeline",
    ]
    unsafe_tokens = [
        "dismemberment",
        "beheaded",
        "torture porn",
        "snuff",
    ]

    k1: float = 55 + sum(9 for t in stop_tokens if t in lower)
    if len(hook_text) > 115:
        k1 -= 8
    if primary.get("visual_available"):
        k1 += 8
    if angle in {"contradiction", "timeline_twist", "record_conflict"}:
        k1 += 9
    k1 = clamp(k1)

    k3: float = 50 + sum(9 for t in curiosity_tokens if t in lower)
    if hook_text.endswith("?"):
        k3 += 8
    if angle in {"curiosity_question", "loop_reveal"}:
        k3 += 10
    k3 = clamp(k3)

    k2: float = 62 + sum(6 for t in clarity_tokens if t in lower)
    if len(hook_text.split()) < 8:
        k2 -= 10
    if len(hook_text.split()) > 22:
        k2 -= 8
    k2 = clamp(k2)

    k4: float = 68
    if "?" in hook_text:
        k4 += 5
    if "case" in lower or "record" in lower:
        k4 += 7
    if "blood" in lower or "gore" in lower:
        k4 -= 30
    k4 = clamp(k4)

    hard_k5 = not any(tok in lower for tok in unsafe_tokens)

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


def evaluate_and_select_hooks(
    *,
    candidates: list[dict[str, Any]],
    primary: dict[str, Any],
    score_spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    for cand in candidates:
        total, criteria, hard = score_hook_candidate(
            hook_text=cand["hook_text"],
            angle=cand["angle"],
            primary=primary,
            score_spec=score_spec,
        )
        rows.append(
            {
                "hook_id": cand["hook_id"],
                "total_score": total,
                "criteria": criteria,
                "hard_constraints": hard,
                "hard_pass": all(hard.values()),
            }
        )

    passing = [r for r in rows if r["hard_pass"]]
    tie_break_order = score_spec.get("selection_rule", {}).get("tie_break_order", ["K1", "K3"])
    min_accept_score = float(score_spec.get("selection_rule", {}).get("min_accept_score", 0))

    def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple([row["total_score"]] + [row["criteria"].get(k, 0.0) for k in tie_break_order])

    selected_row = max(passing if passing else rows, key=rank_key)
    selected_cand = next(c for c in candidates if c["hook_id"] == selected_row["hook_id"])
    pass_threshold = selected_row["total_score"] >= min_accept_score and selected_row["hard_pass"]

    evaluation_report = {
        "schema_version": "1.0.0",
        "task_id": "hook_selection",
        "created_at": now_utc_iso(),
        "candidate_evaluations": rows,
        "selected_hook_id": selected_row["hook_id"],
        "min_accept_score": min_accept_score,
        "pass_threshold_met": pass_threshold,
    }

    selection_trace = {
        "schema_version": "1.0.0",
        "task_id": "hook_selection",
        "created_at": now_utc_iso(),
        "selected_hook_id": selected_row["hook_id"],
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


def build_narrative_seed(
    topic_research_bundle: dict[str, Any],
    selected_hook: dict[str, Any],
) -> dict[str, str]:
    claims = topic_research_bundle.get("claims", [])
    setup = (
        normalize_text(claims[0]["claim_text"])
        if claims
        else "A normal day turned into an unexpected criminal case."
    )
    twist = selected_hook["candidate"]["hook_text"]
    resolution = "Final records and verified sources reveal what actually happened."
    loop_line = "Watch the first line again. It means something different now."
    return {
        "setup": setup,
        "twist": twist,
        "resolution": resolution,
        "loop_line": loop_line,
    }


def build_script_seed_pack(
    *,
    session_input: dict[str, Any],
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    topic_research_bundle: dict[str, Any],
) -> dict[str, Any]:
    selected_hook = selected["candidate"]
    selected_eval = selected["evaluation"]

    trigger_words = []
    for cand in candidates:
        for token in cand["trigger_words"]:
            if token not in trigger_words:
                trigger_words.append(token)

    return {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "selected_hook": {
            "hook_id": selected_hook["hook_id"],
            "hook_text": selected_hook["hook_text"],
            "total_score": selected_eval["total_score"],
            "criteria": selected_eval["criteria"],
            "hard_constraints": selected_eval["hard_constraints"],
            "hard_pass": selected_eval["hard_pass"],
        },
        "hook_bank": candidates,
        "narrative_seed": build_narrative_seed(topic_research_bundle, selected),
        "trigger_words": trigger_words[:12],
        "banned_expressions": [
            "explicit dismemberment details",
            "graphic torture description",
            "victim-blaming phrasing"
        ],
        "policy_notes": [
            "Use psychological and investigative framing instead of graphic depictions.",
            "Maintain respect for victims and avoid glorifying perpetrators."
        ],
        "quality_status": "READY_FOR_SCRIPT"
        if selected["pass_threshold_met"]
        else "NEEDS_HOOK_REWORK",
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    topic_research_bundle_path = Path(args.topic_research_bundle).resolve()
    entity_profile_pack_path = Path(args.entity_profile_pack).resolve()
    score_spec_path = Path(args.score_spec).resolve()

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
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l4-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    topic_research_bundle = read_json(topic_research_bundle_path)
    entity_profile_pack = read_json(entity_profile_pack_path)
    score_spec = read_json(score_spec_path)
    _assert_session_alignment(
        session_input=session_input,
        session_objective=session_objective,
        topic_research_bundle=topic_research_bundle,
        entity_profile_pack=entity_profile_pack,
    )

    primary = extract_primary_facts(topic_research_bundle, entity_profile_pack)
    candidates = build_hook_candidates(primary)
    selected, evaluation_report, selection_trace = evaluate_and_select_hooks(
        candidates=candidates,
        primary=primary,
        score_spec=score_spec,
    )
    script_seed_pack = build_script_seed_pack(
        session_input=session_input,
        selected=selected,
        candidates=candidates,
        topic_research_bundle=topic_research_bundle,
    )

    candidates_doc = {
        "schema_version": "1.0.0",
        "task_id": "hook_selection",
        "created_at": now_utc_iso(),
        "candidates": candidates,
    }

    write_json(out_dir / "candidates_hook.json", candidates_doc)
    write_json(out_dir / "evaluation_report_hook.json", evaluation_report)
    write_json(out_dir / "selection_trace_hook.json", selection_trace)
    write_json(out_dir / "script_seed_pack.json", script_seed_pack)

    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS",
        "output_artifact_refs": [
            {
                "name": "candidates_hook",
                "path": str((out_dir / "candidates_hook.json").resolve()),
            },
            {
                "name": "evaluation_report_hook",
                "path": str((out_dir / "evaluation_report_hook.json").resolve()),
            },
            {
                "name": "selection_trace_hook",
                "path": str((out_dir / "selection_trace_hook.json").resolve()),
            },
            {
                "name": "script_seed_pack",
                "path": str((out_dir / "script_seed_pack.json").resolve()),
            },
        ],
        "metrics": {
            "hook_candidate_count": len(candidates),
            "selected_hook_id": selected["candidate"]["hook_id"],
            "selected_hook_score": selected["evaluation"]["total_score"],
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
                "script_seed_pack",
                out_dir / "script_seed_pack.json",
                root / "schemas" / "script_seed_pack.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l4-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    print(f"[l4-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L4 Hook Session with score-based selection."
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
        "--score-spec",
        default=str(root / "score_specs" / "score_spec_hook.json"),
        help="Path to score_spec_hook.json",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ContractError as err:
        print(f"[l4-runner] contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l4-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
