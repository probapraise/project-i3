#!/usr/bin/env python3
"""L1 Topic Research runner (config-driven, schema-gated)."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gate_validator import run_gate_validation
from runner_common import now_utc_iso, read_json, read_yaml, write_gate_manifest, write_json


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class AgentRuntimeInfo:
    agent_id: str
    provider: str
    model: str
    tools: list[str]
    purpose: str


class ConfigError(Exception):
    """Raised when config payload is malformed."""


class ContractError(Exception):
    """Raised when cross-artifact session contracts are violated."""


def load_agent_runtime_infos(model_map_path: Path) -> dict[str, AgentRuntimeInfo]:
    payload = read_yaml(model_map_path)
    if not isinstance(payload, dict):
        raise ConfigError("agent_model_map root must be an object")

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
        "SearchAugmenter",
        "SourceNormalizer",
        "ClaimVerifier",
        "ResearchSelector",
    }
    missing = required_agents.difference(infos)
    if missing:
        raise ConfigError(f"missing required agents: {sorted(missing)}")

    return infos


def _assert_session_alignment(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    topic_intake: dict[str, Any],
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

    intake_session_id = str(topic_intake.get("session_id", "")).strip()
    intake_objective_id = str(topic_intake.get("objective_id", "")).strip()
    if intake_session_id != input_session_id:
        raise ContractError("topic_intake.session_id must match session_input.session_id")
    if intake_objective_id != input_objective_id:
        raise ContractError("topic_intake.objective_id must match session_input.objective_id")


def material_to_source(material: dict[str, Any], idx: int) -> dict[str, Any]:
    material_id = material.get("material_id") or f"material_{idx:03d}"
    material_type = str(material.get("material_type") or "OTHER")
    source_name = str(material.get("source_name") or "UserCollected")
    url = material.get("url")
    if not isinstance(url, str) or not url.strip():
        url = f"https://local.invalid/material/{material_id}"

    source_type_map = {
        "URL": "NEWS",
        "ARTICLE_SNIPPET": "NEWS",
        "VIDEO_DESC": "VIDEO_PLATFORM",
        "NOTE": "OTHER",
        "IMAGE_DESC": "OTHER",
        "OTHER": "OTHER",
    }
    credibility_map = {
        "URL": "MEDIUM",
        "ARTICLE_SNIPPET": "MEDIUM",
        "VIDEO_DESC": "LOW",
        "NOTE": "LOW",
        "IMAGE_DESC": "LOW",
        "OTHER": "UNKNOWN",
    }

    return {
        "source_id": f"src_{idx:03d}",
        "url": url,
        "source_name": source_name,
        "source_type": source_type_map.get(material_type, "OTHER"),
        "credibility_tier": credibility_map.get(material_type, "UNKNOWN"),
    }


def extract_claim_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    chunks = re.split(r"[.!?]\s+", clean)
    claim = chunks[0].strip()
    if len(claim) > 180:
        claim = claim[:177].rstrip() + "..."
    return claim


def claim_dedup_key(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip().lower()
    if not clean:
        return ""
    normalized = re.sub(r"[^\w]+", " ", clean, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or clean


def material_to_claim(material: dict[str, Any], source_id: str, idx: int) -> dict[str, Any]:
    material_type = str(material.get("material_type") or "OTHER")
    content = str(material.get("content") or "").strip()
    claim_text = extract_claim_text(content)
    if not claim_text:
        claim_text = f"User provided material #{idx} indicates relevant case context."

    if material_type == "URL":
        status = "VERIFIED"
    elif material_type in {"ARTICLE_SNIPPET", "VIDEO_DESC"}:
        status = "PARTIALLY_VERIFIED"
    else:
        status = "UNVERIFIED"

    return {
        "claim_id": f"clm_{idx:03d}",
        "claim_text": claim_text,
        "verification_status": status,
        "supporting_source_ids": [source_id],
    }


def build_visual_assets(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keywords = [
        ("cctv", "CCTV"),
        ("camera", "CCTV"),
        ("footage", "CCTV"),
        ("mugshot", "MUGSHOT"),
        ("court", "COURT_VIDEO"),
        ("trial", "COURT_VIDEO"),
        ("instagram", "SNS_POST"),
        ("facebook", "SNS_POST"),
        ("sns", "SNS_POST"),
        ("photo", "NEWS_STILL"),
        ("image", "NEWS_STILL"),
        ("interview", "INTERVIEW_VIDEO"),
        ("video", "INTERVIEW_VIDEO"),
    ]

    assets: list[dict[str, Any]] = []
    asset_idx = 1
    for material in materials:
        blob = " ".join(
            [
                str(material.get("content") or ""),
                str(material.get("url") or ""),
            ]
        ).lower()
        for token, asset_type in keywords:
            if token in blob:
                asset = {
                    "asset_id": f"asset_{asset_idx:03d}",
                    "asset_type": asset_type,
                    "availability": "CONFIRMED"
                    if material.get("url")
                    else "PROBABLE",
                }
                if material.get("url"):
                    asset["url"] = material["url"]
                assets.append(asset)
                asset_idx += 1
                break

    if not assets:
        assets.append(
            {
                "asset_id": "asset_001",
                "asset_type": "OTHER",
                "availability": "UNKNOWN",
                "description": "No explicit visual evidence marker in current materials.",
            }
        )

    return assets


def build_candidate(
    *,
    candidate_id: str,
    topic_query: str,
    materials: list[dict[str, Any]],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    source_id_by_key: dict[str, str] = {}
    material_source_ids: list[str] = []
    for idx, material in enumerate(materials):
        source = material_to_source(material, idx + 1)
        key = f"{source['url']}::{source['source_name']}"
        if key not in source_id_by_key:
            source_id_by_key[key] = source["source_id"]
            sources.append(source)
        material_source_ids.append(source_id_by_key[key])

    claims = [
        material_to_claim(material, material_source_ids[idx], idx + 1)
        for idx, material in enumerate(materials)
    ]

    # Deduplicate by claim text while preserving merged evidence links.
    dedup_claims: list[dict[str, Any]] = []
    claim_index_by_key: dict[str, int] = {}
    verification_rank = {
        "UNVERIFIED": 0,
        "PARTIALLY_VERIFIED": 1,
        "VERIFIED": 2,
    }
    for claim in claims:
        key = claim_dedup_key(str(claim.get("claim_text", "")))
        existing_idx = claim_index_by_key.get(key)
        if existing_idx is None:
            claim_index_by_key[key] = len(dedup_claims)
            dedup_claims.append(dict(claim))
            continue

        existing = dedup_claims[existing_idx]
        merged_sources = []
        seen_sources: set[str] = set()
        for source_id in list(existing.get("supporting_source_ids", [])) + list(
            claim.get("supporting_source_ids", [])
        ):
            source_key = str(source_id)
            if not source_key or source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            merged_sources.append(source_key)
        existing["supporting_source_ids"] = merged_sources

        existing_status = str(existing.get("verification_status", "UNVERIFIED"))
        incoming_status = str(claim.get("verification_status", "UNVERIFIED"))
        if verification_rank.get(incoming_status, -1) > verification_rank.get(existing_status, -1):
            existing["verification_status"] = incoming_status

    if not dedup_claims:
        dedup_claims = [
            {
                "claim_id": "clm_001",
                "claim_text": topic_query,
                "verification_status": "UNVERIFIED",
                "supporting_source_ids": [sources[0]["source_id"]],
            }
        ]

    visual_assets = build_visual_assets(materials)

    return {
        "candidate_id": candidate_id,
        "topic_query": topic_query,
        "normalized_sources": sources,
        "claims": dedup_claims,
        "visual_assets": visual_assets,
    }


def score_candidate(
    candidate: dict[str, Any], score_spec: dict[str, Any]
) -> tuple[float, dict[str, float], dict[str, bool]]:
    text_blob = " ".join(
        [candidate.get("topic_query", "")]
        + [c.get("claim_text", "") for c in candidate.get("claims", [])]
    ).lower()

    contradiction_tokens = [
        "but",
        "however",
        "actually",
        "revealed",
        "unexpected",
        "irony",
        "contradiction",
        "turned out",
        "while",
    ]
    policy_block_tokens = [
        "snuff",
        "graphic dismemberment",
        "torture porn",
    ]

    contradiction_hits = sum(1 for t in contradiction_tokens if t in text_blob)
    k2 = clamp(30 + contradiction_hits * 12)

    confirmed_assets = sum(
        1 for a in candidate.get("visual_assets", []) if a.get("availability") == "CONFIRMED"
    )
    probable_assets = sum(
        1 for a in candidate.get("visual_assets", []) if a.get("availability") == "PROBABLE"
    )
    k4 = clamp(20 + confirmed_assets * 20 + probable_assets * 10)

    claims = candidate.get("claims", [])
    verified_claims = sum(1 for c in claims if c.get("verification_status") == "VERIFIED")
    verification_ratio = verified_claims / len(claims) if claims else 0.0

    k1 = clamp(40 + (k2 * 0.25) + (k4 * 0.35) + (verification_ratio * 25))
    k3 = clamp(35 + (k2 * 0.20) + (k4 * 0.25) + (len(claims) * 5) + (verification_ratio * 20))
    k5_pass = not any(t in text_blob for t in policy_block_tokens)

    criterion_scores = {
        "K1": round(k1, 2),
        "K2": round(k2, 2),
        "K3": round(k3, 2),
        "K4": round(k4, 2),
    }
    hard_constraints = {"K5": k5_pass}

    weight_by_id = {c["id"]: float(c["weight"]) for c in score_spec.get("criteria", [])}
    total = 0.0
    for cid, value in criterion_scores.items():
        total += weight_by_id.get(cid, 0.0) * value
    total = round(total, 2)

    return total, criterion_scores, hard_constraints


def select_candidate(
    candidates: list[dict[str, Any]], score_spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for item in candidates:
        total, criterion_scores, hard = score_candidate(item, score_spec)
        row = {
            "candidate_id": item["candidate_id"],
            "total_score": total,
            "criteria": criterion_scores,
            "hard_constraints": hard,
            "hard_pass": all(hard.values()),
        }
        rows.append(row)

    passing = [r for r in rows if r["hard_pass"]]
    if not passing:
        selected_row = max(rows, key=lambda r: r["total_score"])
    else:
        tie_break_order = score_spec.get("selection_rule", {}).get("tie_break_order", ["K1", "K3"])

        def key_fn(r: dict[str, Any]) -> tuple[Any, ...]:
            return tuple([r["total_score"]] + [r["criteria"].get(k, 0.0) for k in tie_break_order])

        selected_row = max(passing, key=key_fn)

    selected = next(c for c in candidates if c["candidate_id"] == selected_row["candidate_id"])
    evaluation_report = {
        "schema_version": "1.0.0",
        "task_id": score_spec.get("task_id", "topic_intake"),
        "created_at": now_utc_iso(),
        "candidate_evaluations": rows,
        "selected_candidate_id": selected_row["candidate_id"],
    }
    return selected, evaluation_report


def build_l1_outputs(
    *,
    session_input: dict[str, Any],
    session_objective: dict[str, Any],
    topic_intake: dict[str, Any],
    score_spec: dict[str, Any],
) -> dict[str, Any]:
    materials = list(topic_intake["user_collected_materials"])
    topic_query = topic_intake["topic_query"]

    candidates = [
        build_candidate(candidate_id="cand_all", topic_query=topic_query, materials=materials)
    ]
    url_materials = [m for m in materials if str(m.get("material_type")) == "URL"]
    if url_materials and len(url_materials) < len(materials):
        candidates.append(
            build_candidate(
                candidate_id="cand_url_only",
                topic_query=topic_query,
                materials=url_materials,
            )
        )

    selected, evaluation = select_candidate(candidates, score_spec)

    verified_count = sum(
        1 for c in selected["claims"] if c.get("verification_status") == "VERIFIED"
    )
    unresolved_questions = []
    if verified_count == 0:
        unresolved_questions.append("No fully verified claims yet. Add stronger sources.")
    if not any(v.get("availability") == "CONFIRMED" for v in selected["visual_assets"]):
        unresolved_questions.append("Visual evidence is not confirmed by source URL.")

    research_status = "READY_FOR_L2" if verified_count > 0 else "NEEDS_MORE_RESEARCH"

    topic_research_bundle = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "topic_query": topic_query,
        "normalized_sources": selected["normalized_sources"],
        "claims": selected["claims"],
        "visual_assets": selected["visual_assets"],
        "unresolved_questions": unresolved_questions,
        "research_status": research_status,
    }

    claim_verification_report = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "claims": [
            {
                "claim_id": c["claim_id"],
                "claim_text": c["claim_text"],
                "verification_status": c["verification_status"],
                "supporting_source_ids": c["supporting_source_ids"],
            }
            for c in selected["claims"]
        ],
        "summary": {
            "total": len(selected["claims"]),
            "verified": verified_count,
            "partially_verified": sum(
                1
                for c in selected["claims"]
                if c.get("verification_status") == "PARTIALLY_VERIFIED"
            ),
            "unverified": sum(
                1 for c in selected["claims"] if c.get("verification_status") == "UNVERIFIED"
            ),
        },
    }

    selection_trace = {
        "schema_version": "1.0.0",
        "task_id": score_spec.get("task_id", "topic_intake"),
        "created_at": now_utc_iso(),
        "selected_candidate_id": evaluation["selected_candidate_id"],
        "selection_reason": "Highest weighted score among hard-constraint passing candidates.",
    }

    candidates_doc = {
        "schema_version": "1.0.0",
        "task_id": score_spec.get("task_id", "topic_intake"),
        "created_at": now_utc_iso(),
        "candidates": candidates,
    }

    session_output = {
        "schema_version": "1.0.0",
        "session_id": session_input["session_id"],
        "objective_id": session_input["objective_id"],
        "created_at": now_utc_iso(),
        "status": "SUCCESS" if research_status != "BLOCKED" else "BLOCKED",
        "output_artifact_refs": [
            {
                "name": "topic_research_bundle",
                "path": "topic_research_bundle.json",
            },
            {
                "name": "claim_verification_report",
                "path": "claim_verification_report.json",
            },
            {
                "name": "evaluation_report_topic_intake",
                "path": "evaluation_report_topic_intake.json",
            },
            {
                "name": "selection_trace_topic_intake",
                "path": "selection_trace_topic_intake.json",
            },
        ],
        "metrics": {
            "candidate_count": len(candidates),
            "verified_claim_count": verified_count,
            "selected_candidate_id": evaluation["selected_candidate_id"],
        },
    }

    return {
        "topic_research_bundle": topic_research_bundle,
        "claim_verification_report": claim_verification_report,
        "evaluation_report_topic_intake": evaluation,
        "selection_trace_topic_intake": selection_trace,
        "candidates_topic_intake": candidates_doc,
        "session_output": session_output,
        "session_objective_echo": session_objective,
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_input_path = Path(args.session_input).resolve()
    session_objective_path = Path(args.session_objective).resolve()
    topic_intake_path = Path(args.topic_intake).resolve()
    model_map_path = Path(args.model_map).resolve()
    score_spec_path = Path(args.score_spec).resolve()

    # 1) Gate-In
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
                "topic_intake_input",
                topic_intake_path,
                root / "schemas" / "topic_intake_input.schema.json",
            ),
        ],
    )
    gate_in_report = run_gate_validation(
        manifest_path=gate_in_manifest,
        requested_gate="in",
    )
    write_json(out_dir / "gate_in_report.json", gate_in_report)
    if gate_in_report["overall_status"] != "PASS":
        print("[l1-runner] Gate-In failed. See gate_in_report.json", file=sys.stderr)
        return 1

    # 2) Load runtime configs
    agent_runtime_infos = load_agent_runtime_infos(model_map_path)
    score_spec = read_json(score_spec_path)

    session_input = read_json(session_input_path)
    session_objective = read_json(session_objective_path)
    topic_intake = read_json(topic_intake_path)
    _assert_session_alignment(
        session_input=session_input,
        session_objective=session_objective,
        topic_intake=topic_intake,
    )

    # 3) L1 execution
    outputs = build_l1_outputs(
        session_input=session_input,
        session_objective=session_objective,
        topic_intake=topic_intake,
        score_spec=score_spec,
    )

    write_json(out_dir / "topic_research_bundle.json", outputs["topic_research_bundle"])
    write_json(out_dir / "claim_verification_report.json", outputs["claim_verification_report"])
    write_json(
        out_dir / "evaluation_report_topic_intake.json",
        outputs["evaluation_report_topic_intake"],
    )
    write_json(
        out_dir / "selection_trace_topic_intake.json",
        outputs["selection_trace_topic_intake"],
    )
    write_json(out_dir / "candidates_topic_intake.json", outputs["candidates_topic_intake"])

    session_output = outputs["session_output"]
    # Rewrite artifact paths as absolute runtime output paths.
    for ref in session_output["output_artifact_refs"]:
        ref["path"] = str((out_dir / ref["path"]).resolve())
    write_json(out_dir / "session_output.json", session_output)

    runtime_log = {
        "schema_version": "1.0.0",
        "created_at": now_utc_iso(),
        "pipeline_stage": "L1 Topic Research",
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
            "topic_intake_input": str(topic_intake_path),
            "score_spec": str(score_spec_path),
            "model_map": str(model_map_path),
        },
        "outputs": {
            "out_dir": str(out_dir),
        },
    }
    write_json(out_dir / "runtime_log_l1.json", runtime_log)

    # 4) Gate-Out
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
                "topic_research_bundle",
                out_dir / "topic_research_bundle.json",
                root / "schemas" / "topic_research_bundle.schema.json",
            ),
        ],
    )
    gate_out_report = run_gate_validation(
        manifest_path=gate_out_manifest,
        requested_gate="out",
    )
    write_json(out_dir / "gate_out_report.json", gate_out_report)
    if gate_out_report["overall_status"] != "PASS":
        print("[l1-runner] Gate-Out failed. See gate_out_report.json", file=sys.stderr)
        return 1

    print(f"[l1-runner] completed: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run L1 Topic Research with config-driven agents and schema gates."
    )
    parser.add_argument(
        "--session-input",
        required=True,
        help="Path to session_input.json",
    )
    parser.add_argument(
        "--session-objective",
        required=True,
        help="Path to session_objective.json",
    )
    parser.add_argument(
        "--topic-intake",
        required=True,
        help="Path to topic_intake_input.json",
    )
    parser.add_argument(
        "--model-map",
        default=str(root / "config" / "agent_model_map.yaml"),
        help="Path to agent_model_map.yaml",
    )
    parser.add_argument(
        "--score-spec",
        default=str(root / "score_specs" / "score_spec_topic_intake.json"),
        help="Path to topic intake score spec JSON",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for L1 artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ConfigError, ContractError) as err:
        print(f"[l1-runner] config/contract error: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[l1-runner] unexpected error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
