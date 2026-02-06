#!/usr/bin/env python3
"""Field lock rules for REAL > INFERRED > NO_INFO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TAG_PRIORITY = {
    "NO_INFO": 0,
    "INFERRED": 1,
    "REAL": 2,
}


@dataclass
class MergeDecision:
    field: str
    action: str
    before_tag: str
    incoming_tag: str
    after_tag: str
    reason: str


class LockRuleError(Exception):
    """Raised when source tag is invalid for lock rules."""


def _priority(tag: str) -> int:
    if tag not in TAG_PRIORITY:
        raise LockRuleError(f"invalid source_tag: {tag}")
    return TAG_PRIORITY[tag]


def _confidence(value: dict[str, Any]) -> float:
    raw = value.get("confidence", 0.0)
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


def _merge_real_same_priority(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    # REAL vs REAL: keep existing value, merge unique sources.
    out = dict(existing)
    old_sources = existing.get("sources", []) or []
    new_sources = incoming.get("sources", []) or []
    merged_sources = []
    seen = set()
    for src in list(old_sources) + list(new_sources):
        if src in seen:
            continue
        seen.add(src)
        merged_sources.append(src)
    if merged_sources:
        out["sources"] = merged_sources
    return out, "REAL_CONFLICT_KEEP_EXISTING"


def merge_field(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> tuple[dict[str, Any], MergeDecision]:
    field = str(existing.get("field", incoming.get("field", "unknown")))
    before_tag = str(existing.get("source_tag", "NO_INFO"))
    incoming_tag = str(incoming.get("source_tag", "NO_INFO"))

    before_rank = _priority(before_tag)
    incoming_rank = _priority(incoming_tag)

    # Hard lock: REAL cannot be downgraded or replaced by non-REAL.
    if before_tag == "REAL" and incoming_tag != "REAL":
        decision = MergeDecision(
            field=field,
            action="KEEP_EXISTING",
            before_tag=before_tag,
            incoming_tag=incoming_tag,
            after_tag=before_tag,
            reason="REAL_LOCKED",
        )
        return dict(existing), decision

    # Upgrade by tag priority.
    if incoming_rank > before_rank:
        decision = MergeDecision(
            field=field,
            action="APPLY_INCOMING",
            before_tag=before_tag,
            incoming_tag=incoming_tag,
            after_tag=incoming_tag,
            reason="TAG_UPGRADE",
        )
        return dict(incoming), decision

    # Prevent downgrade.
    if incoming_rank < before_rank:
        decision = MergeDecision(
            field=field,
            action="KEEP_EXISTING",
            before_tag=before_tag,
            incoming_tag=incoming_tag,
            after_tag=before_tag,
            reason="NO_DOWNGRADE",
        )
        return dict(existing), decision

    # Same priority.
    if before_tag == "INFERRED":
        incoming_conf = _confidence(incoming)
        existing_conf = _confidence(existing)
        if incoming_conf > existing_conf:
            decision = MergeDecision(
                field=field,
                action="APPLY_INCOMING",
                before_tag=before_tag,
                incoming_tag=incoming_tag,
                after_tag=incoming_tag,
                reason="HIGHER_CONFIDENCE_INFERRED",
            )
            return dict(incoming), decision
        decision = MergeDecision(
            field=field,
            action="KEEP_EXISTING",
            before_tag=before_tag,
            incoming_tag=incoming_tag,
            after_tag=before_tag,
            reason="LOWER_OR_EQUAL_CONFIDENCE_INFERRED",
        )
        return dict(existing), decision

    if before_tag == "REAL":
        merged, reason = _merge_real_same_priority(existing, incoming)
        decision = MergeDecision(
            field=field,
            action="KEEP_EXISTING",
            before_tag=before_tag,
            incoming_tag=incoming_tag,
            after_tag=merged.get("source_tag", before_tag),
            reason=reason,
        )
        return merged, decision

    # NO_INFO vs NO_INFO
    decision = MergeDecision(
        field=field,
        action="KEEP_EXISTING",
        before_tag=before_tag,
        incoming_tag=incoming_tag,
        after_tag=before_tag,
        reason="NO_INFO_STABLE",
    )
    return dict(existing), decision


def merge_field_sets(
    existing_fields: list[dict[str, Any]],
    incoming_fields: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[MergeDecision]]:
    existing_by_name = {str(f["field"]): dict(f) for f in existing_fields}
    decisions: list[MergeDecision] = []

    for incoming in incoming_fields:
        name = str(incoming["field"])
        if name not in existing_by_name:
            existing_by_name[name] = dict(incoming)
            decisions.append(
                MergeDecision(
                    field=name,
                    action="ADD_NEW",
                    before_tag="NONE",
                    incoming_tag=str(incoming.get("source_tag", "NO_INFO")),
                    after_tag=str(incoming.get("source_tag", "NO_INFO")),
                    reason="FIELD_ADDED",
                )
            )
            continue

        merged, decision = merge_field(existing_by_name[name], incoming)
        existing_by_name[name] = merged
        decisions.append(decision)

    merged_fields = list(existing_by_name.values())
    return merged_fields, decisions
