#!/usr/bin/env python3
"""Shared IO and gate-manifest helpers for pipeline runners."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_gate_manifest(
    *,
    manifest_path: Path,
    gate: str,
    artifacts: list[tuple[str, Path, Path]],
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "gate": gate,
        "artifacts": [
            {
                "name": name,
                "artifact_path": str(artifact_path.resolve()),
                "schema_path": str(schema_path.resolve()),
            }
            for name, artifact_path, schema_path in artifacts
        ],
    }
    write_json(manifest_path, payload)
