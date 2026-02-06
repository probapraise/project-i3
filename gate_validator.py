#!/usr/bin/env python3
"""Gate-In/Out contract validation for pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


@dataclass
class ValidationIssue:
    path: str
    message: str
    validator: str


@dataclass
class ArtifactCheckResult:
    name: str
    artifact_path: str
    schema_path: str
    status: str
    error_count: int
    errors: list[ValidationIssue]


class ManifestError(Exception):
    """Raised when gate manifest shape is invalid."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_schema_id(schema: Any, schema_path: Path) -> Any:
    if not isinstance(schema, dict):
        return schema
    current_id = schema.get("$id")
    if not isinstance(current_id, str) or "://" not in current_id:
        out = dict(schema)
        out["$id"] = schema_path.resolve().as_uri()
        return out
    return schema


def _format_error_path(error_path: list[Any]) -> str:
    if not error_path:
        return "$"
    parts = ["$"]
    for part in error_path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}")
    return "".join(parts)


def _validate_manifest(manifest: dict[str, Any], requested_gate: str) -> None:
    required_top_level = {"schema_version", "gate", "artifacts"}
    missing = required_top_level.difference(manifest.keys())
    if missing:
        raise ManifestError(f"manifest missing required keys: {sorted(missing)}")

    gate = manifest["gate"]
    if gate not in {"in", "out"}:
        raise ManifestError("manifest gate must be one of: in, out")
    if gate != requested_gate:
        raise ManifestError(
            f"requested gate '{requested_gate}' does not match manifest gate '{gate}'"
        )

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("manifest artifacts must be a non-empty array")

    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ManifestError(f"artifacts[{idx}] must be an object")
        for key in ("name", "artifact_path", "schema_path"):
            if key not in item or not isinstance(item[key], str) or not item[key].strip():
                raise ManifestError(f"artifacts[{idx}].{key} must be a non-empty string")


def validate_artifact(
    *,
    name: str,
    artifact_path: Path,
    schema_path: Path,
) -> ArtifactCheckResult:
    issues: list[ValidationIssue] = []

    if not artifact_path.exists():
        issues.append(
            ValidationIssue(
                path="$",
                message=f"artifact file not found: {artifact_path}",
                validator="file_exists",
            )
        )
        return ArtifactCheckResult(
            name=name,
            artifact_path=str(artifact_path),
            schema_path=str(schema_path),
            status="FAIL",
            error_count=len(issues),
            errors=issues,
        )

    if not schema_path.exists():
        issues.append(
            ValidationIssue(
                path="$",
                message=f"schema file not found: {schema_path}",
                validator="file_exists",
            )
        )
        return ArtifactCheckResult(
            name=name,
            artifact_path=str(artifact_path),
            schema_path=str(schema_path),
            status="FAIL",
            error_count=len(issues),
            errors=issues,
        )

    try:
        artifact = _read_json(artifact_path)
    except json.JSONDecodeError as err:
        issues.append(
            ValidationIssue(
                path="$",
                message=f"invalid JSON artifact: {err}",
                validator="json_parse",
            )
        )
        return ArtifactCheckResult(
            name=name,
            artifact_path=str(artifact_path),
            schema_path=str(schema_path),
            status="FAIL",
            error_count=len(issues),
            errors=issues,
        )

    try:
        schema = _read_json(schema_path)
    except json.JSONDecodeError as err:
        issues.append(
            ValidationIssue(
                path="$",
                message=f"invalid JSON schema: {err}",
                validator="json_parse",
            )
        )
        return ArtifactCheckResult(
            name=name,
            artifact_path=str(artifact_path),
            schema_path=str(schema_path),
            status="FAIL",
            error_count=len(issues),
            errors=issues,
        )

    schema = _normalize_schema_id(schema, schema_path)
    validator = Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(artifact), key=lambda e: list(e.path))
    for schema_err in schema_errors:
        issues.append(
            ValidationIssue(
                path=_format_error_path(list(schema_err.path)),
                message=schema_err.message,
                validator=schema_err.validator,
            )
        )

    status = "PASS" if not issues else "FAIL"
    return ArtifactCheckResult(
        name=name,
        artifact_path=str(artifact_path),
        schema_path=str(schema_path),
        status=status,
        error_count=len(issues),
        errors=issues,
    )


def run_gate_validation(
    *,
    manifest_path: Path,
    requested_gate: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be an object")

    _validate_manifest(manifest, requested_gate)

    manifest_base = manifest_path.parent.resolve()
    checks: list[ArtifactCheckResult] = []
    for item in manifest["artifacts"]:
        artifact_path = (manifest_base / item["artifact_path"]).resolve()
        schema_path = (manifest_base / item["schema_path"]).resolve()
        result = validate_artifact(
            name=item["name"],
            artifact_path=artifact_path,
            schema_path=schema_path,
        )
        checks.append(result)

    passed = sum(1 for c in checks if c.status == "PASS")
    failed = len(checks) - passed
    overall = "PASS" if failed == 0 else "FAIL"

    return {
        "schema_version": "1.0.0",
        "gate": requested_gate,
        "manifest_path": str(manifest_path.resolve()),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "summary": {
            "total": len(checks),
            "passed": passed,
            "failed": failed,
        },
        "checks": [
            {
                **{
                    k: v
                    for k, v in asdict(check).items()
                    if k != "errors"
                },
                "errors": [asdict(e) for e in check.errors],
            }
            for check in checks
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pipeline artifacts at Gate-In/Out using JSON Schema."
    )
    parser.add_argument(
        "--gate",
        required=True,
        choices=["in", "out"],
        help="Gate type to validate against manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to gate manifest JSON.",
    )
    parser.add_argument(
        "--report-out",
        help="Optional path to write report JSON.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_gate_validation(
            manifest_path=Path(args.manifest),
            requested_gate=args.gate,
        )
    except FileNotFoundError as err:
        print(f"[gate-validator] file error: {err}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as err:
        print(f"[gate-validator] invalid JSON: {err}", file=sys.stderr)
        return 2
    except ManifestError as err:
        print(f"[gate-validator] invalid manifest: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # pragma: no cover
        print(f"[gate-validator] unexpected error: {err}", file=sys.stderr)
        return 2

    output = json.dumps(report, indent=2 if args.pretty else None, ensure_ascii=True)
    print(output)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
