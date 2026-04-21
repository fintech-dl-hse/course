#!/usr/bin/env python3
"""Validate a critic-output JSON file against the embedded Draft 2020-12 schema.

Usage:
    validate_critic.py <json_path>

Exit codes:
    0 — valid
    1 — schema-invalid (error detail is written as JSON to stderr)
    2 — file not found or malformed JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Draft 2020-12 schema for critic output.
# Kept in sync with the contract block in .claude/agents/manim-frame-critic.md.
CRITIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ManimFrameCriticOutput",
    "type": "object",
    "required": ["approved", "video_hash", "issues"],
    "additionalProperties": False,
    "properties": {
        "approved": {"type": "boolean"},
        "video_hash": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "frame",
                    "severity",
                    "category",
                    "description",
                    "suggested_fix",
                ],
                "additionalProperties": False,
                "properties": {
                    "frame": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["high", "med", "low"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "overlap",
                            "text-clip",
                            "offscreen",
                            "z-fight",
                            "other",
                        ],
                    },
                    "description": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
            },
        },
    },
}


def validate_obj(obj: Any) -> Optional[str]:
    """Return None if valid, else error message string."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as e:
        return f"jsonschema not installed: {e}"

    validator = Draft202012Validator(CRITIC_SCHEMA)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    # Emit first error as the canonical failure reason.
    first = errors[0]
    return f"{list(first.absolute_path)}: {first.message}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate manim-frame-critic JSON output.")
    parser.add_argument("json_path", type=Path, help="Path to critic JSON output")
    args = parser.parse_args(argv)

    if not args.json_path.exists():
        json.dump(
            {"ok": False, "error": f"file not found: {args.json_path}"},
            sys.stderr,
        )
        sys.stderr.write("\n")
        return 2

    try:
        obj = json.loads(args.json_path.read_text())
    except json.JSONDecodeError as e:
        json.dump({"ok": False, "error": f"invalid JSON: {e}"}, sys.stderr)
        sys.stderr.write("\n")
        return 1

    err = validate_obj(obj)
    if err is None:
        return 0

    # Additional guard: approved=true forbidden when any high severity issue exists.
    if obj.get("approved") is True and any(
        isinstance(i, dict) and i.get("severity") == "high"
        for i in obj.get("issues", [])
    ):
        err = "approved=true is forbidden when any issue has severity=high"

    json.dump({"ok": False, "error": err}, sys.stderr)
    sys.stderr.write("\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
