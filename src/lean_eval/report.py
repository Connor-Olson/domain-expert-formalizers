from __future__ import annotations

# Summarizes Lean verification results.
#
# PYTHONPATH=src python -m lean_eval.report \
#     --verified outputs/verified.jsonl \
#     --output-json outputs/summary.json

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from .io import read_jsonl


# Normalize a possibly missing boolean-like value into a reporting label.
# value: Raw field value to label.
# true_label: Label to use when the value is exactly True.
# false_label: Label to use when the value is exactly False.
def _label(value: Any, *, true_label: str = "true", false_label: str = "false") -> str:
    if value is True:
        return true_label
    if value is False:
        return false_label
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


# Infer the completion type label for a verification row.
# row: Verification result row to categorize.
def _completion_type(row: dict[str, Any]) -> str:
    value = row.get("completion_type")
    if value:
        return str(value)
    state_after = str(row.get("state_after", "")).strip().lower()
    if state_after == "no goals":
        return "single_tactic_closure"
    if state_after:
        return "proof_suffix"
    return "unknown"


# Compute total, passed count, and pass rate for a list of rows.
# rows: Verification rows to summarize.
def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("pass") is True)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }


# Group rows by a derived key and compute pass-rate stats per group.
# rows: Verification rows to group.
# key_fn: Function that returns the grouping key for each row.
def _group_rate(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return {key: _rate(value) for key, value in sorted(grouped.items())}


# Build the full JSON summary for a set of verification rows.
# rows: Verification rows to summarize.
def summarize_verification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[str(row.get("status", "unknown"))] += 1
    timeout_count = by_status.get("timeout", 0)
    return {
        "overall": _rate(rows),
        "by_status": dict(sorted(by_status.items())),
        "timeout_rate": timeout_count / len(rows) if rows else 0.0,
        "by_geometry": _group_rate(
            rows,
            lambda row: _label(row.get("is_geometry"), true_label="geometry", false_label="non_geometry"),
        ),
        "by_completion_type": _group_rate(rows, _completion_type),
        "by_source_dataset": _group_rate(rows, lambda row: _label(row.get("source_dataset"))),
    }


# Parse CLI arguments for the verification summary command.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Lean verification results.")
    parser.add_argument("--verified", type=Path, required=True, help="Verification JSONL.")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


# Load verification rows, write the JSON summary, and print it.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = read_jsonl(args.verified)
    summary = summarize_verification(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
