#!/usr/bin/env python
from __future__ import annotations

# Runs second pass evaluation: does first-pass generation and verification, then repairs
# generation for failed rows using Lean feedback, and finally aggregate pass@2-style results.
#
# python scripts/evaluate_model_pass2.py \
#     --input data/processed/test.jsonl \
#     --model Qwen/Qwen3.5-9B \
#     --adapter adapters/run_01 \
#     --output-dir outputs/eval_pass2_run_01 \
#     --max-seq-length 2048 \
#     --max-new-tokens 512 \
#     --workers 4 \
#     --lake-project-dir lean

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from lean_eval.context_sizing import compute_context_stats
from lean_eval.generate import generate_rows, load_model_and_tokenizer
from lean_eval.harness import verify_files
from lean_eval.io import read_jsonl, write_jsonl
from lean_eval.prompts import build_repair_chatml_prompt
from lean_eval.report import summarize_verification


# Parse CLI arguments for the two-pass evaluation pipeline.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-attempt Lean evaluation pipeline: normal generation/verification for pass@1, "
            "then repair-only generation for failed rows using Lean compiler feedback."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Dataset JSONL to evaluate.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Base model id or local path.")
    parser.add_argument("--adapter", help="Optional PEFT adapter directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--l4-cap", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--header-timeout", type=float, default=180.0)
    parser.add_argument("--lake-project-dir", type=Path, default=Path("lean"))
    parser.add_argument("--keep-failures-dir", type=Path)
    parser.add_argument("--no-keep-failures", action="store_true")
    parser.add_argument("--max-feedback-lines", type=int, default=24)
    parser.add_argument("--max-feedback-chars", type=int, default=2500)
    parser.add_argument("--max-previous-chars", type=int, default=1800)
    return parser.parse_args(argv)


# Write a JSON value to disk with stable formatting.
# path: Destination JSON file.
# value: Python value to serialize.
def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Normalize one line of Lean feedback before it is shown to the repair model.
# line: Raw stderr line produced during verification.
def _normalize_feedback_line(line: str) -> str:
    line = line.rstrip()
    line = re.sub(r"verify_[^:\s]+\.lean", "proof.lean", line)
    return line


# Extract a compact Lean feedback string from one failed verification row.
# verified_row: Verification result row containing stderr and status fields.
# max_lines: Maximum number of feedback lines to keep.
# max_chars: Maximum number of characters to keep in the final feedback string.
def extract_lean_feedback(
    verified_row: dict[str, Any],
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    lines = [_normalize_feedback_line(line) for line in verified_row.get("stderr", []) or [] if str(line).strip()]
    error = str(verified_row.get("error", "")).strip()
    if error:
        lines.insert(0, error)
    if not lines:
        lines = [f"Previous attempt failed with status: {verified_row.get('status', 'unknown')}"]

    informative = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in (
                "error:",
                "unsolved goals",
                "unknown",
                "type mismatch",
                "application type mismatch",
                "expected",
                "tactic",
                "failed",
            )
        )
    ]
    selected = informative[-max_lines:] if informative else lines[-max_lines:]
    text = "\n".join(selected).strip()
    if len(text) > max_chars:
        text = text[-max_chars:].lstrip()
    return text or f"Previous attempt failed with status: {verified_row.get('status', 'unknown')}"


# Truncate a previous prediction so it fits within the repair prompt budget.
# text: Previous model prediction text to trim.
# max_chars: Maximum number of characters to retain.
def trim_previous_prediction(text: str, *, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


# Build the dataset rows used for second-pass repair generation.
# dataset_rows: First-pass filtered dataset rows that were originally evaluated.
# pass1_predictions: First-pass prediction rows keyed by item_id.
# pass1_verified: First-pass verification rows used to identify failed examples.
# max_feedback_lines: Maximum number of Lean feedback lines to keep per row.
# max_feedback_chars: Maximum number of Lean feedback characters to keep per row.
# max_previous_chars: Maximum number of previous-prediction characters to keep per row.
def build_repair_rows(
    dataset_rows: list[dict[str, Any]],
    pass1_predictions: list[dict[str, Any]],
    pass1_verified: list[dict[str, Any]],
    *,
    max_feedback_lines: int,
    max_feedback_chars: int,
    max_previous_chars: int,
) -> list[dict[str, Any]]:
    by_dataset = {str(row["item_id"]): row for row in dataset_rows}
    by_pred = {str(row["item_id"]): row for row in pass1_predictions}
    repair_rows: list[dict[str, Any]] = []
    for verified in pass1_verified:
        if verified.get("pass") is True:
            continue
        item_id = str(verified["item_id"])
        pred = by_pred[item_id]
        previous_prediction = trim_previous_prediction(pred.get("prediction", ""), max_chars=max_previous_chars)
        repair_row = {
            **by_dataset[item_id],
            "previous_prediction": previous_prediction,
            "lean_feedback": extract_lean_feedback(
                verified,
                max_lines=max_feedback_lines,
                max_chars=max_feedback_chars,
            ),
            "pass1_status": verified.get("status"),
            "pass1_error": verified.get("error"),
        }
        repair_rows.append(repair_row)
    return repair_rows


# Build a synthetic verification row for a repair example filtered before verification.
# row: Measured repair dataset row associated with the filtered example.
# prediction_row: Pass-2 prediction row containing the filtering metadata.
def build_synthetic_filtered_verify_row(
    row: dict[str, Any],
    prediction_row: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "item_id": str(row.get("item_id")),
        "original_id": str(row.get("original_id", row.get("item_id"))),
        "pass": False,
        "status": "filtered_context_length",
        "error": "repair prompt exceeded max_seq_length",
        "elapsed_sec": 0.0,
        "prompt_tokens": int(prediction_row.get("prompt_tokens", 0)),
        "attempt": 2,
    }
    for key in ("source_dataset", "is_geometry", "state_after"):
        if key in row:
            result[key] = row[key]
    return result


# Merge pass-1 and pass-2 verification outputs into the final per-item results.
# pass1_verified: First-pass verification rows.
# pass1_predictions: First-pass prediction rows.
# pass2_verified: Second-pass verification rows.
# pass2_predictions: Second-pass prediction rows.
def merge_final_rows(
    pass1_verified: list[dict[str, Any]],
    pass1_predictions: list[dict[str, Any]],
    pass2_verified: list[dict[str, Any]],
    pass2_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pass2 = {str(row["item_id"]): row for row in pass2_verified}
    by_pass1_pred = {str(row["item_id"]): row for row in pass1_predictions}
    by_pass2_pred = {str(row["item_id"]): row for row in pass2_predictions}
    merged: list[dict[str, Any]] = []
    for pass1 in pass1_verified:
        item_id = str(pass1["item_id"])
        pass2 = by_pass2.get(item_id)
        final = dict(pass2 if pass2 and pass2.get("pass") else pass1)
        final["pass1_status"] = pass1.get("status")
        final["pass1_pass"] = pass1.get("pass")
        final["pass1_prediction"] = by_pass1_pred.get(item_id, {}).get("prediction", "")
        if pass2 is not None:
            final["pass2_status"] = pass2.get("status")
            final["pass2_pass"] = pass2.get("pass")
            final["pass2_prediction"] = by_pass2_pred.get(item_id, {}).get("prediction", "")
        else:
            final["pass2_status"] = None
            final["pass2_pass"] = None
            final["pass2_prediction"] = ""
        final["solved_on_attempt"] = 1 if pass1.get("pass") else (2 if pass2 and pass2.get("pass") else None)
        final["attempts_used"] = 1 if pass1.get("pass") else 2
        merged.append(final)
    return merged


# Run the full two-pass evaluation pipeline and write all derived artifacts.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    retry_max_new_tokens = args.retry_max_new_tokens or args.max_new_tokens

    output_dir = args.output_dir
    pass1_dir = output_dir / "pass1"
    pass2_dir = output_dir / "pass2"
    final_dir = output_dir / "final"
    for path in (pass1_dir, pass2_dir, final_dir):
        path.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    pass1_stats, pass1_measured = compute_context_stats(rows, tokenizer, cap=args.l4_cap)
    max_seq_length = args.max_seq_length or int(pass1_stats["recommended_max_seq_length"])
    pass1_filtered = [row for row in pass1_measured if int(row["prompt_tokens"]) <= max_seq_length]

    pass1_stats = dict(pass1_stats)
    pass1_stats["selected_max_seq_length"] = max_seq_length
    pass1_stats["kept_at_selected"] = len(pass1_filtered)
    pass1_stats["filtered_at_selected"] = len(pass1_measured) - len(pass1_filtered)
    pass1_stats["filtered_fraction_at_selected"] = (
        (len(pass1_measured) - len(pass1_filtered)) / len(pass1_measured) if pass1_measured else 0.0
    )

    write_json(pass1_dir / "context_stats.json", pass1_stats)
    write_jsonl(pass1_dir / "dataset.measured.jsonl", pass1_measured)
    write_jsonl(pass1_dir / "dataset.filtered.jsonl", pass1_filtered)

    pass1_predictions = generate_rows(
        pass1_filtered,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(pass1_dir / "predictions.jsonl", pass1_predictions)

    keep_failures_dir = None
    if not args.no_keep_failures:
        root = args.keep_failures_dir or (output_dir / "failed_lean")
        keep_failures_dir = {"pass1": root / "pass1", "pass2": root / "pass2"}

    pass1_raw_summary = verify_files(
        input_path=None,
        dataset_path=pass1_dir / "dataset.filtered.jsonl",
        predictions_path=pass1_dir / "predictions.jsonl",
        output_path=pass1_dir / "verified.jsonl",
        summary_path=pass1_dir / "verified.raw_summary.json",
        lake_project_dir=args.lake_project_dir,
        repl_cmd=["lake", "env", "repl"],
        workers=args.workers,
        timeout=args.timeout,
        header_timeout=args.header_timeout,
        split_mode="full",
        backend="lean",
        keep_failures_dir=keep_failures_dir["pass1"] if keep_failures_dir else None,
        completion_field="prediction",
    )
    pass1_verified = read_jsonl(pass1_dir / "verified.jsonl")
    pass1_summary = summarize_verification(pass1_verified)
    write_json(pass1_dir / "summary.json", pass1_summary)

    repair_rows = build_repair_rows(
        pass1_filtered,
        pass1_predictions,
        pass1_verified,
        max_feedback_lines=args.max_feedback_lines,
        max_feedback_chars=args.max_feedback_chars,
        max_previous_chars=args.max_previous_chars,
    )
    write_jsonl(pass2_dir / "repair_dataset.jsonl", repair_rows)
    if repair_rows:
        pass2_stats, pass2_measured = compute_context_stats(
            repair_rows,
            tokenizer,
            cap=max_seq_length,
            prompt_builder=build_repair_chatml_prompt,
        )
        pass2_filtered = [row for row in pass2_measured if int(row["prompt_tokens"]) <= max_seq_length]
        pass2_stats = dict(pass2_stats)
        pass2_stats["selected_max_seq_length"] = max_seq_length
        pass2_stats["kept_at_selected"] = len(pass2_filtered)
        pass2_stats["filtered_at_selected"] = len(pass2_measured) - len(pass2_filtered)
        pass2_stats["filtered_fraction_at_selected"] = (
            (len(pass2_measured) - len(pass2_filtered)) / len(pass2_measured) if pass2_measured else 0.0
        )
        write_json(pass2_dir / "context_stats.json", pass2_stats)
        write_jsonl(pass2_dir / "dataset.measured.jsonl", pass2_measured)
        write_jsonl(pass2_dir / "dataset.filtered.jsonl", pass2_filtered)

        pass2_predictions = generate_rows(
            pass2_filtered,
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_new_tokens=retry_max_new_tokens,
            prompt_builder=build_repair_chatml_prompt,
        )
        write_jsonl(pass2_dir / "predictions.jsonl", pass2_predictions)

        generated_predictions = [row for row in pass2_predictions if row.get("generation_status") == "generated"]
        generated_ids = {str(row["item_id"]) for row in generated_predictions}
        verify_dataset_rows = [row for row in pass2_filtered if str(row["item_id"]) in generated_ids]

        pass2_generated_verified: list[dict[str, Any]] = []
        if generated_predictions:
            write_jsonl(pass2_dir / "verify_dataset.jsonl", verify_dataset_rows)
            write_jsonl(pass2_dir / "verify_predictions.jsonl", generated_predictions)
            verify_files(
                input_path=None,
                dataset_path=pass2_dir / "verify_dataset.jsonl",
                predictions_path=pass2_dir / "verify_predictions.jsonl",
                output_path=pass2_dir / "verified.generated.jsonl",
                summary_path=pass2_dir / "verified.generated.raw_summary.json",
                lake_project_dir=args.lake_project_dir,
                repl_cmd=["lake", "env", "repl"],
                workers=args.workers,
                timeout=args.timeout,
                header_timeout=args.header_timeout,
                split_mode="full",
                backend="lean",
                keep_failures_dir=keep_failures_dir["pass2"] if keep_failures_dir else None,
                completion_field="prediction",
            )
            pass2_generated_verified = read_jsonl(pass2_dir / "verified.generated.jsonl")

        by_verified_generated = {str(row["item_id"]): row for row in pass2_generated_verified}
        by_prediction = {str(row["item_id"]): row for row in pass2_predictions}
        pass2_verified: list[dict[str, Any]] = []
        for row in pass2_measured:
            item_id = str(row["item_id"])
            if item_id in by_verified_generated:
                merged = dict(by_verified_generated[item_id])
                merged["attempt"] = 2
                pass2_verified.append(merged)
                continue
            prediction_row = by_prediction.get(item_id)
            if prediction_row is None:
                continue
            pass2_verified.append(build_synthetic_filtered_verify_row(row, prediction_row))
    else:
        pass2_stats = {
            "count": 0,
            "min": 0,
            "max": 0,
            "percentiles": {},
            "recommended_max_seq_length": max_seq_length,
            "recommended_rule": f"P90 rounded up to a power of two, capped at {max_seq_length}",
            "selected_max_seq_length": max_seq_length,
            "kept_at_selected": 0,
            "filtered_at_selected": 0,
            "filtered_fraction_at_selected": 0.0,
        }
        pass2_measured = []
        pass2_filtered = []
        pass2_predictions = []
        generated_predictions = []
        pass2_verified = []
        write_json(pass2_dir / "context_stats.json", pass2_stats)
        write_jsonl(pass2_dir / "dataset.measured.jsonl", pass2_measured)
        write_jsonl(pass2_dir / "dataset.filtered.jsonl", pass2_filtered)
        write_jsonl(pass2_dir / "predictions.jsonl", pass2_predictions)

    write_jsonl(pass2_dir / "verified.jsonl", pass2_verified)
    pass2_summary = summarize_verification(pass2_verified)
    write_json(pass2_dir / "summary.json", pass2_summary)

    final_verified = merge_final_rows(
        pass1_verified,
        pass1_predictions,
        pass2_verified,
        pass2_predictions,
    )
    write_jsonl(final_dir / "verified.jsonl", final_verified)
    final_summary = summarize_verification(final_verified)
    write_json(final_dir / "summary.json", final_summary)

    repair_successes = sum(
        1 for row in pass2_verified if row.get("pass") is True and row.get("status") == "pass"
    )
    repair_metrics = {
        "repair_candidates": len(repair_rows),
        "repair_prompt_filtered": sum(1 for row in pass2_verified if row.get("status") == "filtered_context_length"),
        "repair_attempts_generated": len(generated_predictions),
        "repair_successes": repair_successes,
        "repair_success_rate": repair_successes / len(generated_predictions) if generated_predictions else 0.0,
    }
    write_json(output_dir / "repair_metrics.json", repair_metrics)

    metadata = {
        "run_name": args.run_name or output_dir.name,
        "input": str(args.input),
        "model": args.model,
        "adapter": args.adapter,
        "output_dir": str(output_dir),
        "max_seq_length": max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": retry_max_new_tokens,
        "workers": args.workers,
        "timeout": args.timeout,
        "header_timeout": args.header_timeout,
        "lake_project_dir": str(args.lake_project_dir),
        "input_rows": len(rows),
        "pass1_rows": len(pass1_verified),
        "pass2_candidates": len(repair_rows),
        "pass2_rows": len(pass2_verified),
        "pass1_raw_verification_summary": pass1_raw_summary,
        "repair_metrics": repair_metrics,
        "pass1_summary": pass1_summary,
        "pass2_summary": pass2_summary,
        "final_summary": final_summary,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "artifacts": {
            "pass1_dir": str(pass1_dir),
            "pass2_dir": str(pass2_dir),
            "final_dir": str(final_dir),
            "repair_metrics": str(output_dir / "repair_metrics.json"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
