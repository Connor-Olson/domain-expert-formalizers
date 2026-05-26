#!/usr/bin/env python
from __future__ import annotations

# Runs the full evaluation pipeline for a base model or adapter: context measurement, generation,
# verification, and summary writing.
#
# To run with a base model:
# python scripts/evaluate_model.py \
#     --input outputs/test.l4.jsonl \
#     --model Qwen/Qwen3.5-9B \
#     --output-dir outputs/eval_base_99 \
#     --max-seq-length 2048 \
#     --max-new-tokens 512 \
#     --workers 4 \
#     --header-timeout 180 \
#     --timeout 20 \
#     --lake-project-dir lean
# To run with an adapater:
# python scripts/evaluate_model.py \
# --input data/processed/test.jsonl \
# --model Qwen/Qwen3.5-9B \
# --adapter adapters/run_01 \
# --output-dir outputs/eval_run_01 \
# --max-seq-length 2048 \
# --max-new-tokens 512 \
# --workers 4 \
# --lake-project-dir lean

import argparse
import json
import sys
import time
from pathlib import Path
from lean_eval.context_sizing import compute_context_stats
from lean_eval.generate import generate_rows, load_model_and_tokenizer
from lean_eval.harness import verify_files
from lean_eval.io import read_jsonl, write_jsonl
from lean_eval.report import summarize_verification


# Parse CLI arguments for the end-to-end evaluation pipeline.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full L4 evaluation pipeline: measure/filter prompts, generate completions, "
            "clean model output, verify with Lean, and summarize results."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Dataset JSONL to evaluate.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Base model id or local path.")
    parser.add_argument("--adapter", help="Optional PEFT adapter directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where all intermediate and final evaluation artifacts are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional metadata label. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Override prompt length cap. Defaults to P90 rounded up, capped by --l4-cap.",
    )
    parser.add_argument("--l4-cap", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit before filtering.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--header-timeout", type=float, default=180.0)
    parser.add_argument("--lake-project-dir", type=Path, default=Path("lean"))
    parser.add_argument(
        "--keep-failures-dir",
        type=Path,
        help="Optional directory for failed generated Lean files. Defaults to OUTPUT_DIR/failed_lean.",
    )
    parser.add_argument(
        "--no-keep-failures",
        action="store_true",
        help="Do not retain failed generated Lean files.",
    )
    return parser.parse_args(argv)


# Write a JSON value to disk with stable formatting.
# path: Destination JSON file.
# value: Python value to serialize.
def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Run the full evaluation pipeline and write all derived artifacts.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    context_stats, measured = compute_context_stats(rows, tokenizer, cap=args.l4_cap)
    max_seq_length = args.max_seq_length or int(context_stats["recommended_max_seq_length"])
    filtered = [row for row in measured if int(row["prompt_tokens"]) <= max_seq_length]

    context_stats = dict(context_stats)
    context_stats["selected_max_seq_length"] = max_seq_length
    context_stats["kept_at_selected"] = len(filtered)
    context_stats["filtered_at_selected"] = len(measured) - len(filtered)
    context_stats["filtered_fraction_at_selected"] = (
        (len(measured) - len(filtered)) / len(measured) if measured else 0.0
    )

    measured_path = output_dir / "dataset.measured.jsonl"
    filtered_path = output_dir / "dataset.filtered.jsonl"
    context_stats_path = output_dir / "context_stats.json"
    predictions_path = output_dir / "predictions.jsonl"
    verified_path = output_dir / "verified.jsonl"
    raw_summary_path = output_dir / "verified.raw_summary.json"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"

    write_json(context_stats_path, context_stats)
    write_jsonl(measured_path, measured)
    write_jsonl(filtered_path, filtered)

    predictions = generate_rows(
        filtered,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(predictions_path, predictions)

    keep_failures_dir = None
    if not args.no_keep_failures:
        keep_failures_dir = args.keep_failures_dir or (output_dir / "failed_lean")

    raw_summary = verify_files(
        input_path=None,
        dataset_path=filtered_path,
        predictions_path=predictions_path,
        output_path=verified_path,
        summary_path=raw_summary_path,
        lake_project_dir=args.lake_project_dir,
        repl_cmd=["lake", "env", "repl"],
        workers=args.workers,
        timeout=args.timeout,
        header_timeout=args.header_timeout,
        split_mode="full",
        backend="lean",
        keep_failures_dir=keep_failures_dir,
        completion_field="prediction",
    )

    verified_rows = read_jsonl(verified_path)
    summary = summarize_verification(verified_rows)
    write_json(summary_path, summary)

    metadata = {
        "run_name": args.run_name or output_dir.name,
        "input": str(args.input),
        "model": args.model,
        "adapter": args.adapter,
        "output_dir": str(output_dir),
        "max_seq_length": max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "workers": args.workers,
        "timeout": args.timeout,
        "header_timeout": args.header_timeout,
        "lake_project_dir": str(args.lake_project_dir),
        "input_rows": len(rows),
        "measured_rows": len(measured),
        "filtered_rows": len(filtered),
        "generated_rows": len(predictions),
        "raw_verification_summary": raw_summary,
        "summary": summary,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "artifacts": {
            "context_stats": str(context_stats_path),
            "measured_dataset": str(measured_path),
            "filtered_dataset": str(filtered_path),
            "predictions": str(predictions_path),
            "verified": str(verified_path),
            "raw_summary": str(raw_summary_path),
            "summary": str(summary_path),
            "failed_lean": str(keep_failures_dir) if keep_failures_dir else None,
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
