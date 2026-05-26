from __future__ import annotations

# Measure prompt token lengths for data.
#
# PYTHONPATH=src python -m lean_eval.context_sizing \
#     --input data/processed/test.jsonl \
#     --output-stats outputs/context_stats.json \
#     --output-measured outputs/test.measured.jsonl \
#     --output-filtered outputs/test.filtered.jsonl

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from .io import read_jsonl, write_jsonl
from .prompts import build_chatml_prompt


PERCENTILES = (50, 75, 90, 95, 99, 100)


# Round a value up to the next power of two, optionally capped at a maximum.
# value: Token length to round up.
# cap: Optional upper bound for the returned value.
def nearest_power_of_two_at_least(value: int, *, cap: int | None = None) -> int:
    if value <= 1:
        result = 1
    else:
        result = 1 << (value - 1).bit_length()
    return min(result, cap) if cap is not None else result


# Compute an integer percentile from a list of token lengths.
# values: Token lengths to summarize.
# pct: Percentile to compute on the 0-100 scale.
def percentile(values: list[int], pct: int) -> int:
    if not values:
        raise ValueError("cannot compute percentiles for an empty dataset")
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * (pct / 100)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    interpolated = ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
    return int(round(interpolated))


# Measure prompt token lengths and derive filtering statistics for a dataset.
# rows: Dataset rows whose prompts will be tokenized.
# tokenizer: Tokenizer used to encode prompts.
# cap: Maximum allowed recommended sequence length.
# prompt_builder: Function that formats each row into the prompt to measure.
def compute_context_stats(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    cap: int,
    prompt_builder: Any = build_chatml_prompt,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    measured: list[dict[str, Any]] = []
    lengths: list[int] = []
    for idx, row in enumerate(rows):
        prompt = prompt_builder(row)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        length = len(token_ids)
        lengths.append(length)
        measured_row = dict(row)
        measured_row["prompt_tokens"] = length
        measured_row.setdefault("item_id", str(idx))
        measured.append(measured_row)

    percentiles = {f"p{pct}": percentile(lengths, pct) for pct in PERCENTILES}
    recommended = nearest_power_of_two_at_least(percentiles["p90"], cap=cap)
    kept = sum(1 for length in lengths if length <= recommended)
    filtered = len(lengths) - kept
    stats = {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "percentiles": percentiles,
        "recommended_max_seq_length": recommended,
        "recommended_rule": f"P90 rounded up to a power of two, capped at {cap}",
        "kept_at_recommended": kept,
        "filtered_at_recommended": filtered,
        "filtered_fraction_at_recommended": filtered / len(lengths) if lengths else 0.0,
    }
    return stats, measured


# Parse CLI arguments for prompt-length measurement.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure ChatML prompt token lengths.")
    parser.add_argument("--input", type=Path, required=True, help="Dataset JSONL.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Tokenizer/model id.")
    parser.add_argument("--output-stats", type=Path, required=True, help="Stats JSON output.")
    parser.add_argument("--output-measured", type=Path, help="Optional JSONL with prompt_tokens.")
    parser.add_argument(
        "--output-filtered",
        type=Path,
        help="Optional JSONL containing rows with prompt_tokens <= recommended length.",
    )
    parser.add_argument("--l4-cap", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args(argv)


# Load a dataset, measure prompt lengths, and write the requested outputs.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    rows = read_jsonl(args.input)
    stats, measured = compute_context_stats(rows, tokenizer, cap=args.l4_cap)

    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_measured:
        write_jsonl(args.output_measured, measured)
    if args.output_filtered:
        max_len = int(stats["recommended_max_seq_length"])
        write_jsonl(args.output_filtered, (row for row in measured if row["prompt_tokens"] <= max_len))
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
