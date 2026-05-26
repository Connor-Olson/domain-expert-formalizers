#!/usr/bin/env python
from __future__ import annotations

# Create theorem-aware train/test splits and CoT seed files from one or more processed JSONL datasets.
# The outputs are the intermediate fine-tuning artifacts consumed by later training and evaluation scripts.
#
# python scripts/prepare_finetune_splits.py \
#     --input data/processed/train.jsonl \
#     --output-dir data/raw/splits/finetune \
#     --test-geometry-proofs 100 \
#     --test-non-geometry-proofs 200

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from lean_eval.geometry import proof_domain, row_is_geometry
from lean_eval.io import read_jsonl, write_jsonl
from lean_eval.prompts import build_chatml_prompt, build_chatml_training_text


# Split rows into geometry and non-geometry partitions under the requested rule.
# rows: Rows to partition by domain.
# geometry_rule: Geometry-labeling rule to apply.
def split_rows_by_geometry(
    rows: list[dict[str, Any]],
    *,
    geometry_rule: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    geometry: list[dict[str, Any]] = []
    non_geometry: list[dict[str, Any]] = []
    for row in rows:
        if row_is_geometry(row, rule=geometry_rule):
            geometry.append(row)
        else:
            non_geometry.append(row)
    return geometry, non_geometry


# Group rows by original proof identifier.
# rows: Rows to bucket by original_id.
def group_by_original_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        original_id = str(row.get("original_id", row.get("item_id", idx)))
        grouped[original_id].append(row)
    return dict(grouped)


# Sample proof ids for the held-out test split with exact geometry/non-geometry counts.
# grouped: Rows grouped by proof id.
# geometry_proofs: Number of geometry proofs to sample.
# non_geometry_proofs: Number of non-geometry proofs to sample.
# seed: Random seed controlling sampling.
# geometry_rule: Geometry-labeling rule used to classify proofs.
def sample_test_proofs(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    geometry_proofs: int,
    non_geometry_proofs: int,
    seed: int,
    geometry_rule: str,
) -> set[str]:
    geometry = [
        proof_id
        for proof_id, rows in grouped.items()
        if proof_domain(rows, geometry_rule=geometry_rule)
    ]
    non_geometry = [
        proof_id
        for proof_id, rows in grouped.items()
        if not proof_domain(rows, geometry_rule=geometry_rule)
    ]
    if len(geometry) < geometry_proofs:
        raise ValueError(f"requested {geometry_proofs} geometry test proofs, found only {len(geometry)}")
    if len(non_geometry) < non_geometry_proofs:
        raise ValueError(
            f"requested {non_geometry_proofs} non-geometry test proofs, found only {len(non_geometry)}"
        )

    rng = random.Random(seed)
    return set(rng.sample(geometry, geometry_proofs)) | set(rng.sample(non_geometry, non_geometry_proofs))


# Flatten selected proof groups back into a row list in proof-id order.
# grouped: Rows grouped by proof id.
# proof_ids: Proof ids to expand into rows.
def flatten_groups(grouped: dict[str, list[dict[str, Any]]], proof_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proof_id in sorted(proof_ids):
        rows.extend(grouped[proof_id])
    return rows


EMPTY_THINK_PREFIX = "<think>\n</think>\n"
DEFAULT_COT_PLACEHOLDER = "__COT_REASONING_TO_FILL__"


# Prepend an empty <think> block to completions that do not already have one.
# rows: Rows whose formal_completion fields may be updated.
def with_empty_think(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        completion = str(out.get("formal_completion", ""))
        if not completion.startswith("<think>"):
            out["formal_completion"] = f"{EMPTY_THINK_PREFIX}{completion}"
        updated.append(out)
    return updated


# Prepend a placeholder <think> block to completions that do not already have one.
# rows: Rows whose formal_completion fields may be updated.
# placeholder: Placeholder reasoning text to insert inside the think block.
def with_cot_placeholder(rows: list[dict[str, Any]], *, placeholder: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        completion = str(out.get("formal_completion", ""))
        if not completion.startswith("<think>"):
            out["formal_completion"] = f"<think>\n{placeholder}\n</think>\n{completion}"
        updated.append(out)
    return updated


# Measure prompt and training token counts for each row with a specific tokenizer.
# rows: Rows to annotate with token counts.
# model: Tokenizer/model identifier used for measurement.
def compute_prompt_tokens(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    measured: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["prompt_tokens"] = len(tokenizer.encode(build_chatml_prompt(row), add_special_tokens=False))
        out["training_tokens"] = len(
            tokenizer.encode(build_chatml_training_text(row), add_special_tokens=False)
        )
        measured.append(out)
    return measured


# Sample exact row counts for geometry and non-geometry examples.
# rows: Candidate rows to sample from.
# geometry_count: Number of geometry rows to sample.
# non_geometry_count: Number of non-geometry rows to sample.
# seed: Random seed controlling sampling.
# geometry_rule: Geometry-labeling rule used to classify rows.
def sample_rows_exact(
    rows: list[dict[str, Any]],
    *,
    geometry_count: int,
    non_geometry_count: int,
    seed: int,
    geometry_rule: str,
) -> list[dict[str, Any]]:
    geometry, non_geometry = split_rows_by_geometry(rows, geometry_rule=geometry_rule)
    if len(geometry) < geometry_count:
        raise ValueError(f"requested {geometry_count} geometry rows, found only {len(geometry)}")
    if len(non_geometry) < non_geometry_count:
        raise ValueError(f"requested {non_geometry_count} non-geometry rows, found only {len(non_geometry)}")

    rng = random.Random(seed)
    return rng.sample(geometry, geometry_count) + rng.sample(non_geometry, non_geometry_count)


# Choose short CoT seed rows with exact geometry and non-geometry counts.
# rows: Candidate rows to choose from.
# geometry_count: Number of geometry rows to keep.
# non_geometry_count: Number of non-geometry rows to keep.
# model: Optional tokenizer/model id used to rank rows by length.
# seed: Random seed used when length ranking is disabled.
# geometry_rule: Geometry-labeling rule used to classify rows.
def select_short_rows_exact(
    rows: list[dict[str, Any]],
    *,
    geometry_count: int,
    non_geometry_count: int,
    model: str | None,
    seed: int,
    geometry_rule: str,
) -> list[dict[str, Any]]:
    geometry, non_geometry = split_rows_by_geometry(rows, geometry_rule=geometry_rule)
    if len(geometry) < geometry_count:
        raise ValueError(f"requested {geometry_count} geometry CoT rows, found only {len(geometry)}")
    if len(non_geometry) < non_geometry_count:
        raise ValueError(
            f"requested {non_geometry_count} non-geometry CoT rows, found only {len(non_geometry)}"
        )

    if model:
        measured = compute_prompt_tokens(rows, model)
        measured_by_item = {str(row.get("item_id", "")): row for row in measured}

        # Rank candidate rows by training length, prompt length, and item id.
        # row: Candidate row to rank.
        def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
            measured_row = measured_by_item[str(row.get("item_id", ""))]
            return (
                int(measured_row["training_tokens"]),
                int(measured_row["prompt_tokens"]),
                str(row.get("item_id", "")),
            )

        geometry = sorted(geometry, key=sort_key)
        non_geometry = sorted(non_geometry, key=sort_key)
    else:
        rng = random.Random(seed)
        rng.shuffle(geometry)
        rng.shuffle(non_geometry)

    return geometry[:geometry_count] + non_geometry[:non_geometry_count]


# Choose proof ids for CoT seeding while balancing domains and preferring short proofs.
# grouped: Rows grouped by proof id.
# train_ids: Proof ids available for CoT selection.
# geometry_examples: Target geometry row count to accumulate.
# non_geometry_examples: Target non-geometry row count to accumulate.
# model: Optional tokenizer/model id used to rank proofs by length.
# seed: Random seed used when length ranking is disabled.
# geometry_rule: Geometry-labeling rule used to classify proofs.
def select_cot_proof_ids(
    grouped: dict[str, list[dict[str, Any]]],
    train_ids: set[str],
    *,
    geometry_examples: int,
    non_geometry_examples: int,
    model: str | None,
    seed: int,
    geometry_rule: str,
) -> set[str]:
    train_rows = flatten_groups(grouped, train_ids)
    measured_prompt_by_item: dict[str, int] = {}
    measured_training_by_item: dict[str, int] = {}
    if model:
        for row in compute_prompt_tokens(train_rows, model):
            item_id = str(row.get("item_id", ""))
            measured_prompt_by_item[item_id] = int(row["prompt_tokens"])
            measured_training_by_item[item_id] = int(row["training_tokens"])

    geometry = [
        proof_id
        for proof_id in train_ids
        if proof_domain(grouped[proof_id], geometry_rule=geometry_rule)
    ]
    non_geometry = [
        proof_id
        for proof_id in train_ids
        if not proof_domain(grouped[proof_id], geometry_rule=geometry_rule)
    ]

    # Rank candidate proofs by measured length, row count, and proof id.
    # proof_id: Proof identifier to rank.
    def proof_sort_key(proof_id: str) -> tuple[int, int, int, str]:
        rows = grouped[proof_id]
        if measured_prompt_by_item:
            prompt_lengths = [
                measured_prompt_by_item[str(row.get("item_id", ""))]
                for row in rows
                if str(row.get("item_id", "")) in measured_prompt_by_item
            ]
            shortest_prompt = min(prompt_lengths) if prompt_lengths else 10**12
            training_lengths = []
            for row in rows:
                item_id = str(row.get("item_id", ""))
                if item_id in measured_training_by_item:
                    training_lengths.append(measured_training_by_item[item_id])
            max_training = max(training_lengths) if training_lengths else 10**12
        else:
            shortest_prompt = 10**12
            max_training = 10**12
        return max_training, shortest_prompt, len(rows), proof_id

    if measured_prompt_by_item:
        geometry.sort(key=proof_sort_key)
        non_geometry.sort(key=proof_sort_key)
    else:
        rng = random.Random(seed)
        rng.shuffle(geometry)
        rng.shuffle(non_geometry)

    # Select proofs in order until their combined row count reaches the target.
    # proof_ids: Ordered proof identifiers to consume.
    # target_examples: Minimum total row count to accumulate.
    def take_until(proof_ids: list[str], target_examples: int) -> set[str]:
        selected: set[str] = set()
        count = 0
        for proof_id in proof_ids:
            if count >= target_examples:
                break
            selected.add(proof_id)
            count += len(grouped[proof_id])
        return selected

    return take_until(geometry, geometry_examples) | take_until(non_geometry, non_geometry_examples)


# Build the prompt used to ask a model for chain-of-thought reasoning.
# row: Training row whose context and completion define the prompt.
def cot_prompt(row: dict[str, Any]) -> str:
    completion = row.get("formal_completion", "")
    return (
        "Given this Lean 4 tactic state, informal context, and target Lean proof closure, "
        "write concise step-by-step reasoning that explains how to arrive at the Lean code. "
        "Output only the reasoning. Do not include Markdown, code fences, imports, theorem "
        "statements, or the final Lean code.\n\n"
        "INFORMAL CONTEXT:\n"
        f"{row.get('informal_context', '')}\n\n"
        "CURRENT TACTIC STATE:\n"
        f"{row.get('tactic_state', '')}\n\n"
        "LEAN CLOSURE:\n"
        f"{completion}"
    )


# Convert rows into the auxiliary CoT prompt dataset format.
# rows: Rows to convert into CoT-generation prompts.
def build_cot_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "item_id": row.get("item_id"),
                "original_id": row.get("original_id"),
                "is_geometry": row.get("is_geometry"),
                "prompt_tokens": row.get("prompt_tokens"),
                "training_tokens": row.get("training_tokens"),
                "formal_completion": row.get("formal_completion"),
                "cot_prompt": cot_prompt(row),
            }
        )
    return result


# Compute row and proof statistics for a split.
# rows: Rows to summarize.
# grouped: Optional proof-grouped view used to count proof domains.
# geometry_rule: Geometry-labeling rule used to classify rows and proofs.
def split_stats(
    rows: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]] | None = None,
    *,
    geometry_rule: str = "field_or_mathlib_geometry",
) -> dict[str, Any]:
    proof_ids = {str(row.get("original_id", row.get("item_id", ""))) for row in rows}
    stats = Counter()
    for row in rows:
        stats["rows"] += 1
        if row_is_geometry(row, rule=geometry_rule):
            stats["geometry_rows"] += 1
        else:
            stats["non_geometry_rows"] += 1
    result: dict[str, Any] = dict(sorted(stats.items()))
    result["proofs"] = len(proof_ids)
    if grouped is not None:
        result["geometry_proofs"] = sum(
            1
            for proof_id in proof_ids
            if proof_domain(grouped[proof_id], geometry_rule=geometry_rule)
        )
        result["non_geometry_proofs"] = result["proofs"] - result["geometry_proofs"]
    return result


# Parse CLI arguments for theorem-aware split generation.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create row-level train/test splits and CoT seed files for fine-tuning."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Training-pool JSONL. Repeatable. These rows are used to build train_lobotomy/train_CoT.",
    )
    parser.add_argument(
        "--test-pool-input",
        type=Path,
        action="append",
        help=(
            "Optional JSONL used only for test-row sampling. Repeatable. "
            "Defaults to --input when omitted."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=599)
    parser.add_argument("--test-geometry-proofs", type=int, default=100)
    parser.add_argument("--test-non-geometry-proofs", type=int, default=200)
    parser.add_argument(
        "--geometry-rule",
        choices=("field", "mathlib_geometry", "field_or_mathlib_geometry"),
        default="field_or_mathlib_geometry",
        help=(
            "How to label geometry. 'field' uses is_geometry; 'mathlib_geometry' uses "
            "Mathlib.Geometry module/import paths; 'field_or_mathlib_geometry' uses either."
        ),
    )
    parser.add_argument(
        "--cot-geometry-examples",
        type=int,
        default=100,
        help="Exact number of geometry rows for train_CoT.jsonl.",
    )
    parser.add_argument(
        "--cot-non-geometry-examples",
        type=int,
        default=100,
        help="Exact number of non-geometry rows for train_CoT.jsonl.",
    )
    parser.add_argument(
        "--cot-per-domain",
        type=int,
        help="Backward-compatible shorthand setting both CoT domain targets.",
    )
    parser.add_argument("--cot-placeholder", default=DEFAULT_COT_PLACEHOLDER)
    parser.add_argument(
        "--cot-model",
        default="Qwen/Qwen3.5-9B",
        help="Tokenizer used to choose the shortest CoT seed examples. Use --no-cot-token-sort to skip.",
    )
    parser.add_argument("--no-cot-token-sort", action="store_true")
    return parser.parse_args(argv)


# Build train/test/CoT split artifacts and write them to the output directory.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit("No input rows found.")

    test_pool_rows: list[dict[str, Any]] = []
    test_pool_paths = args.test_pool_input or args.input
    for path in test_pool_paths:
        test_pool_rows.extend(read_jsonl(path))
    if not test_pool_rows:
        raise SystemExit("No test-pool rows found.")

    test = sample_rows_exact(
        test_pool_rows,
        geometry_count=args.test_geometry_proofs,
        non_geometry_count=args.test_non_geometry_proofs,
        seed=args.seed,
        geometry_rule=args.geometry_rule,
    )
    test_item_ids = {str(row.get("item_id", "")) for row in test}

    # Train rows are drawn from the filtered training pool, excluding any row already
    # assigned to test when the filtered file overlaps the full test-pool file.
    available_train_rows = [
        row for row in rows if str(row.get("item_id", "")) not in test_item_ids
    ]

    cot_geometry_examples = args.cot_per_domain or args.cot_geometry_examples
    cot_non_geometry_examples = args.cot_per_domain or args.cot_non_geometry_examples
    cot_model = None if args.no_cot_token_sort else args.cot_model
    train_cot_base = select_short_rows_exact(
        available_train_rows,
        geometry_count=cot_geometry_examples,
        non_geometry_count=cot_non_geometry_examples,
        model=cot_model,
        seed=args.seed,
        geometry_rule=args.geometry_rule,
    )
    cot_item_ids = {str(row.get("item_id", "")) for row in train_cot_base}
    train_lobotomy_base = [
        row for row in available_train_rows if str(row.get("item_id", "")) not in cot_item_ids
    ]
    train_cot = with_cot_placeholder(train_cot_base, placeholder=args.cot_placeholder)
    train_lobotomy = with_empty_think(train_lobotomy_base)
    cot_prompts = build_cot_prompt_rows(compute_prompt_tokens(train_cot_base, cot_model)) if cot_model else build_cot_prompt_rows(train_cot_base)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_lobotomy.jsonl", train_lobotomy)
    write_jsonl(args.output_dir / "train_CoT.jsonl", train_cot)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_jsonl(args.output_dir / "cot_generation_prompts.jsonl", cot_prompts)

    metadata: dict[str, Any] = {
        "seed": args.seed,
        "inputs": [str(path) for path in args.input],
        "test_pool_inputs": [str(path) for path in test_pool_paths],
        "test_geometry_proofs_requested": args.test_geometry_proofs,
        "test_non_geometry_proofs_requested": args.test_non_geometry_proofs,
        "geometry_rule": args.geometry_rule,
        "cot_geometry_examples_target": cot_geometry_examples,
        "cot_non_geometry_examples_target": cot_non_geometry_examples,
        "cot_placeholder": args.cot_placeholder,
        "cot_token_sort_model": cot_model,
        "all": split_stats(rows, geometry_rule=args.geometry_rule),
        "test_pool_all": split_stats(test_pool_rows, geometry_rule=args.geometry_rule),
        "available_train_pool": split_stats(available_train_rows, geometry_rule=args.geometry_rule),
        "train_lobotomy": split_stats(train_lobotomy, geometry_rule=args.geometry_rule),
        "train_CoT": split_stats(train_cot, geometry_rule=args.geometry_rule),
        "test": split_stats(test, geometry_rule=args.geometry_rule),
        "train_total": split_stats(
            train_lobotomy + train_cot,
            geometry_rule=args.geometry_rule,
        ),
        "disjoint_row_split": {
            "test_and_lobotomy_overlap": len(test_item_ids & {str(r.get("item_id", "")) for r in train_lobotomy}),
            "test_and_cot_overlap": len(test_item_ids & cot_item_ids),
            "lobotomy_and_cot_overlap": len(
                {str(r.get("item_id", "")) for r in train_lobotomy} & cot_item_ids
            ),
        },
        "artifacts": {
            "train_lobotomy": "train_lobotomy.jsonl",
            "train_CoT": "train_CoT.jsonl",
            "test": "test.jsonl",
            "cot_generation_prompts": "cot_generation_prompts.jsonl",
        },
        "notes": [
            "train_lobotomy.jsonl is ready for non-CoT training and has <think>\\n</think>\\n prepended to formal_completion.",
            "train_CoT.jsonl is not ready for final CoT training until __COT_REASONING_TO_FILL__ is replaced with synthetic reasoning.",
            "test.jsonl preserves the original column structure and is sampled by row counts, not proof counts.",
        ],
    }

    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
