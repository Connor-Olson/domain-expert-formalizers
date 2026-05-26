#!/usr/bin/env python
from __future__ import annotations

# Build the final per-model phase splits used to train the three Qwen variants in this repository.
# It consumes the row-level lobotomy/CoT pools and writes model-specific phase 1/phase 2 JSONL files.
#
# python scripts/prepare_splits.py \
#     --raw-dir data/raw/splits/finetune \
#     --output-dir data/splits \
#     --seed 42

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from lean_eval.io import read_jsonl, write_jsonl


# Group rows by theorem identifier so later sampling can happen at proof granularity.
# rows: Input rows to bucket by theorem.
def group_by_theorem(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        theorem_id = str(row.get("original_id", row.get("item_id", idx)))
        groups[theorem_id].append(row)
    return dict(groups)


# Split theorem ids into phase-1-only ids and ids that also have CoT rows.
# lobotomy_groups: Theorem-grouped non-CoT rows.
# cot_groups: Theorem-grouped CoT rows.
def partition_theorem_ids(
    lobotomy_groups: dict[str, list[dict[str, Any]]],
    cot_groups: dict[str, list[dict[str, Any]]],
) -> tuple[set[str], set[str]]:
    cot_ids = set(cot_groups)
    lobotomy_only_ids = set(lobotomy_groups) - cot_ids
    return lobotomy_only_ids, cot_ids


# Keep only theorem groups whose ids are present in the allowed set.
# groups: Theorem-grouped rows to filter.
# allowed_ids: Theorem ids that should remain.
def filter_groups(
    groups: dict[str, list[dict[str, Any]]],
    allowed_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    return {key: rows for key, rows in groups.items() if key in allowed_ids}


# Sample a target number of rows from a theorem-grouped pool.
# pool: Theorem-grouped rows to sample from.
# n_rows: Number of rows to return.
# label: Human-readable pool label for error messages.
# rng: Random generator controlling sampling order.
def sample_from_pool(
    pool: dict[str, list[dict[str, Any]]],
    n_rows: int,
    *,
    label: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if n_rows == 0:
        return []
    keys = list(pool)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    for key in keys:
        if len(selected) >= n_rows:
            break
        selected.extend(pool[key])
    if len(selected) < n_rows:
        raise ValueError(f"Pool too small for {label!r}: need {n_rows} rows, have {len(selected)}")
    return rng.sample(selected, n_rows)


# Add a prompt mode tag to every row in a split.
# rows: Rows to annotate.
# mode: Prompt mode value to assign.
def with_prompt_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [{**row, "prompt_mode": mode} for row in rows]


# Split rows into geometry and non-geometry subsets using the existing row flag.
# rows: Rows to partition by domain.
def split_by_domain(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    geometry = [row for row in rows if row.get("is_geometry") is True]
    non_geometry = [row for row in rows if row.get("is_geometry") is not True]
    return geometry, non_geometry


# Compute row and proof counts for a split, including geometry totals.
# rows: Rows to summarize.
def domain_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    proof_ids = {str(row.get("original_id", row.get("item_id", idx))) for idx, row in enumerate(rows)}
    geometry_rows = sum(1 for row in rows if row.get("is_geometry") is True)
    return {
        "rows": len(rows),
        "proofs": len(proof_ids),
        "geometry_rows": geometry_rows,
        "non_geometry_rows": len(rows) - geometry_rows,
    }


# Count prompt-mode labels in a set of rows.
# rows: Rows whose prompt_mode fields will be counted.
def prompt_mode_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("prompt_mode", "missing")) for row in rows).items()))


# Ensure that phase 1 and phase 2 CoT rows do not share theorem ids.
# phase1_rows: Phase 1 rows to check.
# phase2_cot_rows: Phase 2 CoT rows to check.
# label: Human-readable split label for error messages.
def assert_no_phase_overlap(
    phase1_rows: list[dict[str, Any]],
    phase2_cot_rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    phase1_ids = {str(row.get("original_id", row.get("item_id", ""))) for row in phase1_rows}
    cot_ids = {str(row.get("original_id", row.get("item_id", ""))) for row in phase2_cot_rows}
    overlap = phase1_ids & cot_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"{label}: phase1 and CoT theorem IDs overlap: {sample}")


# Write split metadata to disk as formatted JSON.
# path: Destination metadata file.
# metadata: Metadata object to serialize.
def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Parse CLI arguments for model-specific split generation.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create model-specific phase splits for QLoRA fine-tuning.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/splits/finetune"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a-phase1-rows", type=int, default=1000)
    parser.add_argument("--model-a-cot-rows", type=int, default=100)
    parser.add_argument("--model-a-replay-rows", type=int, default=100)
    parser.add_argument("--model-b-phase1-rows", type=int, default=1000)
    parser.add_argument("--model-b-cot-rows", type=int, default=100)
    parser.add_argument("--model-b-replay-rows", type=int, default=100)
    parser.add_argument("--model-c-phase1-geo-rows", type=int, default=500)
    parser.add_argument("--model-c-phase1-non-geo-rows", type=int, default=500)
    parser.add_argument("--model-c-cot-geo-rows", type=int, default=50)
    parser.add_argument("--model-c-cot-non-geo-rows", type=int, default=50)
    parser.add_argument("--model-c-replay-rows", type=int, default=100)
    return parser.parse_args(argv)


# Build all per-model phase splits and write their JSONL and metadata outputs.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rng = random.Random(args.seed)

    lobotomy = read_jsonl(args.raw_dir / "train_lobotomy.jsonl")
    cot = read_jsonl(args.raw_dir / "train_CoT.jsonl")
    if not lobotomy:
        raise SystemExit(f"No rows found in {args.raw_dir / 'train_lobotomy.jsonl'}")
    if not cot:
        raise SystemExit(f"No rows found in {args.raw_dir / 'train_CoT.jsonl'}")

    lob_geo, lob_non_geo = split_by_domain(lobotomy)
    cot_geo, cot_non_geo = split_by_domain(cot)

    lob_geo_groups = group_by_theorem(lob_geo)
    lob_non_geo_groups = group_by_theorem(lob_non_geo)
    cot_geo_groups = group_by_theorem(cot_geo)
    cot_non_geo_groups = group_by_theorem(cot_non_geo)

    geo_p1_ids, geo_p2_ids = partition_theorem_ids(lob_geo_groups, cot_geo_groups)
    non_geo_p1_ids, non_geo_p2_ids = partition_theorem_ids(lob_non_geo_groups, cot_non_geo_groups)

    lob_geo_p1 = filter_groups(lob_geo_groups, geo_p1_ids)
    lob_non_geo_p1 = filter_groups(lob_non_geo_groups, non_geo_p1_ids)
    cot_geo_p2 = filter_groups(cot_geo_groups, geo_p2_ids)
    cot_non_geo_p2 = filter_groups(cot_non_geo_groups, non_geo_p2_ids)

    model_a_p1 = with_prompt_mode(
        sample_from_pool(lob_geo_p1, args.model_a_phase1_rows, label="model_a phase1", rng=rng),
        "lobotomy",
    )
    model_a_cot = with_prompt_mode(
        sample_from_pool(cot_geo_p2, args.model_a_cot_rows, label="model_a phase2 CoT", rng=rng),
        "cot",
    )
    model_a_p2 = model_a_cot + rng.sample(model_a_p1, args.model_a_replay_rows)
    rng.shuffle(model_a_p2)

    model_b_p1 = with_prompt_mode(
        sample_from_pool(lob_non_geo_p1, args.model_b_phase1_rows, label="model_b phase1", rng=rng),
        "lobotomy",
    )
    model_b_cot = with_prompt_mode(
        sample_from_pool(cot_non_geo_p2, args.model_b_cot_rows, label="model_b phase2 CoT", rng=rng),
        "cot",
    )
    model_b_p2 = model_b_cot + rng.sample(model_b_p1, args.model_b_replay_rows)
    rng.shuffle(model_b_p2)

    model_c_p1 = with_prompt_mode(
        sample_from_pool(lob_geo_p1, args.model_c_phase1_geo_rows, label="model_c geo phase1", rng=rng)
        + sample_from_pool(
            lob_non_geo_p1,
            args.model_c_phase1_non_geo_rows,
            label="model_c non-geo phase1",
            rng=rng,
        ),
        "lobotomy",
    )
    rng.shuffle(model_c_p1)
    model_c_cot = with_prompt_mode(
        sample_from_pool(cot_geo_p2, args.model_c_cot_geo_rows, label="model_c geo phase2 CoT", rng=rng)
        + sample_from_pool(
            cot_non_geo_p2,
            args.model_c_cot_non_geo_rows,
            label="model_c non-geo phase2 CoT",
            rng=rng,
        ),
        "cot",
    )
    model_c_p2 = model_c_cot + rng.sample(model_c_p1, args.model_c_replay_rows)
    rng.shuffle(model_c_p2)

    for label, phase1_rows, cot_rows in (
        ("model_a", model_a_p1, model_a_cot),
        ("model_b", model_b_p1, model_b_cot),
        ("model_c", model_c_p1, model_c_cot),
    ):
        assert_no_phase_overlap(phase1_rows, cot_rows, label=label)

    outputs = {
        "model_a_phase1.jsonl": model_a_p1,
        "model_a_phase2.jsonl": model_a_p2,
        "model_b_phase1.jsonl": model_b_p1,
        "model_b_phase2.jsonl": model_b_p2,
        "model_c_phase1.jsonl": model_c_p1,
        "model_c_phase2.jsonl": model_c_p2,
    }
    for filename, rows in outputs.items():
        write_jsonl(args.output_dir / filename, rows)

    metadata = {
        "seed": args.seed,
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "source": {
            "train_lobotomy": domain_stats(lobotomy),
            "train_CoT": domain_stats(cot),
            "cot_prompt_modes": prompt_mode_stats(cot),
        },
        "pools": {
            "lob_geo_p1_rows": sum(len(rows) for rows in lob_geo_p1.values()),
            "lob_non_geo_p1_rows": sum(len(rows) for rows in lob_non_geo_p1.values()),
            "cot_geo_p2_rows": sum(len(rows) for rows in cot_geo_p2.values()),
            "cot_non_geo_p2_rows": sum(len(rows) for rows in cot_non_geo_p2.values()),
        },
        "outputs": {
            name: {
                **domain_stats(rows),
                "prompt_modes": prompt_mode_stats(rows),
            }
            for name, rows in outputs.items()
        },
    }
    write_metadata(args.output_dir / "split_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
