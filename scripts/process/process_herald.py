# Converts raw Herald proofs plus pre-extracted LeanDojo traces into training examples for subgoal closure.
#
# python scripts/process/process_herald.py \
#     --input data/raw/herald_raw.jsonl \
#     --output data/processed/herald_sample.jsonl \
#     --leandojo-data-dir raid/data/mathlib4_20c73142afa995ac9c8fb80a9bb585a55ca38308

import argparse
import glob
import json
import os
import random
import re
import uuid
from collections import Counter

LEANDOJO_DATA_DIR = "raid/data/mathlib4_20c73142afa995ac9c8fb80a9bb585a55ca38308"
RAW_HERALD_PATH = "data/raw/herald_raw.jsonl"
PROCESSED_HERALD_PATH = "data/processed/herald_sample.jsonl"
MAX_SAMPLES_PER_PROOF = 5
DEFAULT_SPLIT_DIR = "random"
SORRY_RE = re.compile(r"\bsorry\b")


# List LeanDojo trace JSON files while optionally restricting to one split scheme.
# data_dir: Root directory containing extracted LeanDojo JSON files.
# split_dir: Optional split subdirectory to restrict the search.
def iter_trace_json_files(data_dir: str, split_dir: str | None) -> list[str]:
    if split_dir:
        json_files = glob.glob(os.path.join(data_dir, split_dir, "*.json"))
    else:
        json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    return sorted(path for path in json_files if not path.endswith("metadata.json"))


# Build a theorem-name index from the extracted LeanDojo trace JSON files.
# data_dir: Root directory containing extracted LeanDojo JSON files.
# split_dir: Split subdirectory to index, defaulting to the repo's preferred split.
def build_static_index(
    data_dir: str, split_dir: str | None = DEFAULT_SPLIT_DIR
) -> tuple[dict, Counter]:
    index = {}
    duplicate_counts = Counter()
    json_files = iter_trace_json_files(data_dir, split_dir)
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for theorem in data:
                full_name = theorem.get("full_name")
                if not full_name:
                    continue
                if full_name in index:
                    duplicate_counts[full_name] += 1
                    continue
                index[full_name] = theorem

    return index, duplicate_counts


# Ensure that the parent directory for an output path exists.
# path: Output file path whose parent directory should be created.
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# Check whether a Herald header is present and usable for extraction.
# header: Header text taken from a Herald row.
def is_usable_header(header: str) -> bool:
    return bool(header.strip()) and "Unable to analyze" not in header


# Count occurrences of the token `sorry` in a Lean code string.
# code: Lean code to inspect.
def sorry_count(code: str) -> int:
    return len(SORRY_RE.findall(code))


# Decide whether a traced tactic starts at a top-level proof position.
# proof: Full formal proof text.
# tactic_idx: Character offset where the tactic begins.
# top_level_indent: Expected indentation depth for top-level tactics.
def is_top_level_tactic_position(
    proof: str, tactic_idx: int, top_level_indent: int | None
) -> bool:
    line_start = proof.rfind("\n", 0, tactic_idx) + 1
    prefix = proof[line_start:tactic_idx]
    if prefix.strip():
        return False
    if top_level_indent is None:
        return True
    return len(prefix.replace("\t", "  ")) == top_level_indent


# Infer the indentation depth used by top-level tactics in a proof body.
# proof: Full formal proof text to inspect.
def infer_top_level_indent(proof: str) -> int | None:
    by_idx = proof.find(":= by")
    if by_idx == -1:
        return None

    body_start = proof.find("\n", by_idx)
    if body_start == -1:
        return None

    indents = []
    for line in proof[body_start + 1:].splitlines():
        if not line.strip():
            continue
        stripped = line.lstrip(" \t")
        if stripped.startswith("--") or stripped.startswith("/-"):
            continue
        indents.append(len(line[: len(line) - len(stripped)].replace("\t", "  ")))
    return min(indents) if indents else None


# Locate top-level tactic cut points inside a proof using the traced tactics list.
# full_formal_proof: Full formal proof text to search.
# tactics_trace: LeanDojo tactic trace entries for the proof.
def locate_tactic_positions(
    full_formal_proof: str, tactics_trace: list[dict]
) -> tuple[list[tuple[int, str]], Counter]:
    stats = Counter()
    cursor = 0
    tactic_positions = []
    top_level_indent = infer_top_level_indent(full_formal_proof)

    for step in tactics_trace:
        state_before = step.get("state_before", "").strip()
        tactic = step.get("tactic", "").strip()

        if state_before == "no goals" or not tactic:
            stats["skipped_empty_or_no_goal_trace_step"] += 1
            continue

        tactic_idx = full_formal_proof.find(tactic, cursor)
        if tactic_idx == -1:
            stats["skipped_text_mismatch"] += 1
            continue

        cursor = tactic_idx + len(tactic)
        if not is_top_level_tactic_position(full_formal_proof, tactic_idx, top_level_indent):
            stats["skipped_nested_tactic_position"] += 1
            continue

        tactic_positions.append((tactic_idx, state_before))

    return tactic_positions, stats


# Convert one Herald theorem row plus its trace into zero or more training datapoints.
# row: Herald dataset row to process.
# traced_thm: Matching traced theorem metadata from LeanDojo.
def make_datapoints(row: dict, traced_thm: dict) -> tuple[list[dict], Counter]:
    stats = Counter()
    thm_name = row["name"]
    informal_thm = row.get("informal_theorem", "")
    informal_prf = row.get("informal_proof", "")
    full_informal_context = f"{informal_thm}\n\n{informal_prf}".strip()
    is_geo = "geometry" in full_informal_context.lower()

    file_path = traced_thm["file_path"].replace(".lean", "").replace("/", ".")
    original_id = f"{file_path}::{thm_name}"
    header = row.get("header", "").rstrip()
    full_formal_proof = row.get("formal_proof", "").strip()

    if not full_formal_proof:
        stats["skipped_empty_formal_proof"] += 1
        return [], stats
    if sorry_count(full_formal_proof) > 0:
        stats["skipped_source_proof_contains_sorry"] += 1
        return [], stats

    tactics_trace = traced_thm.get("traced_tactics", traced_thm.get("tactics", []))
    if not tactics_trace:
        stats["skipped_missing_tactics_trace"] += 1

    tactic_positions, locate_stats = locate_tactic_positions(full_formal_proof, tactics_trace)
    stats.update(locate_stats)

    theorem_datapoints = []
    for tactic_idx, state_before in tactic_positions:
        proof_prefix = full_formal_proof[:tactic_idx]
        proof_with_sorry = proof_prefix + "sorry"
        context_with_sorry = f"{header}\n\n{proof_with_sorry}"

        if sorry_count(context_with_sorry) != 1:
            stats["skipped_wrong_sorry_count"] += 1
            continue

        formal_completion = full_formal_proof[tactic_idx:].strip()
        if not formal_completion or sorry_count(formal_completion) > 0:
            stats["skipped_bad_completion"] += 1
            continue

        theorem_datapoints.append(
            {
                "item_id": str(uuid.uuid4()),
                "source_dataset": "herald",
                "original_id": original_id,
                "informal_context": full_informal_context,
                "formal_context_with_sorry": context_with_sorry,
                "formal_history": proof_prefix.strip(),
                "tactic_state": state_before,
                "formal_completion": formal_completion,
                "is_geometry": is_geo,
            }
        )

    return theorem_datapoints, stats


# Load theorem ids already present in an existing processed output file.
# output_path: Output JSONL path to scan for existing rows.
def load_processed_original_ids(output_path: str) -> set[str]:
    processed_original_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_original_ids.add(json.loads(line)["original_id"])
    return processed_original_ids


# Parse CLI arguments for Herald datapoint extraction.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Herald subgoal-closing datapoints from pre-extracted LeanDojo traces."
    )
    parser.add_argument("--leandojo-data-dir", default=LEANDOJO_DATA_DIR)
    parser.add_argument("--input", default=RAW_HERALD_PATH)
    parser.add_argument("--output", default=PROCESSED_HERALD_PATH)
    parser.add_argument("--max-samples-per-proof", type=int, default=MAX_SAMPLES_PER_PROOF)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many Herald rows.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite the output JSONL from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction without writing output.")
    return parser.parse_args()


# Run the Herald-to-training-data extraction pipeline.
def main():
    args = parse_args()
    random.seed(args.seed)

    print("1. Building static theorem index from LeanDojo extracted data...")
    if not os.path.exists(args.leandojo_data_dir):
        print(f"Error: Could not find extracted data at {args.leandojo_data_dir}")
        return
    if not os.path.exists(args.input):
        print(f"Error: Could not find Herald raw data at {args.input}")
        return

    theorem_index, duplicate_counts = build_static_index(args.leandojo_data_dir, args.split_dir)
    print(f"Successfully indexed {len(theorem_index)} extracted Mathlib declarations.")
    if duplicate_counts:
        print(f"Skipped {sum(duplicate_counts.values())} duplicate trace entries while indexing.")

    ensure_parent_dir(args.output)
    if args.overwrite and os.path.exists(args.output) and not args.dry_run:
        os.remove(args.output)

    processed_original_ids = set() if args.overwrite else load_processed_original_ids(args.output)
    if processed_original_ids:
        print(f"Resuming: {len(processed_original_ids)} original theorem IDs already processed.")

    print("\n2. Processing Herald dataset...")

    stats = Counter()
    output_mode = "w" if args.overwrite else "a"
    out_f = None if args.dry_run else open(args.output, output_mode, encoding="utf-8")

    try:
        with open(args.input, "r", encoding="utf-8") as in_f:
            for row_idx, line in enumerate(in_f):
                if args.limit is not None and row_idx >= args.limit:
                    break
                if not line.strip():
                    continue
                stats["rows_seen"] += 1
                row = json.loads(line)
                thm_name = row["name"]

                header = row.get("header", "")
                if not is_usable_header(header):
                    stats["skipped_missing_header"] += 1
                    continue

                traced_thm = theorem_index.get(thm_name)
                if not traced_thm:
                    stats["skipped_not_found"] += 1
                    continue

                file_path = traced_thm["file_path"].replace(".lean", "").replace("/", ".")
                original_id = f"{file_path}::{thm_name}"
                if original_id in processed_original_ids:
                    stats["skipped_already_processed"] += 1
                    continue

                theorem_datapoints, theorem_stats = make_datapoints(row, traced_thm)
                stats.update(theorem_stats)

                if len(theorem_datapoints) > args.max_samples_per_proof:
                    theorem_datapoints = random.sample(theorem_datapoints, args.max_samples_per_proof)

                for dp in theorem_datapoints:
                    if out_f is not None:
                        out_f.write(json.dumps(dp, ensure_ascii=False) + "\n")
                    stats["extracted_datapoints"] += 1

                if out_f is not None:
                    out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()

    print("\nExtraction Complete!")
    print(f"Generated {stats['extracted_datapoints']} state-tactic training pairs.")
    print(f"Rows seen: {stats['rows_seen']}")
    print(f"Skipped missing/corrupted header: {stats['skipped_missing_header']}")
    print(f"Skipped not found in LeanDojo trace: {stats['skipped_not_found']}")
    print(f"Skipped already processed: {stats['skipped_already_processed']}")
    print(f"Skipped source proofs containing sorry: {stats['skipped_source_proof_contains_sorry']}")
    print(f"Skipped missing tactics trace: {stats['skipped_missing_tactics_trace']}")
    print(f"Skipped tactic text mismatches: {stats['skipped_text_mismatch']}")
    print(f"Skipped nested tactic positions: {stats['skipped_nested_tactic_position']}")
    print(f"Skipped wrong sorry count: {stats['skipped_wrong_sorry_count']}")
    if args.dry_run:
        print("Dry run only; no data was written.")
    else:
        print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
