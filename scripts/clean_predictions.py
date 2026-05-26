#!/usr/bin/env python
from __future__ import annotations

# Normalizes model outputs before verification by stripping ChatML wrappers and preserving the raw text.
#
# python scripts/clean_predictions.py \
#     --input outputs/predictions.jsonl \
#     --output outputs/predictions.cleaned.jsonl

import argparse
import json
import sys
from pathlib import Path
from lean_eval.io import read_jsonl, write_jsonl
from lean_eval.prompts import strip_chatml_completion


# Parse CLI arguments for prediction cleaning.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated Lean completions before verification.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", default="prediction")
    parser.add_argument("--raw-field", default="raw_prediction")
    return parser.parse_args(argv)


# Clean prediction text fields and write the normalized JSONL output.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = read_jsonl(args.input)
    cleaned = []
    changed = 0
    for row in rows:
        out = dict(row)
        value = str(out.get(args.field, ""))
        if args.raw_field not in out:
            out[args.raw_field] = value
        new_value = strip_chatml_completion(value)
        if new_value != value:
            changed += 1
        out[args.field] = new_value
        cleaned.append(out)
    write_jsonl(args.output, cleaned)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(cleaned),
                "changed": changed,
                "field": args.field,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
