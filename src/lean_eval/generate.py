from __future__ import annotations

# Generates Lean proof completions from a base model or adapter-backed model.
#
# PYTHONPATH=src python -m lean_eval.generate \
#     --input data/processed/test.jsonl \
#     --output outputs/predictions.jsonl \
#     --model Qwen/Qwen3.5-9B \
#     --max-seq-length 2048 \
#     --max-new-tokens 512

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from .io import read_jsonl, write_jsonl
from .prompts import build_chatml_prompt, strip_chatml_completion


# Resolve a tokenizer token to its id, returning None when it maps to unknown.
# tokenizer: Tokenizer that defines the token vocabulary.
# token: Token string to resolve.
def _maybe_token(tokenizer: Any, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        return None
    return token_id


# Generate prediction rows for a dataset with a base model or adapter-backed model.
# rows: Dataset rows to run through generation.
# model: Loaded causal language model used for generation.
# tokenizer: Tokenizer paired with the model.
# max_seq_length: Maximum allowed prompt length before filtering a row.
# max_new_tokens: Maximum number of tokens to generate per row.
# limit: Optional prefix length of rows to evaluate.
# prompt_builder: Function that formats each row into the prompt to generate from.
def generate_rows(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    max_seq_length: int,
    max_new_tokens: int,
    limit: int | None = None,
    prompt_builder: Any = build_chatml_prompt,
) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    im_end_id = _maybe_token(tokenizer, "<|im_end|>")
    eos_token_id = im_end_id if im_end_id is not None else tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    selected = rows[:limit] if limit is not None else rows
    for idx, row in enumerate(selected):
        prompt = prompt_builder(row)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        base = {
            "item_id": str(row.get("item_id", idx)),
            "original_id": str(row.get("original_id", row.get("item_id", idx))),
            "prompt_tokens": prompt_tokens,
            "model_input_filtered": prompt_tokens > max_seq_length,
        }
        for key in ("source_dataset", "is_geometry", "state_after"):
            if key in row:
                base[key] = row[key]

        if prompt_tokens > max_seq_length:
            base.update(
                {
                    "prediction": "",
                    "generation_status": "filtered_context_length",
                    "generated_tokens": 0,
                    "elapsed_sec": 0.0,
                }
            )
            results.append(base)
            continue

        inputs = {key: value.to(model.device) for key, value in encoded.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        elapsed = time.perf_counter() - start
        generated_ids = output[0, prompt_tokens:]
        raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
        completion = strip_chatml_completion(raw_completion)
        base.update(
            {
                "prediction": completion,
                "raw_prediction": raw_completion,
                "generation_status": "generated",
                "generated_tokens": int(generated_ids.shape[0]),
                "elapsed_sec": round(elapsed, 6),
            }
        )
        results.append(base)
    return results


# Load the generation model and tokenizer, optionally attaching a PEFT adapter.
# model_id: Base model identifier or local path.
# adapter: Optional adapter directory to load on top of the base model.
def load_model_and_tokenizer(model_id: str, adapter: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


# Parse CLI arguments for batch generation.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lean completions with 4-bit Transformers.")
    parser.add_argument("--input", type=Path, required=True, help="Evaluation dataset JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL output.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", help="Optional PEFT adapter directory.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--metadata", type=Path, help="Optional generation metadata JSON output.")
    return parser.parse_args(argv)


# Load input rows, generate predictions, and write the outputs.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = read_jsonl(args.input)
    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    results = generate_rows(
        rows,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    write_jsonl(args.output, results)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "adapter": args.adapter,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "rows": len(results),
        "generated": sum(1 for row in results if row.get("generation_status") == "generated"),
        "filtered_context_length": sum(
            1 for row in results if row.get("generation_status") == "filtered_context_length"
        ),
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
