#!/usr/bin/env python
from __future__ import annotations

# Tunes a Qwen3.5-9B base model or adapter with QLoRA for Lean 4 subgoal completion.
# Formats rows into ChatML training examples, tokenizes them, and launches training.
#
# From base model:
# python scripts/train_qlora.py \
#     --input data/splits/model_a_phase1.jsonl \
#     --output-dir adapters/model_a_phase1 \
#     --phase 1
# From existing adapter:
# python scripts/train_qlora.py \
#     --input data/splits/model_a_phase2.jsonl \
#     --output-dir adapters/model_a_phase2 \
#     --phase 2 \
#     --from-adapter adapters/model_a_phase1

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from lean_eval.io import read_jsonl


BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM_PROMPT = {
    "lobotomy": (
        "You are an expert Lean 4 theorem prover. Your task is to complete the formal "
        "proof by providing the tactic or sequence of tactics required to close all goals "
        "from the `sorry` position. Respond directly with the tactic(s) without extended reasoning."
    ),
    "cot": (
        "You are an expert Lean 4 theorem prover. Your task is to complete the formal "
        "proof by providing the tactic or sequence of tactics required to close all goals "
        "from the `sorry` position. Think carefully through the proof strategy before "
        "providing your answer."
    ),
}
DEFAULT_PROMPT_MODE = {1: "lobotomy", 2: "cot"}


# Resolve the prompt mode to use for a row in the requested training phase.
# row: Training row that may override the default prompt mode.
# phase: Training phase number used to choose the default mode.
def prompt_mode_for_row(row: dict[str, Any], phase: int) -> str:
    mode = str(row.get("prompt_mode", DEFAULT_PROMPT_MODE[phase]))
    if mode not in SYSTEM_PROMPT:
        raise ValueError(f"unknown prompt_mode {mode!r}; expected one of {sorted(SYSTEM_PROMPT)}")
    return mode


# Build the user prompt text from the row fields expected by training.
# row: Training row containing the informal context, tactic state, and Lean code.
def build_user_prompt(row: dict[str, Any]) -> str:
    missing = [
        key
        for key in ("informal_context", "tactic_state", "formal_context_with_sorry")
        if key not in row
    ]
    if missing:
        raise ValueError(f"row is missing prompt fields: {', '.join(missing)}")
    return (
        "INFORMAL CONTEXT:\n"
        f"{row['informal_context']}\n\n"
        "CURRENT TACTIC STATE:\n"
        f"{row['tactic_state']}\n\n"
        "LEAN CODE:\n"
        f"{row['formal_context_with_sorry']}"
    )


# Build the full ChatML prompt prefix for a training row.
# row: Training row to format.
# phase: Training phase number used to choose the system prompt.
def build_prompt(row: dict[str, Any], phase: int) -> str:
    mode = prompt_mode_for_row(row, phase)
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT[mode]}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{build_user_prompt(row)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Build the supervised training text including the target completion.
# row: Training row containing the target formal completion.
# phase: Training phase number used to choose the system prompt.
def format_training_text(row: dict[str, Any], phase: int) -> str:
    completion = str(row.get("formal_completion", ""))
    if not completion:
        raise ValueError("row has empty formal_completion")
    return build_prompt(row, phase) + completion + "<|im_end|>\n"


# Build the token-id pattern that marks where the assistant response begins.
# tokenizer: Tokenizer used to encode the response template.
def response_template_ids(tokenizer: Any) -> list[int]:
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer does not expose <|im_start|> as a known token")
    return [im_start_id] + tokenizer.encode("assistant\n", add_special_tokens=False)


# Find the last occurrence of a token subsequence inside a token list.
# values: Token ids to search within.
# pattern: Token-id subsequence to locate.
def find_last_subsequence(values: list[int], pattern: list[int]) -> int:
    if not pattern:
        raise ValueError("empty pattern")
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return -1


# Tokenize and mask a single training row for causal language-model fine-tuning.
# row: Training row to tokenize.
# tokenizer: Tokenizer used to encode the formatted example.
# phase: Training phase number used to choose prompt formatting.
# max_seq_length: Maximum allowed tokenized length.
# template_ids: Token-id sequence marking the assistant response boundary.
def tokenize_training_row(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    phase: int,
    max_seq_length: int,
    template_ids: list[int],
) -> dict[str, Any]:
    text = format_training_text(row, phase)
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = list(encoded["input_ids"])
    if len(input_ids) > max_seq_length:
        return {
            "too_long": True,
            "length": len(input_ids),
            "item_id": str(row.get("item_id", "")),
            "original_id": str(row.get("original_id", "")),
        }

    labels = list(input_ids)
    start = find_last_subsequence(input_ids, template_ids)
    if start < 0:
        raise ValueError(f"assistant response template was not found for item_id={row.get('item_id')}")
    response_start = start + len(template_ids)
    labels[:response_start] = [-100] * response_start
    if all(label == -100 for label in labels):
        raise ValueError(f"no completion tokens left after masking for item_id={row.get('item_id')}")
    return {
        "too_long": False,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    # Pad a batch of tokenized examples into tensors for Trainer consumption.
    # self: Collator instance holding the tokenizer.
    # features: Tokenized examples to pad and stack.
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# Parse the target-modules setting into either a sentinel string or a module list.
# raw: Raw CLI value for --target-modules.
def parse_target_modules(raw: str) -> str | list[str]:
    if raw == "all-linear":
        return raw
    return [item.strip() for item in raw.split(",") if item.strip()]


# Parse CLI arguments for QLoRA training.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Lean 4 subgoal completion.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--from-adapter", type=Path)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use for one-batch smoke tests.")
    parser.add_argument("--limit", type=int, help="Limit rows before tokenization; useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--drop-overlength", action="store_true")
    parser.add_argument("--dry-run-format", action="store_true", help="Print formatted examples and exit.")
    parser.add_argument(
        "--validate-tokenization",
        action="store_true",
        help="Tokenize and report masking/length stats, then exit before loading the model.",
    )
    return parser.parse_args(argv)


# Load the tokenizer used for training and set its padding behavior.
# model_id: Base model identifier or local path.
def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# Tokenize all rows and collect summary statistics for training-time filtering.
# rows: Training rows to tokenize.
# tokenizer: Tokenizer used to encode the examples.
# phase: Training phase number used to choose prompt formatting.
# max_seq_length: Maximum allowed tokenized length.
# drop_overlength: Whether to drop overlength rows instead of aborting.
def prepare_tokenized_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    phase: int,
    max_seq_length: int,
    drop_overlength: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template_ids = response_template_ids(tokenizer)
    tokenized: list[dict[str, Any]] = []
    too_long: list[dict[str, Any]] = []
    lengths: list[int] = []
    supervised_tokens = 0
    for row in rows:
        item = tokenize_training_row(
            row,
            tokenizer=tokenizer,
            phase=phase,
            max_seq_length=max_seq_length,
            template_ids=template_ids,
        )
        if item.pop("too_long"):
            too_long.append(item)
            continue
        lengths.append(int(item["length"]))
        supervised_tokens += sum(1 for label in item["labels"] if label != -100)
        tokenized.append(item)

    if too_long and not drop_overlength:
        sample = ", ".join(str(row.get("item_id") or row.get("original_id")) for row in too_long[:5])
        raise SystemExit(
            f"{len(too_long)} rows exceed max_seq_length={max_seq_length}. "
            f"Rerun with --drop-overlength to skip them. Sample: {sample}"
        )

    stats = {
        "input_rows": len(rows),
        "kept_rows": len(tokenized),
        "dropped_overlength": len(too_long),
        "max_seq_length": max_seq_length,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
        "supervised_tokens": supervised_tokens,
        "response_template_ids": template_ids,
    }
    return tokenized, stats


# Load the quantized base model and attach either a fresh or existing LoRA adapter.
def load_model(args: argparse.Namespace) -> Any:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    if args.phase == 2:
        if args.from_adapter is None:
            raise SystemExit("--from-adapter is required for --phase 2")
        model = PeftModel.from_pretrained(model, str(args.from_adapter), is_trainable=True)
    else:
        target_modules = parse_target_modules(args.target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    return model


# Run tokenization, model loading, training, and metadata export for a training job.
# rows: Training rows to fit on.
def train(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    tokenizer = load_tokenizer(args.model)
    tokenized, token_stats = prepare_tokenized_rows(
        rows,
        tokenizer=tokenizer,
        phase=args.phase,
        max_seq_length=args.max_seq_length,
        drop_overlength=args.drop_overlength,
    )
    if not tokenized:
        raise SystemExit("No rows left after tokenization.")

    if args.validate_tokenization:
        return token_stats

    model = load_model(args)
    model.print_trainable_parameters()

    total_steps_override = args.max_steps if args.max_steps and args.max_steps > 0 else None
    if total_steps_override is not None:
        warmup_steps = args.warmup_steps
    else:
        estimated_optimizer_steps = max(
            1,
            int((len(tokenized) * args.epochs) / max(1, args.batch_size * args.grad_accum)),
        )
        warmup_steps = (
            args.warmup_steps
            if args.warmup_steps > 0
            else int(round(estimated_optimizer_steps * args.warmup_ratio))
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        remove_unused_columns=False,
    )
    dataset = Dataset.from_list(tokenized)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    resume_from_checkpoint = (
        str(args.resume_from_checkpoint) if args.resume_from_checkpoint is not None else None
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "input": str(args.input),
                "phase": args.phase,
                "from_adapter": str(args.from_adapter) if args.from_adapter else None,
                "resume_from_checkpoint": (
                    str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
                ),
                "max_seq_length": args.max_seq_length,
                "target_modules": parse_target_modules(args.target_modules),
                "tokenization": token_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return token_stats


# Orchestrate CLI validation, data loading, optional previewing, and training.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase == 1 and args.from_adapter is not None:
        raise SystemExit("--from-adapter is only valid for --phase 2")

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    if args.dry_run_format:
        for idx, row in enumerate(rows[: min(3, len(rows))]):
            print(f"===== row {idx} item_id={row.get('item_id')} prompt_mode={prompt_mode_for_row(row, args.phase)} =====")
            print(format_training_text(row, args.phase)[:4000])
        return 0

    stats = train(args, rows)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
