from __future__ import annotations

import re
from typing import Any


# Default system prompt for first-pass proof-completion generation.
SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover. Your task is to complete the formal proof by "
    "providing the tactic or sequence of tactics required to complete the proof from the "
    "`sorry`'s position and close the goal."
)

# Specialized system prompt for second-pass repair generation using compiler feedback.
REPAIR_SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover. A previous proof attempt failed to compile. "
    "Given the original theorem context, the previous attempt, and Lean's feedback, produce a "
    "corrected Lean proof completion that replaces the same `sorry`. Output only Lean code. "
    "Do not include explanations, Markdown, or code fences."
)


# Build the user-facing prompt body from a dataset row.
# row: Dataset row containing the informal context, tactic state, and Lean code.
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


# Build the repair prompt body from a failed attempt plus Lean feedback.
# row: Dataset row containing the original context, previous attempt, and compiler feedback.
def build_repair_user_prompt(row: dict[str, Any]) -> str:
    missing = [
        key
        for key in (
            "informal_context",
            "tactic_state",
            "formal_context_with_sorry",
            "previous_prediction",
            "lean_feedback",
        )
        if key not in row
    ]
    if missing:
        raise ValueError(f"row is missing repair prompt fields: {', '.join(missing)}")
    return (
        "INFORMAL CONTEXT:\n"
        f"{row['informal_context']}\n\n"
        "CURRENT TACTIC STATE:\n"
        f"{row['tactic_state']}\n\n"
        "LEAN CODE WITH SORRY:\n"
        f"{row['formal_context_with_sorry']}\n\n"
        "PREVIOUS ATTEMPT:\n"
        f"{row['previous_prediction']}\n\n"
        "LEAN FEEDBACK:\n"
        f"{row['lean_feedback']}"
    )


# Build chat message objects for a dataset row.
# row: Dataset row to convert into system and user messages.
def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]


# Build a ChatML prompt prefix using configurable system and user prompt builders.
# row: Dataset row whose fields are inserted into the prompt.
# system_prompt: System instruction placed in the ChatML system message.
# user_prompt_builder: Function that builds the user message content from the row.
def build_chatml_prompt(
    row: dict[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt_builder: Any = build_user_prompt,
) -> str:
    user = user_prompt_builder(row)
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Build the ChatML prompt prefix used for second-pass repair generation.
# row: Dataset row containing the failed attempt and Lean feedback.
def build_repair_chatml_prompt(row: dict[str, Any]) -> str:
    return build_chatml_prompt(
        row,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_prompt_builder=build_repair_user_prompt,
    )


# Build the full ChatML training text including the target completion.
# row: Dataset row to format as a supervised example.
# completion_field: Row field containing the target Lean completion text.
def build_chatml_training_text(
    row: dict[str, Any],
    *,
    completion_field: str = "formal_completion",
) -> str:
    completion = str(row.get(completion_field, ""))
    return build_chatml_prompt(row) + completion + "<|im_end|>\n"


# Strip ChatML wrappers and stop markers from a model completion.
# text: Raw model output to clean.
def strip_chatml_completion(text: str) -> str:
    text = strip_reasoning_and_fences(text)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[0]
    return text.strip()


# Remove reasoning tags and Markdown code fences from model output.
# text: Raw model output that may contain reasoning or fenced code.
def strip_reasoning_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL).strip()

    fence = re.search(r"```(?:lean4|lean)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence is not None:
        return fence.group(1).strip()

    open_fence = re.search(r"```(?:lean4|lean)?\s*\n?", text, flags=re.IGNORECASE)
    if open_fence is not None:
        return text[open_fence.end() :].strip()

    text = re.sub(r"^```(?:lean4|lean)?\s*\n?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n?```\s*$", "", text).strip()
    return text
