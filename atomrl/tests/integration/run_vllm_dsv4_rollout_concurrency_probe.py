# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Probe DSV4 vLLM with the exact heterogeneous GRPO rollout prompts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

EXPECTED_FIRST_INDICES = [13572, 7694, 16062, 13448, 14644, 2000, 1845, 120]


def _prompt_text(row: dict[str, Any], tokenizer: Any) -> str:
    prompt = row.get("prompt") or row.get("question") or row.get("input") or ""
    if isinstance(prompt, list):
        prompt_text = "\n".join(
            message.get("content", "")
            for message in prompt
            if isinstance(message, dict)
        )
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (RuntimeError, TypeError, ValueError):
            return prompt_text
    if isinstance(prompt, str) and prompt.startswith("["):
        try:
            messages = json.loads(prompt)
            return "\n".join(
                message.get("content", "")
                for message in messages
                if isinstance(message, dict)
            )
        except (json.JSONDecodeError, TypeError):
            pass
    return str(prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8018/v1/completions")
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--shuffle-seed", type=int, default=10086)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset_kind = "parquet" if args.dataset.endswith(".parquet") else "json"
    dataset = load_dataset(dataset_kind, data_files=args.dataset, split="train")
    generator = torch.Generator().manual_seed(args.shuffle_seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    if args.shuffle_seed == 10086 and len(dataset) == 17398:
        assert permutation[:8] == EXPECTED_FIRST_INDICES

    selected = permutation[: args.num_prompts]
    prompt_ids = [
        tokenizer(
            _prompt_text(dataset[index], tokenizer),
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_prompt_tokens,
        )["input_ids"]
        for index in selected
    ]
    expanded = [
        list(token_ids) for token_ids in prompt_ids for _ in range(args.num_generations)
    ]
    payload = {
        "model": args.model,
        "prompt": expanded,
        "max_tokens": args.max_tokens,
        "temperature": 0.8,
        "top_p": 1.0,
    }
    print(
        f"request prompts={args.num_prompts} generations={args.num_generations} "
        f"sequences={len(expanded)} prompt_lengths="
        f"{[len(ids) for ids in prompt_ids]} indices={selected}",
        flush=True,
    )
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    choices = result["choices"]
    assert len(choices) == len(
        expanded
    ), f"expected {len(expanded)} choices, got {len(choices)}"
    lengths = [
        len(tokenizer.encode(choice["text"], add_special_tokens=False))
        for choice in choices
    ]
    print(
        f"PASS sequences={len(choices)} generated_min={min(lengths)} "
        f"generated_max={max(lengths)} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
