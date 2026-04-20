# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Qwen3.5 text-only offline inference (native ATOM engine).

Uses the same model path as multimodal inference but without any image input.
Useful for verifying language model accuracy independently of the vision encoder.
"""

import argparse

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Qwen3.5 text-only offline inference (native ATOM engine)",
)

EngineArgs.add_cli_args(parser)

parser.add_argument(
    "--temperature", type=float, default=0.1, help="Temperature for sampling"
)
parser.add_argument(
    "--max-tokens", type=int, default=256, help="Max tokens to generate"
)


def main():
    args = parser.parse_args()
    args.cudagraph_capture_sizes = "[1]"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [
        "What is the capital of France? Answer in one word.",
    ]

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]

    token_ids_list = [tokenizer.encode(f) for f in formatted]
    for p, ids in zip(prompts, token_ids_list):
        print(f"Prompt: {p!r}  ->  {len(ids)} tokens")

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    outputs = llm.generate(token_ids_list, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n" + "=" * 70)
        print(f"Prompt: {prompt}")
        print(f"Generated: {output['text']}")
        print(f"Tokens: {output['num_tokens_input']} in / {output['num_tokens_output']} out")
        print(f"Finish reason: {output['finish_reason']}")
        print("=" * 70)

    llm.close()


if __name__ == "__main__":
    main()
