# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

# Add engine arguments
EngineArgs.add_cli_args(parser)

# Add example-specific arguments
parser.add_argument(
    "--temperature", type=float, default=0.6, help="temperature for sampling"
)


def generate_cuda_graph_sizes(max_size):
    # This is for DP split batch size
    sizes = []
    power = 1
    while power <= max_size:
        sizes.append(power)
        power *= 2
    return sizes


def main():
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        "如何在一个月内增肌10公斤",
    ]
    args = parser.parse_args()
    # Generate power of 2 sizes for CUDA graph: [1, 2, 4, 8, ...]
    args.cudagraph_capture_sizes = str(generate_cuda_graph_sizes(len(prompts)))

    # Create engine from args
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=256)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    print("This is prompts:", prompts)
    print("Warming up...")
    _ = llm.generate(["warmup"], sampling_params)
    print("Warm up done")

    print("\n" + "=" * 70)
    print("Starting profiling...")
    print("=" * 70)

    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
