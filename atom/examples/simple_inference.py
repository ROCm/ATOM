# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

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
parser.add_argument(
    "--max-tokens",
    type=int,
    default=300,
    help="max sampled tokens per prompt",
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
        "+".join([f"{i}-{i+1}" for i in range(1000)]) + "=? 最后结果是什么",
        "+".join([f"{i}+{i+1}" for i in range(1500)]) + "=? 最后结果是什么",
    ]
    args = parser.parse_args()
    # Generate power of 2 sizes for CUDA graph: [1, 2, 4, 8, ...]
    args.cudagraph_capture_sizes = str(generate_cuda_graph_sizes(len(prompts)))

    # Create engine from args
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    # Apply chat template. DeepSeek-V4 ships without a HuggingFace
    # chat_template; use its custom encoding_dsv4.py if available.
    if getattr(tokenizer, "chat_template", None):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt in prompts
        ]
    else:
        try:
            import importlib.util

            enc_path = os.path.join(args.model, "encoding", "encoding_dsv4.py")
            if os.path.exists(enc_path):
                spec = importlib.util.spec_from_file_location("encoding_dsv4", enc_path)
                enc_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(enc_mod)
                prompts = [
                    enc_mod.encode_messages(
                        [{"role": "user", "content": p}], thinking_mode="chat"
                    )
                    for p in prompts
                ]
                print(
                    f"  (applied V4 encoding_dsv4, prompt tokens: "
                    f"{[len(tokenizer.encode(p)) for p in prompts]})"
                )
            else:
                print("  (tokenizer has no chat_template — feeding raw prompts as-is)")
        except Exception as e:
            print(f"  (V4 encoding failed: {e} — feeding raw prompts as-is)")
    print("This is prompts:", prompts)
    # print("Warming up...")
    # _ = llm.generate(["warmup"], sampling_params)
    # print("Warm up done")

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

    llm.print_mtp_statistics()


if __name__ == "__main__":
    main()
