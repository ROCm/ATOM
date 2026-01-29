# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import random
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Offline profiling example for Atom LLM Engine",
)

EngineArgs.add_cli_args(parser)

parser.add_argument(
    "--input-length",
    type=int,
    default=128,
    help="Input prompt length in tokens (approximate, used with --random-input)",
)
parser.add_argument(
    "--output-length",
    type=int,
    default=32,
    help="Output generation length in tokens",
)
parser.add_argument(
    "--bs",
    type=int,
    default=1,
    help="Batch size (number of requests to process in parallel)",
)
parser.add_argument(
    "--random-input",
    action="store_true",
    help="Use random repeated words as input. Otherwise use a predefined meaningful text.",
)


def main():
    args = parser.parse_args()

    # Set default torch_profiler_dir to current directory if not provided
    if args.torch_profiler_dir is None:
        args.torch_profiler_dir = "./profiler_traces"
        print(
            "Warning: --torch-profiler-dir not specified, using default: ./profiler_traces"
        )

    print("\nInitializing LLM engine...")
    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.random_input:
        vocab_size = tokenizer.vocab_size
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(args.input_length)
        ]
        prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
        # Re-encode to verify and truncate to exact length
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)[
            : args.input_length
        ]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"Using randomly generated input with length: {len(token_ids)} tokens")
    else:
        prompt = """Artificial intelligence has revolutionized numerous fields in recent years, transforming the way we live, work, and interact with technology. Machine learning algorithms have become increasingly sophisticated, enabling computers to perform tasks that were once thought to require human intelligence. From natural language processing to computer vision, AI systems are now capable of understanding and generating human language, recognizing objects in images, and even creating original artwork and music. The applications of AI continue to expand rapidly across industries."""
        print(f"Input_prompt: {prompt}")

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=args.output_length,
        ignore_eos=True,
    )

    # TODO: remove warm up and skip cuda capture process in engine while profiling.
    print("Warming up...")
    _ = llm.generate(["warmup"], sampling_params)
    print("Warm up done")

    print("\n" + "=" * 70)
    print("Starting profiling...")
    print("=" * 70)

    llm.start_profile()

    # Create batch of prompts based on batch size
    prompts = [prompt] * args.bs
    print(f"Processing batch of {args.bs} requests...")

    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()

    print("\n" + "=" * 70)
    print("Profiling completed!")
    print("Profiler traces have been saved to:", args.torch_profiler_dir)
    print("=" * 70)

    if not args.random_input:
        print("Generated Output:")
        for i, output in enumerate(outputs):
            generated_text = output["text"]
            if args.bs > 1:
                print(f"Output [{i+1}/{args.bs}]: {generated_text}\n")
            else:
                print(f"Output: {generated_text}\n")


if __name__ == "__main__":
    main()
