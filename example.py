import argparse
import os
from typing import List

import torch
from transformers import AutoTokenizer

from atom import LLMEngine, SamplingParams
from atom.config import CompilationConfig

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

parser.add_argument(
    "--kv_cache_dtype",
    choices=["bf16", "fp8"],
    type=str,
    default="bf16",
    help="""KV cache type. Default is 'bf16'.
    e.g.: -kv_cache_dtype fp8""",
)

parser.add_argument(
    "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
)

parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)

parser.add_argument(
    "--enforce-eager", action="store_true", help="Enforce eager mode execution."
)

parser.add_argument(
    "--enable_prefix_caching", action="store_true", help="Enable prefix caching."
)

parser.add_argument("--port", type=int, default=8006, help="API server port")


parser.add_argument(
    "--cudagraph-capture-sizes", type=str, default="[1,2,4,8,16]", help="Sizes to capture cudagraph."
)

parser.add_argument("--level", type=int, default=3, help="The level of compilation")

parser.add_argument(
    "--torch-profiler-dir", type=str, default=None, help="Directory to save torch profiler traces"
)

def parse_size_list(size_str: str) -> List[int]:
    import ast
    try:
        return ast.literal_eval(size_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Error list size: {size_str}") from e

def main():
    args = parser.parse_args()
    model_name_or_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # If you want to use torch.compile, please --level=3
    llm = LLMEngine(
        model_name_or_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        port=args.port,
        torch_profiler_dir=args.torch_profiler_dir,
        compilation_config=CompilationConfig(
            level = args.level,
            cudagraph_capture_sizes=parse_size_list(args.cudagraph_capture_sizes)
    )
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "1+2+3=?",
        "如何在一个月内增肌10公斤",
    ]
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

    # # Calculate total token number of prompts
    # prompt_token_ids = [llm.tokenizer.encode(prompt) for prompt in prompts]
    # prompt_lens = [len(token_ids) for token_ids in prompt_token_ids]
    # # Create warmup inputs with same shapes as all prompts
    # # Generate random token IDs for each prompt length to match expected input shapes
    # warmup_prompts = []
    # for i, prompt_len in enumerate(prompt_lens):
    #     warmup_prompt = torch.randint(
    #         0, llm.tokenizer.vocab_size, size=(prompt_len,)
    #     ).tolist()
    #     warmup_prompts.append(warmup_prompt)
    # # Run warmup with the same batch structure as the actual prompts (no profiling)
    # _ = llm.generate(warmup_prompts, sampling_params)

    # generate (with profiling)
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
