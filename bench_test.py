import argparse
import os
import pstats
import time
from random import randint, seed

from transformers import AutoTokenizer

from atom import LLMEngine, SamplingParams
from atom.config import CompilationConfig

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config of test",
)

parser.add_argument(
    "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path."
)

# A very long prompt, total number of tokens is about 15k.
LONG_PROMPT = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 64


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    args = parser.parse_args()
    path = args.model
    llm = LLMEngine(
        path,
        enforce_eager=False,
        max_model_len=4096,
        port=12345,
        torch_profiler_dir="./log",
        compilation_config=CompilationConfig(
            level=3,
        ),
    )


    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)
        for _ in range(num_seqs)
    ]

    print("------start generating------")

    #warm up
    outputs_ = llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs = llm.generate(LONG_PROMPT, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
