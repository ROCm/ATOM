# SPDX-License-Identifier: MIT
"""Reproducible offline prefill comparison using exact token counts.

Run each parallel configuration in a fresh process. Prefix caching must be off;
the warmup uses the same shape as measurement, but never supplies cached KV.
Wall time includes scheduling and one output token; it is not GPU kernel time.
Use --torch-profiler-dir to capture a separate, untimed iteration.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main():
    parser = FlexibleArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--input-length", type=int, default=131072)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument(
        "--prompt-file", help="Optional text file with at least --input-length tokens"
    )
    args = parser.parse_args()
    if args.enable_prefix_caching:
        parser.error("pass --no-enable_prefix_caching to measure uncached prefill")
    if min(args.input_length, args.repeats, args.batch_size) < 1:
        parser.error("input length, repeats and batch size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Pass token IDs directly: decoding and re-tokenizing does not preserve length.
    paragraph = tokenizer.encode(
        "A research team recorded the temperature, pressure and humidity every "
        "hour. They compared the observations across seasons and checked each "
        "measurement before publishing the results.\n",
        add_special_tokens=False,
    )
    prompts = []
    supplied_tokens = None
    if args.prompt_file:
        supplied_tokens = tokenizer.encode(
            Path(args.prompt_file).read_text(), add_special_tokens=False
        )
        if len(supplied_tokens) < args.input_length:
            parser.error("prompt file has fewer tokens than --input-length")
    for i in range(args.batch_size):
        if supplied_tokens is not None:
            prompts.append(supplied_tokens[: args.input_length])
            continue
        prefix = tokenizer.encode(f"Document {i}:\n", add_special_tokens=False)
        prompts.append(
            (prefix + paragraph * (args.input_length // len(paragraph) + 1))[
                : args.input_length
            ]
        )
    prompt_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    engine_args = EngineArgs.from_cli_args(args)
    params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, logprobs=20)
    source_root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "args": vars(args),
        "prompt_sha256": prompt_hash,
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "atom/config.py",
                "atom/distributed/ulysses_sp.py",
                "atom/distributed/sp_kernels.py",
                "atom/model_ops/moe.py",
                "atom/model_ops/paged_attention.py",
                "atom/model_ops/topK.py",
                "atom/benchmarks/benchmark_prefill_parallel.py",
            )
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(),
            "visible_gpus": torch.cuda.device_count(),
        },
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ("ATOM_", "AITER_", "NCCL_", "RCCL_", "HIP_VISIBLE_", "ROCR_VISIBLE_")
            )
        },
        "iterations": [],
    }
    llm = engine_args.create_engine()
    try:
        print("PREFILL_WARMUP", flush=True)
        llm.generate(prompts, params)
        for i in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params)
            elapsed = time.perf_counter() - start
            if len(outputs) != args.batch_size or any(
                len(o["token_ids"]) != 1 for o in outputs
            ):
                raise RuntimeError(f"incomplete generation: {outputs}")
            if any(
                not math.isfinite(value)
                for output in outputs
                for value in output["logprobs"]
            ):
                raise RuntimeError("nonfinite output logprobs; timing is invalid")
            item = {"wall_seconds": elapsed, "outputs": outputs}
            result["iterations"].append(item)
            print(f"PREFILL_ITERATION {i}: {elapsed:.6f}s", flush=True)
        result["median_seconds"] = statistics.median(
            x["wall_seconds"] for x in result["iterations"]
        )
        result["input_tokens_per_second"] = (
            args.batch_size * args.input_length / result["median_seconds"]
        )
        result["median_engine_ttft_seconds"] = statistics.median(
            output["ttft"]
            for item in result["iterations"]
            for output in item["outputs"]
        )
        if args.torch_profiler_dir:
            llm.start_profile()
            llm.generate(prompts, params)
            llm.stop_profile()
        Path(args.result_file).write_text(
            json.dumps(result, indent=2, default=str, allow_nan=False) + "\n"
        )
        print(
            f"PREFILL_RESULT {args.result_file}: {result['median_seconds']:.6f}s",
            flush=True,
        )
    finally:
        llm.close()


if __name__ == "__main__":
    main()
