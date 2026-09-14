# SPDX-License-Identifier: MIT
"""TP ModelRunner/Scheduler latency, excluding model loading and graph capture.

Run this same script against the accepted baseline and candidate checkouts.
Each repeat starts a fresh scheduler to avoid prefix-cache hits. Timings include
scheduling, Engram preparation, model execution, sampling and postprocessing;
there is no HTTP server or arrival queue. Outputs are recorded for comparison.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def run_case(runner, prompts, output_tokens, *, multimodal_data=None):
    scheduler = Scheduler(runner.config, state_runtime=runner.state_runtime)
    sequences = [
        Sequence(
            tokens,
            runner.block_size,
            sampling_params=SamplingParams(
                temperature=0, max_tokens=output_tokens, ignore_eos=True
            ),
            has_per_req_cache=True,
            multimodal_data=None if multimodal_data is None else multimodal_data[i],
        )
        for i, tokens in enumerate(prompts)
    ]
    for seq in sequences:
        scheduler.add(seq)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    arrivals = {seq.id: [] for seq in sequences}
    steps = 0
    while not scheduler.is_finished():
        item = scheduler.schedule()
        if item is None:
            raise RuntimeError("Scheduler stalled with unfinished requests")
        batch, seqs = item
        counts = {seq.id: seq.num_completion_tokens for seq in sequences}
        output = runner.forward(batch)
        finished = scheduler.postprocess(list(seqs.values()), output, batch=batch)
        runner.release_multimodal_requests(
            [seq.id for seq in finished if seq.cache_seed != -1]
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        for seq in sequences:
            if seq.num_completion_tokens > counts[seq.id]:
                arrivals[seq.id].append(elapsed)
        steps += 1
    elapsed = time.perf_counter() - start
    scheduler.block_manager.complete_previous_state_batch()
    outputs = [list(seq.token_ids)[seq.num_prompt_tokens :] for seq in sequences]
    if any(len(tokens) != output_tokens for tokens in outputs):
        raise AssertionError("Generation ended before the requested token count")
    first = [arrivals[seq.id][0] for seq in sequences]
    tpot = [
        (arrivals[seq.id][-1] - arrivals[seq.id][0]) / (output_tokens - 1)
        for seq in sequences
    ]
    graphs = getattr(runner.model, "dense_graphs", None)
    return {
        "ttft_ms": statistics.mean(first) * 1000,
        "tpot_ms": statistics.mean(tpot) * 1000,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": len(sequences) * output_tokens / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "steps": steps,
        "graph_replays": 0 if graphs is None else graphs.replays,
        "output_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache-dtype", choices=("bf16", "fp4"), default="bf16")
    parser.add_argument("--expert-backend", choices=("eager", "aiter"), default="eager")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--cases", default="1:128,4:128,8:128,1:1024")
    args = parser.parse_args()
    if args.repeats < 1 or args.output_tokens < 2:
        parser.error("Use at least one repeat and two output tokens")
    cases = [tuple(map(int, case.split(":"))) for case in args.cases.split(",")]
    if any(batch < 1 or length < 1 for batch, length in cases):
        parser.error("Case batch size and prompt length must be positive")
    max_batch = max(batch for batch, _ in cases)
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    buckets = [1 << power for power in range((max_batch - 1).bit_length() + 1)]
    config = Config(
        model=args.model,
        tensor_parallel_size=size,
        enable_expert_parallel=True,
        enforce_eager=not args.graph,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.PIECEWISE if args.graph else None,
            cudagraph_capture_sizes=buckets,
        ),
        kv_cache_dtype=args.cache_dtype,
        index_cache_dtype=args.cache_dtype,
        max_num_batched_tokens=1024,
        max_model_len=max(length for _, length in cases) + args.output_tokens,
        max_num_seqs=max(4, max_batch),
        long_prefill_token_threshold=128,
        state_checkpoint_interval_tokens=128,
        enable_log_stats=False,
        port=port,
    )
    config.hf_config.expert_backend = args.expert_backend
    config.parallel_config.data_parallel_base_port = port
    runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        pages = max(
            1024, 2 * max_batch * (config.max_model_len // runner.block_size + 1)
        )
        runner.pool_plan = runner.pool_plan.with_paged_entries(pages)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(pages)
        if args.graph:
            runner.capture_cudagraph()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        report = {
            "label": args.label,
            "cache_dtype": args.cache_dtype,
            "expert_backend": args.expert_backend,
            "graph": args.graph,
            "tp": size,
            "output_tokens": args.output_tokens,
            "page_bytes": runner.attn_metadata_builder.geometry.page_bytes,
            "state_bytes": runner.attn_metadata_builder.geometry.state_bytes,
            "cases": [],
        }
        for batch, length in cases:
            prompts = []
            for i in range(batch):
                seed = tokenizer.encode(
                    f"Request {i}. Explain this carefully: Water freezes, clouds form, "
                    "and sunlight changes the temperature. Give concrete examples. "
                )
                prompts.append((seed * (-(-length // len(seed))))[:length])
            run_case(runner, prompts, args.output_tokens)  # untimed warmup
            records = []
            for repeat in range(args.repeats):
                record = run_case(runner, prompts, args.output_tokens)
                all_ranks = [None] * size
                torch.distributed.all_gather_object(all_ranks, record)
                if len({r["output_sha256"] for r in all_ranks}) != 1:
                    raise AssertionError(
                        "Tensor-parallel ranks generated different tokens"
                    )
                record = dict(record)
                for metric in (
                    "ttft_ms",
                    "tpot_ms",
                    "elapsed_seconds",
                    "peak_allocated_gib",
                ):
                    record[metric] = max(r[metric] for r in all_ranks)
                record["output_tokens_per_second"] = (
                    batch * args.output_tokens / record["elapsed_seconds"]
                )
                records.append(record)
                if rank == 0:
                    print(
                        "BENCH",
                        args.label,
                        batch,
                        length,
                        repeat,
                        json.dumps({k: v for k, v in record.items() if k != "outputs"}),
                        flush=True,
                    )
            if len({r["output_sha256"] for r in records}) != 1:
                raise AssertionError("Repeated generation was not deterministic")
            report["cases"].append(
                {
                    "batch": batch,
                    "prompt_tokens": length,
                    "repeats": records,
                    **{
                        key: statistics.median(r[key] for r in records)
                        for key in (
                            "ttft_ms",
                            "tpot_ms",
                            "output_tokens_per_second",
                            "peak_allocated_gib",
                        )
                    },
                }
            )
            if rank == 0:
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
