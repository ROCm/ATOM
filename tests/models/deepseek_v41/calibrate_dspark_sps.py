# SPDX-License-Identifier: MIT
"""Measure verification cost through real request lifecycles in isolation."""

import argparse
import copy
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import torch
from tests.attentions.deepseek_v41.benchmark_runtime import run_case
from transformers import AutoTokenizer

from atom.config import (
    CompilationConfig,
    Config,
    CUDAGraphMode,
    DSparkConfig,
    SpeculativeConfig,
)
from atom.model_engine.model_runner import ModelRunner
from atom.spec_decode.dspark_scheduler import build_sps_table

from .dspark_runtime_stats import RuntimeStats
from .validate_dspark_runtime import diagnostic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workload", type=Path)
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])

    def check(config):
        checked = copy.copy(config)
        checked.dspark = copy.copy(config.dspark)
        checked.dspark.confidence_schedule = False
        diagnostic_config(checked)

    with patch("atom.models.deepseek_v41.config.validate_runtime_config", check):
        config = Config(
            model=args.model,
            tensor_parallel_size=size,
            enable_expert_parallel=True,
            enforce_eager=False,
            compilation_config=CompilationConfig(
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_capture_sizes=[1, 2, 4],
            ),
            speculative_config=SpeculativeConfig(
                method="dspark", model=args.model, num_speculative_tokens=5
            ),
            dspark=DSparkConfig(
                confidence_schedule=True, ragged=True, disable_sps_calib=True
            ),
            kv_cache_dtype="bf16",
            index_cache_dtype="bf16",
            max_num_batched_tokens=1024,
            max_model_len=512,
            max_num_seqs=4,
            long_prefill_token_threshold=256,
            state_checkpoint_interval_tokens=128,
            enable_log_stats=False,
            port=port,
        )
    config.parallel_config.data_parallel_base_port = port
    report = {
        "completed": False,
        "tp": size,
        "cache_dtype": "bf16",
        "graph": True,
        "gpu_name": torch.cuda.get_device_name(rank),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "context_tokens": 128,
        "max_num_seqs": 4,
        "draft_width": 5,
        "cases": [],
        "measurement": "max-rank target forward CUDA events; one TP4 instance; 3 repeats after shape warmup",
    }
    runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        runner.capture_cudagraph()
        stats = RuntimeStats(runner)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prompts = []
        for i in range(4):
            seed = tokenizer.encode(
                f"Request {i}. Explain why water expands when it freezes. "
            )
            prompts.append((seed * 30)[:128])
        if args.workload is not None:
            prompts = json.loads(args.workload.read_text())["batches"][0]["input_ids"]
            report["input_lengths"] = [len(ids) for ids in prompts]
            report.pop("context_tokens")
        buckets = defaultdict(list)
        completed = set()
        if args.resume is not None:
            previous = json.loads(args.resume.read_text())
            report["resumed_from"] = str(args.resume)
            for row in previous["cases"]:
                buckets[row["tokens"]].extend(row["samples_ms"])
                report["cases"].append(row)
                completed.add((row["requests"], row["query_width"]))
        for batch in (1, 2, 4):
            for drafts in range(6):
                if (batch, drafts + 1) in completed:
                    continue
                runner.drafter.verify_scheduler.compute_ell = (
                    lambda confidence, drafts=drafts: torch.full_like(
                        confidence[:, 0], drafts, dtype=torch.int64
                    )
                )
                run_case(runner, prompts[:batch], 12)
                stats.take()
                times = []
                for repeat in range(3):
                    run_case(runner, prompts[:batch], 96)
                    local = stats.take()["verify"]["measurements"]
                    ranks = [None] * size
                    torch.distributed.all_gather_object(ranks, local)
                    assert all(len(rows) == len(local) for rows in ranks)
                    for samples in zip(*ranks):
                        assert (
                            len({(r["requests"], r["target_tokens"]) for r in samples})
                            == 1
                        )
                        row = samples[0]
                        if row["requests"] == batch and row[
                            "target_tokens"
                        ] == batch * (drafts + 1):
                            times.append(max(r["milliseconds"] for r in samples))
                if len(times) < 3:
                    raise AssertionError("Insufficient actual verification samples")
                tokens = batch * (drafts + 1)
                buckets[tokens].extend(times)
                row = {
                    "requests": batch,
                    "query_width": drafts + 1,
                    "tokens": tokens,
                    "samples_ms": times,
                    "median_ms": statistics.median(times),
                }
                report["cases"].append(row)
                if rank == 0:
                    print(
                        json.dumps({k: v for k, v in row.items() if k != "samples_ms"}),
                        flush=True,
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        points = sorted(buckets)
        # A conservative monotone envelope keeps the scheduler's cost contract.
        measured = [statistics.median(buckets[point]) for point in points]
        envelope = []
        for value in measured:
            envelope.append(max(value, envelope[-1] if envelope else 0))
        report.update(
            completed=True,
            token_points=points,
            measured_median_ms=measured,
            monotone_median_ms=envelope,
            sps_table=build_sps_table(
                points, [1000 / value for value in envelope], 24
            ).tolist(),
        )
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
