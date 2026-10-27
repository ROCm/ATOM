"""Measure V4.1 target graph costs in a disposable TP process.

This process never serves requests: graph replays write synthetic cache slots,
then the entire process/cache exits. Do not invoke on a live serving runner.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from atom.config import CompilationConfig, Config, CUDAGraphMode, SpeculativeConfig
from atom.model_engine.model_runner import ModelRunner
from atom.spec_decode.calibration import calibration_identity
from atom.utils.forward_context import (
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)
from atom.utils.region_graph import TokenGraph


class ProfileRunner(ModelRunner):
    def _record_prefill_graphs(self):
        # Disposable decode-only calibration. Serving records its prefill
        # graphs normally; none of these synthetic caches are served.
        pass

    def _dspark_capture_q_buckets(self, full_q):
        # Capture the actual supported target buckets without enabling a
        # confidence scheduler that would itself require this output profile.
        assert full_q == 6
        return [1, 3, 6]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    config = Config(
        model=args.model,
        tensor_parallel_size=size,
        enable_expert_parallel=False,
        gpu_memory_utilization=0.7,
        compilation_config=CompilationConfig(
            level=0,
            cudagraph_mode=CUDAGraphMode.TR_HYBRID,
            cudagraph_capture_sizes=list(range(1, 17)) + [24, 32, 48, 64, 128],
        ),
        speculative_config=SpeculativeConfig(
            method="dspark", model=args.model, num_speculative_tokens=5
        ),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        long_prefill_token_threshold=16384,
        state_checkpoint_interval_tokens=8192,
        enable_log_stats=False,
        port=port,
    )
    config.parallel_config.data_parallel_base_port = port
    # The loader uses worker threads. Allocate ordinary parameters there;
    # inference mode is thread-local and cannot cover their staging copies.
    with torch.inference_mode(False):
        runner = ProfileRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        runner.capture_cudagraph()
        points = []
        shapes = []
        for bs, q in sorted((bs, q) for bs in runner.capture_sizes for q in [1, 3, 6]):
            rows = bs * q
            assert rows in TokenGraph.captured_sizes("decode"), (bs, q)
            metadata, context = runner._build_capture(
                runner.attn_metadata_builder.build_for_cudagraph_capture,
                bs=bs,
                max_q_len=q,
            )
            runner.running_tokens_per_dp_rank.fill(rows)
            set_forward_context(
                attn_metadata=metadata,
                atom_config=config,
                context=context,
                running_tokens_per_dp_rank=runner.running_tokens_per_dp_rank,
                in_hipgraph=True,
            )
            get_forward_context().tr_graph = "replay"
            torch.cuda.synchronize()

            def forward(rows=rows, positions=context.positions):
                hidden = runner.model(
                    runner.forward_vars["input_ids"].gpu[:rows], positions
                )
                runner.model.compute_logits(hidden)

            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            times = []
            for _ in range(15):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                forward()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            value = torch.tensor(
                [statistics.median(times)], device="cpu", dtype=torch.float64
            )
            from aiter.dist.parallel_state import get_tp_group

            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
            points.append([bs * q, value.item()])
            shapes.append(
                {"batch": bs, "q": q, "rows": bs * q, "target_ms": value.item()}
            )
            reset_forward_context()
        if rank == 0:
            args.output.write_text(
                json.dumps(
                    {
                        "identity": calibration_identity(config, runner.device),
                        "max_num_seqs": 128,
                        "points": points,
                        "shapes": shapes,
                        "scope": "max TP rank median; real V4.1 weights; disposable TR_HYBRID token graphs at startup window context; target including eager regions and LM head; no sampler/draft",
                    },
                    indent=2,
                )
                + "\n"
            )
            print("SPS_PROFILE_COMPLETE", len(points))
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
