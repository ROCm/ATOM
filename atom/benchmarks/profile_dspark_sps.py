"""Measure V4.1 target graph costs in disposable TP/DP workers.

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

from atom.config import (
    CompilationConfig,
    Config,
    CUDAGraphMode,
    SpeculativeConfig,
)
from atom.model_engine.model_runner import ModelRunner
from atom.spec_decode.calibration import (
    calibration_identity,
    single_node_parallel_config,
)
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
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--moe-all2all-backend", default="auto")
    parser.add_argument("--max-num-seqs", type=int, default=128)
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if int(os.environ.get("LOCAL_WORLD_SIZE", size)) != size:
        parser.error("This disposable calibration launcher is single-node")
    if args.dp_size < 1 or size % args.dp_size:
        parser.error("WORLD_SIZE must be divisible by --dp-size")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    parallel, tp_size, tp_rank = single_node_parallel_config(size, args.dp_size, rank)
    port = int(os.environ["MASTER_PORT"])
    config = Config(
        model=args.model,
        tensor_parallel_size=tp_size,
        parallel_config=parallel,
        enable_dp_attention=args.enable_dp_attention,
        enable_expert_parallel=args.enable_expert_parallel,
        moe_all2all_backend=args.moe_all2all_backend,
        gpu_memory_utilization=0.7,
        compilation_config=CompilationConfig(
            level=0,
            cudagraph_mode=CUDAGraphMode.TR_HYBRID,
            cudagraph_capture_sizes=sorted(
                set(
                    [
                        b
                        for b in list(range(1, 17)) + [24, 32, 48, 64, 128]
                        if b <= args.max_num_seqs
                    ]
                    + [args.max_num_seqs]
                )
            ),
        ),
        speculative_config=SpeculativeConfig(
            method="dspark", model=args.model, num_speculative_tokens=5
        ),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        max_num_seqs=args.max_num_seqs,
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
        runner = ProfileRunner(tp_rank, config)
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
            from aiter.dist.parallel_state import get_dp_group, get_tp_group

            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=get_dp_group().cpu_group)
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
                        "max_num_seqs": args.max_num_seqs,
                        "points": points,
                        "shapes": shapes,
                        "scope": "max TP/DP rank median at equal local rows; real V4.1 weights; disposable TR_HYBRID token graphs at startup window context; target including eager regions and LM head; no sampler/draft",
                    },
                    indent=2,
                )
                + "\n"
            )
            print("SPS_PROFILE_COMPLETE", len(points))
    finally:
        torch.cuda.synchronize()
        runner.exit()


if __name__ == "__main__":
    main()
    # Release graph/runner cycles before torch's operator registry is torn down.
    import gc

    gc.collect()
