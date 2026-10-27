"""Production-admitted TP V4.1 online-SPS lifecycle probe with real weights."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from atom.config import (
    CompilationConfig,
    Config,
    CUDAGraphMode,
    DSparkConfig,
    SpeculativeConfig,
)
from atom.model_engine.model_runner import ModelRunner
from atom.utils.forward_context import get_forward_context
from atom.utils.region_graph import TokenGraph
from tests.models.deepseek_v41.validate_dspark_lifecycle import run_scenarios


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controlled-lengths", action="store_true")
    args = p.parse_args()
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
            cudagraph_capture_sizes=[1, 2, 3, 4],
        ),
        speculative_config=SpeculativeConfig(
            method="dspark", model=args.model, num_speculative_tokens=5
        ),
        dspark=DSparkConfig(
            confidence_schedule=True,
            ragged=True,
            ragged_graph_sizes="1,3,6",
            calibration_profile=args.profile,
            online_sps=True,
            bucket_global=True,
            online_sps_window_steps=64,
            online_sps_sample_interval=2,
            online_sps_update_interval=16,
        ),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        max_num_batched_tokens=256,
        max_model_len=512,
        max_num_seqs=4,
        long_prefill_token_threshold=128,
        state_checkpoint_interval_tokens=256,
        # These scenarios explicitly assert checkpoint restore. The serving
        # default can rebuild encoder state with bounded replay instead.
        enable_encoder_swa_bounded_replay=False,
        enable_log_stats=False,
        port=port,
    )
    config.parallel_config.data_parallel_base_port = port
    report = {
        "completed": False,
        "tp": size,
        "level": 0,
        "production_config": True,
        "cases": [],
        "controlled_lengths": args.controlled_lengths,
    }
    with torch.inference_mode(False):
        runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        runner.capture_cudagraph()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        shapes = Counter()
        replay_counts = Counter()
        prepare = runner.prepare_model
        replay = TokenGraph.replay

        def observed_prepare(batch):
            original_rows = batch.total_tokens_num
            result = prepare(batch)
            mode = get_forward_context().context.forward_mode
            if not batch.is_dummy_run and not batch.total_tokens_num_prefill:
                shapes[
                    (
                        int(original_rows),
                        int(batch.total_tokens_num),
                        int(mode.running_tokens),
                        bool(mode.token_graph),
                    )
                ] += 1
            return result

        def observed_replay(graph, *a, **kw):
            # ReqGraph subclasses TokenGraph; count the token graphs themselves.
            if type(graph) is TokenGraph:
                replay_counts[str(kw.get("phase"))] += 1
            return replay(graph, *a, **kw)

        runner.prepare_model = observed_prepare
        TokenGraph.replay = observed_replay
        vs = runner.drafter.verify_scheduler
        compute = vs.compute_ell
        if args.controlled_lengths:
            calls = 0

            def controlled(confidence):
                nonlocal calls
                compute(
                    confidence
                )  # Keep live SPS updates, but isolate layout correctness.
                phase = calls // 8
                calls += 1
                if phase == 0:
                    return torch.zeros_like(confidence[:, 0], dtype=torch.int64)
                return (
                    torch.arange(confidence.shape[0], device=confidence.device) + phase
                ) % (confidence.shape[1] + 1)

            vs.compute_ell = controlled
        try:
            run_scenarios(
                runner,
                tokenizer,
                report,
                args.output,
                ragged_probe=args.controlled_lengths,
                sampling_probe=True,
            )
        finally:
            runner.prepare_model = prepare
            TokenGraph.replay = replay
            vs.compute_ell = compute
        report["decode_shapes"] = [
            {"original": a, "scheduled": b, "executed": c, "token_graph": d, "steps": n}
            for (a, b, c, d), n in sorted(shapes.items())
        ]
        report["token_graph_replays"] = dict(replay_counts)
        assert sum(replay_counts.values()) > 0
        assert all(d for (a, b, c, d) in shapes), "decode left the token graph path"
        if args.controlled_lengths:
            assert any(
                b < a and c < a for (a, b, c, d) in shapes
            ), "no actual execution reduction"
        runtime = runner.drafter.verify_scheduler.runtime
        snapshot = runtime.snapshot()
        snapshots = [None] * size
        from aiter.dist.parallel_state import get_tp_group

        dist.all_gather_object(snapshots, snapshot, group=get_tp_group().cpu_group)
        keys = [
            "version",
            "accepted",
            "slope_ms_per_row_by_batch",
            "fixed_ms_by_batch",
            "fixed_ms_global",
            "slope_support_by_batch",
            "post_ms_by_batch",
        ]
        assert snapshot["version"] > 0 and snapshot["accepted"] > 0, snapshot
        assert all(
            {k: s[k] for k in keys} == {k: snapshots[0][k] for k in keys}
            for s in snapshots
        )
        report["online_ranks"] = snapshots
        report["target_token_graph_rows"] = sorted(TokenGraph.captured_sizes("decode"))
        if rank == 0:
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print("DSPARK_ONLINE_LIFECYCLE_PASS", snapshot)
    finally:
        # The diagnostic's last forward can leave draft/event work in flight.
        # Finish it before destroying the distributed buffers.
        torch.cuda.synchronize()
        runner.exit()


if __name__ == "__main__":
    main()
    # Collect runner/proposer cycles while torch's operators are still alive.
    import gc

    gc.collect()
