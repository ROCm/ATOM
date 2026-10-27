"""V4.1 TP x DP model/scheduler integration; writes counts and hashes only.

Use a separately measured SPS profile for this topology. The reference run
keeps full-width verification; --dynamic exercises online group planning.
--controlled-confidence exercises graph contraction independently of the
model's natural confidence distribution, without changing sampled tokens.
"""

import argparse
import copy
import faulthandler
import hashlib
import json
import os
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

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
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams
from atom.spec_decode.calibration import single_node_parallel_config
from atom.utils.forward_context import get_forward_context


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--dp-size", type=int, default=2)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--controlled-confidence", action="store_true")
    p.add_argument("--diagnostic-prefixes", action="store_true")
    p.add_argument("--compare-with", type=Path)
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    parallel, tp, local = single_node_parallel_config(world, args.dp_size, rank)
    parallel.data_parallel_base_port = int(os.environ["MASTER_PORT"])
    config = Config(
        model=args.model,
        tensor_parallel_size=tp,
        parallel_config=parallel,
        enable_dp_attention=True,
        enable_expert_parallel=args.enable_expert_parallel,
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
            online_sps=args.dynamic,
            bucket_global=args.dynamic,
            online_sps_window_steps=32,
            online_sps_sample_interval=1,
            online_sps_update_interval=4,
        ),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        max_num_batched_tokens=512,
        max_model_len=1024,
        max_num_seqs=4,
        long_prefill_token_threshold=64,
        state_checkpoint_interval_tokens=256,
        enable_encoder_swa_bounded_replay=False,
        enable_log_stats=False,
        port=int(os.environ["MASTER_PORT"]),
    )
    with torch.inference_mode(False):
        runner = ModelRunner(local, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        runner.capture_cudagraph()
        from aiter.dist.parallel_state import get_dp_group, get_tp_group

        cpu_group = get_dp_group().cpu_group
        tp_group = get_tp_group().cpu_group
        leader = dist.get_rank(tp_group) == 0
        tp_source = dist.get_global_rank(tp_group, 0)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        seed = tokenizer.encode("Explain why water expands when it freezes.")
        scheduler = (
            Scheduler(
                config,
                state_runtime=runner.state_runtime,
                prefill_token_graphs=runner.prefill_token_graphs(),
            )
            if leader
            else None
        )
        # Use production prefill alignment; a bare Scheduler omits the
        # cross-DP delayer and can exercise unsupported mixed graph collectives
        # before the optimization is even enabled.
        from atom.model_engine.engine_core import EngineCore

        if leader:
            EngineCore._init_prefill_delayer(
                SimpleNamespace(scheduler=scheduler), config, cpu_group
            )
        # Different prompt lengths and decode lifetimes create real mixed P/D
        # and idle forwards rather than relabeling identical worker batches.
        dp_rank = parallel.data_parallel_rank
        specs = [(321, 80)] if dp_rank == 0 else [(33, 128), (65, 128), (97, 128)]
        sequences = []
        for length, limit in specs if leader else []:
            prompt = (seed * -(-length // len(seed)))[:length]
            seq = Sequence(
                prompt,
                runner.block_size,
                SamplingParams(temperature=0, max_tokens=limit, ignore_eos=True),
                has_per_req_cache=True,
                num_draft_tokens=runner.num_spec_tokens,
            )
            sequences.append(seq)
            scheduler.add(seq)
        counts, shapes = Counter(), Counter()
        first_shrink = []
        original_prepare = runner.prepare_model

        def observed_prepare(batch):
            old = batch.total_tokens_num
            result = original_prepare(batch)
            mode = get_forward_context().context.forward_mode
            counts[
                (
                    "dummy"
                    if batch.is_dummy_run
                    else "prefill" if batch.total_tokens_num_prefill else "decode"
                )
            ] += 1
            counts["mixed_group"] += int(
                mode.sync.any_rank_has_prefill and not all(mode.sync.prefill_across_dp)
            )
            if not batch.total_tokens_num_prefill:
                shapes[
                    (
                        old,
                        batch.total_tokens_num,
                        mode.running_tokens,
                        bool(mode.token_graph),
                    )
                ] += 1
                if old > batch.total_tokens_num and not first_shrink:
                    first_shrink.append(
                        [len(s.completion_token_ids) for s in sequences]
                    )
            return result

        runner.prepare_model = observed_prepare
        if not args.dynamic:
            # Keep capture/layout identical to the dynamic arm. Only the
            # verification budget differs: reference always verifies all drafts.
            runner.drafter.verify_scheduler.compute_ell = lambda confidence: (
                torch.full(
                    (confidence.shape[0],),
                    confidence.shape[1],
                    dtype=torch.long,
                    device=confidence.device,
                )
            )
        if args.dynamic and args.controlled_confidence:
            verify = runner.drafter.verify_scheduler
            original_compute = verify.compute_ell

            def controlled(confidence):
                return original_compute(torch.full_like(confidence, 0.05))

            verify.compute_ell = controlled
        faulthandler.dump_traceback_later(90, repeat=True)
        forward_digest = hashlib.sha256()
        for tick in range(400):
            # Production has one scheduler per TP group. In particular the
            # wall-clock prefill delayer must not run independently on peers.
            control = [None]
            if leader:
                active = torch.tensor(
                    [not scheduler.is_finished()], dtype=torch.int32, device="cpu"
                )
                dist.all_reduce(active, op=dist.ReduceOp.MAX, group=cpu_group)
                item = scheduler.schedule() if active.item() else None
                batch = item[0] if item is not None else None
                if batch is not None and not batch.req_ids:
                    batch = None
                control[0] = (bool(active.item()), batch, scheduler.dummy_prefill)
            dist.broadcast_object_list(control, src=tp_source, group=tp_group)
            active, batch, dummy_prefill = control[0]
            if not active:
                break
            if batch is None:
                output = runner.dummy_execution(prefill=dummy_prefill)
                if leader and dummy_prefill and scheduler.running:
                    scheduler.postprocess([], output, batch=None)
            else:
                # ModelRunner mutates the worker's snapshot for ragged input.
                # Keep the scheduler's snapshot separate, as IPC does in serving.
                output = runner.forward(copy.deepcopy(batch))
                runner.attn_metadata_builder.cache.require_committed()
                if leader:
                    scheduler.postprocess(list(item[1].values()), output, batch=batch)
                    if (
                        scheduler.prefill_delayer is not None
                        and batch.total_tokens_num_prefill
                    ):
                        scheduler.prefill_delayer.notify_prefill_executed()
            forward_digest.update(repr(output.token_ids).encode())
            if leader and tick in (18, 36):
                scheduler.running = deque(reversed(scheduler.running))
        else:
            raise AssertionError("DP schedulers did not drain")
        if leader:
            scheduler.block_manager.complete_previous_state_batch()
            assert scheduler.is_finished()
        output = [
            {
                "tokens": len(s.completion_token_ids),
                "sha256": hashlib.sha256(
                    json.dumps(list(s.completion_token_ids)).encode()
                ).hexdigest(),
            }
            for s in sequences
        ]
        if args.diagnostic_prefixes:
            for record, seq in zip(output, sequences):
                tokens = list(seq.completion_token_ids)
                record["prefix_sha256"] = [
                    hashlib.sha256(json.dumps(tokens[:i]).encode()).hexdigest()
                    for i in range(1, len(tokens) + 1)
                ]
        completed = [output if leader else None]
        dist.broadcast_object_list(completed, src=tp_source, group=tp_group)
        output = completed[0]
        report = {
            "completed": True,
            "dynamic": args.dynamic,
            "controlled_confidence": args.controlled_confidence,
            "tp": tp,
            "dp": args.dp_size,
            "dp_rank": dp_rank,
            "counts": dict(counts),
            "outputs": output,
            "first_shrink_at_output_counts": first_shrink,
            "shapes": [
                {
                    "before": a,
                    "scheduled": b,
                    "executed": c,
                    "token_graph": d,
                    "steps": n,
                }
                for (a, b, c, d), n in sorted(shapes.items())
            ],
        }
        if args.dynamic:
            verify = runner.drafter.verify_scheduler
            report["sps"] = verify.runtime.model.snapshot()
            report["planned"], report["contracted"] = (
                verify.dp.planned,
                verify.dp.contracted,
            )
            assert report["sps"]["accepted"] > 0
        peers = [None] * args.dp_size
        dist.all_gather_object(peers, report, group=cpu_group)
        if args.dynamic:
            assert all(x["sps"] == peers[0]["sps"] for x in peers)
        tp_reports = [None] * tp
        dist.all_gather_object(tp_reports, forward_digest.hexdigest(), group=tp_group)
        assert all(x == tp_reports[0] for x in tp_reports)
        if rank == 0:
            args.output.write_text(
                json.dumps({"completed": True, "ranks": peers}, indent=2) + "\n"
            )
            print(
                "DSPARK_DP_MODEL_EXECUTION_COMPLETE",
                {"tp": tp, "dp": args.dp_size, "dynamic": args.dynamic},
            )
        if args.compare_with is not None:
            reference = json.loads(args.compare_with.read_text())["ranks"]
            # Compare final hashes, independent of optional diagnostic fields.
            actual = [[(o["tokens"], o["sha256"]) for o in r["outputs"]] for r in peers]
            expected = [
                [(o["tokens"], o["sha256"]) for o in r["outputs"]] for r in reference
            ]
            assert actual == expected, "Model output hashes differ from reference"
    finally:
        faulthandler.cancel_dump_traceback_later()
        torch.cuda.synchronize()
        runner.exit()


if __name__ == "__main__":
    main()
    import gc

    gc.collect()
