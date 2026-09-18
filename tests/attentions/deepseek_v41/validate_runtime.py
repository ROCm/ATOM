# SPDX-License-Identifier: MIT
"""Real TP4 scheduler lifecycle acceptance.

Every case asserts that some other route -- chunked and reordered, forked,
replayed from a missing image, preempted -- finishes with the tokens an
uninterrupted scheduled run produced.

Run with torchrun --nproc_per_node=4 -m tests.attentions.deepseek_v41.validate_runtime.
One full model instance per rank; PAGE allocation is capped for this test.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import torch
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
)
from atom.sampling_params import SamplingParams


def scheduler_cases(runner, tokenizer):
    scheduler = Scheduler(runner.config, state_runtime=runner.state_runtime)
    events = []
    serial = 100

    def sequence(tokens, count=4):
        nonlocal serial
        serial += 1
        return Sequence(
            tokens,
            runner.block_size,
            sampling_params=SamplingParams(
                temperature=0, max_tokens=count, ignore_eos=True
            ),
            has_per_req_cache=True,
            id=serial,
        )

    def run(name, sequences, *, reorder=False, preempt=False, cancel=False):
        for seq in sequences:
            scheduler.add(seq)
        changed = False
        begin = len(events)
        for tick in range(100):
            item = scheduler.schedule()
            if item is None:
                break
            batch, seqs = item
            event = {
                "case": name,
                "ids": list(batch.req_ids),
                "lengths": batch.num_scheduled_tokens.tolist(),
                "cached": list(batch.num_cached_tokens),
                "stores": len(batch.state_maintenance_ops.checkpoint_stores),
                "restores": len(batch.state_maintenance_ops.checkpoint_restores),
            }
            events.append(event)
            output = runner.forward(batch)
            scheduler.postprocess(list(seqs.values()), output, batch=batch)
            if reorder:
                scheduler.running = deque(reversed(scheduler.running))
            target = sequences[0]
            boundary = 64 if preempt else 32
            if (
                not changed
                and target.is_partial_prefill
                and target.num_cached_tokens >= boundary
            ):
                if preempt:
                    scheduler.block_manager.complete_previous_state_batch()
                    scheduler.running.remove(target)
                    assert scheduler.preempt(target)
                    changed = True
                elif cancel:
                    target.status = SequenceStatus.ABORTED
                    changed = True
        else:
            raise AssertionError(f"{name}: scheduler did not drain")
        assert scheduler.is_finished(), name
        if preempt or cancel:
            assert changed, name
        if runner.rank == 0:
            print("SCHEDULER", name, json.dumps(events[begin:]), flush=True)
        return events[begin:]

    base = (
        tokenizer.encode("Facts about water, clouds, sunlight and the atmosphere. ")
        * 12
    )[:96]
    other = (
        tokenizer.encode("A Python function can return a list of prime numbers. ") * 10
    )[:81]
    # An uninterrupted scheduled run, which is what every case below claims to
    # reproduce. Previously an offline oracle, which stopped matching the
    # engine past ~600 tokens and so capped how long these prompts could be.
    baseline = [sequence(base), sequence(other)]
    run("baseline", baseline)
    gold = [list(seq.token_ids)[seq.num_prompt_tokens :] for seq in baseline]
    controls = [sequence(base), sequence(other)]
    run("chunked_batched_reorder", controls, reorder=True)
    for seq, expected in zip(controls, gold):
        assert list(seq.token_ids)[seq.num_prompt_tokens :] == expected
    forks = [sequence(base), sequence(base)]
    fork_events = run("prefix_fork", forks)
    assert any(any(n > 0 for n in event["cached"]) for event in fork_events)
    for seq in forks:
        assert list(seq.token_ids)[seq.num_prompt_tokens :] == gold[0]
    # A state checkpoint needs a prompt longer than one interval, and the
    # interval is a whole PAGE: 256 tokens since P3 made that the block size,
    # against the 96 above. So this asserts under its own precondition rather
    # than unconditionally, and comes back on its own if the prompt grows.
    # The restore path is therefore NOT covered in this file at a 256-token
    # PAGE, and nothing else here covers it either.
    if len(base) > runner.config.state_checkpoint_interval_tokens:
        assert sum(event["restores"] for event in fork_events) >= 2
    # KV-only hits must rewind if no matching state image remains.
    scheduler.block_manager._state_checkpoint_cache.clear_index()
    fallback = sequence(base)
    replay = run("missing_image_replay", [fallback])
    assert replay[0]["cached"] == [0]
    assert list(fallback.token_ids)[fallback.num_prompt_tokens :] == gold[0]
    unique = tokenizer.encode("Discuss a different topic: binary search. ") + base
    uninterrupted = sequence(unique)
    run("preempt_baseline", [uninterrupted])
    expected = list(uninterrupted.token_ids)[uninterrupted.num_prompt_tokens :]
    preempted = sequence(unique)
    resumed = run("preempt_resume", [preempted], preempt=True)
    # Resuming a preempted request restores a checkpoint too, so it carries
    # the same precondition as the fork case above -- and, either way, that
    # the request finishes with the tokens it would have produced uninterrupted
    # is the claim this case exists for.
    if len(unique) > runner.config.state_checkpoint_interval_tokens:
        assert sum(event["restores"] for event in resumed) >= 1
    assert list(preempted.token_ids)[preempted.num_prompt_tokens :] == expected
    cancelled = sequence(tokenizer.encode("A cancelled request: ") + other, 12)
    run("cancel", [cancelled], cancel=True)
    assert cancelled.leave_reason == "aborted"
    recycled = sequence(other)
    run("slot_recycle", [recycled])
    assert list(recycled.token_ids)[recycled.num_prompt_tokens :] == gold[1]
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dtype", choices=("bf16", "fp4"), default="bf16")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    capture_sizes = [1, 2, 4]
    # The index plane is FP8 whatever the main pool is: the paged scorer is
    # the only reader and it is the format that scorer takes.
    index_dtype = "fp8"
    config = Config(
        model=args.model,
        tensor_parallel_size=size,
        enable_expert_parallel=True,
        enforce_eager=not args.graph,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL if args.graph else None,
            cudagraph_capture_sizes=capture_sizes,
        ),
        kv_cache_dtype=args.cache_dtype,
        index_cache_dtype=index_dtype,
        max_num_batched_tokens=256,
        max_model_len=1024,
        max_num_seqs=4,
        long_prefill_token_threshold=32,
        # A multiple of the PAGE size, which is the prefix-cache hash block:
        # anything else is snapped to off, and then this file's fork case is
        # asserting on a feature that was never running.
        state_checkpoint_interval_tokens=256,
        enable_log_stats=False,
        port=port,
    )
    config.parallel_config.data_parallel_base_port = port
    start = time.perf_counter()
    runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        if args.graph:
            # Capture starts each synthetic request behind a full window, so it
            # declares `ceil((window_size + max_q_len) / block_size)` pages and
            # STATE slots `[0, bs)` as scratch and scribbles on them, as V4 does
            # ("the data is throwaway", deepseek_v4_attn.py) -- safe only
            # because capture precedes admission. What must stay pristine is
            # every page it never named, so the bound is computed the way the
            # builder computes it rather than spelled: a wider window, a bigger
            # block or draft tokens all move it, and this assert is where a
            # capture that outgrew its scratch is found.
            builder = runner.attn_metadata_builder
            cache = builder.cache
            max_q_len = runner.drafter.mtp_k + 1 if hasattr(runner, "drafter") else 1
            scratch = -(-(cache.geometry.window_size + max_q_len) // builder.block_size)
            before = cache.page_bytes[scratch:].clone()
            runner.capture_cudagraph()
            torch.testing.assert_close(
                cache.page_bytes[scratch:], before, rtol=0, atol=0
            )
            del before
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        events = scheduler_cases(runner, tokenizer)
        shapes = sorted(getattr(runner, "graphs", {}))
        if args.graph:
            # One whole-forward graph per declared shape, and nothing per
            # layer: the target is a single replay now. A shape that failed to
            # capture would be missing here, and a decode step that reached for
            # it would have raised a KeyError inside `scheduler_cases` above --
            # so those cases passing is the other half of this claim.
            buckets = runner._dspark_capture_q_buckets(
                runner.drafter.mtp_k + 1 if hasattr(runner, "drafter") else 1
            )
            assert shapes == sorted(
                (bs, q) for q in buckets for bs in capture_sizes
            ), f"captured {shapes}, declared {capture_sizes} x {buckets}"
        report = {
            "graph": args.graph,
            "reference": "uncaptured execution, private BF16 cache",
            "target_graphs": len(shapes),
            "target_graph_shapes": shapes,
            "passed": True,
            "tp": size,
            "cache_dtype": args.cache_dtype,
            "index_cache_dtype": index_dtype,
            "scheduler": events,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "page_bytes": runner.attn_metadata_builder.geometry.page_bytes,
            "state_bytes": runner.attn_metadata_builder.geometry.state_bytes,
        }
        if rank == 0:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print("RUNTIME_ACCEPTANCE_PASSED", flush=True)
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
