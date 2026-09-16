# SPDX-License-Identifier: MIT
"""Real TP4 paged/private-cache parity and scheduler lifecycle acceptance.

Parity is measured against the engine's own irreproducibility rather than
against zero; see `compare_private_cache`.

Run with torchrun --nproc_per_node=4 -m tests.attentions.deepseek_v41.validate_runtime.
One full model instance per rank; PAGE allocation is capped for this test.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.examples.deepseek_v41_offline import offline_forward_context, prepare_engram
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)
from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM
from atom.sampling_params import SamplingParams
from atom.utils.forward_context import reset_forward_context


def reference_forward(model, *args, **kwargs):
    # Hold model arithmetic fixed: quality against P05 is a separate comparison.
    # This oracle uses an uncaptured forward and a private BF16 cache.
    return DeepseekV41ForCausalLM.forward(model, *args, **kwargs)


def divergence(actual, expected):
    """The two numbers this file judges on, for any pair of logit tensors."""
    return {
        "relative_l2": float(
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ),
        "top1_mismatches": int((actual.argmax(-1) != expected.argmax(-1)).sum()),
    }


def compare_private_cache(runner, tokenizer):
    """Teacher-force identical chunks, including long SWA wraps, and ask whether
    the paged path sits inside the spread the engine shows against itself."""
    prompts = [
        (
            "english",
            tokenizer.encode("Explain why water expands when it freezes. " * 4),
        ),
        (
            "chinese",
            tokenizer.encode(
                "请解释为什么天空是蓝色的，并说明日落时颜色发生变化的原因。" * 4
            ),
        ),
        (
            "code",
            tokenizer.encode(
                "def fibonacci(n):\n    # Return the nth Fibonacci number.\n" * 4
            ),
        ),
        (
            "long",
            (tokenizer.encode("The quick brown fox jumps over the lazy dog. ") * 64)[
                :525
            ],
        ),
    ]
    records = []
    prepare = runner.attn_metadata_builder.engram
    for name, tokens in prompts:
        # Two oracles on independently advanced caches, fed the same chunks.
        # The second one measures what the engine cannot reproduce about
        # itself, which is the floor the paged path is judged against.
        private = runner.model.new_cache(1)
        control_cache = runner.model.new_cache(1)
        history = np.full((1, 3), -1, dtype=np.int64)
        seq = Sequence(tokens, runner.block_size, has_per_req_cache=True)
        seq.type, seq.status, seq.state_slot = (
            SequenceType.PREFILL,
            SequenceStatus.RUNNING,
            2,
        )
        count = -(-len(tokens) // runner.block_size)
        seq.block_table = new_block_table(list(range(300, 300 + 2 * count, 2))[::-1])
        at = 0
        lengths = iter((3, 1, 129, 7))
        while at < len(tokens):
            length = min(next(lengths, 128), len(tokens) - at)
            seq.num_cached_tokens = at
            batch = ScheduledBatch(
                {seq.id: seq},
                [length],
                length,
                total_tokens_num_prefill=length,
                total_seqs_num=1,
                total_seqs_num_prefill=1,
            )
            runner.tokenID_processor.clean()
            input_ids, *_ = runner.prepare_model(batch)
            with torch.inference_mode():
                if name == "english" and at == 0:
                    padded = torch.nn.functional.pad(input_ids, (0, 8))
                    hidden = runner.model(padded, torch.zeros_like(padded))
                    assert hidden[length:].count_nonzero() == 0
                    hidden = hidden[:length]
                else:
                    _, hidden = runner.run_model(input_ids, batch)
            reset_forward_context()
            values, history = prepare_engram(
                tokens[at : at + length], at, history, prepare.mapping, prepare.host
            )
            # Both sides want a logit per token, which is what the offline
            # contract says; the serving context this forward ran under would
            # have kept one row per sequence.
            chunk = torch.tensor(tokens[at : at + length], device=runner.device)
            with offline_forward_context(runner.config), torch.inference_mode():
                actual = runner.model.head.get_logits(runner.model.norm(hidden))
                expected = reference_forward(
                    runner.model, chunk[None], private, values, full_logits=True
                )[0]
                repeated = reference_forward(
                    runner.model, chunk[None], control_cache, values, full_logits=True
                )[0]
            record = {
                "case": name,
                "position": at,
                "length": length,
                "max_logit_error": (actual - expected).abs().max().item(),
                **{f"paged_{k}": v for k, v in divergence(actual, expected).items()},
                **{
                    f"engine_noise_{k}": v
                    for k, v in divergence(repeated, expected).items()
                },
            }
            records.append(record)
            if runner.rank == 0:
                print("PARITY", json.dumps(record), flush=True)
            cursor = runner.attn_metadata_builder.cache.cursor[2].cpu().numpy()
            np.testing.assert_array_equal(
                cursor, np.concatenate(([at + length], history[0]))
            )
            at += length
        del private, control_cache
    runner.tokenID_processor.clean()
    # Judged against the measured floor, not against zero: V4's fused MoE is
    # not reproducible on TP4+EP, so the same forward twice already differs and
    # a bit-exact comparison can never pass. An order of magnitude above that
    # floor is still far below what this file exists to catch -- a stale
    # window or a wrong cache row collapses decode outright, not by 10x.
    # Judged over the whole table, because which chunks diverge is the
    # diagnosis and aborting on chunk one throws it away.
    over = [
        record
        for record in records
        if record["paged_relative_l2"] > 10 * record["engine_noise_relative_l2"]
    ]
    assert not over, json.dumps(over, indent=2)
    return records


def offline_completion(runner, tokens, count):
    cache = runner.model.new_cache(1)
    prepare = runner.attn_metadata_builder.engram
    history = np.full((1, 3), -1, dtype=np.int64)
    output = []
    with offline_forward_context(runner.config):
        for _ in range(count):
            values, history = prepare_engram(
                tokens, cache.position, history, prepare.mapping, prepare.host
            )
            logits = reference_forward(
                runner.model,
                torch.tensor([tokens], device=runner.device),
                cache,
                values,
            )
            token = int(logits.argmax(-1).item())
            output.append(token)
            tokens = [token]
    return output


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
    gold = [offline_completion(runner, tokens, 4) for tokens in (base, other)]
    controls = [sequence(base), sequence(other)]
    run("chunked_batched_reorder", controls, reorder=True)
    for seq, expected in zip(controls, gold):
        assert list(seq.token_ids)[seq.num_prompt_tokens :] == expected
    forks = [sequence(base), sequence(base)]
    fork_events = run("prefix_fork", forks)
    assert any(any(n > 0 for n in event["cached"]) for event in fork_events)
    assert sum(event["restores"] for event in fork_events) >= 2
    for seq in forks:
        assert list(seq.token_ids)[seq.num_prompt_tokens :] == gold[0]
    # KV-only hits must rewind if no matching state image remains.
    scheduler.block_manager._state_checkpoint_cache.clear_index()
    fallback = sequence(base)
    replay = run("missing_image_replay", [fallback])
    assert replay[0]["cached"] == [0]
    assert list(fallback.token_ids)[fallback.num_prompt_tokens :] == gold[0]
    unique = tokenizer.encode("Discuss a different topic: binary search. ") + base
    preempted = sequence(unique)
    expected = offline_completion(runner, unique, 4)
    resumed = run("preempt_resume", [preempted], preempt=True)
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
    config = Config(
        model=args.model,
        tensor_parallel_size=size,
        enable_expert_parallel=True,
        enforce_eager=not args.graph,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.PIECEWISE if args.graph else None,
            cudagraph_capture_sizes=[1, 2, 4],
        ),
        kv_cache_dtype=args.cache_dtype,
        index_cache_dtype=args.cache_dtype,
        max_num_batched_tokens=256,
        max_model_len=1024,
        max_num_seqs=4,
        long_prefill_token_threshold=32,
        state_checkpoint_interval_tokens=32,
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
        parity = compare_private_cache(runner, tokenizer)
        events = scheduler_cases(runner, tokenizer)
        graphs = runner.model.dense_graphs
        if args.graph:
            # Every layer captures the same stages, so the entry count is a
            # whole multiple of the layer count -- a layer that silently fell
            # back to eager breaks that. The multiple itself is not asserted:
            # it is the stage set, and 41011da8a already changed it once by
            # letting the decode FFN into the graphs.
            layers = config.hf_config.num_hidden_layers
            assert (
                graphs.entries and len(graphs.entries) % layers == 0
            ), f"{len(graphs.entries)} dense-graph entries over {layers} layers"
            assert graphs.replays > len(graphs.entries)
        report = {
            "graph": args.graph,
            "reference": "uncaptured execution, private BF16 cache",
            "dense_graphs": 0 if graphs is None else len(graphs.entries),
            "dense_graphs_per_layer": (
                0
                if graphs is None
                else len(graphs.entries) / config.hf_config.num_hidden_layers
            ),
            "dense_graph_replays": 0 if graphs is None else graphs.replays,
            "passed": True,
            "tp": size,
            "cache_dtype": args.cache_dtype,
            "parity": parity,
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
