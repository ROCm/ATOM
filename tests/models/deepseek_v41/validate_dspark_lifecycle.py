# SPDX-License-Identifier: MIT
"""Real TP scheduler lifecycle checks for native DSpark verification."""

import argparse
import copy
import json
import os
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
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
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.models.deepseek_v41.config import validate_runtime_config
from atom.sampling_params import SamplingParams


def diagnostic_config(config):
    """Validate the runtime as if the draft were absent.

    `validate_runtime_config` rejects the combinations the engine cannot serve;
    a DSpark-5 config has to clear that bar on its target alone, so the draft is
    taken off the copy before the check rather than special-cased inside it.
    """
    speculative = config.speculative_config
    assert speculative.method == "dspark" and speculative.num_speculative_tokens == 5
    checked = copy.copy(config)
    checked.speculative_config = None
    validate_runtime_config(checked)


def run_scenarios(
    runner, tokenizer, report, output_path, *, ragged_probe=False, sampling_probe=False
):
    scheduler = Scheduler(runner.config, state_runtime=runner.state_runtime)
    seed = tokenizer.encode("Explain why water expands when it freezes.")
    interval = runner.config.state_checkpoint_interval_tokens
    manager = scheduler.block_manager
    assert manager.state_checkpoint_interval_tokens == interval > 0
    assert interval % manager.hash_block_size == 0
    # Both requests must cross a published checkpoint. A 129-token prompt
    # cannot restore a 256-token PAGE, even if the requested interval is 128:
    # BlockManager snaps that interval to zero and disables checkpointing.
    lengths = (interval + 1, interval + runner.config.long_prefill_token_threshold + 1)
    prompts = [(seed * -(-length // len(seed)))[:length] for length in lengths]
    builder = runner.attn_metadata_builder
    commit = builder.commit_speculative_state
    accepted = []
    last_commit = {}
    verify_shapes = []

    def observe_commit(metadata, last_indices):
        step = metadata.step
        pending = metadata.cache.pending
        commit(metadata, last_indices)
        if pending is not None:
            verify_shapes.append([span.length for span in step.requests])
            counts = (last_indices - step.cu_seqlens_q[:-1] + 1).tolist()
            expected = [
                span.position + count for span, count in zip(step.requests, counts)
            ]
            assert metadata.cache.cursor[step.slots.long(), 0].tolist() == expected
            accepted.extend(count - 1 for count in counts)
            last_commit.update(
                {span.request_id: count for span, count in zip(step.requests, counts)}
            )

    builder.commit_speculative_state = observe_commit

    def sequence(
        tokens,
        limit=24,
        *,
        stop=None,
        ignore_eos=True,
        temperature=0,
        top_k=-1,
        top_p=1.0,
    ):
        return Sequence(
            tokens,
            runner.block_size,
            SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=limit,
                ignore_eos=ignore_eos,
            ),
            has_per_req_cache=True,
            num_draft_tokens=runner.num_spec_tokens,
            stop_token_sequences=stop,
        )

    def episode(
        name,
        sequences,
        *,
        preempt=False,
        abort=False,
        reorder=False,
        after_verify=False,
    ):
        for seq in sequences:
            scheduler.add(seq)
        changed = False
        events = []
        report["active_case"] = {"name": name, "events": events}

        def snapshot(seq):
            return {
                "id": seq.id,
                "type": seq.type.name,
                **{
                    field: int(getattr(seq, field))
                    for field in (
                        "num_tokens",
                        "num_prompt_tokens",
                        "num_cached_tokens",
                        "num_rejected",
                        "num_bonus_tokens",
                        "num_placeholder_tokens",
                        "num_finalized_tokens",
                    )
                },
                "tail": list(seq.token_ids[-16:]),
            }

        def save_progress():
            if torch.distributed.get_rank() == 0:
                output_path.write_text(json.dumps(report, indent=2) + "\n")

        accept_start = len(accepted)
        for tick in range(160):
            if scheduler.is_finished():
                break
            item = scheduler.schedule()
            if item is None:
                assert scheduler.is_finished(), "scheduler stalled"
                break
            batch, active = item
            event = {
                "ids": list(batch.req_ids),
                "cached": [int(n) for n in batch.num_cached_tokens],
                "lengths": batch.num_scheduled_tokens.tolist(),
                "prefill_requests": batch.total_seqs_num_prefill,
                "restores": len(batch.state_maintenance_ops.checkpoint_restores),
                "stores": len(batch.state_maintenance_ops.checkpoint_stores),
                "context_lens": batch.context_lens.tolist(),
                "before": [snapshot(seq) for seq in sequences],
            }
            events.append(event)
            save_progress()
            last_commit.clear()
            output = runner.forward(batch)
            event["accepted_inputs"] = dict(last_commit)
            builder.cache.require_committed()
            finished = scheduler.postprocess(list(active.values()), output, batch=batch)
            runner.release_multimodal_requests([seq.id for seq in finished])
            event["sampled"] = {
                str(req): list(tokens)
                for req, tokens in zip(output.req_ids, output.token_ids)
            }
            event["after"] = [snapshot(seq) for seq in sequences]
            if reorder and (not ragged_probe or tick % 7 == 6):
                scheduler.running = deque(reversed(scheduler.running))
            first = sequences[0]
            if (
                not changed
                and first.status == SequenceStatus.RUNNING
                and first.num_completion_tokens >= 2
                and (not after_verify or first.id in last_commit)
            ):
                if preempt:
                    scheduler.block_manager.complete_previous_state_batch()
                    scheduler.running.remove(first)
                    assert scheduler.preempt(first)
                    changed = True
                    event["preempted"] = first.id
                    event["after_preempt"] = snapshot(first)
                elif abort:
                    first.status = SequenceStatus.ABORTED
                    changed = True
                    event["aborted"] = first.id
            save_progress()
        else:
            raise AssertionError("scheduler did not drain")
        scheduler.block_manager.complete_previous_state_batch()
        assert scheduler.is_finished()
        if preempt or abort:
            assert changed
        if abort:
            assert sequences[0].leave_reason == "aborted"
        outputs = [list(seq.completion_token_ids) for seq in sequences]
        rank_outputs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rank_outputs, outputs)
        assert all(value == outputs for value in rank_outputs)
        row = {
            "name": name,
            "events": events,
            "outputs": outputs,
            "leave_reasons": [seq.leave_reason for seq in sequences],
            "accepted_drafts": dict(Counter(accepted[accept_start:])),
        }
        report["cases"].append(row)
        report.pop("active_case", None)
        if torch.distributed.get_rank() == 0:
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        return row

    try:
        if sampling_probe:
            chinese = tokenizer.encode("请用中文解释为什么天空是蓝色的。")
            stochastic = episode(
                "mixed_stochastic_requests",
                [
                    sequence(chinese, limit=32, temperature=0.6, top_k=20, top_p=0.9),
                    sequence(prompts[0], limit=32, temperature=1.1),
                ],
                reorder=True,
            )
            assert all(len(tokens) == 32 for tokens in stochastic["outputs"])
            assert all(
                token >= 0 for tokens in stochastic["outputs"] for token in tokens
            )
        if ragged_probe:
            chinese = tokenizer.encode("请用中文解释为什么天空是蓝色的。")
            ragged = episode(
                "varying_verify_lengths",
                [sequence(chinese, limit=96), sequence(prompts[1], limit=96)],
                reorder=True,
            )
            assert all(len(tokens) == 96 for tokens in ragged["outputs"])
            report["verify_shapes"] = verify_shapes.copy()
            assert any(
                len(set(shape)) > 1 for shape in verify_shapes
            ), "The probe did not exercise unequal per-request query lengths"
            assert any(
                min(shape) < 6 for shape in verify_shapes
            ), "The probe stayed at full verification width"
        first = episode(
            "prefill_transition_preempt_reorder",
            [sequence(p) for p in prompts],
            preempt=True,
            reorder=True,
        )
        assert sum(event["restores"] for event in first["events"]) >= 1
        assert all(len(tokens) == 24 for tokens in first["outputs"])
        decoded = episode(
            "decode_preempt_reorder",
            [sequence(p) for p in prompts],
            preempt=True,
            reorder=True,
            after_verify=True,
        )
        assert all(len(tokens) == 24 for tokens in decoded["outputs"])
        assert any(
            event.get("preempted") in event["accepted_inputs"]
            for event in decoded["events"]
        )
        fork = episode(
            "exact_prefix_fork",
            [sequence(prompts[0]), sequence(prompts[0])],
            reorder=True,
        )
        assert sum(event["restores"] for event in fork["events"]) >= 2
        assert any(
            any(count > 0 for count in event["cached"]) for event in fork["events"]
        )
        assert all(len(tokens) == 24 for tokens in fork["outputs"])
        aborted = episode(
            "abort_and_survivor",
            [sequence(p) for p in prompts],
            abort=True,
            reorder=True,
        )
        assert len(aborted["outputs"][1]) == 24
        aborted_decode = episode(
            "abort_after_verify",
            [sequence(p) for p in prompts],
            abort=True,
            reorder=True,
            after_verify=True,
        )
        assert len(aborted_decode["outputs"][1]) == 24
        capped = episode("slot_reuse_and_output_cap", [sequence(prompts[0], limit=3)])
        assert len(capped["outputs"][0]) == 3
        assert capped["leave_reasons"][0] == "max_tokens"
        stop = fork["outputs"][0][2:4]
        stopped = episode(
            "stop_inside_accepted_block", [sequence(prompts[0], stop=[stop])]
        )
        assert stopped["leave_reasons"][0] == "stop_sequence"
        assert stopped["outputs"][0][-2:] == stop
        assert len(stopped["outputs"][0]) == 4
        old_eos = scheduler.eos_token_id
        try:
            # The target still produces real logits. Treat its known first
            # continuation as EOS to exercise termination within accepted rows.
            scheduler.eos_token_id = fork["outputs"][0][0]
            eos = episode(
                "eos_inside_accepted_block", [sequence(prompts[0], ignore_eos=False)]
            )
            assert eos["leave_reasons"][0] == "eos"
            assert eos["outputs"][0] == [scheduler.eos_token_id]
        finally:
            scheduler.eos_token_id = old_eos
        report["completed"] = True
    finally:
        builder.commit_speculative_state = commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--cache-dtype", choices=("bf16", "fp4"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-probe", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--calibration-profile")
    parser.add_argument("--production", action="store_true")
    parser.add_argument(
        "--ragged-probe",
        action="store_true",
        help="Exercise the real deferred length pipeline with a controlled schedule",
    )
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])

    def check_config(config):
        # This diagnostic exercises scheduling, not confidence calibration.
        from copy import copy

        checked = copy(config)
        checked.dspark = copy(config.dspark)
        checked.dspark.confidence_schedule = False
        diagnostic_config(checked)

    if args.calibration_profile and args.ragged_probe:
        parser.error("Use calibrated scheduling or a controlled probe separately")
    guard = (
        nullcontext()
        if args.production
        else patch(
            "atom.models.deepseek_v41.config.validate_runtime_config", check_config
        )
    )
    with guard:
        config = Config(
            model=args.model,
            tensor_parallel_size=size,
            enable_expert_parallel=True,
            enforce_eager=not args.graph,
            compilation_config=CompilationConfig(
                level=0,
                cudagraph_mode=CUDAGraphMode.FULL if args.graph else None,
                cudagraph_capture_sizes=[1, 2, 4],
            ),
            speculative_config=SpeculativeConfig(
                method="dspark", model=args.model, num_speculative_tokens=5
            ),
            dspark=DSparkConfig(
                confidence_schedule=args.ragged_probe or bool(args.calibration_profile),
                ragged=args.ragged_probe or bool(args.calibration_profile),
                disable_sps_calib=args.ragged_probe,
                calibration_profile=args.calibration_profile,
            ),
            kv_cache_dtype=args.cache_dtype,
            max_num_batched_tokens=256,
            max_model_len=512,
            max_num_seqs=4,
            long_prefill_token_threshold=128,
            state_checkpoint_interval_tokens=256,
            enable_log_stats=False,
            port=port,
        )
    config.parallel_config.data_parallel_base_port = port
    report = {
        "completed": False,
        "level": 0,
        "tp": size,
        "cache_dtype": args.cache_dtype,
        "graph": args.graph,
        "production_config": args.production,
        "calibration_profile": args.calibration_profile,
        "controlled_verify_schedule": args.ragged_probe,
        "stochastic_sampling_probe": args.sampling_probe,
        "cases": [],
    }
    runner = ModelRunner(rank, config)
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(1024)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(1024)
        if args.graph:
            runner.capture_cudagraph()
            report["target_graph_shapes"] = sorted(runner.graphs)
            assert report["target_graph_shapes"]
            from atom.utils import envs

            expected = runner.capture_sizes if envs.ATOM_DRAFT_CUDAGRAPH else []
            assert sorted(runner.drafter.block._cuda_graphs) == expected
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if args.ragged_probe:
            calls = 0

            def controlled_lengths(confidence):
                nonlocal calls
                # Each length travels through the normal async, request-keyed
                # scheduler. Hold short prefixes first to escape an all-accepted
                # block's anchor floor, then exercise contraction and expansion.
                phase = calls // 8
                calls += 1
                if phase == 0:
                    return torch.zeros_like(confidence[:, 0], dtype=torch.int64)
                return (
                    torch.arange(confidence.shape[0], device=confidence.device) + phase
                ) % (confidence.shape[1] + 1)

            runner.drafter.verify_scheduler.compute_ell = controlled_lengths
        run_scenarios(
            runner,
            tokenizer,
            report,
            args.output,
            ragged_probe=args.ragged_probe,
            sampling_probe=args.sampling_probe,
        )
        # No replay counter to check any more, and none to add: a FULL decode
        # step indexes `runner.graphs` and replays it, so a shape that was
        # never captured raises here rather than falling back to eager. The
        # scenarios above completing is that claim.
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
