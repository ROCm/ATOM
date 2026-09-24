# SPDX-License-Identifier: MIT
"""Real scheduler/worker media leases through regroup, preemption, prefix and abort."""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from atom.config import CompilationConfig, Config, CUDAGraphMode
from atom.entrypoints.openai.chat_encoders import load_custom_message_encoder
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.models.deepseek_v41.image_processing import DeepseekV41ImageProcessor
from atom.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requests", type=int, choices=(2, 3), nargs="+", default=[2, 3]
    )
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
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
        kv_cache_dtype="fp4",
        index_cache_dtype="fp8",
        max_num_batched_tokens=max(256, args.chunk_size * max(args.requests)),
        max_model_len=2048,
        max_num_seqs=4,
        long_prefill_token_threshold=args.chunk_size,
        state_checkpoint_interval_tokens=256,
        enable_log_stats=False,
        port=port,
    )
    config.parallel_config.data_parallel_base_port = port
    runner = ModelRunner(rank, config)
    report = {
        "completed": False,
        "chunk_size": args.chunk_size,
        "tp": size,
        "cases": [],
        "level": 0,
        "graph": args.graph,
        "requests": args.requests,
    }
    try:
        runner.get_num_blocks()
        runner.pool_plan = runner.pool_plan.with_paged_entries(512)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(512)
        if args.graph:
            runner.capture_cudagraph()
        tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        processor = DeepseekV41ImageProcessor(
            config, tokenizer, load_custom_message_encoder(config.model)
        )
        images = [Image.new("RGB", (256, 256), c) for c in ("green", "blue")]
        parts = [
            {"type": "text", "text": "Image A:"},
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "Image B:"},
            {"type": "image", "image": images[1]},
            {
                "type": "text",
                "text": "Give the color of A, then B. Answer briefly in English.",
            },
        ]
        ids, data = processor.prepare(
            [{"role": "user", "content": parts}], images, {"thinking_mode": "chat"}
        )
        assert len(ids) > args.chunk_size
        scheduler = Scheduler(config, state_runtime=runner.state_runtime)

        def sequence():
            return Sequence(
                ids,
                runner.block_size,
                SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
                has_per_req_cache=True,
                multimodal_data=data,
            )

        def run(name, seqs, *, preempt=False, cancel=False, reorder=False):
            before = runner.vision_embeddings.encodes
            for seq in seqs:
                scheduler.add(seq)
            changed = False
            events = []
            peak_leases = 0
            for tick in range(100):
                item = scheduler.schedule()
                rejected = scheduler.take_rejected()
                runner.release_multimodal_requests([seq.id for seq in rejected])
                if item is None:
                    break
                batch, active = item
                event = {
                    "ids": list(batch.req_ids),
                    "lengths": batch.num_scheduled_tokens.tolist(),
                    "cached": list(batch.num_cached_tokens),
                    "restores": len(batch.state_maintenance_ops.checkpoint_restores),
                    "stores": len(batch.state_maintenance_ops.checkpoint_stores),
                    "pixels": [
                        i
                        for i, d in batch.multimodal_data.items()
                        if "pixel_values" in d
                    ],
                    "prefill_requests": batch.total_seqs_num_prefill,
                }
                for req, data_now in batch.multimodal_data.items():
                    assert "token_types" not in data_now
                    if req in runner.vision_embeddings.leases:
                        assert "pixel_values" not in data_now
                output = runner.forward(batch)
                peak_leases = max(peak_leases, len(runner.vision_embeddings.leases))
                finished = scheduler.postprocess(
                    list(active.values()), output, batch=batch
                )
                runner.release_multimodal_requests([seq.id for seq in finished])
                if reorder:
                    scheduler.running = deque(reversed(scheduler.running))
                first = seqs[0]
                if (
                    not changed
                    and first.is_partial_prefill
                    # Stores are issued at the start of the next chunk.
                    # Wait until that chunk executes before preempting.
                    and first.num_cached_tokens
                    > config.state_checkpoint_interval_tokens
                ):
                    if preempt:
                        scheduler.block_manager.complete_previous_state_batch()
                        scheduler.running.remove(first)
                        assert scheduler.preempt(first)
                        changed = True
                    elif cancel:
                        first.status = SequenceStatus.ABORTED
                        changed = True
                events.append(event)
            else:
                raise AssertionError("scheduler did not drain")
            assert scheduler.is_finished()
            scheduler.block_manager.complete_previous_state_batch()
            assert (
                not runner.vision_embeddings.entries
                and not runner.vision_embeddings.leases
            )
            if preempt or cancel:
                assert changed
            if cancel:
                assert seqs[0].leave_reason == "aborted"
            outputs = []
            for seq in seqs:
                tokens = list(seq.token_ids)[seq.num_prompt_tokens :]
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(tokens)
                outputs.append(text)
                if seq.leave_reason != "aborted":
                    assert (
                        0 <= text.lower().find("green") < text.lower().find("blue")
                    ), text
            row = {
                "name": name,
                "outputs": outputs,
                "events": events,
                "vision_encodes": runner.vision_embeddings.encodes - before,
                "peak_leases": peak_leases,
            }
            outputs_by_rank = [None] * size
            torch.distributed.all_gather_object(outputs_by_rank, outputs)
            assert all(value == outputs for value in outputs_by_rank)
            report["cases"].append(row)
            if rank == 0:
                print("LEASE", json.dumps(row), flush=True)
                args.output.write_text(json.dumps(report, indent=2))
            return row

        for request_count in args.requests:
            scheduler.block_manager._state_checkpoint_cache.clear_index()
            first = run(
                f"regroup_preempt_shared_{request_count}",
                [sequence() for _ in range(request_count)],
                preempt=True,
                reorder=True,
            )
            assert (
                first["vision_encodes"] == 2 and first["peak_leases"] == request_count
            )
            assert any(
                sum(e["lengths"]) >= request_count * args.chunk_size
                for e in first["events"]
                if e["prefill_requests"]
            )
            assert any(
                len(e["ids"]) == request_count
                for e in first["events"]
                if not e["prefill_requests"]
            )
            assert sum(e["restores"] for e in first["events"]) >= 1
            fork = run(
                f"prefix_fork_{request_count}",
                [sequence() for _ in range(request_count)],
                reorder=True,
            )
            assert any(any(p > 0 for p in e["cached"]) for e in fork["events"])
            assert sum(e["restores"] for e in fork["events"]) >= request_count
            scheduler.block_manager._state_checkpoint_cache.clear_index()
            aborted = run(
                f"abort_final_request_{request_count}", [sequence()], cancel=True
            )
            assert aborted["vision_encodes"] == 2
        report["completed"] = True
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2))
    finally:
        runner.exit()


if __name__ == "__main__":
    main()
