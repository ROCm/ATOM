# SPDX-License-Identifier: MIT
"""TP4 DSpark scheduler validation, profiling and optional arithmetic diagnostics."""

import argparse
import copy
import hashlib
import json
import os
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
from atom.models.deepseek_v41.config import validate_runtime_config
from tests.attentions.deepseek_v41.benchmark_runtime import run_case


def diagnostic_config(config):
    speculative = config.speculative_config
    assert speculative.method == "dspark" and speculative.num_speculative_tokens == 5
    checked = copy.copy(config)
    checked.speculative_config = None
    validate_runtime_config(checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/mnt/DeepSeek-V4.1-Flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--calibration-profile")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-persistent", action="store_true")
    parser.add_argument(
        "--trace-layers", type=int, help="Number of initial layers to trace"
    )
    parser.add_argument(
        "--trace-batched",
        action="store_true",
        help="Replay one target token per active request together",
    )
    parser.add_argument(
        "--quality-config",
        action="store_true",
        help="Match the frozen lm_eval scheduler and pool capacities",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--torch-profiler-dir",
        type=str,
        help="ModelRunner trace; exclude profiled runs from throughput comparisons",
    )
    parser.add_argument("--collect-confidence", action="store_true")
    parser.add_argument("--prefill-trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inspect-position", type=int)
    parser.add_argument("--cases", default="0,1,2")
    parser.add_argument(
        "--prompts", type=Path, help="JSON workload with tokenized batches"
    )
    parser.add_argument("--pad-single-row", action="store_true")
    parser.add_argument("--ordered-reductions", action="store_true")
    parser.add_argument("--fixed-attention-reduction", action="store_true")
    parser.add_argument(
        "--serial-verify-ops", nargs="*", choices=("engram", "attention"), default=[]
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--cache-dtype", choices=("bf16", "fp4"), default="bf16")
    args = parser.parse_args()
    if args.trace_layers is not None and args.trace_layers < 1:
        parser.error("--trace-layers must be positive")
    if args.trace and args.graph and args.inspect_position is not None:
        parser.error("Layer tracing requires eager execution so Python hooks run")
    if args.trace_persistent and (
        not args.trace or not args.trace_batched or args.baseline
    ):
        parser.error("--trace-persistent requires batched speculative tracing")
    if args.trace_batched and not args.trace:
        parser.error("--trace-batched requires --trace")
    if args.repeats < 1 or args.output_tokens < 2:
        parser.error("Use at least one repeat and two output tokens")
    if args.production and (
        args.pad_single_row
        or args.ordered_reductions
        or args.fixed_attention_reduction
        or args.serial_verify_ops
    ):
        parser.error("Production validation cannot patch model arithmetic")
    if args.profile and args.trace:
        parser.error("Profiling and serial oracle tracing must run separately")
    if args.collect_confidence and (args.baseline or args.profile or args.trace):
        parser.error("Confidence collection requires a separate speculative run")
    if args.prefill_trace and (not args.baseline or args.profile):
        parser.error("Prefill tracing requires an unprofiled baseline")
    if args.pad_single_row:
        from tests.models.deepseek_v41.dspark_projection_probe import (
            pad_single_row_projections,
        )

        pad_single_row_projections()
    if args.fixed_attention_reduction:
        from tests.models.deepseek_v41.dspark_projection_probe import (
            fixed_attention_reduction,
        )

        fixed_attention_reduction()
    if args.serial_verify_ops:
        from tests.models.deepseek_v41.dspark_projection_probe import (
            serial_verify_operations,
        )

        serial_verify_operations(args.serial_verify_ops)
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    port = int(os.environ["MASTER_PORT"])
    speculative = (
        None
        if args.baseline
        else SpeculativeConfig(
            method="dspark", model=args.model, num_speculative_tokens=5
        )
    )
    guard = (
        nullcontext()
        if args.production
        else patch(
            "atom.models.deepseek_v41.config.validate_runtime_config",
            validate_runtime_config if args.baseline else diagnostic_config,
        )
    )
    with guard:
        config = Config(
            model=args.model,
            tensor_parallel_size=size,
            enable_expert_parallel=True,
            enforce_eager=not args.graph,
            compilation_config=CompilationConfig(
                cudagraph_mode=CUDAGraphMode.PIECEWISE if args.graph else None,
                cudagraph_capture_sizes=[1, 2, 4],
            ),
            speculative_config=speculative,
            dspark=DSparkConfig(
                confidence_schedule=bool(args.calibration_profile),
                ragged=bool(args.calibration_profile),
                calibration_profile=args.calibration_profile,
            ),
            torch_profiler_dir=args.torch_profiler_dir,
            kv_cache_dtype=args.cache_dtype,
            index_cache_dtype=args.cache_dtype,
            max_num_batched_tokens=512 if args.quality_config else 256,
            max_model_len=4096 if args.quality_config else 512,
            max_num_seqs=4,
            long_prefill_token_threshold=128,
            state_checkpoint_interval_tokens=128,
            enable_log_stats=False,
            port=port,
        )
    config.parallel_config.data_parallel_base_port = port
    report = {
        "completed": False,
        "baseline": args.baseline,
        "graph": args.graph,
        "torch_profiler_dir": args.torch_profiler_dir,
        "tp": size,
        "runtime_guard_bypassed_for_diagnostic": not args.baseline
        and not args.production,
        "calibration_profile": args.calibration_profile,
        "cache_dtype": args.cache_dtype,
        "single_row_padding_diagnostic": args.pad_single_row,
        "ordered_reductions_diagnostic": args.ordered_reductions,
        "fixed_attention_reduction_diagnostic": args.fixed_attention_reduction,
        "serial_verify_operations_diagnostic": args.serial_verify_ops,
        "cases": [],
        "quality_config": args.quality_config,
        "trace_batched": args.trace_batched,
        "trace_persistent": args.trace_persistent,
    }
    runner = ModelRunner(rank, config)
    if args.ordered_reductions:
        from tests.models.deepseek_v41.dspark_projection_probe import ordered_reductions

        ordered_reductions()
    try:
        runner.get_num_blocks()
        pool_blocks = 2048 if args.quality_config else 1024
        runner.pool_plan = runner.pool_plan.with_paged_entries(pool_blocks)
        config.pool_entries = dict(runner.pool_plan.entries)
        runner.allocate_kv_cache(pool_blocks)
        if args.graph:
            runner.capture_cudagraph()
            report["target_graph_tokens"] = sorted(runner._piecewise_captured_tokens)
            # Stages-per-layer, which is the number a capture built on the wrong
            # step kind gets wrong: `decode_ffn` is only offered to the
            # execution policy on a decode step, and a speculative bucket is
            # only a decode step because the capture asks for one.
            dense = runner.model.dense_graphs
            report["dense_graphs_per_layer"] = (
                0
                if dense is None
                else len(dense.entries) / config.hf_config.num_hidden_layers
            )
            report["draft_graph_sizes"] = (
                [] if args.baseline else sorted(runner.drafter.block._cuda_graphs)
            )
            assert report["target_graph_tokens"]
            if not args.baseline:
                from atom.utils import envs

                expected = runner.capture_sizes if envs.ATOM_DRAFT_CUDAGRAPH else []
                assert report["draft_graph_sizes"] == expected
        stats = None
        if args.profile:
            from tests.models.deepseek_v41.dspark_runtime_stats import RuntimeStats

            stats = RuntimeStats(runner)
        trace = None
        confidence_trace = None
        if args.collect_confidence:
            from tests.models.deepseek_v41.dspark_confidence_trace import (
                ConfidenceTrace,
            )

            confidence_trace = ConfidenceTrace(runner)
        if args.trace:
            from tests.models.deepseek_v41.dspark_runtime_trace import VerifyTrace

            trace = VerifyTrace(
                runner,
                inspect_position=args.inspect_position,
                dump_directory=args.output.parent,
                replay_batch=args.trace_batched,
                trace_layers=args.trace_layers,
                persistent_shadow=args.trace_persistent,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        seed = tokenizer.encode("Explain why water expands when it freezes.")
        chinese = tokenizer.encode("请用中文解释为什么天空是蓝色的。")
        prefill_trace = None
        if args.prefill_trace:
            from tests.models.deepseek_v41.dspark_prefill_trace import PrefillTrace

            prefill_trace = PrefillTrace(runner, chinese, args.output.parent)
        prompts = [
            [seed],
            [(seed * 30)[:129]],
            [chinese, (seed * 30)[:257]],
            [chinese],
            [chinese, chinese],
        ]
        if args.prompts is not None:
            payload = args.prompts.read_bytes()
            workload = json.loads(payload)
            prompts = [row["input_ids"] for row in workload["batches"]]
            if any(
                not batch
                or len(batch) > config.max_num_seqs
                or any(
                    not tokens
                    or len(tokens) + args.output_tokens > config.max_model_len
                    or any(
                        type(token) is not int
                        or not 0 <= token < config.hf_config.vocab_size
                        for token in tokens
                    )
                    for tokens in batch
                )
                for batch in prompts
            ):
                raise ValueError("Prompt workload exceeds the diagnostic capacity")
            report["workload"] = {
                "path": str(args.prompts),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "metadata": workload.get("metadata", {}),
                "batches": [
                    {key: value for key, value in row.items() if key != "input_ids"}
                    for row in workload["batches"]
                ],
            }
        selected_cases = (
            set(range(len(prompts)))
            if args.cases == "all"
            else {int(value) for value in args.cases.split(",")}
        )
        if args.trace_persistent and (len(selected_cases) != 1 or args.repeats != 1):
            raise ValueError("Persistent shadow supports one case and one repetition")
        report["state_bytes"] = runner.attn_metadata_builder.geometry.state_bytes
        report["physical_window"] = runner.attn_metadata_builder.geometry.ring_slots
        if args.torch_profiler_dir:
            runner.start_profiler("v41_runtime")
        for case_id, inputs in enumerate(prompts):
            if case_id not in selected_cases:
                continue
            if args.profile:
                run_case(runner, inputs, args.output_tokens)
                stats.take()  # exclude warmup
            for repeat in range(args.repeats):
                if prefill_trace is not None:
                    prefill_trace.case = case_id
                    prefill_trace.repeat = repeat
                row = run_case(runner, inputs, args.output_tokens)
                row.update(
                    case=case_id,
                    repeat=repeat,
                    input_lengths=[len(tokens) for tokens in inputs],
                )
                if stats is not None:
                    row["dspark_stats"] = stats.take()
                if confidence_trace is not None:
                    row["confidence_observations"] = confidence_trace.take()
                ranks = [None] * size
                torch.distributed.all_gather_object(ranks, row)
                assert all(value["outputs"] == ranks[0]["outputs"] for value in ranks)
                for metric in (
                    "ttft_ms",
                    "tpot_ms",
                    "elapsed_seconds",
                    "peak_allocated_gib",
                ):
                    row[metric] = max(value[metric] for value in ranks)
                row["output_tokens_per_second"] = (
                    len(inputs) * args.output_tokens / row["elapsed_seconds"]
                )
                if stats is not None:
                    row["rank_stats"] = [value["dspark_stats"] for value in ranks]
                report["cases"].append(row)
                if trace is not None:
                    report["verify_trace"] = trace.records
                if prefill_trace is not None:
                    report["prefill_trace"] = prefill_trace.records
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.torch_profiler_dir:
            report["torch_profiler_traces"] = runner.stop_profiler()
        report["completed"] = True
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if args.torch_profiler_dir and runner.profiler is not None:
            runner.stop_profiler()
        runner.exit()


if __name__ == "__main__":
    main()
