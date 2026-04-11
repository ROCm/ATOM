"""
CLI entry point for the ROCm Autotuner.

Usage::

    # Full autonomous tuning (model-only estimation, no GPU required)
    python -m atom.autotuner.cli run --model meta-llama/Llama-3.1-70B \\
        --system mi355x --total-gpus 8 --budget 600

    # With real GPU benchmarks via ATOM
    python -m atom.autotuner.cli run --model meta-llama/Llama-3.1-70B \\
        --system mi355x --total-gpus 8 --adapter atom --eval-mode real_bench

    # Collect kernel benchmark data
    python -m atom.autotuner.cli collect --system mi355x --kernels gemm,attention

    # Resume from checkpoint
    python -m atom.autotuner.cli run --resume autotuner_results/latest_checkpoint.json

    # Use with vLLM
    python -m atom.autotuner.cli run --model meta-llama/Llama-3.1-70B \\
        --adapter vllm --total-gpus 8 --eval-mode real_bench
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("atom.autotuner")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rocm-autotuner",
        description="Autonomous kernel & inference configuration tuning for AMD GPUs",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run_p = sub.add_parser("run", help="Run the autonomous tuning loop")
    run_p.add_argument("--model", required=True, help="HuggingFace model ID or path")
    run_p.add_argument("--system", default="mi355x", choices=["mi355x", "mi325x", "mi300x", "auto"])
    run_p.add_argument("--total-gpus", type=int, default=8)
    run_p.add_argument("--budget", type=int, default=600, help="Time budget in seconds")
    run_p.add_argument("--max-experiments", type=int, default=500)
    run_p.add_argument("--adapter", default="none", choices=["none", "atom", "vllm", "sglang"])
    run_p.add_argument("--eval-mode", default="model_only", choices=["model_only", "real_bench", "hybrid_eval"])
    run_p.add_argument("--strategy", default="agent_guided", choices=["grid", "bayesian", "agent_guided"])
    run_p.add_argument("--isl", type=int, default=4000, help="Input sequence length")
    run_p.add_argument("--osl", type=int, default=1000, help="Output sequence length")
    run_p.add_argument("--ttft", type=float, default=None, help="TTFT SLA limit (ms)")
    run_p.add_argument("--tpot", type=float, default=None, help="TPOT SLA limit (ms)")
    run_p.add_argument("--output-dir", default="autotuner_results", help="Output directory")
    run_p.add_argument("--resume", default=None, help="Resume from checkpoint file")
    run_p.add_argument("--db-mode", default="hybrid", choices=["silicon", "hybrid", "empirical", "sol"])

    # ---- collect ----
    col_p = sub.add_parser("collect", help="Collect kernel benchmark data")
    col_p.add_argument("--system", default="auto")
    col_p.add_argument("--kernels", default="gemm,attention,moe,communication")
    col_p.add_argument("--output", default="data/benchmarks")
    col_p.add_argument("--warmup", type=int, default=10)
    col_p.add_argument("--iters", type=int, default=100)

    # ---- report ----
    rep_p = sub.add_parser("report", help="Generate report from previous run")
    rep_p.add_argument("--input-dir", required=True)
    rep_p.add_argument("--format", default="text", choices=["text", "csv", "json"])

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "collect":
        return _cmd_collect(args)
    if args.command == "report":
        return _cmd_report(args)

    return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the autonomous tuning loop."""
    from atom.autotuner.types import DatabaseMode, GPUInfo
    from atom.autotuner.database.storage import PerfStorage
    from atom.autotuner.database.perf_model import PerformanceModel
    from atom.autotuner.database.estimator import ModelArch
    from atom.autotuner.agent.loop import AgentLoop, EvalMode, LoopConfig

    gpu_info = _resolve_gpu(args.system, args.total_gpus)
    model_arch = ModelArch.from_hf_config(args.model)

    output_dir = Path(args.output_dir)
    db_path = output_dir / "perf.db"
    storage = PerfStorage(db_path)

    db_mode = DatabaseMode(args.db_mode)
    perf_model = PerformanceModel(storage, args.system, gpu_info, db_mode)

    real_bench_fn = None
    if args.adapter != "none":
        adapter = _build_adapter(args.adapter)
        real_bench_fn = lambda config: adapter.run_full(config)

    loop_config = LoopConfig(
        budget_sec=args.budget,
        max_experiments=args.max_experiments,
        eval_mode=EvalMode(args.eval_mode),
        strategy=args.strategy,
        ttft_limit_ms=args.ttft,
        tpot_limit_ms=args.tpot,
        log_dir=output_dir,
    )

    loop = AgentLoop(
        model_arch=model_arch,
        gpu_info=gpu_info,
        total_gpus=args.total_gpus,
        loop_config=loop_config,
        perf_model=perf_model,
        real_bench_fn=real_bench_fn,
    )

    print(f"\n{'='*80}")
    print(f"  ROCm Autotuner")
    print(f"  Model:    {args.model}")
    print(f"  System:   {args.system} × {args.total_gpus} GPUs")
    print(f"  Strategy: {args.strategy}")
    print(f"  Eval:     {args.eval_mode}")
    print(f"  Budget:   {args.budget}s ({args.max_experiments} max experiments)")
    print(f"  ISL/OSL:  {args.isl}/{args.osl}")
    if args.ttft:
        print(f"  TTFT SLA: {args.ttft}ms")
    if args.tpot:
        print(f"  TPOT SLA: {args.tpot}ms")
    print(f"{'='*80}\n")

    start = time.time()
    tracker = loop.run()
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    storage.close()
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    """Collect kernel benchmark data."""
    from atom.autotuner.types import GPUInfo
    from atom.autotuner.database.storage import PerfStorage
    from atom.autotuner.collector import (
        GEMMCollector,
        AttentionCollector,
        MoECollector,
        CommunicationCollector,
        GPUStateManager,
    )

    gpu_info = _resolve_gpu(args.system, 1)
    output_dir = Path(args.output)
    db_path = output_dir / "perf.db"
    storage = PerfStorage(db_path)

    kernels = args.kernels.split(",")
    gpu_mgr = GPUStateManager()

    with gpu_mgr.pinned():
        for kernel in kernels:
            kernel = kernel.strip()
            collector = {
                "gemm": lambda: GEMMCollector(gpu_info, warmup_iters=args.warmup, bench_iters=args.iters),
                "attention": lambda: AttentionCollector(gpu_info, warmup_iters=args.warmup, bench_iters=args.iters),
                "moe": lambda: MoECollector(gpu_info, warmup_iters=args.warmup, bench_iters=args.iters),
                "communication": lambda: CommunicationCollector(gpu_info, warmup_iters=args.warmup, bench_iters=args.iters),
            }.get(kernel)

            if collector is None:
                logger.warning("Unknown kernel type: %s", kernel)
                continue

            c = collector()
            results = c.collect_all()
            storage.insert_batch(args.system, results)
            c.save_results(results, output_dir / f"{kernel}_results.jsonl")

    storage.close()
    print(f"Collection complete. Data saved to {output_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate report from a previous autotuner run."""
    from atom.autotuner.agent.experiment import ExperimentTracker

    tracker = ExperimentTracker(Path(args.input_dir))
    loaded = tracker.load_checkpoint()
    if not loaded:
        print("No checkpoint found in", args.input_dir)
        return 1

    print(tracker.format_summary())
    return 0


def _resolve_gpu(system: str, num_gpus: int):
    from atom.autotuner.types import GPUInfo

    if system == "auto":
        from atom.autotuner.utils.gpu import ROCmGPU
        return ROCmGPU.detect()

    factory = {
        "mi355x": GPUInfo.mi355x,
        "mi325x": GPUInfo.mi325x,
        "mi300x": GPUInfo.mi300x,
    }.get(system, GPUInfo.mi300x)
    return factory(num_gpus)


def _build_adapter(name: str):
    if name == "atom":
        from atom.autotuner.adapters.atom_adapter import ATOMAdapter
        return ATOMAdapter()
    if name == "vllm":
        from atom.autotuner.adapters.vllm_adapter import VLLMAdapter
        return VLLMAdapter()
    if name == "sglang":
        from atom.autotuner.adapters.sglang_adapter import SGLangAdapter
        return SGLangAdapter()
    raise ValueError(f"Unknown adapter: {name}")


if __name__ == "__main__":
    sys.exit(main())
