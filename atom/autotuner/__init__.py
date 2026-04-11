"""
ROCm Autotuner — autonomous kernel & inference configuration tuning for AMD GPUs.

Inspired by NVIDIA AIConfigurator (offline perf modeling + config search) and
Karpathy's autoresearch (agent-driven experiment loop).  Designed to be
framework-agnostic: adapters exist for ATOM, vLLM, and SGLang.

Usage::

    # CLI (model-only, no GPU needed)
    python -m atom.autotuner.cli run --model gpt-oss-120b --system mi355x --total-gpus 8

    # CLI (real GPU benchmarks via ATOM)
    python -m atom.autotuner.cli run --model <hf_id> --system mi355x --adapter atom --eval-mode real_bench

    # Python API
    from atom.autotuner.agent.loop import AgentLoop, LoopConfig
    from atom.autotuner.database.estimator import ModelArch
    from atom.autotuner.types import GPUInfo

    loop = AgentLoop(
        model_arch=ModelArch.from_hf_config("gpt-oss-120b"),
        gpu_info=GPUInfo.mi355x(num_gpus=8),
        total_gpus=8,
        loop_config=LoopConfig(budget_sec=300),
        perf_model=perf_model,
    )
    results = loop.run()
"""

from atom.autotuner.types import (
    KernelType,
    QuantFormat,
    DatabaseMode,
    SearchStrategy,
    KernelConfig,
    KernelBenchResult,
    InferenceConfig,
    BenchmarkResult,
    Experiment,
    ParetoPoint,
    GPUInfo,
    TunerState,
)

__all__ = [
    "KernelType",
    "QuantFormat",
    "DatabaseMode",
    "SearchStrategy",
    "KernelConfig",
    "KernelBenchResult",
    "InferenceConfig",
    "BenchmarkResult",
    "Experiment",
    "ParetoPoint",
    "GPUInfo",
    "TunerState",
]

__version__ = "0.1.0"
