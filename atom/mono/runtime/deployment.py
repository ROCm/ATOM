# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether a deployment can run a mono decode at all: the refusals every model
shares (one TP group on one node, decode steps in the model's own graph), which
a model's own list extends."""

from atom.mono.plan.shard import shard_refusal
from atom.plugin.prepare import is_sglang, is_vllm


def _requires_piecewise_compilation(atom_config) -> bool:
    """``atom_config.compilation_config.cudagraph_mode``'s own answer, or False
    when either is absent. That absence is the real state at ``--level`` 0/1/2
    (``CompilationConfig`` is still built, but nothing sets a piecewise
    ``cudagraph_mode`` at those levels) and under SGLang, which builds its own
    ``atom_config`` without a ``compilation_config`` at all (see
    ``atom/plugin/sglang/minimax_m3_bridge.py``) -- neither one requires
    piecewise compilation, so treating the missing attribute as "not
    piecewise" is correct, not just a fallback for incomplete test fixtures."""
    cudagraph_mode = getattr(
        getattr(atom_config, "compilation_config", None), "cudagraph_mode", None
    )
    return bool(cudagraph_mode) and cudagraph_mode.requires_piecewise_compilation()


def common_refusals(
    atom_config, dims, supported_tp, *, piecewise_cudagraph_required: bool = True
) -> list[tuple[bool, str]]:
    """``(holds, why not)`` of the conditions every mono model needs: ``dims``
    (the model's ``Dims``) shards at this TP, which is one of ``supported_tp``;
    no expert, data, pipeline or context parallelism, no TBO, not a plugin.

    ``piecewise_cudagraph_required`` gates one more check: no piecewise
    cudagraph. MiniMax-M3 and DeepSeek-V4.1 mono wraps the *compiled* inner
    model, so a piecewise-compiled graph of that module runs underneath mono
    and the two conflict -- this refusal is correct for them, and is the
    default. GLM-5.2 mono instead routes inside the *uncompiled* outer
    ``GlmMoeDsaForCausalLM`` and bypasses the compiled inner
    ``DeepseekV2Model`` entirely, so piecewise compilation of that module
    never runs and cannot conflict with mono; GLM-5.2 passes
    ``piecewise_cudagraph_required=False`` to opt out of this one check while
    keeping every other shared refusal."""
    tp = atom_config.tensor_parallel_size
    shard = shard_refusal(dims, tp)
    checks = [
        (shard is None, shard),
        (tp in supported_tp, f"TP {tp} not in {supported_tp}"),
        (not atom_config.enable_expert_parallel, "expert parallel"),
        (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        (atom_config.pipeline_parallel_size == 1, "PP > 1"),
        (atom_config.decode_context_parallel_size == 1, "decode CP"),
        (atom_config.prefill_context_parallel_size == 1, "prefill CP"),
        (not (atom_config.enable_tbo or atom_config.enable_tbo_decode), "TBO"),
        (not (is_vllm() or is_sglang()), "plugin mode"),
    ]
    if piecewise_cudagraph_required:
        checks.append(
            (not _requires_piecewise_compilation(atom_config), "piecewise cudagraph")
        )
    return checks


def first_refusal(checks) -> str | None:
    """The reason of the first check that does not hold, or None."""
    return next((why for ok, why in checks if not ok), None)
