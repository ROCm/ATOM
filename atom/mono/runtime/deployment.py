# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether a deployment can run a mono decode at all: the refusals every model
shares (one TP group on one node, decode steps in the model's own graph), which
a model's own list extends."""

from atom.config import CompilationLevel
from atom.mono.plan.shard import shard_refusal
from atom.plugin.prepare import is_sglang, is_vllm
from atom.utils import resolve_obj_by_qualname


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


def _takes_weight_updates(atom_config) -> bool:
    """Whether this deployment's model runner can be handed a rollout weight
    sync (``WeightUpdaterMixin``: ``RLHFModelRunner``, not the base runner).

    A sync rewrites the model's parameter buffers IN PLACE -- deliberately, so
    a captured decode graph keeps its addresses (`weight_updater`'s
    `_await_readers_of`). Mono does not read every weight in place: GLM-5.2
    requantizes the attention projections at bind time and K3 dequantizes the
    latent MoE, and both cache the result for the life of the runner
    (`glm52/mono/weights.py`'s `layer_weights`, kept in `_weights` /
    `_prepared`). After a sync those copies are the OLD weights, nothing
    changes address, and no shape or dtype check can notice -- the server goes
    on answering, wrongly.

    Invalidating the copies from `weight_updater` is the real fix. Until that
    lands with numerics behind it, the two features are kept apart so they can
    never silently combine.

    Resolving the qualname is free in a live deployment: `engine_core` has
    already imported that class to construct the runner. A name that will not
    resolve is treated as updatable, because this refusal only costs speed
    while guessing wrong costs correctness.
    """
    qualname = getattr(atom_config, "runner_qualname", None)
    if not qualname:
        return False
    try:
        return hasattr(resolve_obj_by_qualname(qualname), "update_weights")
    except Exception:  # noqa: BLE001 - an unresolvable runner is not provably safe
        return True


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
        (
            not _takes_weight_updates(atom_config),
            "rollout weight updates (mono caches copies a sync cannot reach)",
        ),
        # `model_runner.py` wraps the model -- and the drafter, which K3 mono
        # needs -- in `torch.compile(fullgraph=True)` at this level and only
        # this level (levels 2 and 3 go through the decorator instead, and
        # `fullgraph` appears nowhere else). Under `fullgraph` a graph break
        # raises, and every mono path calls `Tensor.data_ptr()` -- GLM's
        # contract checks and buffer aliasing, K3's kernel argument packing.
        # Unconditional, not folded into the piecewise check below: that one
        # is opt-out for GLM-5.2, while this applies to every mono model.
        # Read like `_requires_piecewise_compilation`: an absent
        # `compilation_config` is a real production state (SGLang builds one
        # without it), and it means no `torch.compile` wrapper at all, so
        # absent reads as "not this level" rather than as a refusal.
        (
            getattr(getattr(atom_config, "compilation_config", None), "level", None)
            != CompilationLevel.DYNAMO_AS_IS,
            "DYNAMO_AS_IS compilation (fullgraph breaks on data_ptr)",
        ),
    ]
    if piecewise_cudagraph_required:
        checks.append(
            (not _requires_piecewise_compilation(atom_config), "piecewise cudagraph")
        )
    return checks


def first_refusal(checks) -> str | None:
    """The reason of the first check that does not hold, or None."""
    return next((why for ok, why in checks if not ok), None)
