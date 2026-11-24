# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""TRAINING-phase Megatron hooks (setup_model_and_optimizer / evaluate)."""

from __future__ import annotations

import os

from atomrl.lumen.patches.registry import PatchPhase, register_patch


def install_fp8_param_gather_hook() -> None:
    """Install the canonical Megatron optimizer hook for FP8 param gather."""
    import megatron.training.training as _mt_training
    from megatron.training import get_args

    from atomrl.lumen.models.fp8_param_storage import register_fp8_param_optimizer_hook

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_fp8_param_gather_hook", False):
        return

    def _setup_with_fp8_hook(*args, **kwargs):
        model, optimizer, scheduler = current_setup(*args, **kwargs)
        train_args = get_args()
        _weight_quant_once = os.environ.get("LUMEN_WEIGHT_QUANT_ONCE", "0") == "1"
        if (
            getattr(train_args, "lumen_fp8_param_gather", False) or _weight_quant_once
        ) and model:
            target = model[0] if isinstance(model, list) else model
            register_fp8_param_optimizer_hook(target, optimizer)
        return model, optimizer, scheduler

    _setup_with_fp8_hook._lumen_fp8_param_gather_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_fp8_hook


def install_mxfp4_weight_cache_hook() -> None:
    """Install the Megatron optimizer hook that invalidates MXFP4 weight caches.

    The MXFP4 linear path caches each layer's quantized weight so that
    gradient-accumulation micro-batches (and gradient-checkpoint recomputes)
    reuse it instead of re-deriving an identical FP4 tensor.  The cache is
    keyed on nothing but the module, so it has to be dropped once the BF16
    master weight changes — otherwise the run silently trains against the
    step-0 weights.
    """
    import megatron.training.training as _mt_training
    from megatron.training import get_args, print_rank_0

    from atomrl.lumen.models.megatron import resolve_quant_format

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_mxfp4_weight_cache_hook", False):
        return

    def _setup_with_mxfp4_hook(*args, **kwargs):
        model, optimizer, scheduler = current_setup(*args, **kwargs)
        train_args = get_args()

        if (
            getattr(train_args, "linear_fp4", False)
            and resolve_quant_format(train_args) == "mxfp4"
            and os.environ.get("LUMEN_MXFP4_DISABLE_WEIGHT_CACHE") != "1"
            and model
        ):
            from atomrl.lumen.quantize import register_mxfp4_weight_optimizer_hooks

            register_mxfp4_weight_optimizer_hooks(model, optimizer)
            print_rank_0("> MXFP4 weight cache enabled (invalidated on optimizer step)")

        return model, optimizer, scheduler

    _setup_with_mxfp4_hook._lumen_mxfp4_weight_cache_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_mxfp4_hook


def install_gc_freeze_hook(warmup_steps: int = 20) -> None:
    """Take the process's permanent Python objects out of the collector's reach.

    A step allocates enough short-lived Python objects to reach a generation-2
    collection every few steps, and that collection walks *every* tracked object
    in the process: the imported modules, the Triton and AITER kernel caches,
    every parameter and every autograd node. Here it measures ~1.2 s, landing on
    whichever step it happens to fall in — the run's step time is fine at the
    median and has a tail of steps that take twice as long.

    Almost all of what it walks is alive for the whole run. ``gc.freeze()`` moves
    the objects that exist when it is called into a permanent generation the
    collector never visits, so later collections scan only the step's own
    garbage. It runs after ``warmup_steps`` so that the kernel caches, which fill
    in on the shapes' first call, are frozen too.

    Objects created after the freeze are still collected normally, but anything
    alive at that moment is never revisited, so a reference cycle formed during
    warmup is retained for the rest of the run.

    Set ``LUMEN_GC_FREEZE=0`` to keep stock collector behaviour.
    """
    if os.environ.get("LUMEN_GC_FREEZE", "1") == "0":
        return

    import gc

    import megatron.training.training as _mt_training
    from megatron.training import print_rank_0

    current_train_step = _mt_training.train_step
    if getattr(current_train_step, "_lumen_gc_freeze_hook", False):
        return

    state = {"calls": 0}

    def _train_step_with_gc_freeze(*args, **kwargs):
        out = current_train_step(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == warmup_steps:
            gc.collect()
            gc.freeze()
            print_rank_0(
                f"> GC: gc.freeze() after {warmup_steps} steps "
                f"(freeze_count={gc.get_freeze_count()}; later collections skip frozen objects)"
            )
            # Only unwrap ourselves. Restoring the snapshot unconditionally would
            # drop any wrapper installed on top of this one after we did.
            if _mt_training.train_step is _train_step_with_gc_freeze:
                _mt_training.train_step = current_train_step
        return out

    _train_step_with_gc_freeze._lumen_gc_freeze_hook = True
    _mt_training.train_step = _train_step_with_gc_freeze


def install_val_loss_early_stop_hook() -> None:
    """Stop training when reduced validation loss reaches ``val_loss_target``."""
    try:
        import megatron.training.training as _train_mod
        from megatron.training import get_args, print_rank_0
    except ImportError:
        return
    if getattr(_train_mod, "_lumen_val_early_stop_patched", False):
        return

    _orig_evaluate = _train_mod.evaluate

    def _evaluate_with_early_stop(*a, **kw):
        result = _orig_evaluate(*a, **kw)
        try:
            args = get_args()
            target = getattr(args, "val_loss_target", None)
            if target is not None and result and isinstance(result[0], dict):
                v = result[0].get("lm loss")
                if v is not None:
                    val = v.item() if hasattr(v, "item") else float(v)
                    if val <= float(target):
                        cur = getattr(args, "iteration", None)
                        if cur:
                            args.train_iters = cur
                        print_rank_0(
                            f"> [Early Stop] validation loss ({val:.4f}) <= "
                            f"target ({float(target):.4f}) -> stopping at iter {cur}."
                        )
        except (AttributeError, RuntimeError) as e:  # never let the hook break eval
            print_rank_0(f"> [Early Stop] hook skipped ({e})")
        return result

    _train_mod.evaluate = _evaluate_with_early_stop
    _train_mod._lumen_val_early_stop_patched = True


def install_fp8_param_storage_hook() -> None:
    """Hook training setup for FP8 parameter storage (meta init + FP8 placeholders)."""
    import megatron.training.training as _mt_training
    from megatron.training import get_args, print_rank_0

    from atomrl.lumen.models.fp8_param_storage import (
        install_embedding_output_fp8_hooks,
        patch_float16_module,
        patch_load_checkpoint_for_fp8,
        patch_meta_materializer,
        prepare_hipblaslt_for_fp8_storage,
    )

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_fp8_param_storage_hook", False):
        return

    def _setup_with_fp8_storage(*a, **kw):
        train_args = get_args()
        if not getattr(train_args, "fp8_param_storage", False):
            return current_setup(*a, **kw)

        prepare_hipblaslt_for_fp8_storage(train_args)
        train_args.init_model_with_meta_device = True
        print_rank_0("> FP8 param storage: forcing init_model_with_meta_device=True")
        patch_meta_materializer()
        patch_float16_module()
        patch_load_checkpoint_for_fp8()
        model, optimizer, scheduler = current_setup(*a, **kw)

        targets = model if isinstance(model, list) else [model]
        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            install_embedding_output_fp8_hooks(unwrapped)

        print_rank_0(
            "> FP8 param storage: linear layers handled inline by "
            "FP8StoredLinearFunction (no per-layer forward hooks needed)"
        )

        return model, optimizer, scheduler

    _setup_with_fp8_storage._lumen_fp8_param_storage_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_fp8_storage


def install_hip_graphs_hook() -> None:
    """Hook setup_model_and_optimizer for lazy HIP graph capture on transformer layers."""
    import megatron.training.training as _mt_training
    from megatron.training import get_args, print_rank_0

    current_setup = _mt_training.setup_model_and_optimizer
    if getattr(current_setup, "_lumen_hip_graphs_hook", False):
        return

    def _setup_with_hip_graphs(*args, **kwargs):
        model, optimizer, scheduler = current_setup(*args, **kwargs)
        train_args = get_args()

        if not getattr(train_args, "lumen_hip_graphs", False):
            return model, optimizer, scheduler

        if not model:
            return model, optimizer, scheduler

        from atomrl.lumen.utils.hip_graphs import install_lazy_graph_capture

        warmup_steps = getattr(train_args, "warmup_steps", 5)
        num_warmup = max(warmup_steps, 3)

        recompute_num = 0
        if (
            getattr(train_args, "recompute_granularity", None) == "full"
            and getattr(train_args, "recompute_method", None) == "block"
        ):
            recompute_num = getattr(train_args, "recompute_num_layers", 0)

        targets = model if isinstance(model, list) else [model]
        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            max_graphed = int(os.environ.get("LUMEN_HIP_GRAPHS_MAX_LAYERS", "10"))
            count = install_lazy_graph_capture(
                unwrapped,
                num_warmup=num_warmup,
                skip_recomputed_layers=recompute_num,
                max_graphed_layers=max_graphed,
            )
            if count > 0:
                print_rank_0(
                    f"> HIP graphs: installed lazy capture on {count} "
                    f"transformer layers (skipped {recompute_num} recomputed, "
                    f"capture after step {num_warmup})"
                )

        return model, optimizer, scheduler

    _setup_with_hip_graphs._lumen_hip_graphs_hook = True
    _mt_training.setup_model_and_optimizer = _setup_with_hip_graphs


register_patch(
    "fp8_param_gather_hook",
    PatchPhase.TRAINING,
    description="FP8 param gather optimizer post-step hook on setup_model_and_optimizer",
    tags=frozenset({"fp8", "training", "megatron"}),
    default=False,
)(install_fp8_param_gather_hook)

register_patch(
    "mxfp4_weight_cache_hook",
    PatchPhase.TRAINING,
    description="Invalidate the per-module MXFP4 weight cache on each optimizer step",
    tags=frozenset({"mxfp4", "training", "megatron"}),
    default=False,
)(install_mxfp4_weight_cache_hook)

register_patch(
    "gc_freeze_hook",
    PatchPhase.TRAINING,
    description="gc.freeze() the long-lived objects after warmup to cut gen-2 pauses",
    tags=frozenset({"training", "megatron"}),
    default=False,
)(install_gc_freeze_hook)

register_patch(
    "fp8_param_storage_hook",
    PatchPhase.TRAINING,
    description="FP8 param storage via meta init and checkpoint-time quantization",
    tags=frozenset({"fp8", "training", "megatron"}),
    depends_on=("fp8_param_gather_hook",),
    default=False,
)(install_fp8_param_storage_hook)

register_patch(
    "hip_graphs_hook",
    PatchPhase.TRAINING,
    description="Lazy HIP graph capture on transformer layers after model setup",
    tags=frozenset({"training", "megatron"}),
    depends_on=("fp8_param_gather_hook", "fp8_param_storage_hook"),
    default=False,
)(install_hip_graphs_hook)

register_patch(
    "val_loss_early_stop_hook",
    PatchPhase.TRAINING,
    description="Collective early stop when reduced validation loss hits target",
    tags=frozenset({"training", "megatron", "eval"}),
    default=False,
)(install_val_loss_early_stop_hook)
