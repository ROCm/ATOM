# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 parameter storage for Megatron training (meta init, checkpoint quant, hooks)."""

from __future__ import annotations

import os

import torch

from atomrl.lumen.quantize.config import _get_float8_e4m3
from atomrl.lumen.quantize.descriptor import FP8Descriptor


def shrink_frozen_weights_to_fp8(model) -> None:
    """Tag frozen 2-D weights for FP8 storage.

    At this point the model might still be on meta device or already on CUDA.
    We tag the weights with metadata and install load/forward hooks.  If the
    weight is already materialized on CUDA, we also shrink it to a 1-element
    FP8 placeholder.  If on meta device, we just tag it — the patched
    materializer will create FP8-sized tensors later.
    """
    import torch
    from megatron.training import print_rank_0

    fp8_dtype = _get_float8_e4m3()
    count = 0
    for _name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, torch.nn.Parameter):
            continue
        if w.requires_grad:
            continue
        if w.ndim < 2:
            continue

        orig_shape = w.shape
        orig_dtype = w.dtype
        w._fp8_orig_shape = orig_shape
        w._fp8_original_dtype = orig_dtype
        w._fp8_dtype = fp8_dtype
        w._fp8_storage_enabled = True

        if str(w.device) != "meta":
            device = w.device
            tiny = torch.zeros(1, dtype=fp8_dtype, device=device)
            w.data = tiny

        wrap_load_from_state_dict(module, fp8_dtype)
        install_fp8_forward_hooks(module, fp8_dtype)
        count += 1

    print_rank_0(f"> FP8 param storage: tagged {count} frozen weights for FP8 storage")


def patch_meta_materializer() -> None:
    """Replace to_empty_if_meta_device with a version that materializes
    FP8-tagged parameters as tiny 1-element FP8 tensors (saving ~70GB).

    The trick: ``Module._apply`` iterates parameters in order.  We build
    a lookup of which Parameter objects are FP8-tagged, then inside the
    per-tensor callback we look up the enclosing Parameter via the module
    tree to decide whether to shrink it.

    We also directly patch the local name binding in the already-imported
    ``megatron.training.training`` module via ``sys.modules``.
    """
    import sys

    import megatron.training.utils as _mu
    import torch

    _orig_to_empty = _mu.to_empty_if_meta_device
    if getattr(_orig_to_empty, "_fp8_patched", False):
        return

    def _fp8_aware_to_empty(module, *, device, recurse=True):
        fp8_data_map = {}
        for _n, p in module.named_parameters(recurse=recurse):
            if getattr(p, "_fp8_storage_enabled", False):
                fp8_data_map[id(p)] = p

        orig_apply = torch.nn.Module._apply

        def _custom_apply(mod, fn, recurse_inner=True):
            for key, param in mod._parameters.items():
                if param is None:
                    continue
                if id(param) in fp8_data_map:
                    if param.data.device == torch.device("meta"):
                        fp8_dtype = _get_float8_e4m3()
                        new_data = torch.zeros(1, dtype=fp8_dtype, device=device)
                    else:
                        new_data = param.data.to(device)
                    param_out = torch.nn.Parameter(
                        new_data, requires_grad=param.requires_grad
                    )
                    for attr in (
                        "_fp8_storage_enabled",
                        "_fp8_orig_shape",
                        "_fp8_original_dtype",
                        "_fp8_dtype",
                        "_fp8_scale",
                    ):
                        if hasattr(param, attr):
                            setattr(param_out, attr, getattr(param, attr))
                    if (
                        getattr(param_out, "_fp8_scale", None) is not None
                        and getattr(param_out, "_fp8_dtype", None) is not None
                    ):
                        sc = param_out._fp8_scale
                        if torch.is_tensor(sc):
                            sc = sc.to(param_out.device)
                            param_out._fp8_scale = sc
                        param_out._fp8_desc = FP8Descriptor(
                            data=param_out.data,
                            scale=sc,
                            fp8_dtype=param_out._fp8_dtype,
                        )
                    mod._parameters[key] = param_out
                    fp8_data_map[id(param_out)] = param_out
                else:
                    with torch.no_grad():
                        new_data = fn(param.data)
                    if new_data is not param.data:
                        param_out = torch.nn.Parameter(
                            new_data, requires_grad=param.requires_grad
                        )
                        mod._parameters[key] = param_out
            for key, buf in mod._buffers.items():
                if buf is not None:
                    mod._buffers[key] = fn(buf)
            if recurse_inner:
                for child in mod.children():
                    _custom_apply(child, fn, recurse_inner)
            return mod

        def _empty_fn(tensor):
            if tensor.device == torch.device("meta"):
                return torch.empty_like(tensor, device=device)
            return tensor.to(device)

        _custom_apply(module, _empty_fn, recurse)
        torch.nn.Module._apply = orig_apply
        return module

    _fp8_aware_to_empty._fp8_patched = True
    _mu.to_empty_if_meta_device = _fp8_aware_to_empty
    training_mod = sys.modules.get("megatron.training.training")
    if training_mod is not None:
        training_mod.to_empty_if_meta_device = _fp8_aware_to_empty


def patch_float16_module() -> None:
    """Patch Float16Module.__init__ so .bfloat16() skips FP8-tagged params.

    Float16Module wraps the model via ``module.bfloat16()``, which casts
    every parameter to BF16.  For FP8-tagged weights (tiny placeholders),
    we collect them, let .bfloat16() run, then restore FP8 data and
    re-attach the custom attributes.
    """
    import torch
    from megatron.core.transformer.module import Float16Module

    _orig_init = Float16Module.__init__
    if getattr(_orig_init, "_fp8_patched", False):
        return

    def _fp8_safe_init(self, config, module):
        fp8_info = {}
        for name, mod in module.named_modules():
            for pname, p in mod._parameters.items():
                if p is not None and getattr(p, "_fp8_storage_enabled", False):
                    fp8_info[(name, pname)] = {
                        "data": p.data.clone(),
                        "_fp8_storage_enabled": True,
                        "_fp8_orig_shape": getattr(p, "_fp8_orig_shape", None),
                        "_fp8_original_dtype": getattr(p, "_fp8_original_dtype", None),
                        "_fp8_dtype": getattr(p, "_fp8_dtype", None),
                        "_fp8_scale": getattr(p, "_fp8_scale", None),
                    }

        _orig_init(self, config, module)

        inner = self.module if hasattr(self, "module") else module
        for (mod_name, pname), info in fp8_info.items():
            parts = mod_name.split(".") if mod_name else []
            target = inner
            for part in parts:
                target = getattr(target, part, target)
            p = target._parameters.get(pname)
            if p is not None:
                p.data = info["data"].to(p.device)
                for attr in (
                    "_fp8_storage_enabled",
                    "_fp8_orig_shape",
                    "_fp8_original_dtype",
                    "_fp8_dtype",
                    "_fp8_scale",
                ):
                    if info.get(attr) is not None:
                        setattr(p, attr, info[attr])
                if (
                    info.get("_fp8_scale") is not None
                    and info.get("_fp8_dtype") is not None
                ):
                    sc = p._fp8_scale
                    if torch.is_tensor(sc):
                        sc = sc.to(p.device)
                        p._fp8_scale = sc
                    p._fp8_desc = FP8Descriptor(
                        data=p.data, scale=sc, fp8_dtype=info["_fp8_dtype"]
                    )

    _fp8_safe_init._fp8_patched = True
    Float16Module.__init__ = _fp8_safe_init


def get_fp8_store_scaling() -> str:
    """FP8 scaling type for param storage (from Megatron args; default per-tensor)."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "linear_fp8_scaling", "delayed") or "delayed"
    except (AssertionError, ImportError, RuntimeError):
        return "delayed"


def fp8_store_quantize_weight(
    weight_bf16, fp8_dtype, scaling_type, block_size: int = 128
):
    """Quantize a frozen weight for FP8 param storage.

    Returns ``(fp8_data, scale, transpose)``:
      - ``blockwise2d``: ``scale`` is the 2D ``(ceil(N/bs), ceil(K/bs))`` dequant
        factor consumed directly by ``gemm_blockscale``; ``transpose`` is None
        (the blockscale kernel transposes internally).
      - otherwise: per-tensor scalar quant factor (``fp8_max/amax``) plus a
        precomputed transpose for the hipBLASLt per-tensor path.
    """
    import torch

    if scaling_type == "blockwise2d":
        from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight

        fp8_data, scale = _quant_blockwise2d_weight(weight_bf16, fp8_dtype, block_size)
        return fp8_data, scale, None

    amax = weight_bf16.float().abs().amax().clamp(min=1e-12)
    scale = torch.finfo(fp8_dtype).max / amax
    fp8_data = (weight_bf16.float() * scale).to(fp8_dtype)
    return fp8_data, scale, precompute_fp8_transpose(fp8_data)


def precompute_fp8_transpose(fp8_data: torch.Tensor) -> torch.Tensor | None:
    """Pre-compute the transposed layout for an FP8 weight tensor.

    Uses the fast Triton transpose when available, otherwise falls back
    to ``t().contiguous()``.  Called once at checkpoint load time so that
    ``FP8Descriptor.transpose_cached`` never needs to compute it lazily.

    When ``LUMEN_PREFER_HIPBLASLT=1`` we skip the allocation entirely.
    hipBLASLt's C++ kernel (``hipbsolgemm.cu``) detects non-contiguous
    strides from a metadata-only ``.t()`` view and applies ``HIPBLAS_OP_T``
    internally.  ``_gemm_per_tensor_hipblas`` passes ``w.t()`` (zero-cost
    view, no memory copy) directly to ``hipb_mm``.  Storing both NxK and
    KxN would add ~37 GiB on Llama2-70B, causing OOM.
    """
    import os

    if os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1":
        return None
    if fp8_data.dim() == 2 and fp8_data.is_cuda:
        try:
            from atomrl.lumen.ops.quantize.fast_transpose import fast_transpose_fp8

            return fast_transpose_fp8(fp8_data)
        except (ImportError, OSError, RuntimeError):
            pass
    return fp8_data.t().contiguous()


def patch_load_checkpoint_for_fp8() -> None:
    """Monkey-patch Megatron's load_checkpoint to convert weights to FP8 after loading.

    Also integrates LoRA base_layer key remapping and mmap loading, so
    external ``patch_checkpointing.py`` is no longer needed.
    """
    import sys

    import megatron.training.checkpointing as _ckpt
    from megatron.training import print_rank_0

    _original_load = _ckpt.load_checkpoint
    if getattr(_original_load, "_fp8_patched", False):
        return

    from atomrl.lumen.models.megatron_patches import (
        remap_lora_state_dict as _remap_lora_state_dict,
    )

    def _load_with_fp8(ddp_model, optimizer, opt_param_scheduler, **kwargs):
        import gc

        _orig_module_load_sd = torch.nn.Module.load_state_dict

        def _remap_load_state_dict(self_mod, state_dict, strict=True, **kw):
            state_dict = _remap_lora_state_dict(self_mod, state_dict)
            try:
                return _orig_module_load_sd(self_mod, state_dict, strict=strict, **kw)
            except Exception:
                if strict:
                    return _orig_module_load_sd(
                        self_mod, state_dict, strict=False, **kw
                    )
                raise

        torch.nn.Module.load_state_dict = _remap_load_state_dict
        try:
            result = _original_load(ddp_model, optimizer, opt_param_scheduler, **kwargs)
        finally:
            torch.nn.Module.load_state_dict = _orig_module_load_sd

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print_rank_0(f"> GPU memory right after ckpt load: {alloc:.2f}GB")

        targets = ddp_model if isinstance(ddp_model, list) else [ddp_model]
        fp8_dtype = _get_float8_e4m3()
        converted = 0
        freed_bytes = 0
        already_fp8 = 0
        already_fp8_no_desc = 0

        for m in targets:
            unwrapped = m
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            for _name, mod in unwrapped.named_modules():
                w = getattr(mod, "weight", None)
                if w is None or w.requires_grad or w.dim() != 2:
                    continue
                if w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                    torch.float8_e5m2,
                    torch.float8_e5m2fnuz,
                ):
                    already_fp8 += 1
                    if not hasattr(w, "_fp8_desc"):
                        already_fp8_no_desc += 1
                        if already_fp8_no_desc <= 5:
                            print_rank_0(
                                f"  [FP8 BUG] {_name}.weight: FP8 but NO _fp8_desc! "
                                f"shape={tuple(w.shape)} fp8_amax={w.data.float().abs().amax():.4f}"
                            )
                        amax = w.data.float().abs().amax().clamp(min=1e-12)
                        w._fp8_scale = (torch.finfo(fp8_dtype).max / amax).to(w.device)
                        w._fp8_scale_reciprocal = (1.0 / w._fp8_scale).to(w.device)
                        fp8_data_t = precompute_fp8_transpose(w.data)
                        w._fp8_desc = FP8Descriptor(
                            data=w.data,
                            scale=w._fp8_scale,
                            fp8_dtype=fp8_dtype,
                            _transpose=fp8_data_t,
                        )
                        w._fp8_orig_shape = w.shape
                        w._fp8_original_dtype = torch.bfloat16
                        w._fp8_storage_enabled = True
                        install_fp8_forward_hooks(mod, fp8_dtype)
                    continue
                if w.dtype == torch.bfloat16:
                    old_bytes = w.numel() * w.element_size()
                    _scaling = get_fp8_store_scaling()
                    fp8_data, scale, fp8_data_t = fp8_store_quantize_weight(
                        w.data, fp8_dtype, _scaling
                    )
                    w.data = fp8_data
                    w._fp8_scale = scale.to(w.device)
                    if _scaling not in ("blockwise", "blockwise2d"):
                        w._fp8_scale_reciprocal = (1.0 / scale).to(w.device)
                    w._fp8_desc = FP8Descriptor(
                        data=w.data,
                        scale=w._fp8_scale,
                        fp8_dtype=fp8_dtype,
                        _transpose=fp8_data_t,
                    )
                    w._fp8_orig_shape = fp8_data.shape
                    w._fp8_original_dtype = torch.bfloat16
                    w._fp8_storage_enabled = True
                    freed_bytes += (
                        old_bytes - fp8_data.numel() * fp8_data.element_size()
                    )
                    converted += 1
                    if not getattr(mod, "_fp8_hooks_installed", False):
                        install_fp8_forward_hooks(mod, fp8_dtype)
                        mod._fp8_hooks_installed = True

        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print_rank_0(f"> GPU memory after FP8 conversion: {alloc:.2f}GB")

        print_rank_0(
            f"> FP8 param storage: {converted} BF16 weights converted to FP8, "
            f"{already_fp8} already FP8 ({already_fp8_no_desc} had NO _fp8_desc!), "
            f"freed {freed_bytes/(1024**3):.1f}GB"
        )
        if already_fp8_no_desc > 0:
            print_rank_0(
                f"  *** WARNING: {already_fp8_no_desc} FP8 weights lost _fp8_desc "
                f"and got WRONG scale (fp8_max/fp8_amax ≈ 1.0 instead of correct value)! ***"
            )
        return result

    _load_with_fp8._fp8_patched = True
    _ckpt.load_checkpoint = _load_with_fp8
    training_mod = sys.modules.get("megatron.training.training")
    if training_mod is not None:
        training_mod.load_checkpoint = _load_with_fp8


def wrap_load_from_state_dict(module, fp8_dtype):
    """Override _load_from_state_dict to quantize 'weight' on the fly."""
    import torch

    original_load = module._load_from_state_dict

    _fp8_hook_call_count = [0]

    def _fp8_load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        _fp8_hook_call_count[0] += 1
        if _fp8_hook_call_count[0] <= 3:
            import torch.distributed as _dist

            rank = _dist.get_rank() if _dist.is_initialized() else 0
            if rank == 0:
                has_key = weight_key in state_dict
                has_attr = (
                    hasattr(module.weight, "_fp8_orig_shape")
                    if hasattr(module, "weight")
                    else False
                )
                print(
                    f"[FP8 HOOK] prefix={prefix!r} key={weight_key!r} "
                    f"found={has_key} has_fp8_shape={has_attr} "
                    f"w.dtype={module.weight.dtype if hasattr(module, 'weight') else 'N/A'} "
                    f"w.shape={tuple(module.weight.shape) if hasattr(module, 'weight') else 'N/A'}",
                    flush=True,
                )

        if weight_key in state_dict:
            w = module.weight
            if hasattr(w, "_fp8_orig_shape"):
                incoming = state_dict[weight_key]
                if isinstance(incoming, torch.Tensor):
                    device = (
                        w.device if str(w.device) != "meta" else torch.device("cuda")
                    )
                    _scaling = get_fp8_store_scaling()
                    fp8_w, scale, fp8_w_t = fp8_store_quantize_weight(
                        incoming.to(device), fp8_dtype, _scaling
                    )
                    w.data = fp8_w
                    w._fp8_scale = scale.to(device)
                    if _scaling not in ("blockwise", "blockwise2d"):
                        w._fp8_scale_reciprocal = (1.0 / scale).to(device)
                    w._fp8_desc = FP8Descriptor(
                        data=w.data,
                        scale=w._fp8_scale,
                        fp8_dtype=fp8_dtype,
                        _transpose=fp8_w_t,
                    )
                    del state_dict[weight_key]

                    remaining = {
                        k: v
                        for k, v in state_dict.items()
                        if k.startswith(prefix) and k != weight_key
                    }
                    if remaining:
                        original_load(
                            state_dict,
                            prefix,
                            local_metadata,
                            False,
                            missing_keys,
                            unexpected_keys,
                            error_msgs,
                        )
                    return

        original_load(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    module._load_from_state_dict = _fp8_load_from_state_dict


def install_fp8_forward_hooks(module, fp8_dtype):
    """No-op per-linear hooks — layer-level hooks are installed by
    _install_layer_level_fp8_hooks instead, to handle FP8 wrappers
    that bypass individual module forward()."""


def install_embedding_output_fp8_hooks(model):
    """Install dequant hooks on embedding and output_layer only.

    Transformer layer linears are handled by ``FP8StoredLinearFunction``
    inside ``_do_gemm`` — no hooks needed there.  But embedding (index
    lookup) and Megatron's output_layer (``ColumnParallelLinear``) don't
    go through ``_do_gemm``, so they still need pre/post hooks.  These
    are only two weights (~500 MB BF16 each), so the temporary is small.
    """
    import torch
    from megatron.training import print_rank_0

    embed_mod = None
    output_mod = None
    for name, mod in model.named_modules():
        if "word_embeddings" in name and hasattr(mod, "weight"):
            w = mod.weight
            if hasattr(w, "_fp8_desc"):
                embed_mod = mod
        if name.endswith("output_layer") and hasattr(mod, "weight"):
            w = mod.weight
            if hasattr(w, "_fp8_desc"):
                output_mod = mod

    count = 0
    for mod in [embed_mod, output_mod]:
        if mod is None:
            continue

        def _pre(m, inputs, _mod=mod):
            w = _mod.weight
            if hasattr(w, "_fp8_desc"):
                orig_dtype = getattr(w, "_fp8_original_dtype", torch.bfloat16)
                _mod._fp8_emb_saved = w.data
                _sc = w._fp8_desc.scale
                if _sc.numel() > 1:
                    # blockwise2d 2D block scale (dequant factor) -> expand+multiply
                    from atomrl.lumen.ops.quantize.gemm_primitives import (
                        _dequant_fp8_weight,
                    )

                    w.data = _dequant_fp8_weight(w.data, _sc, 128).to(orig_dtype)
                else:
                    w.data = (w.data.to(torch.float32) / _sc).to(orig_dtype)

        def _post(m, inputs, output, _mod=mod):
            if hasattr(_mod, "_fp8_emb_saved"):
                _mod.weight.data = _mod._fp8_emb_saved
                del _mod._fp8_emb_saved

        mod.register_forward_pre_hook(_pre)
        mod.register_forward_hook(_post)
        count += 1

    print_rank_0(
        f"> FP8 param storage: installed embedding/output dequant hooks "
        f"on {count} modules"
    )


# Backward-compatible private aliases (legacy imports from atomrl.lumen.models.megatron).
_shrink_frozen_weights_to_fp8 = shrink_frozen_weights_to_fp8
_patch_meta_materializer = patch_meta_materializer
_patch_float16_module = patch_float16_module
_get_fp8_store_scaling = get_fp8_store_scaling
_fp8_store_quantize_weight = fp8_store_quantize_weight
_precompute_fp8_transpose = precompute_fp8_transpose
_patch_load_checkpoint_for_fp8 = patch_load_checkpoint_for_fp8
_wrap_load_from_state_dict = wrap_load_from_state_dict
_install_fp8_forward_hooks = install_fp8_forward_hooks
_install_embedding_output_fp8_hooks = install_embedding_output_fp8_hooks


def _find_scaling_manager(model):
    for module in model.modules():
        sm = getattr(module, "_quant_manager", None)
        if sm is not None:
            return sm
    return None


def register_fp8_param_optimizer_hook(model, optimizer):
    """Register optimizer post-step hook for FP8 param staleness marking."""
    from megatron.training import print_rank_0

    sm = _find_scaling_manager(model)
    if sm is not None and sm.num_fp8_params > 0:
        sm.register_fp8_optimizer_hook(optimizer)
        print_rank_0("> FP8 param optimizer hook registered")


def prepare_hipblaslt_for_fp8_storage(train_args) -> None:
    """Pre-allocate hipBLASLt workspace when hybrid FP8 / env requests it."""
    from megatron.training import print_rank_0

    fmt = (
        getattr(train_args, "lumen_fp8_format", "")
        or getattr(train_args, "fp8", "")
        or getattr(train_args, "linear_fp8_format", "")
    )
    want_hipblaslt = (
        fmt == "hybrid" or os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1"
    )
    if not want_hipblaslt:
        return
    os.environ["LUMEN_PREFER_HIPBLASLT"] = "1"
    try:
        import atomrl.lumen.ops.quantize.linear as qlinear

        qlinear._PREFER_HIPBLASLT = True
        qlinear.ensure_hipblaslt_ready()
        reason = (
            "LUMEN_PREFER_HIPBLASLT"
            if os.environ.get("LUMEN_PREFER_HIPBLASLT") == "1"
            else "hybrid FP8 backward"
        )
        print_rank_0(f"> hipBLASLt workspace pre-allocated for {reason}")
    except RuntimeError as e:
        print_rank_0(f"> WARNING: hipBLASLt pre-init failed: {e}")
