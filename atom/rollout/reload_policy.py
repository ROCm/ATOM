# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whether a model's weights can be reloaded in place, decided before any
sync writes, and how a full-precision weight is requantised into the FP8
formats a reload can rewrite."""

import torch


def _requantise_per_1x128(module, weight: torch.Tensor, fp8_dtype):
    """A true 128x128 block scale of shape (N//128, K//128), the layout the
    load-time online_quantize_weight builds and the blockscale GEMM reads.
    ``scale_dtype`` is decided here: casting an FP32 scale to E8M0 afterwards
    does not round it up to a power of two, and the weight would have been
    quantised against the unrounded scale."""
    from atom.quantization.quark.utils import (
        quantize_weight_to_fp8_128x128_blockscale,
    )

    return quantize_weight_to_fp8_128x128_blockscale(
        weight,
        fp8_dtype,
        # Online quant rebuilds its own (maybe E8M0) scale; a checkpoint's
        # scale stays FP32.
        scale_dtype=(
            module.weight_scale.dtype
            if getattr(module.quant_config, "online_quant", False)
            else torch.float32
        ),
    )


def _requantise_per_tensor(module, weight: torch.Tensor, fp8_dtype):
    scale = (weight.abs().max() / torch.finfo(fp8_dtype).max).clamp(min=1e-12)
    return (weight / scale).to(fp8_dtype), scale


def _requantise_per_token(module, weight: torch.Tensor, fp8_dtype):
    row_amax = weight.abs().amax(dim=-1, keepdim=True)
    scale = (row_amax / torch.finfo(fp8_dtype).max).clamp(min=1e-12)
    return (weight / scale).to(fp8_dtype), scale


# How a full-precision weight is requantised into an FP8 quant type, by the
# type's name, so that asking which parameters a reload can rewrite needs no
# aiter. A type missing here is one a reload refuses.
FP8_REQUANTISERS = {
    "per_1x128": _requantise_per_1x128,
    "per_Tensor": _requantise_per_tensor,
    "per_Token": _requantise_per_token,
}


def is_expert_parallel(module: torch.nn.Module) -> bool:
    return getattr(module, "expert_map", None) is not None or bool(
        getattr(module, "num_redundant_experts", 0)
    )


class ReloadPolicyMixin:
    """What a weight updater asks before a sync's first write. Its host, a
    ModelRunner, gives it the model (``_sync_target_model``), ``config`` and
    ``attn_family``; ``WeightUpdaterMixin`` gives it ``_is_fp8_param``."""

    def _unreloadable_params(self) -> list[str]:
        """Parameters a full-precision write cannot rewrite in place.

        A floating-point parameter of two bytes or more is written as sent,
        and an FP8 one is requantised if ``FP8_REQUANTISERS`` has its quant
        type. Anything else -- FP4 or MXFP4 packed bytes, integer quantisation,
        FP8 of another scheme -- is cast into garbage where the shapes agree.
        A module that writes its weights itself (``sync_weight``) knows their
        format, and an int64 parameter is a constant, an index or a hash: no
        quantisation stores weights in it.
        """
        unreloadable = []
        for prefix, module in self._sync_target_model().named_modules():
            if callable(getattr(module, "sync_weight", None)):
                continue
            quant_type = getattr(getattr(module, "quant_type", None), "name", None)
            for name, param in module.named_parameters(recurse=False):
                if (
                    name.endswith("input_scale")
                    or param.dtype == torch.int64
                    or (param.dtype.is_floating_point and param.element_size() >= 2)
                ):
                    continue
                if self._is_fp8_param(module, param) and quant_type in FP8_REQUANTISERS:
                    continue
                unreloadable.append(f"{prefix}.{name}" if prefix else name)
        return unreloadable

    def _reload_refusal(self) -> str | None:
        """Why this model's weights cannot be reloaded in place, if they cannot.

        Asked before any sync writes, whatever carries it.
        """
        config = getattr(self, "config", None)
        pcp = int(getattr(config, "prefill_context_parallel_size", 1) or 1)
        if pcp > 1:
            return (
                f"prefill_context_parallel_size={pcp}: a worker's rank counts "
                f"every prefill-context rank, but the updater slices a full "
                f"tensor by it as by a TP rank, so some workers would load "
                f"another rank's shard"
            )
        model = self._sync_target_model()
        # ModelRunner's, declared by the config it was built from.
        family = getattr(self, "attn_family", None)
        if family is not None and family.is_mla:
            return (
                "the model attends through MLA, whose decode reads W_K and W_V "
                "derived from kv_b_proj once, at load; a reload rewrites "
                "kv_b_proj and leaves them as they were"
            )
        expert_parallel = [
            prefix
            for prefix, module in model.named_modules()
            if is_expert_parallel(module)
        ]
        if expert_parallel:
            return (
                f"{len(expert_parallel)} MoE layer(s), {expert_parallel[0]!r} "
                f"among them, are expert-parallel or carry redundant expert "
                f"replicas: incoming expert ids then address a rank's local "
                f"slots through expert_map, and the replicas are filled after "
                f"loading, neither of which the updater tracks"
            )
        held = self._unreloadable_params()
        if held:
            return (
                f"{len(held)} parameter(s), {held[:3]} among them, are held in "
                f"a format a full-precision write cannot produce: FP4 or MXFP4 "
                f"packed bytes, integer quantisation, or FP8 of a scheme the "
                f"updater does not requantise"
            )
        return None
