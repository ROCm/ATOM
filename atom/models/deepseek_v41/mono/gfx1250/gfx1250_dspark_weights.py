# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bind native gfx1250 parameters to the frozen gfx950 decode layout.

Native/prefill keeps its row-interleaved, padded weights. The reference runner
owns a second, immutable decode binding, prepared once before graph capture.
Only the routed MXFP4 tensors need translation: tile16 GUGU, 128-K padding and
the original shuffle_scale order. All other tensors alias loaded parameters.
"""

from dataclasses import dataclass

import torch

from atom.models.deepseek_v41.mono.weights import (
    E8M0, FP4X2, LayerWeights, expected,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.models.deepseek_v41.mono.kernels.dims import Dims


@dataclass(frozen=True)
class ReferenceExpertWeights:
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.__dict__.values())


def _tensor(tensor, name, shape, dtypes, device=None):
    if (
        tensor is None
        or tuple(tensor.shape) != shape
        or tensor.dtype not in dtypes
        or not tensor.is_contiguous()
        or (device is not None and tensor.device != device)
    ):
        actual = None if tensor is None else (tuple(tensor.shape), tensor.dtype, tensor.device)
        raise MonoUnsupported(f"reference binding {name}: {actual}, expected contiguous {shape} {dtypes}")
    return tensor


def _unshuffle_weight(weight, rows, packed_k):
    """Inverse of native shuffle_weight(layout=(16,16)), one expert only."""
    return (
        weight.view(torch.uint8)
        .reshape(rows // 16, packed_k // 32, 2, 16, 16)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(rows, packed_k)
    )


def _unshuffle_scale(scale, rows, cols):
    """Inverse of native n32k4, including the actual padded K stride."""
    return (
        scale.view(torch.uint8)
        .reshape(rows // 32, cols // 4, 32, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(rows, cols)
    )


def _tile16_gu(rows, inter):
    """[g0,u0,g1,u1,...] -> [g0..g15,u0..u15,...], dropping padded rows."""
    return (
        rows.reshape(-1, 2, rows.shape[-1])[:inter]
        .reshape(inter // 16, 16, 2, rows.shape[-1])
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(2 * inter, rows.shape[-1])
    )


@torch.no_grad()
def reference_expert_weights(w13, w13_scale, w2, w2_scale, *, inter=1152, inter_real=None):
    """Translate the native FlyDSL A8W4 binding, without changing its storage.

    Validate every input before allocating or dispatching. Conversion proceeds
    one expert at a time, so temporary unshuffled buffers never grow with the
    expert count. Down's 36 live scale columns retain 40 stored columns: the
    mother's full-tile scale loads require the extra four, initialized to 1.
    """
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    if w13.ndim != 3 or w2.ndim != 3:
        raise MonoUnsupported("reference experts require native 3-D weight views")
    inter_real = inter if inter_real is None else inter_real
    experts, rows, packed_hidden = w13.shape
    hidden, source_inter = 2 * packed_hidden, rows // 2
    if (
        experts < 1 or rows % 2 or hidden % 256 or source_inter % 128
        or inter < 128 or inter % 128 or inter > source_inter
        or inter_real < 32 or inter_real % 32 or inter_real > inter
    ):
        raise MonoUnsupported("reference experts have unsupported MXFP4 padding/tile geometry")
    dev = w13.device
    weight_dtypes, scale_dtypes = (FP4X2, torch.uint8), (E8M0, torch.uint8)
    _tensor(w13, "w13", (experts, 2 * source_inter, hidden // 2), weight_dtypes)
    _tensor(w2, "w2", (experts, hidden, source_inter // 2), weight_dtypes, dev)
    _tensor(w13_scale, "w13_scale", (experts, 2 * source_inter // 32, hidden), scale_dtypes, dev)
    _tensor(w2_scale, "w2_scale", (experts, hidden // 32, source_inter), scale_dtypes, dev)
    if dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise MonoUnsupported("prepare reference weights before graph capture")

    scale_cols = -(-(inter // 32) // 8) * 8
    out13 = torch.empty((experts, 2 * inter, hidden // 2), dtype=torch.uint8, device=dev)
    out13s = torch.empty((experts, 2 * inter, hidden // 32), dtype=torch.uint8, device=dev)
    out2 = torch.empty((experts, hidden, inter // 2), dtype=torch.uint8, device=dev)
    out2s = torch.empty((experts, hidden, scale_cols), dtype=torch.uint8, device=dev)
    for e in range(experts):
        gu = _tile16_gu(_unshuffle_weight(w13[e], 2 * source_inter, hidden // 2), inter)
        if inter_real != inter:
            gu[2 * inter_real:].zero_()
        out13[e].copy_(shuffle_weight(gu))
        del gu
        gs = _tile16_gu(_unshuffle_scale(w13_scale[e], 2 * source_inter, hidden // 32), inter)
        if inter_real != inter:
            gs[2 * inter_real:].fill_(127)
        out13s[e].copy_(shuffle_scale(gs))
        del gs
        down = _unshuffle_weight(w2[e], hidden, source_inter // 2)[:, :inter // 2].contiguous()
        if inter_real != inter:
            down[:, inter_real // 2:].zero_()
        out2[e].copy_(shuffle_weight(down))
        del down
        ds = torch.full((hidden, scale_cols), 127, dtype=torch.uint8, device=dev)
        ds[:, :inter_real // 32].copy_(_unshuffle_scale(w2_scale[e], hidden, source_inter // 32)[:, :inter_real // 32])
        out2s[e].copy_(shuffle_scale(ds))
        del ds
    return ReferenceExpertWeights(
        out13.view(FP4X2), out13s.view(E8M0), out2.view(FP4X2), out2s.view(E8M0),
    )


def reference_expert_bytes(config, tp):
    """Persistent extra bytes per layer; used before the model-wide bind."""
    inter = -(-(config.moe_intermediate_size // tp) // 128) * 128
    hidden, experts = config.hidden_size, config.n_routed_experts
    cols = -(-(inter // 32) // 8) * 8
    return experts * (3 * hidden * inter // 2 + 2 * inter * hidden // 32 + hidden * cols)


def bind_layer(block, config, dims, moe_tp=1):
    assert dims.tp == 4 and dims.ep == moe_tp == 1
    tp = dims.tp
    """TP2/TP4 c1 reference binding over the fully loaded native layer.

    The returned LayerWeights owns decode copies; block parameters continue to
    serve native/prefill. The runner retains this object through every replay.
    """
    spec, method = block.attn.spec, block.ffn.experts.quant_method
    if tp != 4 or (config.hidden_size, config.moe_intermediate_size) != (5120, 2304) or config.n_routed_experts not in (128, 384):
        raise MonoUnsupported("reference binding currently supports V4.1 TP2/TP4 c1 geometry")
    if (
        not getattr(method, "is_gfx1250", False)
        or not getattr(method, "is_guinterleave", False)
        or not getattr(method, "fp8_activations", False)
        or getattr(getattr(method, "moe", None), "use_ep", True)
        or any(getattr(method, key, False) for key in ("use_triton", "use_triton_ep", "use_triton_decode"))
        or block.ffn.experts.num_fused_shared_experts != 0
    ):
        raise MonoUnsupported(f"layer {spec.layer_id}: reference binding requires native FlyDSL TP A8W4 GUGU")
    params = dict(block.named_parameters())
    w13 = params.get("ffn.experts.w13_weight")
    if w13 is None:
        raise MonoUnsupported(f"layer {spec.layer_id}: missing routed weight")
    dev = w13.device
    for name, (shape, dtype) in expected(config, spec.mode, dims, moe_tp).items():
        if name.startswith("ffn.experts."):
            continue
        # E8M0 codes may be exposed as uint8 by native loading. The pointer ABI
        # is the same; reinterpretation neither requantizes nor copies them.
        dtypes = (dtype, torch.uint8) if dtype == E8M0 else (dtype,)
        tensor = _tensor(params.get(name), f"layer {spec.layer_id} {name}", shape, dtypes, dev)
        params[name] = tensor.view(E8M0) if dtype == E8M0 else tensor
    prefix = "ffn.experts."
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    routed = [params.get(prefix + name) for name in names]
    if any(t is None for t in routed):
        raise MonoUnsupported(f"layer {spec.layer_id}: missing routed parameter")
    if routed[0].dtype != FP4X2 or routed[2].dtype != FP4X2:
        raise MonoUnsupported(f"layer {spec.layer_id}: native routed weights must be MXFP4")
    inter = config.moe_intermediate_size // tp
    source_inter = getattr(method, "intermediate_size", 0)
    _tensor(routed[0], "native w13", (config.n_routed_experts, 2 * source_inter, 2560), (FP4X2,), dev)
    _tensor(routed[2], "native w2", (config.n_routed_experts, 5120, source_inter // 2), (FP4X2,), dev)
    translated = reference_expert_weights(*routed, inter=-(-inter // 128) * 128, inter_real=inter)
    params.update({prefix + name: getattr(translated, name) for name in names})
    assert translated.nbytes == reference_expert_bytes(config, tp)
    return LayerWeights(spec.layer_id, spec.mode, params)


def bind_model(model, dims, moe_tp=1):
    assert dims.tp == 4 and dims.ep == moe_tp == 1
    tp = dims.tp
    """Prepare every layer once, outside graph capture and decode timing."""
    if model.layers:
        source = dict(model.layers[0].named_parameters())["ffn.experts.w13_weight"]
        if source.device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("prepare reference model before graph capture")
            extra = len(model.layers) * reference_expert_bytes(model.config, tp)
            # Include one expert's temporary raw/shuffled buffers. Completed
            # layers retain only the four decode tensors, never native copies.
            temporary = 8 * source[0].numel() * source.element_size()
            free, _ = torch.cuda.mem_get_info(source.device)
            if free < extra + temporary:
                raise MonoUnsupported(f"reference decode weights need {extra + temporary} free bytes, have {free}")
    return [bind_layer(block, model.config, dims, moe_tp) for block in model.layers]
