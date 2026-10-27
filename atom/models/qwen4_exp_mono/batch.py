# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1/TP8 ordinary batch execution, using the shared Qwen layer lifecycle.

GDN and MoE reuse weights across independent requests. QSA retains native attention and index-cache updates. Batch Q/K norm, mRoPE
and K/V publication share one kernel; TP8 fuses the output gate with projection.
Collectives retain their native implementation. TP1 B2 and both B4 layouts
fuse MLP HC up with the MoE front; attention and final HC retain independent small-row kernels.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce

from atom.models.qwen4_exp_mono import native_qsa
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import (
    batch_front,
    batch_gdn,
    batch_hc,
    batch_hc_front,
    batch_qsa,
    batch_qsa_output,
    batch_route,
    batch_routed,
    batch_shared,
)
from atom.models.qwen4_exp_mono.layout import (
    finish_scratch_bytes,
    finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.native_hc import mix_rows
from atom.mono.runtime.abi import KernelAbi


class GdnKernels(NamedTuple):
    projection: object
    recurrence_norm: object
    output: object


class MoeKernels(NamedTuple):
    route: object
    shared_g2: object
    routed_g1: object
    routed_g2: object


class HcKernels(NamedTuple):
    up: object
    final: object


class GdnWorkspace(NamedTuple):
    projected: torch.Tensor
    normed: torch.Tensor
    output: torch.Tensor


class MoeWorkspace(NamedTuple):
    logits: torch.Tensor
    ids: torch.Tensor
    probabilities: torch.Tensor
    gate: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor
    shared_mid: torch.Tensor
    shared: torch.Tensor
    output: torch.Tensor


# Explicit scalar argument order for compile-only validation. Runtime launches
# below retain the same flat ABI and resolve weights from named static bindings.
_QSA_OUTPUT_ABI = KernelAbi(("x", "gate", "w", "out", "gate_stride"))

_FRONT_ABI = KernelAbi(
    ("x", "router_weight", "logits", "shared_weight", "shared_mid", "q", "scale")
)

_GDN_ABIS = (
    KernelAbi(
        ("x", "w", "out", "conv_weight", "state", "read_indices", "write_indices")
    ),
    KernelAbi(
        (
            "projected",
            "dt_bias",
            "a_log",
            "read",
            "write",
            "state",
            "norm_weight",
            "out",
        )
    ),
    KernelAbi(("normed", "weight", "out")),
)
_MOE_ABIS = (
    KernelAbi(("logits", "ids", "weights", "gate", "expert_map")),
    KernelAbi(("mid", "weight", "gate", "out", "scratch")),
    KernelAbi(("scratch", "weight", "scale")),
    KernelAbi(("scratch", "weight", "scale", "out")),
)
_G2_SHARED_ABI = KernelAbi(("scratch", "weight", "scale", "shared", "out"))
_MERGE_ABI = KernelAbi(("partials", "shared", "out"))
_HC_ABI = KernelAbi(("part", "normed", "weight", "out", "raw", "stride"))
_HC_FRONT_ABI = KernelAbi(
    (
        "part",
        "normed",
        "weight",
        "raw",
        "stride",
        "router_weight",
        "logits",
        "shared_weight",
        "shared_mid",
        "q",
        "scale",
        "payload",
    )
)


@dataclass
class BatchBuild:
    """Ordinary batch launchers and workspace owned by WidthBuilds."""

    batched = True
    rows: int
    front: object
    gdn: GdnKernels
    moe: MoeKernels
    buffers: GdnWorkspace
    arena: torch.Tensor
    values: MoeWorkspace
    routed_partials: torch.Tensor | None = None
    routed_merge: object = None
    hc: HcKernels | None = None
    hc_front: object = None
    hc_payload: torch.Tensor | None = None
    qsa_output: object = None

    def kernels(self):
        if self.qsa_output is not None:
            yield self.qsa_output, _QSA_OUTPUT_ABI
        if self.hc_front is not None:
            yield self.hc_front, _HC_FRONT_ABI
        if self.hc is not None:
            yield self.hc.up, _HC_ABI
            yield self.hc.final, _HC_ABI
        yield from zip(self.gdn, _GDN_ABIS, strict=True)
        if self.front is not None:
            yield self.front, _FRONT_ABI
        for kernel, abi in zip(self.moe, _MOE_ABIS, strict=True):
            if kernel is self.moe.routed_g2 and self.routed_merge is None:
                abi = _G2_SHARED_ABI
            yield kernel, abi
        if self.routed_merge is not None:
            yield self.routed_merge, _MERGE_ABI


def build(rows, device, tp=8):
    dims = Dims(tp)
    bf = lambda width: torch.empty((rows, width), device=device, dtype=torch.bfloat16)
    gdn = GdnKernels(
        batch_gdn.build_projection(rows, tp),
        batch_gdn.build_recurrence_norm(rows, tp),
        batch_gdn.build_output(rows, tp),
    )
    stride = finish_scratch_bytes()
    arena = torch.zeros((rows, stride), device=device, dtype=torch.uint8)
    layout = finish_scratch_layout()

    def view(name, dtype):
        off, size = layout[name]
        return arena[:, off : off + size].view(dtype)

    parallel_g2 = tp == 1 and rows == 2
    moe = MoeKernels(
        batch_route.build_route(rows, tp == 8, stride),
        batch_shared.build_g2_mid_quant(rows, tp),
        batch_routed.build_g1(rows, prefetch=tp == 8),
        (
            batch_routed.build_g2_partials(rows)
            if parallel_g2
            else batch_routed.build_g2_shared(
                rows,
                wave_experts=tp == 1 and rows == 4,
                owned_sparse=tp == 8 and rows == 4,
            )
        ),
    )
    values = MoeWorkspace(
        bf(513),
        view("route_ids", torch.int32),
        view("route_weights", torch.float32),
        torch.empty((rows, 1), device=device, dtype=torch.float32),
        view("xq", torch.uint8),
        view("xq_scale", torch.float32),
        bf(dims.shared_intermediate),
        bf(2560),
        bf(2560),
    )
    return BatchBuild(
        rows,
        batch_front.build(rows, tp, stride) if tp == 8 and rows == 2 else None,
        gdn,
        moe,
        GdnWorkspace(
            bf(dims.gdn_proj_dim),
            bf(dims.gdn_value_dim),
            bf(2560),
        ),
        arena,
        values,
        (
            torch.empty((rows, 10, 2560), device=device, dtype=torch.bfloat16)
            if parallel_g2
            else None
        ),
        batch_routed.build_g2_merge(rows) if parallel_g2 else None,
        (
            HcKernels(batch_hc.build(rows), batch_hc.build(rows, False))
            if rows == 4
            else None
        ),
        batch_hc_front.build(rows, tp, stride) if tp == 1 or rows == 4 else None,
        (
            torch.zeros(rows * 2560, device=device, dtype=torch.int32)
            if tp == 1 or rows == 4
            else None
        ),
        qsa_output=batch_qsa_output.build(rows) if tp == 8 else None,
    )


def mix(module, streams, pending, built):
    block, raw = pending if pending is not None else (None, None)
    injected = module.block_inject_weight is not None
    use_mfma = built.hc is not None
    fn = batch_hc.run if use_mfma else mix_rows
    options = (
        {"kernel": built.hc.up if injected else built.hc.final} if use_mfma else {}
    )
    h, mixed, r = fn(
        streams,
        module.hc_norm.weight,
        module.fused_w_cat,
        module.input_mix_weight_up.weight,
        module.hc_norm.variance_epsilon,
        module.hc_count,
        injected,
        block,
        raw,
        **options,
    )
    return mixed, (h, r)


def attention(built, static, step, mixed):
    k, v = built.gdn, built.buffers
    stream = torch.cuda.current_stream()
    k.projection(
        mixed.data_ptr(),
        static["projection_weight"],
        v.projected.data_ptr(),
        static["conv_weight"],
        step.conv_state.data_ptr(),
        step.state_read.data_ptr(),
        step.state_write.data_ptr(),
        stream=stream,
    )
    k.recurrence_norm(
        v.projected.data_ptr(),
        static["dt_bias"],
        static["a_log"],
        step.state_read.data_ptr(),
        step.state_write.data_ptr(),
        step.recurrent_state.data_ptr(),
        static["norm_weight"],
        v.normed.data_ptr(),
        stream=stream,
    )
    k.output(
        v.normed.data_ptr(), static["output_weight"], v.output.data_ptr(), stream=stream
    )
    return v.output


def qsa_attention(module, positions, mixed, built, static):
    """Project QSA natively, fuse batch norm/cache, then select causal indices."""
    prepared = native_qsa.prepare(module, positions, mixed)
    return qsa_finish(module, prepared, built, static)


def qsa_finish(module, prepared, built=None, static=None):
    """Finish native or Mono K1 with causal indices for an all-visible bucket."""
    metadata = prepared.qsa
    selected = None
    if metadata is not None and 0 < metadata.max_seq_len <= module.indexer.token_topk:
        selected = batch_qsa.select_all(
            metadata,
            prepared.gate.shape[0],
            module.indexer.token_topk,
            module.indexer.compress_ratio,
        )
    if built is None or built.qsa_output is None:
        return native_qsa.finish(module, prepared, selected)
    attention = native_qsa.attend(module, prepared, selected)
    output = built.buffers.output
    built.qsa_output(
        attention.data_ptr(),
        prepared.gate.data_ptr(),
        static["output_weight"],
        output.data_ptr(),
        prepared.gate.stride(0),
        stream=torch.cuda.current_stream(),
    )
    return output


def moe(built, static, x, *, front_ready=False):
    k, v = built.moe, built.values
    scratch = built.arena.data_ptr()
    stream = torch.cuda.current_stream()
    if not front_ready:
        built.front(
            x.data_ptr(),
            static["router"],
            v.logits.data_ptr(),
            static["shared_w13"],
            v.shared_mid.data_ptr(),
            v.quantized.data_ptr(),
            v.scales.data_ptr(),
            stream=stream,
        )
    k.route(
        v.logits.data_ptr(),
        v.ids.data_ptr(),
        v.probabilities.data_ptr(),
        v.gate.data_ptr(),
        static["expert_map"],
        stream=stream,
    )
    k.routed_g1(
        scratch, static["routed_w13"], static["routed_w13_scale"], stream=stream
    )
    k.shared_g2(
        v.shared_mid.data_ptr(),
        static["shared_w2"],
        v.gate.data_ptr(),
        v.shared.data_ptr(),
        scratch,
        stream=stream,
    )
    if built.routed_merge is not None:
        k.routed_g2(
            scratch,
            static["routed_w2"],
            static["routed_w2_scale"],
            built.routed_partials.data_ptr(),
            stream=stream,
        )
        built.routed_merge(
            built.routed_partials.data_ptr(),
            v.shared.data_ptr(),
            v.output.data_ptr(),
            stream=stream,
        )
    else:
        k.routed_g2(
            scratch,
            static["routed_w2"],
            static["routed_w2_scale"],
            v.shared.data_ptr(),
            v.output.data_ptr(),
            stream=stream,
        )
    return tensor_model_parallel_all_reduce(v.output)


def mlp_front(module, streams, pending, built, static):
    block, raw = pending
    streams, normed, part = batch_hc.prepare(
        streams,
        module.hc_norm.weight,
        module.fused_w_cat,
        module.input_mix_weight_up.weight,
        module.hc_norm.variance_epsilon,
        module.hc_count,
        True,
        block,
        raw,
    )
    inject = torch.empty((built.rows, 4), device=streams.device, dtype=streams.dtype)
    v = built.values
    built.hc_front(
        part.data_ptr(),
        normed.data_ptr(),
        static["hc_up"],
        inject.data_ptr(),
        normed.stride(0),
        static["router"],
        v.logits.data_ptr(),
        static["shared_w13"],
        v.shared_mid.data_ptr(),
        v.quantized.data_ptr(),
        v.scales.data_ptr(),
        built.hc_payload.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    return streams, (moe(built, static, None, front_ready=True), inject)
