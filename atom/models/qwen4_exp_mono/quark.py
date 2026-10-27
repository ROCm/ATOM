# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1 Quark decode with MXFP4 preparation and explicit FP8-KV seams.

BF16 GDN and HC enter Mono emitters for rows 1/2/4. QSA fuses norm/RoPE with
FP8 K/V and FP32 scale publication, retaining native index selection and attention.
Mono prepares routing and MXFP4 inputs for the native expert GEMMs. G1 writes
BF16 before native intermediate quantization; expert weights stay packed MXFP4.
"""

import logging
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.qwen4_exp_mono import batch, native_qsa
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import (
    batch_gdn,
    batch_hc,
    gdn_attention,
    quark_hc_router,
    quark_moe,
)
from atom.models.qwen4_exp_mono.launch_args import (
    _GDN_BINDINGS,
    _GDN_TP1_NAMES,
    apply_ple,
    gdn_state_args,
    pointer,
)
from atom.models.qwen4_exp_mono.layout import gdn_scratch_bytes, gdn_scratch_layout
from atom.models.qwen4_exp_mono.plan import make_plan
from atom.models.qwen4_exp_mono.shapes import DECODE_ROWS, HIDDEN
from atom.models.qwen4_exp_mono.step_meta import gdn_step_view
from atom.models.qwen4_exp_mono.weights import _bind_gdn
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


@dataclass
class QuarkBuild:
    gdn: object = None
    buffers: object = None
    hc: object = None
    persistent: object = None
    scratch: torch.Tensor | None = None
    ready: torch.Tensor | None = None
    moe: object = None
    hc_router: object = None
    hc_payload: torch.Tensor | None = None
    mixed: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    raw: torch.Tensor | None = None

    def kernels(self):
        if self.persistent is not None:
            yield self.persistent, gdn_attention.ABI
        else:
            yield from zip(self.gdn, batch._GDN_ABIS, strict=True)
        yield self.hc.up, batch._HC_ABI
        yield self.hc.final, batch._HC_ABI
        yield self.hc_router, quark_hc_router.ABI


class QuarkMonoRunner:
    def __init__(self, model, max_model_len):
        self.model = model
        self.group = get_tp_group().cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        self._running = False
        bind_agreed(self._bind, self.group)
        self.widths = WidthBuilds(
            self._build, lambda built: built.kernels(), self.group, "Qwen Quark seam"
        )
        logger.info(
            "Qwen Quark Mono seam: TP1 rows 1/2/4; GDN/HC Mono; "
            "fused HC/router and MXFP4 route/quant; native G1 intermediate quant; "
            "fused QSA norm/RoPE/FP8 KV with native index/attention "
            "(max_model_len=%s)",
            max_model_len,
        )

    def _bind(self):
        props = torch.cuda.get_device_properties(self.device)
        if (
            torch.version.hip is None
            or props.gcnArchName.split(":")[0] != "gfx950"
            or props.multi_processor_count != BLOCKS
        ):
            raise MonoUnsupported("Qwen Quark seam needs gfx950 with 256 resident CTAs")
        native = self.model.model
        if native.return_hc_state:
            raise MonoUnsupported("Qwen Quark seam does not serve MTP")
        self.layers = native.layers
        self.plan = make_plan(self.layers)
        self.weights = tuple(
            _bind_gdn(layer, task, Dims(1))
            for layer, task in zip(self.layers, self.plan.layers, strict=True)
        )
        self.static = tuple(
            (
                {
                    name: pointer(getattr(weight, attr))
                    for name, attr in _GDN_BINDINGS.items()
                }
                if weight is not None
                else None
            )
            for weight in self.weights
        )
        for layer in self.layers:
            ex = layer.mlp.experts
            if not (
                layer.mlp.fuse_shared
                and layer.mlp.tp_size == 1
                and tuple(ex.w13_weight.shape) == (513, 1280, 1280)
                and tuple(ex.w2_weight.shape) == (513, 2560, 320)
                and ex.w13_weight.dtype == torch.float4_e2m1fn_x2
                and ex.w2_weight.dtype == torch.float4_e2m1fn_x2
                and getattr(ex.w13_weight, "is_shuffled", False)
                and getattr(ex.w2_weight, "is_shuffled", False)
                and ex.w13_weight_scale.element_size() == 1
                and ex.w2_weight_scale.element_size() == 1
                and ex.expert_mask is None
                and ex.expert_map is None
                and ex.w13_bias is None
                and ex.w2_bias is None
                and ex.renormalize
                and not ex.apply_router_weight_on_input
                and not ex.quant_method.fp8_activations
                and not ex.quant_method.is_guinterleave
                and getattr(ex, "swiglu_limit", 0.0) in (None, 0.0)
            ):
                raise MonoUnsupported(
                    "Quark route/quant needs TP1 fused-shared A4W4 layout"
                )
        ex = self.layers[0].mlp.experts
        logger.info(
            "Qwen Quark seam weights: w13=%s/%s w2=%s/%s",
            tuple(ex.w13_weight.shape),
            ex.w13_weight.dtype,
            tuple(ex.w2_weight.shape),
            ex.w2_weight.dtype,
        )

    def _build(self, rows):
        for layer, task in zip(self.layers, self.plan.layers, strict=True):
            if not task.is_gdn:
                attn = layer.self_attn
                for name, dtype in (
                    ("raw_key_cache", torch.bfloat16),
                    ("compressed_key_cache", torch.bfloat16),
                    ("rope_position_cache", torch.int64),
                ):
                    cache = getattr(attn, name, None)
                    if (
                        cache is None
                        or cache.dtype != dtype
                        or not cache.is_contiguous()
                    ):
                        raise MonoUnsupported(f"Quark QSA index cache contract: {name}")
                if attn.compressed_key_scale is not None:
                    raise MonoUnsupported("Quark QSA index fusion requires BF16 index")
        bf = lambda width: torch.empty(
            (rows, width), device=self.device, dtype=torch.bfloat16
        )
        if rows == 1:
            scratch = torch.zeros(
                gdn_scratch_bytes(), device=self.device, dtype=torch.uint8
            )
            offset, size = gdn_scratch_layout()["gdn_ready"]
            built = QuarkBuild(
                persistent=gdn_attention.build(),
                scratch=scratch,
                ready=scratch[offset : offset + size],
            )
        else:
            built = QuarkBuild(
                gdn=batch.GdnKernels(
                    batch_gdn.build_projection(rows, 1),
                    batch_gdn.build_recurrence_norm(rows, 1),
                    batch_gdn.build_output(rows, 1),
                ),
                buffers=batch.GdnWorkspace(
                    bf(Dims(1).gdn_proj_dim), bf(Dims(1).gdn_value_dim), bf(HIDDEN)
                ),
            )
        built.moe = quark_moe.Workspace.allocate(rows, self.device)
        built.hc = batch.HcKernels(
            batch_hc.build(rows),
            batch_hc.build(rows, False),
        )
        built.hc_router = quark_hc_router.build(rows)
        built.hc_payload = torch.zeros(
            rows * 2560, device=self.device, dtype=torch.int32
        )
        built.mixed, built.logits, built.raw = bf(2560), bf(513), bf(4)
        quark_moe.validate_native_config(rows)
        compile_kernels(built.kernels())
        for kernel, _ in built.kernels():
            resources = launcher_resources(kernel)
            if any(r["scratch"] or r.get("vgpr_spill", 0) for r in resources):
                raise MonoUnsupported(f"Qwen Quark seam spills: {resources}")
            logger.info("Qwen Quark seam B%s resources: %s", rows, resources)
        return built

    def prepare(self, rows):
        return rows in DECODE_ROWS and self.widths.prepare(rows)

    def close(self):
        if self._running:
            raise RuntimeError("cannot close active Qwen Quark seam")
        self._closed = True

    def forward(self, input_ids, positions):
        if self._closed or self._running:
            raise RuntimeError("closed or nested Qwen Quark seam")
        self._running = True
        try:
            rows = input_ids.numel()
            built = self.widths[rows]
            context = get_forward_context()
            # Reject incompatible state views before mutating any layer.
            steps = tuple(
                (
                    gdn_step_view(context, task.layer_id, rows, tp=1)
                    if task.is_gdn
                    else None
                )
                for task in self.plan.layers
            )
            if built.ready is not None:
                built.ready.zero_()
            native = self.model.model
            streams = native.get_input_embeddings(input_ids).repeat(1, native.hc_count)
            pending = None
            for layer, task, step, static in zip(
                self.layers, self.plan.layers, steps, self.static, strict=True
            ):
                streams, pending = apply_ple(layer, streams, pending, input_ids)
                mixed, (streams, raw) = batch.mix(
                    layer.attn_hyper_connection, streams, pending, built
                )
                if task.is_gdn:
                    if rows == 1:
                        values = {
                            "gdn_input": pointer(mixed),
                            **{_GDN_TP1_NAMES[n]: v for n, v in static.items()},
                            **gdn_state_args(step, 1),
                            "scratch": pointer(built.scratch),
                            "layer": task.layer_id,
                        }
                        built.persistent(
                            *gdn_attention.ABI.pack(values),
                            stream=torch.cuda.current_stream(),
                        )
                        offset, size = gdn_scratch_layout()["gdn_out"]
                        sub = (
                            built.scratch[offset : offset + size]
                            .view(torch.bfloat16)
                            .view(1, HIDDEN)
                        )
                    else:
                        sub = batch.attention(built, static, step, mixed)
                else:
                    prepared = native_qsa.prepare(
                        layer.self_attn,
                        positions,
                        mixed,
                        fuse_decode_index=True,
                    )
                    sub = native_qsa.finish(layer.self_attn, prepared)
                hc = layer.mlp_hyper_connection
                streams, normed, part = batch_hc.prepare(
                    streams,
                    hc.hc_norm.weight,
                    hc.fused_w_cat,
                    hc.input_mix_weight_up.weight,
                    hc.hc_norm.variance_epsilon,
                    hc.hc_count,
                    True,
                    sub,
                    raw,
                )
                built.hc_router(
                    part.data_ptr(),
                    normed.data_ptr(),
                    hc.input_mix_weight_up.weight.data_ptr(),
                    built.raw.data_ptr(),
                    normed.stride(0),
                    layer.mlp.gate.weight.data_ptr(),
                    built.logits.data_ptr(),
                    built.mixed.data_ptr(),
                    built.hc_payload.data_ptr(),
                    stream=torch.cuda.current_stream(),
                )
                mixed, raw = built.mixed, built.raw
                built.moe.prepare(built.logits, mixed)
                ex = layer.mlp.experts
                pending = (
                    built.moe.experts_forward(
                        mixed,
                        ex.w13_weight,
                        ex.w2_weight,
                        ex.w13_weight_scale,
                        ex.w2_weight_scale,
                    ),
                    raw,
                )
            mixed, _ = batch.mix(native.hyper_connection_mixer, streams, pending, built)
            return mixed
        finally:
            self._running = False
