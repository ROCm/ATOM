# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1/TP8 Quark decode with MXFP4 preparation and explicit FP8-KV seams.

BF16 GDN and HC enter Mono emitters for rows 1/2/4. QSA fuses norm/RoPE with
FP8 K/V and FP32 scale publication, retaining native index selection and attention.
Mono prepares routing and MXFP4 inputs for the native expert GEMMs. G1 writes
BF16 before native intermediate quantization; expert weights stay packed MXFP4.
"""

import logging
from dataclasses import dataclass

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_tp_group

from atom.models.qwen4_exp_mono import batch, native_qsa, quark_binding
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import (
    batch_gdn,
    batch_hc,
    batch_qsa_output,
    gdn_attention,
    quark_hc_pre,
    quark_hc_router,
    quark_moe,
    quark_qsa_projection,
    quark_qsa_select,
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
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.compile import launcher_resources
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
    router_splits: int = 1
    hc_payload: torch.Tensor | None = None
    mixed: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    raw: torch.Tensor | None = None
    qsa_projection: object = None
    qsa_output: object = None
    qsa_select: object = None
    qsa_qkv: torch.Tensor | None = None
    qsa_index_qk: torch.Tensor | None = None

    def kernels(self):
        if self.persistent is not None:
            yield self.persistent, gdn_attention.ABI
        else:
            yield from zip(self.gdn, batch._GDN_ABIS, strict=True)
        yield self.hc.up, batch._HC_ABI
        yield self.hc.final, batch._HC_ABI
        yield self.hc_router, quark_hc_router.ABI
        if self.qsa_projection is not None:
            yield self.qsa_projection, quark_qsa_projection.ABI
            yield self.qsa_output, batch._QSA_OUTPUT_ABI
        if self.qsa_select is not None:
            yield self.qsa_select, quark_qsa_select.ABI


class QuarkMonoRunner:
    def __init__(self, model, max_model_len):
        self.model = model
        self.max_model_len = max_model_len
        group = get_tp_group()
        self.group = group.cpu_group
        self.tp, self.rank = group.world_size, group.rank_in_group
        self.dims = Dims(self.tp)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        self._running = False
        bind_agreed(self._bind, self.group)
        self.widths = WidthBuilds(
            self._build,
            lambda built: built.kernels(),
            self.group,
            "Qwen Quark seam",
            check_resources=self._check_resources,
        )
        logger.info(
            "Qwen Quark Mono seam: TP%s rows 1/2/4; GDN/HC Mono; "
            "fused HC/router and MXFP4 route/quant; native G1 intermediate quant; "
            "fused QSA norm/RoPE/FP8 KV with native index/attention "
            "(max_model_len=%s)",
            self.tp,
            max_model_len,
        )
        if self.tp == 8:
            logger.info(
                "Qwen Quark TP8 QSA: fused BF16 QKV/index projections and gated output; "
                "native sparse selection/attention"
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
        self.intermediate = self.layers[0].mlp.experts.w13_weight.shape[1] // 2
        self.intermediate_pad = self.layers[0].mlp.experts.quant_method.intermediate_pad
        local_experts = 512 // self.tp + 1
        expected_map = [-1] * 514
        for expert in range(64):
            expected_map[self.rank * 64 + expert] = expert
        expected_map[512] = 64
        for layer in self.layers:
            ex = layer.mlp.experts
            ep_ok = ex.expert_map is None and ex.expert_mask is None
            if self.tp == 8:
                ep_ok = (
                    ex.expert_map is not None
                    and ex.expert_mask is not None
                    and ex.expert_map.cpu().tolist() == expected_map
                    and ex.moe_parallel_config.ep_size == 8
                    and ex.moe_parallel_config.tp_size == 1
                    and not ex.moe_parallel_config.use_all2all_kernels
                )
            if not (
                layer.mlp.fuse_shared
                and layer.mlp.tp_size == self.tp
                and self.intermediate in ((640,) if self.tp == 1 else (640, 768))
                and ex.quant_method.hidden_pad == 0
                and ex.quant_method.intermediate_pad == self.intermediate_pad
                and self.intermediate - self.intermediate_pad == 640
                and tuple(ex.w13_weight.shape)
                == (local_experts, 2 * self.intermediate, 1280)
                and tuple(ex.w2_weight.shape)
                == (local_experts, 2560, self.intermediate // 2)
                and ex.w13_weight.dtype == torch.float4_e2m1fn_x2
                and ex.w2_weight.dtype == torch.float4_e2m1fn_x2
                and getattr(ex.w13_weight, "is_shuffled", False)
                and getattr(ex.w2_weight, "is_shuffled", False)
                and ep_ok
                and ex.w13_bias is None
                and ex.w2_bias is None
                and ex.renormalize
                and not ex.apply_router_weight_on_input
                and not ex.quant_method.fp8_activations
                and not ex.quant_method.is_guinterleave
                and getattr(ex, "swiglu_limit", 0.0) in (None, 0.0)
            ):
                raise MonoUnsupported(
                    "Quark route/quant needs complete local A4W4 experts and shared ownership"
                )
        self.bound_weights = tuple(
            quark_binding.bind_layer(
                layer, task, self.dims, self.device, self.max_model_len
            )
            for layer, task in zip(self.layers, self.plan.layers, strict=True)
        )
        self.final_hc = quark_binding.bind_hc(
            native.hyper_connection_mixer, has_inject=False
        )
        quark_binding._check_device(self.final_hc, self.device)
        # Publish pointers only after all weights, numerical parameters and cache
        # layouts have been validated. Records retain the native tensor owners.
        self.static = tuple(
            (
                {
                    name: pointer(getattr(bound.gdn, attr))
                    for name, attr in _GDN_BINDINGS.items()
                }
                if bound.gdn is not None
                else None
            )
            for bound in self.bound_weights
        )
        self.qsa_weights = tuple(
            (
                tuple(
                    pointer(t)
                    for t in (bound.qsa.qkv, bound.qsa.index_qk, bound.qsa.o_proj)
                )
                if self.tp == 8 and bound.qsa is not None
                else None
            )
            for bound in self.bound_weights
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
        bf = lambda width: torch.empty(
            (rows, width), device=self.device, dtype=torch.bfloat16
        )
        if rows == 1 and self.tp == 1:
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
                    batch_gdn.build_projection(rows, self.tp),
                    batch_gdn.build_recurrence_norm(rows, self.tp),
                    batch_gdn.build_output(rows, self.tp),
                ),
                buffers=batch.GdnWorkspace(
                    bf(self.dims.gdn_proj_dim), bf(self.dims.gdn_value_dim), bf(HIDDEN)
                ),
            )
        built.moe = quark_moe.Workspace.allocate(rows, self.device)
        built.hc = batch.HcKernels(
            batch_hc.build(rows),
            batch_hc.build(rows, False),
            quark_hc_pre._hc_pre_dot2 if self.tp == 8 else None,
        )
        if self.tp == 8:
            built.qsa_projection = quark_qsa_projection.build(rows)
            built.qsa_output = batch_qsa_output.build(rows)
            built.qsa_qkv, built.qsa_index_qk = bf(2048), bf(640)
            if 2048 <= self.max_model_len <= 16384:
                built.qsa_select = quark_qsa_select.build(rows)
        built.router_splits = 2 if self.tp == 8 else 1
        built.hc_router = quark_hc_router.build(rows, built.router_splits)
        built.hc_payload = torch.zeros(
            rows * 2560, device=self.device, dtype=torch.int32
        )
        built.mixed, built.raw = bf(2560), bf(4)
        built.logits = (
            torch.empty((2, rows, 513), device=self.device, dtype=torch.float32)
            if built.router_splits == 2
            else bf(513)
        )
        quark_moe.validate_native_config(
            rows, self.intermediate, 512 // self.tp + 1, self.intermediate_pad
        )
        return built

    @staticmethod
    def _check_resources(built, rows):
        for kernel, _ in built.kernels():
            resources = launcher_resources(kernel)
            if any(r["scratch"] or r.get("vgpr_spill", 0) for r in resources):
                raise MonoUnsupported(f"Qwen Quark seam spills: {resources}")
            if not resources:
                raise MonoUnsupported("Qwen Quark seam has no compiled binary")
            logger.info("Qwen Quark seam B%s resources: %s", rows, resources)

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
                    gdn_step_view(context, task.layer_id, rows, tp=self.tp)
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
            for layer, task, step, static, qsa_weights, bound in zip(
                self.layers,
                self.plan.layers,
                steps,
                self.static,
                self.qsa_weights,
                self.bound_weights,
                strict=True,
            ):
                streams, pending = apply_ple(layer, streams, pending, input_ids)
                mixed, (streams, raw) = batch.mix(
                    layer.attn_hyper_connection, streams, pending, built
                )
                if task.is_gdn:
                    if rows == 1 and self.tp == 1:
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
                    if built.qsa_projection is not None:
                        built.qsa_projection(
                            mixed.data_ptr(),
                            qsa_weights[0],
                            qsa_weights[1],
                            built.qsa_qkv.data_ptr(),
                            built.qsa_index_qk.data_ptr(),
                            stream=torch.cuda.current_stream(),
                        )
                    prepared = native_qsa.prepare(
                        layer.self_attn,
                        positions,
                        mixed,
                        fuse_decode_index=True,
                        projected_qkv=built.qsa_qkv,
                        projected_index_qk=built.qsa_index_qk,
                    )
                    selected = None
                    if built.qsa_select is not None:
                        selected = quark_qsa_select.select(
                            layer.self_attn,
                            prepared.index_query,
                            prepared.qsa,
                            built.qsa_select,
                        )
                    if built.qsa_output is None:
                        sub = native_qsa.finish(layer.self_attn, prepared, selected)
                    else:
                        attention = native_qsa.attend(
                            layer.self_attn, prepared, selected
                        )
                        sub = built.buffers.output
                        built.qsa_output(
                            attention.data_ptr(),
                            prepared.gate.data_ptr(),
                            qsa_weights[2],
                            sub.data_ptr(),
                            prepared.gate.stride(0),
                            stream=torch.cuda.current_stream(),
                        )
                if self.tp > 1:
                    sub = tensor_model_parallel_all_reduce(sub)
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
                    pre=built.hc.pre,
                )
                built.hc_router(
                    part.data_ptr(),
                    normed.data_ptr(),
                    bound.mlp_hc.up.data_ptr(),
                    built.raw.data_ptr(),
                    normed.stride(0),
                    bound.router.data_ptr(),
                    built.logits.data_ptr(),
                    built.mixed.data_ptr(),
                    built.hc_payload.data_ptr(),
                    stream=torch.cuda.current_stream(),
                )
                mixed, raw = built.mixed, built.raw
                ex = layer.mlp.experts
                built.moe.prepare(
                    built.logits,
                    mixed,
                    ex.expert_map,
                    self.tp,
                    self.rank,
                    router_splits=built.router_splits,
                )
                output = built.moe.experts_forward(
                    mixed,
                    bound.w13,
                    bound.w2,
                    bound.w13_scale,
                    bound.w2_scale,
                    intermediate_pad=self.intermediate_pad,
                )
                if self.tp > 1:
                    output = tensor_model_parallel_all_reduce(output)
                pending = output, raw
            mixed, _ = batch.mix(native.hyper_connection_mixer, streams, pending, built)
            return mixed
        finally:
            self._running = False
