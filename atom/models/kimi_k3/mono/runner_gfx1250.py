# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 decode with resident gfx1250 K1 and externally reduced K2.

K1 fuses the attention-residual seam, projections, convolution and recurrent
attention into one rank-local launch. K2 keeps the production output projection
and its collective, uses the faster production AttnRes kernel, then runs native
router/latent/shared projections and a persistent finish (routing and the shared
expert) around regular TP collectives. With GUGU expert planes every rank
projects the whole router and latent itself (no all-gather) and AITER's grouped
TDM MoE reads the FP4 latent and the finish's routes for the routed experts.
Otherwise the latent is quantized by its producer before a packed all-gather
and the finish runs the routed experts too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.parallel_state import get_tp_group
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode

from atom.models.kimi_k3.mono.config import (
    EXPERTS,
    GFX1250_BLOCKS,
    HIDDEN,
    LATENT,
    SITU_BETA,
    SITU_LINEAR_BETA,
    TOPK,
)
from atom.models.kimi_k3.mono.external_collectives import FrontPayloadLayout
from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import (
    Gfx1250K1LocalRunner,
    K1LocalRuntime,
)
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    K1LocalBuildSpec,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    build_packed_finish,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_role_lds_gfx1250 import (
    build_role_lds_front,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finalize_tdm_gfx1250 import (
    build_replicated_finalize,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_tdm_gfx1250 import (
    build_replicated_front,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_local_kernels_gfx1250 import (
    build_finalize_local,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.plan import _width, kda_metadata, width_key
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner, state_slots
from atom.models.kimi_k3.mono.weights import bind_external_moe
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import launcher_resources
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")
K1_PTPC_FUSION_MAX_ROWS = 40


def _uses_ptpc_o_proj(layer) -> bool:
    o_proj = layer.self_attn.o_proj
    return (
        o_proj.quant_type == QuantType.per_Token and o_proj.params_dtype == dtypes.fp8
    )


def _fuse_ptpc_core(layer, rows: int) -> bool:
    """Measured crossover against K1 + AITER per-token quant on gfx1250."""
    return rows <= K1_PTPC_FUSION_MAX_ROWS and _uses_ptpc_o_proj(layer)


def _resident_hazards(launcher: object) -> tuple[str, ...]:
    hazards = []
    lds_limit = GFX1250.max_workgroup_lds_bytes
    assert lds_limit is not None
    for resource in launcher_resources(launcher):
        if resource["scratch"]:
            hazards.append(f"{resource['scratch']}B private segment")
        for kind in ("vgpr_spill", "sgpr_spill"):
            count = resource[kind]
            if count:
                hazards.append(f"{count} {kind.replace('_', ' ')}")
        if resource["lds"] > lds_limit:
            hazards.append(f"{resource['lds']}B LDS")
    return tuple(hazards)


def _lean_width(
    built: Gfx1250WidthBuild,
    spilled: list[object],
) -> Gfx1250WidthBuild | None:
    """Rebuild only spilled resident K1 launchers with one weight group live."""

    replacements: dict[int, Gfx1250K1LocalRunner] = {}
    for layer, runner in built.k1.items():
        if runner.entrypoint.launcher in spilled and runner.spec.gemv_unit > 1:
            lean = runner.plan.make_k1_local_runner(replace(runner.spec, gemv_unit=1))
            scratch = getattr(runner, "_scratch", None)
            if scratch is not None:
                lean.prepare(scratch.device)
            replacements[layer] = lean
    if not replacements:
        return None
    return Gfx1250WidthBuild(
        {layer: replacements.get(layer, runner) for layer, runner in built.k1.items()},
        built.k2,
    )


@dataclass
class Gfx1250RoutedK2Build:
    plan: Gfx1250KimiBuildPlan
    front: object
    finish: object
    finalize: object
    buffers: dict[str, torch.Tensor]
    grouped_experts: bool
    replicated_finalize: bool


@dataclass(frozen=True)
class Gfx1250WidthBuild:
    k1: dict[int, Gfx1250K1LocalRunner]
    k2: dict[int, Gfx1250RoutedK2Build]

    def kernels(self) -> list[tuple[object, KernelAbi]]:
        """All local launchers, for compile-time scratch validation."""
        pairs = [(k1.entrypoint.launcher, k1.entrypoint.abi) for k1 in self.k1.values()]
        for k2 in self.k2.values():
            pairs.extend(
                (
                    (k2.front.launcher, k2.front.abi),
                    (k2.finish.launcher, k2.finish.abi),
                    (k2.finalize.launcher, k2.finalize.abi),
                )
            )

        unique = {}
        for launcher, abi in pairs:
            unique.setdefault(id(launcher), (launcher, abi))
        return list(unique.values())


class K3Gfx1250MonoDecodeRunner(K3MonoDecodeRunner):
    """One TP8 rank with resident K1 and production K2."""

    max_rows = GFX1250_MAX_TOKENS
    bind_k2 = False

    def __init__(self, model, drafter=None, metadata_builder=None) -> None:
        group = get_tp_group().cpu_group
        bind_agreed(lambda: self._bind(model, drafter, metadata_builder), group)
        self.widths = WidthBuilds(
            self._build_width,
            Gfx1250WidthBuild.kernels,
            group,
            "Kimi-K3 gfx1250 packed mono",
            lean=_lean_width,
            hazards=_resident_hazards,
        )
        self._served_widths: set[tuple[int, int]] = set()
        logger.info(
            "Kimi-K3 gfx1250 packed mono runner bound (TP%d, %d CUs)",
            self.tp,
            self.blocks,
        )

    def _bind(self, model, drafter=None, metadata_builder=None) -> None:
        # This runner only replaces K1/K2. DSA layers keep the ordinary
        # production indexer, so no shared-indexer hook or peer buffer is
        # installed on the external-collective path.
        super()._bind(model, drafter, None)
        self.ix = None
        if self.debug:
            raise MonoUnsupported("gfx1250 packed mono does not support debug mode")
        if self.replay_cap:
            raise MonoUnsupported("gfx1250 K1 does not support ReplaySSM")
        self.blocks = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        if self.blocks != GFX1250_BLOCKS:
            raise MonoUnsupported(
                f"gfx1250 packed mono needs {GFX1250_BLOCKS} CUs, got {self.blocks}"
            )
        self.routed = {
            plan.layer_id: bind_external_moe(
                self.inner.layers[plan.layer_id],
                plan.layer_id,
                self.dims,
                self.rank,
                GFX1250,
            )
            for plan in self.plans
            if plan.kda is not None
        }

    def _allocate(self) -> None:
        self.ones = torch.ones(self.max_rows, dtype=torch.int32, device=self.device)

    def close(self) -> None:
        """There is no peer mapping to retire on the RCCL path."""

    def prepare(self, rows: int, qlen: int) -> bool:
        if not 0 < rows <= self.max_rows or rows % qlen:
            return False
        return self.widths.prepare(width_key(rows, qlen))

    def _plan(self, rows: int, nblocks: int) -> Gfx1250KimiBuildPlan:
        lds_bytes = GFX1250.max_workgroup_lds_bytes
        assert lds_bytes is not None
        return Gfx1250KimiBuildPlan(
            threads=256,
            blocks=self.blocks,
            lds_bytes=lds_bytes,
            tokens=rows,
            nblocks=nblocks,
        )

    def _build_width(self, key: int) -> Gfx1250WidthBuild:
        rows, qlen = _width(key)
        k1_cache: dict[tuple, Gfx1250K1LocalRunner] = {}
        k1_by_layer: dict[int, Gfx1250K1LocalRunner] = {}
        k2_cache: dict[tuple[int, bool, bool], Gfx1250RoutedK2Build] = {}
        k2_by_layer: dict[int, Gfx1250RoutedK2Build] = {}

        for layer in self.plans:
            if layer.kda is not None:
                weight = layer.kda
                spec = K1LocalBuildSpec(
                    qlen=qlen,
                    delta=False,
                    state_len=self.state_len,
                    eps=weight.seam.eps,
                    out_eps=weight.seam.out_eps,
                    onorm_eps=weight.onorm_eps,
                    lower_bound=weight.lower_bound,
                    write_idx=-1,
                    quant_core=_fuse_ptpc_core(self.inner.layers[layer.layer_id], rows),
                    external_seam=True,
                )
                cache_key = spec
                local = k1_cache.get(cache_key)
                if local is None:
                    local = (
                        self._plan(rows, 1).make_k1_local_runner(spec).prepare(self.device)
                    )
                    k1_cache[cache_key] = local
                k1_by_layer[layer.layer_id] = local

                routed = self.routed[layer.layer_id]
                reset = routed.write_idx >= 0
                nblocks = routed.blocks + reset
                whole_up = routed.w_up_full is not None
                k2_key = (nblocks, reset, routed.gu_interleave, whole_up)
                k2_local = k2_cache.get(k2_key)
                if k2_local is None:
                    k2_local = self._build_routed_k2(
                        rows,
                        nblocks,
                        reset,
                        routed.gu_interleave,
                        whole_up,
                    )
                    k2_cache[k2_key] = k2_local
                k2_by_layer[layer.layer_id] = k2_local

        return Gfx1250WidthBuild(k1_by_layer, k2_by_layer)

    def _build_routed_k2(
        self,
        rows: int,
        nblocks: int,
        reset: bool,
        gu_interleave: bool,
        whole_up: bool = False,
    ) -> Gfx1250RoutedK2Build:
        plan = self._plan(rows, nblocks)
        # AITER's grouped TDM MoE reads GUGU gate/up planes only.
        grouped = gu_interleave
        # Both partials then take one all-reduce and every rank applies the
        # whole up-projection.
        replicated = grouped and whole_up
        front = (
            build_replicated_front(plan)
            if grouped
            else build_role_lds_front(plan, include_attention=False, reset=reset)
        )
        finish = build_packed_finish(
            plan,
            gu_interleave=gu_interleave,
            include_routed=not grouped,
            replicated_front=grouped,
        )
        finalize = (
            build_replicated_finalize(plan)
            if replicated
            else build_finalize_local(plan, input_normalized=True)
        )
        payload = FrontPayloadLayout(rows)
        tiles = plan.token_tiling.tile_count

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        buffers = {
            "shared_gate_up": empty((rows, 2 * self.dims.sh_inter), torch.bfloat16),
            "route": empty((rows, TOPK if grouped else 2 * TOPK), torch.int32),
            "tile_route": (
                empty((rows, TOPK))
                if grouped
                else empty((tiles, 16, 32), torch.int32)
            ),
            "flags": empty((tiles, EXPERTS), torch.int32),
            "experts": empty((tiles, 256), torch.int32),
            "kof": empty((tiles, 256, 16), torch.int32),
            "union_sizes": empty((tiles,), torch.int32),
            "latent_fp4": empty((rows, LATENT // 8), torch.int32),
            "latent_scales": empty(
                (rows, LATENT // 32), torch.uint8 if grouped else torch.int32
            ),
            "intermediate_fp4": empty((tiles, 256, self.dims.inter // 8), torch.int32),
            "intermediate_scales": empty(
                (tiles, 256, self.dims.inter // 32), torch.int32
            ),
            "routed": empty((rows, LATENT), torch.bfloat16),
            "shared_hidden": empty((rows, self.dims.sh_inter), torch.bfloat16),
            "shared": empty((rows, HIDDEN), torch.bfloat16),
            "finish_scratch": torch.zeros(
                finish.scratch_bytes, dtype=torch.uint8, device=self.device
            ),
            "final": empty((rows, HIDDEN), torch.bfloat16),
        }
        if replicated:
            reduce = empty((rows * (LATENT + HIDDEN),), torch.bfloat16)
            buffers["reduce"] = reduce
            buffers["routed"] = reduce[: rows * LATENT].view(rows, LATENT)
            buffers["shared"] = reduce[rows * LATENT :].view(rows, HIDDEN)
        if grouped:
            buffers["router_logits"] = empty((rows, EXPERTS))
        else:
            buffers["payload"] = empty(payload.local_shape, torch.uint8)
            buffers["latent"] = empty((rows, LATENT // self.tp), torch.bfloat16)
            buffers["front_scratch"] = torch.zeros(
                front.scratch_bytes, dtype=torch.uint8, device=self.device
            )
        return Gfx1250RoutedK2Build(
            plan,
            front,
            finish,
            finalize,
            buffers,
            grouped,
            replicated,
        )

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, qlen: int
    ) -> torch.Tensor:
        rows = input_ids.numel()
        width = (rows, qlen)
        if width not in self._served_widths:
            self._served_widths.add(width)
            logger.info(
                "Kimi-K3 gfx1250 packed mono serving rows=%d qlen=%d",
                rows,
                qlen,
            )
        build = self.widths[width_key(rows, qlen)]
        metadata = kda_metadata(get_forward_context())
        prefix = self.inner.get_input_embeddings(input_ids)
        blocks = prefix.new_empty(rows, self.num_blocks, prefix.size(1))
        add = add2 = None
        for plan in self.plans:
            prefix, add, add2 = self._layer(
                build,
                plan,
                positions,
                prefix,
                blocks,
                add,
                add2,
                metadata,
            )
        hidden, _ = self.inner.output_attn_res(prefix, blocks, add, add2)
        return hidden

    def _layer(self, build, plan, positions, prefix, blocks, add, add2, metadata):
        if plan.layer_id not in self.routed:
            out = self._fallback(plan, positions, prefix, blocks, add, add2)
        else:
            layer, normed, prefix, _, write_idx = self._seam(
                plan, prefix, blocks, add, add2
            )
            if write_idx >= 0:
                blocks[:, write_idx].copy_(prefix)
            core = self._k1(build, plan, normed, None, blocks, metadata)
            if isinstance(core, tuple):
                core, core_scale = core
                attn_out = layer.self_attn.o_proj(core, x_scale=core_scale)
            else:
                attn_out = layer.self_attn.o_proj(core)
            prefix, add = self._routed_k2(
                build,
                plan,
                attn_out,
                prefix,
                blocks,
            )
            out = (prefix, add, None)

        rows = blocks.size(0)
        widths = {tensor.size(0) for tensor in out if tensor is not None}
        if widths != {rows}:
            raise MonoUnsupported(
                f"layer {plan.layer_id} returned rows {sorted(widths)} for "
                f"an {rows}-row step"
            )
        self._write_aux(plan.layer_id, out)
        return out

    def _routed_k2(self, build, plan, attn_out, prefix, blocks):
        weight = self.routed[plan.layer_id]
        local = build.k2[plan.layer_id]
        buffers = local.buffers
        stream = torch.cuda.current_stream()
        reset = weight.write_idx >= 0
        x, prefix = self.inner.layers[plan.layer_id].mlp_attn_res(
            None if reset else prefix,
            blocks[:, : weight.blocks + reset],
            attn_out,
        )
        if local.grouped_experts:
            local.front.launcher(
                x.data_ptr(),
                weight.w_gate.data_ptr(),
                weight.w_ld.data_ptr(),
                weight.w_sgu.data_ptr(),
                buffers["router_logits"].data_ptr(),
                buffers["latent_fp4"].data_ptr(),
                buffers["latent_scales"].data_ptr(),
                buffers["shared_hidden"].data_ptr(),
                stream=stream,
            )
            gathered = buffers["router_logits"]
        else:
            gathered = self._sharded_front(
                plan, weight, local, attn_out, prefix, blocks, x
            )
        self._finish(weight, local, gathered, stream)
        moe = self.inner.layers[plan.layer_id].block_sparse_moe
        if local.grouped_experts:
            fused_moe(
                buffers["latent_fp4"].view(torch.uint8).view(dtypes.fp4x2),
                weight.w13.view(dtypes.fp4x2),
                weight.w2.view(dtypes.fp4x2),
                buffers["tile_route"],
                buffers["route"],
                activation=ActivationType.Situv2,
                quant_type=QuantType.per_1x32,
                w1_scale=moe.experts.w13_weight_scale,
                w2_scale=moe.experts.w2_weight_scale,
                a1_scale=buffers["latent_scales"],
                dtype=torch.bfloat16,
                beta=SITU_BETA,
                linear_beta=SITU_LINEAR_BETA,
                gate_mode=GateMode.INTERLEAVE.value,
                output=buffers["routed"],
            )
        if local.replicated_finalize:
            reduced = get_tp_group().all_reduce(buffers["reduce"], ca_fp8_quant=False)
            if reduced is None:
                reduced = buffers["reduce"]
            rows = buffers["routed"].size(0)
            routed = moe.routed_expert_norm(reduced[: rows * LATENT].view(rows, LATENT))
            local.finalize.launcher(
                routed.data_ptr(),
                reduced[rows * LATENT :].data_ptr(),
                weight.w_up_full.data_ptr(),
                buffers["final"].data_ptr(),
                stream=stream,
            )
            return prefix, buffers["final"]
        routed = get_tp_group().all_reduce(buffers["routed"], ca_fp8_quant=False)
        if routed is None:
            routed = buffers["routed"]
        routed = moe.routed_expert_norm(routed)
        local.finalize.launcher(
            routed.data_ptr(),
            buffers["shared"].data_ptr(),
            weight.ln_w.data_ptr(),
            weight.w_up.data_ptr(),
            buffers["final"].data_ptr(),
            self.rank,
            weight.ln_eps,
            stream=stream,
        )
        final = get_tp_group().all_reduce(buffers["final"], ca_fp8_quant=False)
        return prefix, buffers["final"] if final is None else final

    def _sharded_front(self, plan, weight, local, attn_out, prefix, blocks, x):
        """This rank's router/latent rows and the packed all-gather of them."""
        buffers = local.buffers
        stream = torch.cuda.current_stream()
        router = weight.w_gate.narrow(
            0, self.rank * (EXPERTS // self.tp), EXPERTS // self.tp
        )
        latent = weight.w_ld.narrow(
            0, self.rank * (LATENT // self.tp), LATENT // self.tp
        )

        def launch_front(front, scratch, launch_stream):
            front.launcher(
                attn_out.data_ptr(),
                prefix.data_ptr(),
                blocks.data_ptr(),
                blocks.stride(0),
                blocks.stride(1),
                weight.seam.nw.data_ptr(),
                weight.seam.qk.data_ptr(),
                weight.seam.out_nw.data_ptr(),
                router.data_ptr(),
                latent.data_ptr(),
                weight.w_sgu.data_ptr(),
                x.data_ptr(),
                buffers["payload"].data_ptr(),
                buffers["latent"].data_ptr(),
                buffers["shared_gate_up"].data_ptr(),
                scratch.data_ptr(),
                plan.layer_id,
                weight.seam.eps,
                weight.seam.out_eps,
                stream=launch_stream,
            )

        launch_front(local.front, buffers["front_scratch"], stream)

        # RCCL's small float32 all-gather is much faster than its byte path on
        # gfx1250. All-gather only copies elements, so this view preserves every
        # heterogeneous payload bit exactly.
        return get_tp_group().all_gather(
            buffers["payload"].view(torch.float32),
            dim=0,
            use_custom=False,
        ).view(torch.uint8)

    def _finish(self, weight, local, gathered, stream):
        buffers = local.buffers
        local.finish.launcher(
            gathered.data_ptr(),
            weight.bias.data_ptr(),
            buffers["shared_gate_up"].data_ptr(),
            weight.w_sd.data_ptr(),
            weight.w13.data_ptr(),
            weight.w13s.data_ptr(),
            weight.w2.data_ptr(),
            weight.w2s.data_ptr(),
            buffers["shared_hidden"].data_ptr(),
            buffers["route"].data_ptr(),
            buffers["tile_route"].data_ptr(),
            buffers["flags"].data_ptr(),
            buffers["experts"].data_ptr(),
            buffers["kof"].data_ptr(),
            buffers["union_sizes"].data_ptr(),
            buffers["latent_fp4"].data_ptr(),
            buffers["latent_scales"].data_ptr(),
            buffers["intermediate_fp4"].data_ptr(),
            buffers["intermediate_scales"].data_ptr(),
            buffers["routed"].data_ptr(),
            buffers["shared"].data_ptr(),
            buffers["finish_scratch"].data_ptr(),
            stream=stream,
        )

    def _k1(self, build, plan, prefix, delta, blocks, metadata, taps=None):
        if taps is not None:
            raise ValueError("gfx1250 packed mono does not expose K1 taps")
        weight = plan.kda
        local = build.k1[plan.layer_id]
        state = self._conv_state(plan.layer_id)
        recurrent = self._recurrent_state(plan.layer_id)
        slots, accepted = state_slots(metadata, self.ones)
        core = prefix.new_empty(prefix.size(0), self.dims.proj)
        core_q = core_scale = None
        if local.spec.quant_core:
            o_proj = self.inner.layers[plan.layer_id].self_attn.o_proj
            quant_bytes = torch.empty(
                core.numel() + core.size(0) * 4,
                dtype=torch.uint8,
                device=core.device,
            )
            core_q = quant_bytes[: core.numel()].view(o_proj.params_dtype).view_as(core)
            core_scale = quant_bytes[core.numel() :].view(torch.float32).view(-1, 1)
        runtime = K1LocalRuntime(
            prefix=prefix,
            delta=delta,
            blocks=blocks,
            blk_sm=blocks.stride(0),
            blk_sr=blocks.stride(1),
            ares_nw=weight.seam.nw,
            ares_qk=weight.seam.qk,
            in_nw=weight.seam.out_nw,
            w_in=weight.w_in,
            w_fb=weight.w_fb,
            conv_w=weight.conv_w,
            conv_st=state,
            cs_seq=state.stride(0),
            cs_dim=state.stride(1),
            cs_tok=state.stride(2),
            a_log=weight.a_log,
            dt_bias=weight.dt_bias,
            on_w=weight.on_w,
            rstate=recurrent,
            rs_seq=recurrent.stride(0),
            st_idx=slots,
            st_stride=slots.stride(0),
            num_acc=accepted,
            core_out=core,
            core_q=core_q,
            core_scale=core_scale,
        )
        local(runtime, stream=torch.cuda.current_stream())
        return (core_q, core_scale) if local.spec.quant_core else core
