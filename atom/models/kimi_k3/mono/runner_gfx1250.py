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
from aiter.fused_moe import (
    _return_output,
    fused_moe,
    get_inter_dim,
    quant_remap,
    resolve_activation_dtype,
)
from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    flydsl_moe_gather_reduce,
    grouped_gemm_gfx1250_a8w4,
)
from aiter.ops.flydsl.herd_topk import herd_finalize_candidates, herd_route_psum
from aiter.ops.flydsl.moe_common import GateMode

try:
    from aiter.ops.topk import _FLYDSL_HERD_MAX_M, _FLYDSL_HERD_MIN_M, _FLYDSL_USE_HERD
except ImportError:  # an AITER without HERD routes every step exactly
    _FLYDSL_USE_HERD, _FLYDSL_HERD_MIN_M, _FLYDSL_HERD_MAX_M = False, 0, 0

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
from atom.models.kimi_k3.mono.grouped_moe_gfx1250 import GroupedMoe
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
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import (
    GATHER_ALLREDUCE_ABI,
    allreduce_peer_bytes,
    build_allreduce,
    build_gather_allreduce,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import (
    MAX_CTAS as ALLREDUCE_COUNTERS,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finalize_tdm_gfx1250 import (
    build_replicated_finalize,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    build_packed_finish,
    herd_scratch_layout,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_role_lds_gfx1250 import (
    build_role_lds_front,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_tdm_gfx1250 import (
    build_replicated_front,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_local_kernels_gfx1250 import (
    build_finalize_local,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_oproj_ar_gfx1250 import (
    build_oproj_allreduce,
    oproj_peer_bytes,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    FLAG_WORDS,
    INTER_ROW_WORDS,
    MOE_STATE_WORDS,
    build_moe,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.plan import _width, kda_metadata, width_key
from atom.models.kimi_k3.mono.runner import (
    K3MonoDecodeRunner,
    _bf16_attn_res,
    state_slots,
)
from atom.models.kimi_k3.mono.symm_peers import SymmPeers, map_symm_peers
from atom.models.kimi_k3.mono.weights import bind_external_moe, bind_fp8_o_proj
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import launcher_resources
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
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
        built.oproj,
    )


class _PrecomputedHerd:
    """The routes the lite finish finished HERD into (``precompute_herd``),
    standing in for the ``HerdCandidates`` ``fused_moe`` hands the grouped MoE:
    ``route_psum`` returns them when the finish laid them out for the tile the
    MoE asks for, and otherwise finalizes the candidates as AITER would and has
    the finish lay out for that tile from the next launch on."""

    def __init__(self, scratch: torch.Tensor, candidates, values, rows: int) -> None:
        layout = herd_scratch_layout(rows)

        def view(name, dtype=torch.int32):
            at, words = layout[name]
            return scratch[at : at + 4 * words].view(dtype)

        self.candidates, self.values = candidates, values
        # AITER's grouped tile for these decode widths; a different one costs
        # one fallback launch.
        self.tile_m = 16
        self.tile_m_cell = view("tile_m")
        self.tile_m_cell.fill_(self.tile_m)
        view("routed_scaling_factor", torch.float32).fill_(1.0)
        self.routes = (
            view("topk_weights", torch.float32).view(rows, TOPK),
            view("topk_ids").view(rows, TOPK),
            view("masked_m"),
            view("topids_to_rows").view(rows, TOPK),
            view("psum"),
        )

    def fusable(self) -> bool:
        return True

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return herd_finalize_candidates(self.values, self.candidates, EXPERTS)

    def route_psum(self, tile_m: int) -> tuple[torch.Tensor, ...]:
        if tile_m == self.tile_m:
            return self.routes
        self.tile_m = tile_m
        self.tile_m_cell.fill_(tile_m)
        return herd_route_psum(self.values, self.candidates, EXPERTS, tile_m)


@dataclass
class Gfx1250RoutedK2Build:
    plan: Gfx1250KimiBuildPlan
    front: object
    finish: object
    finalize: object
    buffers: dict[str, torch.Tensor]
    grouped_experts: bool
    replicated_finalize: bool
    herd: bool = False
    herd_routes: _PrecomputedHerd | None = None
    # the replicated finalize's all-reduce in a kernel, where the TP group's
    # symm_mem banks map (``K3Gfx1250MonoDecodeRunner.k2_peers``)
    allreduce: object | None = None
    # AITER's grouped MoE launched here (the width's, shared by its builds),
    # where it has a tile for the width; with ``allreduce`` that kernel then
    # does the gather-reduce
    grouped_moe: GroupedMoe | None = None
    # the routed experts as one fused launch (moe_tdm_gfx1250.build_moe) at
    # widths up to ATOM_KIMI_MONO_FUSED_MOE_ROWS, in place of AITER's
    fused_moe: object | None = None


@dataclass(frozen=True)
class Gfx1250WidthBuild:
    k1: dict[int, Gfx1250K1LocalRunner]
    k2: dict[int, Gfx1250RoutedK2Build]
    # o_proj and its all-reduce in one kernel, every KDA layer's
    oproj: object | None = None

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
            if k2.allreduce is not None:
                pairs.append((k2.allreduce.launcher, k2.allreduce.abi))
        if self.oproj is not None:
            pairs.append((self.oproj.launcher, self.oproj.abi))

        unique = {}
        for launcher, abi in pairs:
            unique.setdefault(id(launcher), (launcher, abi))
        return list(unique.values())


class K3Gfx1250MonoDecodeRunner(K3MonoDecodeRunner):
    """One TP8 rank with resident K1 and production K2."""

    max_rows = GFX1250_MAX_TOKENS
    bind_k2 = False
    fp8_in_proj = True
    k2_peers: SymmPeers | None = None
    # the o_proj all-reduce's banks, past the K2 all-reduce's in ``k2_peers``
    oproj_peers: torch.Tensor | None = None
    oproj_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None

    def __init__(self, model, drafter=None, metadata_builder=None) -> None:
        group = get_tp_group().cpu_group
        bind_agreed(lambda: self._bind(model, drafter, metadata_builder), group)
        # Every rank bound: the rendezvous is a collective they all reach.
        if envs.ATOM_KIMI_MONO_KERNEL_AR:
            reduce_bytes = allreduce_peer_bytes(self.tp, self.max_rows)
            self.k2_peers = map_symm_peers(
                reduce_bytes + oproj_peer_bytes(self.tp, self.max_rows),
                ALLREDUCE_COUNTERS,
                group,
                self.tp,
                self.device,
            )
            if self.k2_peers is not None and self.oproj_weights:
                self.oproj_peers = self.k2_peers.peers + reduce_bytes
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
            "Kimi-K3 gfx1250 packed mono runner bound (TP%d, %d CUs, K2 all-reduce %s, "
            "o_proj all-reduce %s)",
            self.tp,
            self.blocks,
            "in kernel" if self.k2_peers is not None else "by aiter",
            "in kernel" if self.oproj_peers is not None else "by aiter",
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
                fp8_shared=True,
            )
            for plan in self.plans
            if plan.kda is not None
        }
        try:
            self.oproj_weights = {
                layer_id: bind_fp8_o_proj(
                    self.inner.layers[layer_id], layer_id, self.dims
                )
                for layer_id in self.routed
            }
        except MonoUnsupported:
            # o_proj stays on AITER's GEMM and all-reduce
            self.oproj_weights = None

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
                    fp8_projection=weight.w_in_scale is not None,
                )
                cache_key = spec
                local = k1_cache.get(cache_key)
                if local is None:
                    local = (
                        self._plan(rows, 1)
                        .make_k1_local_runner(spec)
                        .prepare(self.device)
                    )
                    k1_cache[cache_key] = local
                k1_by_layer[layer.layer_id] = local

                routed = self.routed[layer.layer_id]
                reset = routed.write_idx >= 0
                nblocks = routed.blocks + reset
                whole_up = routed.w_up_full is not None
                fp8_shared = (
                    routed.w_sgu_scale is not None,
                    routed.w_sd_scale is not None,
                )
                k2_key = (nblocks, reset, routed.gu_interleave, whole_up, fp8_shared)
                k2_local = k2_cache.get(k2_key)
                if k2_local is None:
                    k2_local = self._build_routed_k2(
                        rows,
                        nblocks,
                        reset,
                        routed.gu_interleave,
                        whole_up,
                        fp8_shared,
                    )
                    k2_cache[k2_key] = k2_local
                k2_by_layer[layer.layer_id] = k2_local

        oproj = (
            build_oproj_allreduce(self._plan(rows, 1), self.tp, self.dims.proj)
            if self.oproj_peers is not None
            else None
        )
        return Gfx1250WidthBuild(k1_by_layer, k2_by_layer, oproj)

    def _build_routed_k2(
        self,
        rows: int,
        nblocks: int,
        reset: bool,
        gu_interleave: bool,
        whole_up: bool = False,
        fp8_shared: tuple[bool, bool] = (False, False),
    ) -> Gfx1250RoutedK2Build:
        plan = self._plan(rows, nblocks)
        # AITER's grouped TDM MoE reads GUGU gate/up planes only.
        grouped = gu_interleave
        fp8_gate_up, fp8_down = fp8_shared
        if (fp8_gate_up or fp8_down) and not grouped:
            raise MonoUnsupported("only the grouped-MoE K2 reads FP8 shared experts")
        # Both partials then take one all-reduce and every rank applies the
        # whole up-projection.
        replicated = grouped and whole_up
        # The production MoE's HERD window: Top-(K+1) candidates, one dropped
        # by batch popularity inside the grouped MoE.
        herd = (
            grouped
            and _FLYDSL_USE_HERD
            and _FLYDSL_HERD_MIN_M <= rows <= _FLYDSL_HERD_MAX_M
        )
        width = TOPK + 1 if herd else TOPK
        front = (
            build_replicated_front(plan, fp8_shared=fp8_gate_up)
            if grouped
            else build_role_lds_front(plan, include_attention=False, reset=reset)
        )
        finish = build_packed_finish(
            plan,
            gu_interleave=gu_interleave,
            include_routed=not grouped,
            replicated_front=grouped,
            herd=herd,
            fp8_shared_down=fp8_down,
            precompute_herd=herd,
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
            "route": empty((rows, width if grouped else 2 * TOPK), torch.int32),
            "tile_route": (
                empty((rows, width)) if grouped else empty((tiles, 16, 32), torch.int32)
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
        fused_moe = self._fused_moe_for(rows) if grouped else None
        if fused_moe is not None:
            routes = rows * TOPK
            buffers["fused_inter"] = empty((routes, INTER_ROW_WORDS), torch.int32)
            buffers["fused_grouped"] = empty((routes, LATENT), torch.bfloat16)
            # a route's row of the fused output is the route itself
            buffers["fused_rows"] = torch.arange(
                routes, dtype=torch.int32, device=self.device
            ).view(rows, TOPK)
        grouped_moe = (
            self._grouped_moe_for(rows) if herd and fused_moe is None else None
        )
        allreduce = None
        if replicated and self.k2_peers is not None:
            allreduce = (
                build_gather_allreduce(self.tp, rows)
                if grouped_moe is not None or fused_moe is not None
                else build_allreduce(self.tp, rows)
            )
        return Gfx1250RoutedK2Build(
            plan,
            front,
            finish,
            finalize,
            buffers,
            grouped,
            replicated,
            herd,
            (
                _PrecomputedHerd(
                    buffers["finish_scratch"],
                    buffers["route"],
                    buffers["tile_route"],
                    rows,
                )
                if herd
                else None
            ),
            allreduce,
            grouped_moe,
            fused_moe,
        )

    def _fused_moe_for(self, rows: int):
        """The width's fused MoE launcher where ATOM_KIMI_MONO_FUSED_MOE_ROWS
        covers it and its LDS fits, else None; its flags and epoch (one set
        for every layer and width: launches never overlap) made on first use."""
        if rows > envs.ATOM_KIMI_MONO_FUSED_MOE_ROWS:
            return None
        try:
            kernel = build_moe(rows)
        except ValueError:
            return None
        if "fused_moe_sync" not in self.__dict__:
            self.fused_moe_sync = (
                torch.zeros(
                    GFX1250_MAX_TOKENS * TOPK * FLAG_WORDS,
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.zeros(MOE_STATE_WORDS, dtype=torch.int32, device=self.device),
            )
        return kernel

    def _grouped_moe_for(self, rows: int) -> GroupedMoe | None:
        """The width's grouped MoE, shared by its K2 builds; None where AITER
        has no tile for it, and ``grouped_gemm_gfx1250_a8w4`` runs instead."""
        moes = self.__dict__.setdefault("grouped_moes", {})
        if rows not in moes:
            try:
                moes[rows] = GroupedMoe(rows, self.dims.inter, self.device)
            except MonoUnsupported:
                moes[rows] = None
        return moes[rows]

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
                plan, prefix, blocks, add, add2, bf16=True
            )
            if write_idx >= 0:
                blocks[:, write_idx].copy_(prefix)
            core = self._k1(build, plan, normed, None, blocks, metadata)
            if build.oproj is not None:
                attn_out = self._oproj(build.oproj, plan.layer_id, core)
            elif isinstance(core, tuple):
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

    def _oproj(self, kernel, layer_id, core):
        """o_proj of the BF16 core rows and its all-reduce, in one kernel."""
        weight, scale = self.oproj_weights[layer_id]
        attn_out = core.new_empty(core.size(0), HIDDEN)
        kernel.launcher(
            core.data_ptr(),
            weight.data_ptr(),
            scale.data_ptr(),
            attn_out.data_ptr(),
            self.oproj_peers.data_ptr(),
            self.k2_peers.epochs.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )
        return attn_out

    def _routed_k2(self, build, plan, attn_out, prefix, blocks):
        weight = self.routed[plan.layer_id]
        local = build.k2[plan.layer_id]
        buffers = local.buffers
        stream = torch.cuda.current_stream()
        reset = weight.write_idx >= 0
        x, prefix = _bf16_attn_res(
            self.inner.layers[plan.layer_id].mlp_attn_res,
            None if reset else prefix,
            blocks[:, : weight.blocks + reset],
            attn_out,
            None,
        )
        if local.grouped_experts:
            local.front.launcher(
                x.data_ptr(),
                weight.w_gate.data_ptr(),
                weight.w_ld.data_ptr(),
                weight.w_sgu.data_ptr(),
                0 if weight.w_sgu_scale is None else weight.w_sgu_scale.data_ptr(),
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
        # gemm2's grouped rows and their routes, held until the all-reduce that
        # gathers them is queued
        gathered = None
        if local.fused_moe is not None:
            gathered = self._fused_moe(weight, local, moe, stream)
        elif local.grouped_moe is not None:
            gathered = self._grouped_moe(weight, local, moe)
        elif local.herd_routes is not None:
            self._aiter_grouped_moe(weight, local, moe)
        elif local.grouped_experts:
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
                # The mono config requires routed_scaling_factor == 1.
                **(
                    {"herd_topk": TOPK, "herd_routed_scaling_factor": 1.0}
                    if local.herd
                    else {}
                ),
            )
        if local.replicated_finalize:
            if local.allreduce is not None:
                local.allreduce.launcher(
                    buffers["reduce"].data_ptr(),
                    *([] if gathered is None else [t.data_ptr() for t in gathered]),
                    self.k2_peers.peers.data_ptr(),
                    self.k2_peers.epochs.data_ptr(),
                    self.rank,
                    stream=stream,
                )
                reduced = buffers["reduce"]
            else:
                reduced = get_tp_group().all_reduce(
                    buffers["reduce"], ca_fp8_quant=False
                )
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
        return (
            get_tp_group()
            .all_gather(
                buffers["payload"].view(torch.float32),
                dim=0,
                use_custom=False,
            )
            .view(torch.uint8)
        )

    def _fused_moe(self, weight, local, moe, stream):
        """The fused MoE on the lite finish's latent and routes (HERD's, or
        the plain top-k under HERD's widths): each route's BF16 rows, and the
        identity (row, weight) routes the gathering all-reduce takes; or, with
        no such all-reduce, their gather-reduce into ``routed``."""
        buffers = local.buffers
        if local.herd_routes is not None:
            weights, ids = local.herd_routes.routes[:2]
        else:
            weights, ids = buffers["tile_route"], buffers["route"]
        flags, state = self.fused_moe_sync
        local.fused_moe.launcher(
            ids.data_ptr(),
            buffers["latent_fp4"].data_ptr(),
            buffers["latent_scales"].data_ptr(),
            weight.w13.data_ptr(),
            moe.experts.w13_weight_scale.data_ptr(),
            weight.w2.data_ptr(),
            moe.experts.w2_weight_scale.data_ptr(),
            buffers["fused_inter"].data_ptr(),
            buffers["fused_grouped"].data_ptr(),
            flags.data_ptr(),
            state.data_ptr(),
            stream=stream,
        )
        grouped = buffers["fused_grouped"].unsqueeze(0)
        rows = buffers["fused_rows"]
        if local.allreduce is not None and local.allreduce.abi is GATHER_ALLREDUCE_ABI:
            return grouped, rows, weights
        flydsl_moe_gather_reduce(grouped, rows, weights, out=buffers["routed"])
        return None

    def _grouped_moe(self, weight, local, moe):
        """The grouped MoE on the lite finish's latent and HERD routes, into
        ``routed``; or, where the all-reduce gathers, gemm2's grouped rows and
        their routes for it."""
        buffers = local.buffers
        gathers = local.allreduce is not None and (
            local.allreduce.abi is GATHER_ALLREDUCE_ABI
        )
        out = local.grouped_moe(
            buffers["latent_fp4"].view(torch.uint8).view(dtypes.fp4x2),
            buffers["latent_scales"],
            weight.w13.view(dtypes.fp4x2),
            weight.w2.view(dtypes.fp4x2),
            moe.experts.w13_weight_scale,
            moe.experts.w2_weight_scale,
            local.herd_routes,
            buffers["routed"],
            SITU_BETA,
            SITU_LINEAR_BETA,
            gather=not gathers,
        )
        return out if gathers else None

    def _aiter_grouped_moe(self, weight, local, moe):
        """``fused_moe``'s HERD call into the grouped MoE, with the lite
        finish's routes in place of the candidates it would finalize."""
        buffers = local.buffers
        hidden = buffers["latent_fp4"].view(torch.uint8).view(dtypes.fp4x2)
        w1, w2 = weight.w13.view(dtypes.fp4x2), weight.w2.view(dtypes.fp4x2)
        rows = hidden.size(0)
        experts, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        quant_type = quant_remap.get(QuantType.per_1x32, QuantType.per_1x32)
        out = grouped_gemm_gfx1250_a8w4(
            hidden,
            w1,
            w2,
            # Unfilled: the grouped MoE takes the routes from ``herd``.
            torch.empty((rows, TOPK), dtype=torch.float32, device=hidden.device),
            torch.empty((rows, TOPK), dtype=torch.int32, device=hidden.device),
            E=experts,
            model_dim=model_dim,
            inter_dim=inter_dim,
            dtype=torch.bfloat16,
            activation=ActivationType.Situv2,
            quant_type=quant_type,
            q_dtype_a=resolve_activation_dtype(
                quant_type,
                w1.dtype,
                activation=ActivationType.Situv2,
                gate_mode=GateMode.INTERLEAVE,
                M=rows,
                hidden_dtype=hidden.dtype,
                has_a1_scale=True,
                gate_up_interleave=getattr(w1, "gate_up_interleave", None),
            ),
            q_dtype_w=w1.dtype,
            isG1U1=inter_dim != w1.shape[1],
            doweight_stage1=False,
            w1_scale=moe.experts.w13_weight_scale,
            w2_scale=moe.experts.w2_weight_scale,
            a1_scale=buffers["latent_scales"],
            expert_mask=None,
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
            output=buffers["routed"],
            herd=local.herd_routes,
        )
        if out is None:
            raise RuntimeError("the grouped MoE refused the precomputed HERD routes")
        _return_output(out, buffers["routed"])

    def _finish(self, weight, local, gathered, stream):
        buffers = local.buffers
        local.finish.launcher(
            gathered.data_ptr(),
            weight.bias.data_ptr(),
            buffers["shared_gate_up"].data_ptr(),
            weight.w_sd.data_ptr(),
            0 if weight.w_sd_scale is None else weight.w_sd_scale.data_ptr(),
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
            w_in_scale=weight.w_in_scale,
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
