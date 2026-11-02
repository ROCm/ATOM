# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 mono decode on gfx1250 with rank-local kernels and RCCL.

The gfx950 runner keeps TP communication inside its persistent kernels.  That
protocol is wave64- and single-host-specific.  gfx1250 instead runs the
architecture-local K1/K2 compute kernels around ordinary TP collectives:

``K1 -> o_proj AR -> packed router/latent AG -> persistent MoE finish
    -> routed AR -> routed-up/shared merge -> final AR``.

No peer pointer, HIP IPC mapping, or in-kernel cross-rank wait is used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.kimi_k3.mono.config import (
    EXPERTS,
    GFX1250_BLOCKS,
    HIDDEN,
    LATENT,
)
from atom.models.kimi_k3.mono.external_collectives import (
    ExternalCollectivePlan,
    FrontPayloadLayout,
)
from atom.models.kimi_k3.mono.external_runner import (
    ExternalCollectiveK2Runner,
    MoeLocalPartials,
    PackedShardedFrontPartials,
)
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
from atom.models.kimi_k3.mono.kernels.stages.k2_local_kernels_gfx1250 import (
    build_finalize_local,
    build_oproj_local,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.plan import _width, kda_metadata, width_key
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner, state_slots
from atom.models.kimi_k3.mono.weights import bind_k2_expert_scales
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


@dataclass
class Gfx1250K2Build:
    """One row/block specialization and its stable graph-capture buffers."""

    plan: Gfx1250KimiBuildPlan
    runner: ExternalCollectiveK2Runner
    oproj: object
    front: object
    finish: object
    finalize: object
    buffers: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Gfx1250WidthBuild:
    k1: dict[int, Gfx1250K1LocalRunner]
    k2: dict[int, Gfx1250K2Build]

    def kernels(self) -> list:
        # These entry points are FlyDSL jits rather than the ABI-packed
        # launchers consumed by mono.runtime.compile.  Width preparation still
        # emits every specialization; its first eager warmup compiles them
        # before CUDA-graph capture.
        return []


class K3Gfx1250MonoDecodeRunner(K3MonoDecodeRunner):
    """One TP8 rank of the packed, external-collective gfx1250 path."""

    max_rows = GFX1250_MAX_TOKENS

    def __init__(self, model, drafter=None, metadata_builder=None) -> None:
        group = get_tp_group().cpu_group
        bind_agreed(lambda: self._bind(model, drafter, metadata_builder), group)
        self.widths = WidthBuilds(
            self._build_width,
            Gfx1250WidthBuild.kernels,
            group,
            "Kimi-K3 gfx1250 packed mono",
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
        if self.replay_cap:
            raise MonoUnsupported("gfx1250 K1 does not support ReplaySSM")
        self.blocks = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        if self.blocks != GFX1250_BLOCKS:
            raise ValueError(
                f"gfx1250 packed mono needs {GFX1250_BLOCKS} CUs, " f"got {self.blocks}"
            )

        # AITER's gfx1250 expert payload is byte-compatible with the model
        # binder's FP4 planes, but its n32k4 E8M0 scale layout is explicit.
        plans = []
        for plan in self.plans:
            if plan.moe is None:
                plans.append(plan)
                continue
            module = self.inner.layers[plan.layer_id].block_sparse_moe
            scales = bind_k2_expert_scales(
                module, plan.layer_id, self.dims, self.rank, GFX1250
            ).kernel_arguments()
            plans.append(
                replace(
                    plan,
                    moe=replace(
                        plan.moe,
                        w13s=scales["w13s"],
                        w2s=scales["w2s"],
                    ),
                )
            )
        self.plans = plans

    def _allocate(self) -> None:
        # K1's non-debug build still carries these ABI slots.  One stable dummy
        # of the right dtype is enough because no debug store is emitted.
        self._x_debug = torch.empty(1, dtype=torch.bfloat16, device=self.device)
        self._projection_debug = torch.empty_like(self._x_debug)
        self._timeline = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.ones = torch.ones(self.max_rows, dtype=torch.int32, device=self.device)

    def close(self) -> None:
        """There is no peer mapping to retire on the RCCL path."""

    def prepare(self, rows: int, qlen: int) -> bool:
        if not 0 < rows <= self.max_rows or rows % qlen:
            return False
        return self.widths.prepare(width_key(rows, qlen))

    def _plan(self, rows: int, nblocks: int) -> Gfx1250KimiBuildPlan:
        return Gfx1250KimiBuildPlan(
            threads=256,
            blocks=self.blocks,
            lds_bytes=320 * 1024,
            tokens=rows,
            nblocks=nblocks,
        )

    def _build_width(self, key: int) -> Gfx1250WidthBuild:
        rows, qlen = _width(key)
        k1_cache: dict[tuple, Gfx1250K1LocalRunner] = {}
        k2_cache: dict[tuple[int, bool], Gfx1250K2Build] = {}
        k1_by_layer: dict[int, Gfx1250K1LocalRunner] = {}
        k2_by_layer: dict[int, Gfx1250K2Build] = {}

        for layer in self.plans:
            if layer.mono:
                weight = layer.kda
                assert weight is not None
                spec = K1LocalBuildSpec(
                    qlen=qlen,
                    delta=True,
                    state_len=self.state_len,
                    eps=weight.seam.eps,
                    out_eps=weight.seam.out_eps,
                    onorm_eps=weight.onorm_eps,
                    lower_bound=weight.lower_bound,
                    write_idx=weight.write_idx,
                )
                cache_key = (weight.blocks, spec)
                local = k1_cache.get(cache_key)
                if local is None:
                    local = self._plan(rows, weight.blocks).make_k1_local_runner(spec)
                    k1_cache[cache_key] = local
                k1_by_layer[layer.layer_id] = local

            if layer.moe is not None:
                reset = layer.moe.write_idx >= 0
                nblocks = layer.moe.blocks + reset
                cache_key = (nblocks, reset)
                local = k2_cache.get(cache_key)
                if local is None:
                    local = self._build_k2(rows, nblocks, reset)
                    k2_cache[cache_key] = local
                k2_by_layer[layer.layer_id] = local

        return Gfx1250WidthBuild(k1_by_layer, k2_by_layer)

    def _build_k2(self, rows: int, nblocks: int, reset: bool) -> Gfx1250K2Build:
        plan = self._plan(rows, nblocks)
        oproj = build_oproj_local(plan)
        front = build_role_lds_front(plan, reset=reset)
        finish = build_packed_finish(plan)
        finalize = build_finalize_local(plan)
        buffers = self._allocate_k2_buffers(
            plan, front.scratch_bytes, finish.scratch_bytes
        )
        tp_group = get_tp_group()

        def all_gather(tensor: torch.Tensor) -> torch.Tensor:
            return tp_group.all_gather(tensor, dim=0, use_custom=False)

        def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
            torch.distributed.all_reduce(tensor, group=tp_group.device_group)
            return tensor

        runner = ExternalCollectiveK2Runner(
            ExternalCollectivePlan(rows),
            all_gather,
            all_reduce,
        )
        return Gfx1250K2Build(plan, runner, oproj, front, finish, finalize, buffers)

    def _allocate_k2_buffers(
        self,
        plan: Gfx1250KimiBuildPlan,
        front_scratch_bytes: int,
        finish_scratch_bytes: int,
    ) -> dict[str, torch.Tensor]:
        rows = plan.tokens
        tiles = plan.token_tiling.tile_count
        device = self.device

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=device)

        payload = FrontPayloadLayout(rows)
        return {
            "oproj": empty((rows, HIDDEN), torch.bfloat16),
            "x_local": empty((rows, HIDDEN), torch.bfloat16),
            "router_local": empty((rows, EXPERTS // self.tp)),
            "latent_local": empty((rows, LATENT // self.tp), torch.bfloat16),
            "shared_gate_up": empty((rows, 2 * self.dims.sh_inter), torch.bfloat16),
            "latent_logical": empty((rows, LATENT), torch.bfloat16),
            "shared_hidden": empty((rows, self.dims.sh_inter), torch.bfloat16),
            "route": empty((rows, 32), torch.int32),
            "tile_route": empty((tiles, 16, 32), torch.int32),
            "flags": empty((tiles, EXPERTS), torch.int32),
            "experts": empty((tiles, 256), torch.int32),
            "kof": empty((tiles, 256, 16), torch.int32),
            "union_sizes": empty((tiles,), torch.int32),
            "latent_fp4": empty((rows, LATENT // 8), torch.int32),
            "latent_scales": empty((rows, LATENT // 32), torch.int32),
            "routed_intermediate_fp4": empty(
                (tiles, 256, self.dims.inter // 8), torch.int32
            ),
            "routed_intermediate_scales": empty(
                (tiles, 256, self.dims.inter // 32), torch.int32
            ),
            "routed_partial": empty((rows, LATENT), torch.bfloat16),
            "shared_partial": empty((rows, HIDDEN), torch.bfloat16),
            "final": empty((rows, HIDDEN), torch.bfloat16),
            "front_payload": empty(payload.local_shape, torch.uint8),
            "front_scratch": torch.zeros(
                front_scratch_bytes, dtype=torch.uint8, device=device
            ),
            "finish_scratch": torch.zeros(
                finish_scratch_bytes, dtype=torch.uint8, device=device
            ),
        }

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

    def _k1(self, build, plan, prefix, delta, blocks, metadata, taps=None):
        if taps is not None:
            raise ValueError("gfx1250 packed mono does not expose K1 taps")
        weight = plan.kda
        local = build.k1[plan.layer_id]
        state = self._conv_state(plan.layer_id)
        recurrent = self._recurrent_state(plan.layer_id)
        slots, accepted = state_slots(metadata, self.ones)
        core = prefix.new_empty(prefix.size(0), self.dims.proj)
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
            x_dbg=self._x_debug,
            proj_dbg=self._projection_debug,
            layer=plan.layer_id,
            tl=self._timeline,
        )
        return local(runtime, stream=torch.cuda.current_stream())

    def _k2(self, build, plan, core, prefix, blocks) -> torch.Tensor:
        weight = plan.moe
        local = build.k2[plan.layer_id]
        buffers = local.buffers
        stream = torch.cuda.current_stream()

        def oproj_local() -> torch.Tensor:
            local.oproj.launcher(
                core.data_ptr(),
                weight.w_o.data_ptr(),
                buffers["oproj"].data_ptr(),
                stream=stream,
            )
            return buffers["oproj"]

        def front_local(
            oproj_reduced: torch.Tensor,
        ) -> PackedShardedFrontPartials:
            router = weight.w_gate.narrow(
                0, self.rank * (EXPERTS // self.tp), EXPERTS // self.tp
            )
            latent = weight.w_ld.narrow(
                0, self.rank * (LATENT // self.tp), LATENT // self.tp
            )
            local.front.launcher(
                oproj_reduced.data_ptr(),
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
                buffers["x_local"].data_ptr(),
                buffers["router_local"].data_ptr(),
                buffers["latent_local"].data_ptr(),
                buffers["shared_gate_up"].data_ptr(),
                buffers["front_scratch"].data_ptr(),
                plan.layer_id,
                weight.seam.eps,
                weight.seam.out_eps,
                stream=stream,
            )
            torch.cat(
                (
                    buffers["router_local"].view(torch.uint8).flatten(),
                    buffers["latent_local"].view(torch.uint8).flatten(),
                ),
                out=buffers["front_payload"].view(-1),
            )
            return PackedShardedFrontPartials(buffers["front_payload"], buffers)

        def finish_local(gathered: torch.Tensor, retained: object) -> MoeLocalPartials:
            if retained is not buffers:
                raise ValueError("gfx1250 K2 retained buffer mismatch")
            local.finish.launcher(
                gathered.data_ptr(),
                weight.bias.data_ptr(),
                buffers["shared_gate_up"].data_ptr(),
                weight.w_sd.data_ptr(),
                weight.w13.data_ptr(),
                weight.w13s.data_ptr(),
                weight.w2.data_ptr(),
                weight.w2s.data_ptr(),
                buffers["latent_logical"].data_ptr(),
                buffers["shared_hidden"].data_ptr(),
                buffers["route"].data_ptr(),
                buffers["tile_route"].data_ptr(),
                buffers["flags"].data_ptr(),
                buffers["experts"].data_ptr(),
                buffers["kof"].data_ptr(),
                buffers["union_sizes"].data_ptr(),
                buffers["latent_fp4"].data_ptr(),
                buffers["latent_scales"].data_ptr(),
                buffers["routed_intermediate_fp4"].data_ptr(),
                buffers["routed_intermediate_scales"].data_ptr(),
                buffers["routed_partial"].data_ptr(),
                buffers["shared_partial"].data_ptr(),
                buffers["finish_scratch"].data_ptr(),
                stream=stream,
            )
            return MoeLocalPartials(
                buffers["routed_partial"],
                buffers["shared_partial"],
            )

        def finalize_local(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
            local.finalize.launcher(
                routed.data_ptr(),
                shared.data_ptr(),
                weight.ln_w.data_ptr(),
                weight.w_up.data_ptr(),
                buffers["final"].data_ptr(),
                self.rank,
                weight.ln_eps,
                stream=stream,
            )
            return buffers["final"]

        return local.runner.run(
            oproj_local,
            front_local,
            finish_local,
            finalize_local,
        )
