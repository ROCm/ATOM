# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 decode with a resident gfx1250 K1 and production K2.

K1 fuses the attention-residual seam, projections, convolution and recurrent
attention into one rank-local launch.  The output projection and latent MoE
remain on their production paths, retaining K3's tuned FP8/BF16 GEMMs, expert
kernels, collective selection and communication/compute overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.kimi_k3.mono.config import (
    GFX1250_BLOCKS,
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
from atom.models.kimi_k3.mono.kernels.token_tiling import GFX1250_MAX_TOKENS
from atom.models.kimi_k3.mono.plan import _width, kda_metadata, width_key
from atom.models.kimi_k3.mono.runner import K3MonoDecodeRunner, state_slots
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import launcher_resources
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


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
        {layer: replacements.get(layer, runner) for layer, runner in built.k1.items()}
    )


@dataclass(frozen=True)
class Gfx1250WidthBuild:
    k1: dict[int, Gfx1250K1LocalRunner]

    def kernels(self) -> list[tuple[object, KernelAbi]]:
        """All local launchers, for compile-time scratch validation."""
        pairs = [
            (k1.entrypoint.launcher, k1.entrypoint.abi)
            for k1 in self.k1.values()
        ]

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
                f"gfx1250 packed mono needs {GFX1250_BLOCKS} CUs, " f"got {self.blocks}"
            )

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

        for layer in self.plans:
            if layer.kda is not None:
                weight = layer.kda
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
                    local = (
                        self._plan(rows, weight.blocks)
                        .make_k1_local_runner(spec)
                        .prepare(self.device)
                    )
                    k1_cache[cache_key] = local
                k1_by_layer[layer.layer_id] = local

        return Gfx1250WidthBuild(k1_by_layer)

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
        if not plan.k1_only:
            out = self._fallback(plan, positions, prefix, blocks, add, add2)
        else:
            delta = add if add2 is None else (add2 if add is None else add + add2)
            core = self._k1(build, plan, prefix, delta, blocks, metadata)
            layer = self.inner.layers[plan.layer_id]
            attn_out = layer.self_attn.o_proj(core)

            # K1 already banked a closing block. The production MLP seam must
            # therefore restart its prefix exactly as DecoderLayer.forward does.
            weight = plan.kda
            reset = weight.write_idx >= 0
            normed, prefix = layer.mlp_attn_res(
                None if reset else prefix,
                blocks[:, : weight.blocks + reset],
                attn_out,
            )
            add, add2 = layer._ffn(normed)
            out = (prefix, add, add2)

        rows = blocks.size(0)
        widths = {tensor.size(0) for tensor in out if tensor is not None}
        if widths != {rows}:
            raise MonoUnsupported(
                f"layer {plan.layer_id} returned rows {sorted(widths)} for "
                f"an {rows}-row step"
            )
        self._write_aux(plan.layer_id, out)
        return out

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
        )
        return local(runtime, stream=torch.cuda.current_stream())
