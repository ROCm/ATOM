# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER's grouped MXFP4 MoE (``grouped_gemm_gfx1250_a8w4``) for one gfx1250
decode width, on the lite finish's FP4 latent and precomputed HERD routes: the
same kernels, tiles and buffers, launched from here so a later kernel can take
gemm2's grouped rows and do the gather-reduce itself."""

from __future__ import annotations

import os

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.grouped_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked
from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _as_int,
    _find_grouped_config,
    _fit_tile_n,
    _get_padded_m,
    _grouped_weight_uint8,
    _tdm_align_up,
    flydsl_moe_gather_reduce,
    get_wmma_m_rep,
)
from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

from atom.models.kimi_k3.mono.config import EXPERTS, LATENT, TOPK
from atom.mono.runtime.consensus import MonoUnsupported

# the knobs ``grouped_gemm_gfx1250_a8w4`` reads off its CSV row, with its
# defaults: (name, CSV column, default)
_GEMM1_KNOBS = (
    ("tile_m", "tile_m", 64),
    ("tile_k", "tile_k", 128),
    ("m_warp", "m_warp", 1),
    ("n_warp", "n_warp", 4),
    ("num_buffers", "num_buffers", 2),
    ("cluster_m", "cluster_m", -1),
    ("cluster_n", "cluster_n", -1),
    ("waves_per_tensor_tdm", "waves_per_tensor_tdm", -1),
    ("next_stage_prefetch", "next_stage_prefetch", 0),
    ("tdm_as_in_prologue", "tdm_as_in_prologue", 0),
    ("tdm_b_th", "tdm_b_th", 0),
    ("lds_soa_load_interleave", "lds_soa_load_interleave", 0),
    ("rolled_drain", "rolled_drain", 0),
    ("a_preshuffle", "a_preshuffle", 0),
    ("persistent_workers", "persistent_workers", 0),
    ("persistent_workers2", "persistent_workers2", 0),
    ("cluster_m2", "cluster_m2", -1),
)


class GroupedMoe:
    """The grouped MoE of ``rows`` tokens: a1 gather (``quant_preshuffle`` of
    the prequantized latent), gemm1 with its fused SiTUv2 requant, gemm2 into
    grouped rows, and -- unless a later kernel takes them -- the gather-reduce
    into the routed output."""

    def __init__(self, rows: int, inter: int, device: torch.device) -> None:
        if os.environ.get("AITER_FLYDSL_DISABLE_GEMM1_REQUANT", "0") not in ("", "0"):
            raise MonoUnsupported("the grouped MoE needs gemm1's fused requant")
        self.rows, self.inter = rows, inter
        row = _find_grouped_config(
            token_num=_get_padded_m(rows),
            model_dim=LATENT,
            inter_dim=inter,
            experts=EXPERTS,
            topk=TOPK,
            activation=ActivationType.Situv2,
            dtype=torch.bfloat16,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            quant_type=QuantType.per_1x32,
            ep_fused=False,
        )
        if row is None:
            raise MonoUnsupported(f"no grouped MoE config at {rows} rows")
        knobs = {name: _as_int(row.get(column), d) for name, column, d in _GEMM1_KNOBS}
        knobs["tile_n"] = _as_int(row.get("tile_n"), knobs["n_warp"] * 64)
        for name, column in (
            ("tile_m2", "tile_m2"),
            ("tile_n2", "tile_n2"),
            ("tile_k2", "tile_k2"),
            ("m_warp2", "m_warp2"),
            ("n_warp2", "n_warp2"),
            ("num_buffers2", "num_buffer_stage2"),
            ("cluster_n2", "cluster_n2"),
            ("waves_per_tensor_tdm2", "waves_per_tensor_tdm2"),
        ):
            stage1 = name[:-1] if name != "cluster_n2" else "cluster_n"
            knobs[name] = _as_int(row.get(column), knobs[stage1])
        for k in (1, 2):
            knobs[f"k_waves{k}"] = max(1, _as_int(row.get(f"k_waves{k}"), 1))
        knobs["tile_n"], knobs["n_warp"] = _fit_tile_n(
            2 * inter, knobs["tile_n"], knobs["n_warp"]
        )
        knobs["tile_n2"], knobs["n_warp2"] = _fit_tile_n(
            LATENT, knobs["tile_n2"], knobs["n_warp2"]
        )
        self.knobs = knobs
        self.wmma_rep = get_wmma_m_rep(
            knobs["tile_m"], knobs["tile_n"], knobs["m_warp"], knobs["n_warp"], "gemm1"
        )
        self.wmma_rep2 = get_wmma_m_rep(
            knobs["tile_m2"],
            knobs["tile_n2"],
            knobs["m_warp2"],
            knobs["n_warp2"],
            "gemm2",
        )
        self.align_m = max(knobs["tile_m"], knobs["tile_m2"])
        if self.align_m % knobs["tile_m"] or self.align_m % knobs["tile_m2"]:
            raise MonoUnsupported(f"grouped MoE tiles at {rows} rows do not nest")
        routes = rows * TOPK
        self.contiguous_m = max(
            self.align_m,
            _tdm_align_up(
                min(
                    routes + EXPERTS * self.align_m - TOPK,
                    routes + min(EXPERTS, routes) * (self.align_m - 1),
                ),
                self.align_m,
            ),
        )
        self.a_preshuffled = bool(knobs["a_preshuffle"])
        self.device = device

    def __call__(
        self,
        latent_fp4,
        latent_scales,
        w13,
        w2,
        w13_scale,
        w2_scale,
        herd,
        out,
        situ_beta,
        situ_linear_beta,
        gather=True,
    ):
        """The routed output of ``rows`` tokens into ``out`` (``gather``), or
        gemm2's grouped rows and the (row, weight) routes a gather-reduce takes
        -- per-call buffers, as AITER's, held by the caller until that kernel
        is queued."""
        k = self.knobs
        m = self.contiguous_m
        row_to_token = (
            torch.full((m,), -1, dtype=torch.int32, device=self.device)
            if self.a_preshuffled
            else None
        )
        a2_payload = torch.empty(
            (1, m, self.inter // 2), dtype=torch.uint8, device=self.device
        )
        a2_scale = torch.empty(
            (1, m // self.wmma_rep2, (self.inter + 127) // 128 * 4 * self.wmma_rep2),
            dtype=torch.uint8,
            device=self.device,
        )
        grouped = torch.empty((1, m, LATENT), dtype=torch.bfloat16, device=self.device)
        topk_weight, _, _, topids_to_rows, psum = herd.route_psum(self.align_m)
        if latent_scales.dtype == torch.float8_e8m0fnu:
            latent_scales = latent_scales.view(torch.uint8)
        a1_payload, a1_scale = flydsl_moe_fused_quant_preshuffle(
            latent_fp4.reshape(1, self.rows, LATENT // 2),
            1,
            self.contiguous_m,
            wmma_rep=self.wmma_rep,
            quant_mode="fp4",
            topids_to_rows=topids_to_rows,
            source_topk=TOPK,
            prequantized_scale=latent_scales,
            row_to_token=row_to_token,
            payload_preshuffled=self.a_preshuffled,
        )
        common = {
            "n_experts": EXPERTS,
            "contiguous_m": self.contiguous_m,
            "out_is_f16": 0,
            "a_is_fp4": 1,
            "a_preshuffled": int(self.a_preshuffled),
            "next_stage_prefetch": k["next_stage_prefetch"],
            "tdm_as_in_prologue": k["tdm_as_in_prologue"],
            "tdm_b_th": k["tdm_b_th"],
            "lds_soa_load_interleave": k["lds_soa_load_interleave"],
            "rolled_drain": k["rolled_drain"],
        }
        flydsl_grouped_gemm_a8w4_masked(
            a2_payload,
            a1_payload,
            _grouped_weight_uint8(w13),
            a1_scale,
            w13_scale.reshape(-1).view(torch.int32),
            psum,
            N=2 * self.inter,
            K=LATENT,
            tile_m=k["tile_m"],
            tile_n=k["tile_n"],
            tile_k=k["tile_k"],
            m_warp=k["m_warp"],
            n_warp=k["n_warp"],
            stage1_act=3,
            bias=None,
            swiglu_limit=float("inf"),
            num_buffers=k["num_buffers"],
            stage1_quant_out=1,
            quant_scale=a2_scale,
            quant_wmma_rep=self.wmma_rep2,
            persistent_workers=k["persistent_workers"],
            cluster_m=k["cluster_m"],
            cluster_n=k["cluster_n"],
            waves_per_tensor_tdm=k["waves_per_tensor_tdm"],
            k_waves=k["k_waves1"],
            row_major_ascale=0,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            **common,
        )
        flydsl_grouped_gemm_a8w4_masked(
            grouped,
            a2_payload,
            _grouped_weight_uint8(w2),
            a2_scale,
            w2_scale.reshape(-1).view(torch.int32),
            psum,
            N=LATENT,
            K=self.inter,
            tile_m=k["tile_m2"],
            tile_n=k["tile_n2"],
            tile_k=k["tile_k2"],
            m_warp=k["m_warp2"],
            n_warp=k["n_warp2"],
            stage1_act=0,
            bias=None,
            num_buffers=k["num_buffers2"],
            persistent_workers=k["persistent_workers2"],
            cluster_m=k["cluster_m2"],
            cluster_n=k["cluster_n2"],
            waves_per_tensor_tdm=k["waves_per_tensor_tdm2"],
            k_waves=k["k_waves2"],
            **common,
        )
        gather_w = topk_weight.contiguous()
        if gather:
            flydsl_moe_gather_reduce(grouped, topids_to_rows, gather_w, out=out)
        return grouped, topids_to_rows, gather_w
