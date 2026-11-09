# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 QKV sites. The forward of each eligible attention module is
replaced, on the instance and before the first trace, by one splitting op
covering the QKV GEMM through the head gather, followed by ``o_proj``."""

import logging
import types

import torch
from aiter import dtypes

from atom.config import get_current_atom_config
from atom.distributed.ulysses_sp import sp_head_shards
from atom.utils import mark_spliting_op

from .site import QkvSite, rows2d, static_weight_reason

logger = logging.getLogger("atom")


def _site(layer_name):
    ctx = get_current_atom_config().compilation_config.static_forward_context
    return ctx[layer_name].gemm_fusion_site


def _fake(x, x_scale, positions, layer_name, scale):
    width = _site(layer_name).out_width
    return torch.empty((x.shape[0], width), device=x.device, dtype=dtypes.fp8)


# Like sp_attention_fp8: per-token FP8 output, its scales written to `scale`.
@mark_spliting_op(is_custom=True, gen_fake=_fake, mutates_args=["scale"])
def gemm_fusion_qkv_attention(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _site(layer_name).run(x, x_scale, positions, scale)


def _forward(self, positions, hidden_states, hidden_states_scale=None):
    scale = hidden_states.new_empty((hidden_states.shape[0], 1), dtype=torch.float32)
    out = torch.ops.aiter.gemm_fusion_qkv_attention(
        hidden_states, hidden_states_scale, positions, self.attn.layer_name, scale
    )
    return self.o_proj(out, scale)


def _fields(gf, module, world):
    hd = module.head_dim
    widths = [
        ("q", module.num_heads * hd, hd),
        ("k", module.num_kv_heads * hd, hd),
        ("v", module.num_kv_heads * hd, hd),
    ]
    if hasattr(module, "idx_head_dim"):
        idx = module.idx_head_dim
        widths += [("iq", module.num_idx_heads * idx, idx), ("ik", idx, idx)]
    fields, offset = [], 0
    for name, width, head in widths:
        shards = sp_head_shards(width // head, world)[1]
        fields.append(gf.Field(name, offset, width, shards))
        offset += width
    return tuple(fields), offset


def _spec(gf, module, mode, world, page):
    impl = module.attn.impl
    return gf.FusionSpec(
        gf.GemmSpec(n=module.qkv_proj.output_size, k=module.qkv_proj.input_size),
        (
            gf.SPExchange(
                world=world,
                fields=_fields(gf, module, world)[0],
                drop=("iq", "ik") if mode == "reuse" else (),
            ),
            gf.HeadNormRopeCache(
                head_dim=impl.head_dim,
                q_heads=impl.num_heads,
                kv_heads=impl.num_kv_heads,
                rotary_dim=impl.rotary_emb.rotary_dim,
                rope="neox",
                eps=float(module.q_norm.variance_epsilon),
                norm="gemma_rms",
                numerics=(
                    "aiter_fused_qknorm_idxrqknorm_v1"
                    if mode == "indexer"
                    else "triton_norm_rope_cache_v1"
                ),
                kv_cache=f"shuffle_page{page}_{impl.kv_cache_dtype}_ptok",
                index="fp8_unit_page128" if mode == "indexer" else None,
            ),
        ),
    )


def _skip_reason(module, mode, offset):
    from atom.models.minimax_m3.model import _linear_consumes_per_token_fp8

    attn, impl = module.attn, module.attn.impl
    rot = impl.rotary_emb
    if offset != module.qkv_proj.output_size:
        return "field_map"
    if not attn.per_token_fp8_out or not _linear_consumes_per_token_fp8(module.o_proj):
        return "o_proj_input"
    if not impl.gemm_fusion_prologue_supported() or not rot.is_neox_style:
        return "prologue"
    cos = getattr(rot, "cos_cache", None)
    if (
        cos is None
        or cos.dtype != torch.bfloat16
        or 2 * rows2d(cos).shape[1] != rot.rotary_dim
    ):
        return "rope_cache"
    if mode != "dense" and impl.indexer_cp_group is not None:
        return "indexer_cp"
    if mode == "indexer" and (
        impl.index_cache_dtype != "fp8" or impl.index_head_dim != 128
    ):
        return "index_cache"
    return static_weight_reason(module.qkv_proj)


def install_minimax_m3(model, config, world, gf, runtime):
    """Install a site on every eligible attention module; returns the sites."""
    from atom.models.minimax_m3.model import (
        MiniMaxM3Attention,
        MiniMaxM3SparseAttention,
    )

    sites = []
    for module in model.modules():
        if isinstance(module, MiniMaxM3SparseAttention):
            mode = "reuse" if module.skip_index_topk else "indexer"
        elif isinstance(module, MiniMaxM3Attention):
            mode = "dense"
        else:
            continue
        attn = module.attn
        name = attn.layer_name.rsplit(".", 1)[0] + ".qkv"
        offset = _fields(gf, module, world)[1]
        reason = _skip_reason(module, mode, offset)
        if reason is not None:
            logger.info("gemm_fusion %s: native (%s)", name, reason)
            continue
        page = config.kv_cache_block_size if mode == "dense" else 16
        spec = _spec(gf, module, mode, world, page)
        site = QkvSite(
            name,
            attn,
            module.qkv_proj,
            spec,
            mode,
            runtime,
            out_width=module.num_heads * module.head_dim,
        )
        attn.gemm_fusion_site = site
        module.forward = types.MethodType(_forward, module)
        sites.append(site)
    return sites
