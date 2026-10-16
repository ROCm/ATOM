# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's V4 layer."""

from __future__ import annotations

from typing import ClassVar

import torch

from atom.models.deepseek_v4.mono.config import (
    MAX_LAYERS_PER_STEP,
    validate_shard,
)
from atom.models.deepseek_v4.mono.kernels.layer import (
    build_advance_step,
    build_dsv4_kernel,
)
from atom.models.deepseek_v4.mono.layout import layout
from atom.models.deepseek_v4.mono.weights import LayerWeights
from atom.mono.runtime.peer_memory import PeerBuffer

__all__ = ["Dsv4MonoKernel", "Dsv4Variant", "shape_dims"]


def shape_dims(cfg) -> dict:
    """The shape arguments layout() and build_dsv4_kernel() must share identically."""
    return {
        "hidden": cfg.hidden,
        "q_lora": cfg.q_lora,
        "head_dim": cfg.head_dim,
        "o_groups": cfg.o_groups,
        "o_lora": cfg.o_lora,
        "hc_mult": cfg.hc_mult,
        "compress_ratio": cfg.compress_ratio,
        "n_keys": cfg.n_keys,
        "c_coff": cfg.c_coff,
        # 0: no indexer (only CSA has one)
        "index_head_dim": cfg.index_head_dim if cfg.indexed else 0,
        "index_heads": cfg.index_heads if cfg.indexed else 0,
        "max_seq": cfg.max_seq,
        "index_topk": cfg.index_topk if cfg.indexed else 0,
        "indexer_hadamard": cfg.indexer_hadamard,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "inter": cfg.inter,
    }


def _check_i32(name: str, t: torch.Tensor, shape: tuple) -> None:
    """Index tensors are read by raw buffer loads as contiguous int32."""
    if tuple(t.shape) != shape or t.dtype != torch.int32 or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous int32 {shape}, got {tuple(t.shape)} {t.dtype}"
        )


def _variant_key(cfg, samples, npes, spec_steps):
    """Everything the compiled kernel and the scratch layout depend on."""
    return (
        spec_steps,
        tuple(sorted(shape_dims(cfg).items())),
        samples,
        cfg.heads,
        npes,
        cfg.window,
        cfg.cache_rows,
        cfg.softmax_scale,
        cfg.swiglu_limit,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    )


class Dsv4Variant:
    """The compiled kernel, scratch and symmetric buffer shared by all layers of one attention variant.

    Scratch cannot be shared across variants (different mailbox layouts). Within one, fresh epoch
    tags make sharing safe, so ``step`` lives here: one counter per scratch, advanced once per step.
    """

    _cache: ClassVar[dict] = {}
    _adv_cache: ClassVar[dict] = {}

    def __init__(
        self,
        cfg,
        samples: int,
        rank: int,
        npes: int,
        group,
        spec_steps: int = 0,
    ):
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        self.key = _variant_key(cfg, samples, npes, spec_steps)
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(
            samples, cfg.heads, npes, cfg.window, **dims
        )
        self.scratch = torch.zeros(
            self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
        )
        # collective over ``group`` (a CPU group): every rank builds its variants alike
        self.peer_buffer = PeerBuffer(self.sym_layout["_bytes"], group, rank, npes, dev)
        # the tags start from 1: an unwritten pair must read tag 0
        self.peer_buffer.bytes.zero_()
        self.sym = self.peer_buffer.local
        self.peers = self.peer_buffer.addresses
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter
        built = Dsv4Variant._cache.get(self.key)
        if built is None:
            built = build_dsv4_kernel(
                samples,
                cfg.heads,
                npes,
                cfg.window,
                scale=cfg.softmax_scale,
                spec_steps=spec_steps,
                swiglu_limit=cfg.swiglu_limit,
                hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                hc_eps=cfg.hc_eps,
                window_rows=cfg.cache_rows,
                **dims,
            )
            Dsv4Variant._cache[self.key] = built
        self.launch = built
        # the step advance clears mailbox pairs: every scratch and symmetric region is pairs
        adv_key = (self.scr_layout["_bytes"] // 8, self.sym_layout["_bytes"] // 8)
        if adv_key not in Dsv4Variant._adv_cache:
            Dsv4Variant._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Dsv4Variant._adv_cache[adv_key]

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs so tags can wrap; graph-capturable."""
        self._advance(
            self.step.data_ptr(),
            self.scratch.data_ptr(),
            self.sym,
            stream=torch.cuda.current_stream(),
        )


class Dsv4MonoKernel:
    """One rank of one transformer layer over a shared ``Dsv4Variant``.

    Layers sharing a variant need distinct ``layer`` values and one step advance per decode step.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int,
        npes: int,
        variant: Dsv4Variant,
        packed: dict,
        spec_steps: int = 0,
    ):
        """``packed``: the layer's ``pack_layer_weights`` output."""
        cfg = W.cfg
        if cfg.hc_mult > 1 and cfg.hc_mult & (cfg.hc_mult - 1):
            raise ValueError(f"hc_mult must be 1 or a power of two, got {cfg.hc_mult}")
        if variant.key != _variant_key(cfg, samples, npes, spec_steps):
            raise ValueError(
                "this layer's shape is not the one the variant was compiled for; a variant is shared "
                "only by layers of the SAME attention variant (compress_ratio, n_keys, max_seq, ...)"
            )
        # the kernel reads N_EXPERTS f32; a bf16 bias would be read past its end
        if W.t["bias"].dtype != torch.float32:
            raise ValueError(
                f"the router bias must be float32, got {W.t['bias'].dtype}"
            )
        self.W, self.S, self.rank = W, samples, rank
        self.variant = variant
        self.t = dict(W.t, **packed)
        # [3 scales | hc_mix bases] per side, as one f32 vector the kernel indexes
        dev = torch.device("cuda", torch.cuda.current_device())
        self.hc_sb = {}
        for side in ("attn", "ffn"):
            if f"hc_{side}_scale" in W.t:
                self.hc_sb[side] = torch.cat(
                    [W.t[f"hc_{side}_scale"].float(), W.t[f"hc_{side}_base"].float()]
                ).contiguous()
            else:
                self.hc_sb[side] = torch.zeros(1, dtype=torch.float32, device=dev)

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        dest_rows,
        indices,
        cos,
        sin,
        layer,
        tokens,
        block_tables,
        env_rows,
        state,
        run_pos,
    ):
        """One layer; the epoch tag is ``step * MAX_LAYERS_PER_STEP + layer + 1``. Graph-capturable.

        ``state``: the serving engine's compressor (and indexer) state, slots and strides.
        """
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(
                f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}"
            )
        cfg = self.W.cfg
        # offsets derive from self.S: a wrong leading dim reads out of bounds
        hshape = (
            (self.S, cfg.hidden)
            if cfg.hc_mult == 1
            else (self.S, cfg.hc_mult, cfg.hidden)
        )
        if (
            tuple(h.shape) != hshape
            or h.dtype != torch.bfloat16
            or not h.is_contiguous()
        ):
            raise ValueError(
                f"h must be contiguous bfloat16 {hshape}, got {tuple(h.shape)} {h.dtype}"
            )
        # ATOM's fp8 layout: (NoPE plane uint8 [rows, 512], RoPE plane bf16 [rows, rope_dim])
        kv_nope, kv_rope = kv_cache
        if (
            kv_nope.dtype != torch.uint8
            or kv_nope.shape[-1] != 512
            or kv_rope.shape[-1] != cfg.rope_dim
        ):
            raise ValueError(
                f"kv_cache wants (uint8 [rows, 512], bf16 [rows, {cfg.rope_dim}]), "
                f"got {kv_nope.dtype} {tuple(kv_nope.shape)}, {tuple(kv_rope.shape)}"
            )
        _check_i32("dest_rows", dest_rows, (2, self.S))
        _check_i32("indices", indices, (self.S, cfg.n_keys))
        # one position per sample: the samples are independent sequences
        _check_i32("cur_pos", cur_pos, (self.S,))
        _check_i32("run_pos", run_pos, (self.S,))
        if (
            block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or block_tables.shape[0] != self.S
            or block_tables.stride(1) != 1
        ):
            raise ValueError(
                f"block_tables must be int32 [{self.S}, blocks] with unit-stride rows, "
                f"got {tuple(block_tables.shape)} {block_tables.dtype} strides {block_tables.stride()}"
            )
        need = {"state_slots", "state_slots_in"}
        if cfg.compress_ratio:
            need |= {"kv_state", "score_state", "st_kv"}
        if cfg.indexed:
            need |= {
                "i_kv_state",
                "i_score_state",
                "i_cache",
                "i_cache_s",
                "st_i",
            }
        missing = need - set(state)
        if missing:
            raise ValueError(f"forward needs state {sorted(missing)}")
        t = self.t
        use_hash = "tid2eid" in t
        if use_hash:
            _check_i32("tokens", tokens, (self.S,))
        x_out = torch.empty(*hshape, dtype=torch.bfloat16, device=h.device)
        var = self.variant
        p = lambda x: x.data_ptr()
        q = lambda d, k: p(d[k]) if k in d else 0
        var.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_nope),
            p(kv_rope),
            p(dest_rows),
            p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["attn_sink"]),
            q(t, "ape"),
            q(t, "g_ckv"),
            q(state, "kv_state"),
            q(state, "score_state"),
            q(t, "i_ape"),
            q(t, "g_ickv"),
            q(state, "i_kv_state"),
            q(state, "i_score_state"),
            q(state, "i_cache"),
            q(t, "hc_attn_fn"),
            p(self.hc_sb["attn"]),
            q(t, "hc_ffn_fn"),
            p(self.hc_sb["ffn"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            q(t, "w_qkv_c"),
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            q(t, "w_i_q_b"),
            q(t, "s_i_q_b"),
            q(t, "i_w"),
            p(t["w_o_a"]),
            p(t["s_o_a"]),
            p(t["w_o_b"]),
            p(t["s_o_b"]),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            q(t, "w_sug"),
            q(t, "s_sug"),
            q(t, "w_sdn"),
            q(t, "s_sdn"),
            p(var.scratch),
            var.sym,
            p(var.peers),
            p(var.step),
            p(state["state_slots"]),
            p(state["state_slots_in"]),
            p(tokens) if use_hash else 0,
            q(t, "tid2eid"),
            p(block_tables),
            q(state, "i_cache_s"),
            self.rank,
            layer,
            state.get("st_kv", 0),
            state.get("st_i", 0),
            int(use_hash),
            block_tables.stride(0),
            env_rows,
            p(run_pos),
            stream=torch.cuda.current_stream(),
        )
        return x_out
