# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Prepared launch binding for one DeepSeek-V4 Mono layer."""

from __future__ import annotations

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.deepseek_v4.mono.config import (
    MAX_LAYERS_PER_STEP,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.kernels.plan import TL_COLS, stage_tasks
from atom.models.deepseek_v4.mono.state import Dsv4Variant
from atom.models.deepseek_v4.mono.weights import LayerWeights

__all__ = ["Dsv4MonoKernel"]


def _check_i32(name: str, t: torch.Tensor, shape: tuple) -> None:
    """Index tensors are read by raw buffer loads as contiguous int32."""
    if tuple(t.shape) != shape or t.dtype != torch.int32 or not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous int32 {shape}, got {tuple(t.shape)} {t.dtype}"
        )


class Dsv4MonoKernel:
    """One rank of one transformer layer: its weights and rolling state.

    Layers sharing a ``Dsv4Variant`` need distinct ``layer`` values and one step advance per decode step.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int,
        npes: int,
        *,
        variant: Dsv4Variant,
        packed: dict,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
    ):
        """The runner owns prepared variants, packed weights and native state.

        This binding never allocates a synthetic cache or owns IPC mappings.
        """
        tokens_per_seq = variant.tokens_per_seq
        cfg = W.cfg
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if samples == 14 and (tokens_per_seq != 7 or variant.launch_shape != "three"):
            raise ValueError(
                "candidate S14 requires an explicit q7 three-phase variant"
            )
        if cfg.hc_mult > 1 and cfg.hc_mult & (cfg.hc_mult - 1):
            raise ValueError(f"hc_mult must be 1 or a power of two, got {cfg.hc_mult}")
        self.moe_mode = as_moe_mode(moe_mode)
        # the kernel reads N_EXPERTS f32; a bf16 bias would be read past its end
        if W.t["bias"].dtype != torch.float32:
            raise ValueError(
                f"the router bias must be float32, got {W.t['bias'].dtype}"
            )
        self.W, self.S, self.rank = W, samples, rank
        self.packed = packed
        # [3 scales | hc_mix bases] per side, as one f32 vector the kernel indexes
        dev0 = torch.device("cuda", torch.cuda.current_device())
        self.hc_sb = {}
        for side in ("attn", "ffn"):
            if f"hc_{side}_scale" in W.t:
                self.hc_sb[side] = torch.cat(
                    [W.t[f"hc_{side}_scale"].float(), W.t[f"hc_{side}_base"].float()]
                ).contiguous()
            else:
                self.hc_sb[side] = torch.zeros(1, dtype=torch.float32, device=dev0)
        dev = torch.device("cuda", torch.cuda.current_device())
        if variant.key != _variant_key(
            cfg,
            samples,
            npes,
            self.moe_mode,
            timeline,
            variant.poll_timeout_us,
            variant.tokens_per_seq,
            variant.launch_shape,
            variant.qkv_batch_limit,
            variant.qkv_samples_per_pass,
            variant.qkv_norm_handoff,
            variant.attention_mode,
        ):
            raise ValueError(
                "this layer's shape is not the one the variant was compiled for; a variant is shared "
                "only by layers of the SAME attention variant (compress_ratio, n_keys, max_seq, ...)"
            )
        self.variant = variant
        self.scr_layout = variant.scr_layout
        self.scratch, self.sym, self.peers = variant.scratch, variant.sym, variant.peers
        refresh = cfg.index_topk_refresh if cfg.indexed else True
        self.launches = variant.launch_sets[refresh]
        self.stages = stage_tasks(
            samples,
            cfg.heads,
            window=cfg.window,
            qkv_norm_handoff=variant.qkv_norm_handoff,
            index_topk_refresh=refresh,
            **shape_dims(cfg),
        )
        self.tokens_per_seq = tokens_per_seq
        # Inactive state pointers still occupy ABI slots. Real cache/state and
        # block tables always come from the runner's native cache adapter.
        self._unused_state = torch.zeros(1, dtype=torch.float32, device=dev)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = (
            torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev)
            if timeline
            else None
        )
        self.step = variant.step

    def validate_inputs(
        self,
        h,
        cur_pos,
        kv_cache,
        dest_rows,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        tokens=None,
        block_tables=None,
        state=None,
    ):
        """Validate the entire pointer/geometry contract without launching."""
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
        if cfg.kv_fp8:
            # ATOM's fp8 layout: (NoPE plane uint8 [rows, 512], RoPE plane bf16 [rows, rope_dim])
            kv_nope, kv_rope = kv_cache
            if (
                kv_nope.ndim != 2
                or kv_rope.ndim != 2
                or kv_nope.dtype != torch.uint8
                or kv_rope.dtype != torch.bfloat16
                or kv_nope.shape[1] != 512
                or kv_rope.shape[1] != cfg.rope_dim
                or kv_nope.shape[0] != kv_rope.shape[0]
                or not kv_nope.is_contiguous()
                or not kv_rope.is_contiguous()
            ):
                raise ValueError(
                    f"kv_fp8 wants (uint8 [rows, 512], bf16 [rows, {cfg.rope_dim}]), "
                    f"got {kv_nope.dtype} {tuple(kv_nope.shape)}, {tuple(kv_rope.shape)}"
                )
        else:
            kv_nope, kv_rope = kv_cache, None
            if (
                kv_cache.ndim != 2
                or kv_cache.shape[1] != cfg.head_dim
                or kv_cache.dtype != torch.bfloat16
                or not kv_cache.is_contiguous()
            ):
                raise ValueError(
                    f"kv_cache must be one plane [rows, {cfg.head_dim}], got {tuple(kv_cache.shape)} -- "
                    "which rows a sequence owns is the caller's, supplied through indices and dest_rows"
                )
        for name, table in (("cos", cos), ("sin", sin)):
            if (
                table.ndim != 2
                or table.shape[1] != cfg.rope_dim // 2
                or table.dtype != torch.float32
                or not table.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous float32 [positions, {cfg.rope_dim // 2}] table"
                )
        _check_i32("dest_rows", dest_rows, (2, self.S))
        _check_i32("indices", indices, (self.S, cfg.n_keys))
        # one position per sample: the samples are independent sequences
        _check_i32("cur_pos", cur_pos, (self.S,))
        if block_tables is not None and (
            block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or block_tables.shape[0] != self.S
            or block_tables.stride(1) != 1
        ):
            raise ValueError(
                f"block_tables must be int32 [{self.S}, blocks] with unit-stride rows, "
                f"got {tuple(block_tables.shape)} {block_tables.dtype} strides {block_tables.stride()}"
            )
        t = dict(self.W.t, **self.packed)
        if block_tables is None:
            raise ValueError("the native block table is required")
        bt = block_tables
        st = dict.fromkeys(
            (
                "kv_state",
                "score_state",
                "i_kv_state",
                "i_score_state",
                "i_cache",
                "i_cache_s",
            ),
            self._unused_state,
        )
        st.update(dict.fromkeys(("st_kv", "st_sc", "st_i", "st_isc", "st_ic"), 0))
        state = state or {}
        unknown = set(state) - set(st) - {"state_slots", "state_slots_in"}
        if unknown:
            raise ValueError(f"unknown state keys {sorted(unknown)}")
        st.update(state)
        # A split native arena can place values and scores in different
        # planes. Bind each slot stride independently; never reinterpret
        # the score pool using the value pool's address increment.
        for tensor_name, stride_name in (
            ("score_state", "st_sc"),
            ("i_score_state", "st_isc"),
        ):
            if tensor_name in state and stride_name not in state:
                st[stride_name] = state[tensor_name].stride(0)
        # V4 verify may read a different prefix/input slot than it writes.
        if "state_slots" not in st:
            raise ValueError("the native state slots are required")
        st.setdefault("state_slots_in", st["state_slots"])
        need = (
            {"kv_state", "score_state", "state_slots", "st_kv"}
            if cfg.compress_ratio
            else set()
        )
        if cfg.indexed and cfg.index_topk_refresh:
            need |= {
                "i_kv_state",
                "i_score_state",
                "i_cache",
                "i_cache_s",
                "st_i",
                "st_ic",
            }
        missing = need - set(state or ())
        if missing:
            raise ValueError(f"native cache binding needs state {sorted(missing)}")
        _check_i32("state_slots", st["state_slots"], (self.S,))
        _check_i32("state_slots_in", st["state_slots_in"], (self.S,))
        for prefix, stride_name, score_stride_name, width, used in (
            ("", "st_kv", "st_sc", cfg.c_coff * cfg.head_dim, bool(cfg.compress_ratio)),
            (
                "i_",
                "st_i",
                "st_isc",
                cfg.c_coff * cfg.index_head_dim,
                cfg.indexed and cfg.index_topk_refresh,
            ),
        ):
            if not used:
                continue
            ring = cfg.state_ring_rows(self.tokens_per_seq)
            values = st[prefix + "kv_state"]
            scores = st[prefix + "score_state"]
            if (
                values.ndim != 3
                or values.dtype != torch.float32
                or scores.dtype != torch.float32
                or values.shape != scores.shape
                or tuple(values.shape[1:]) != (ring, width)
                or values.stride()[1:] != (width, 1)
                or scores.stride()[1:] != (width, 1)
                or st[stride_name] != values.stride(0)
                or st[score_stride_name] != scores.stride(0)
                or any(
                    not ring * width <= tensor.stride(0) < (1 << 31)
                    for tensor in (values, scores)
                )
            ):
                raise ValueError(
                    f"{prefix}rolling state must match the compiled ring/slot stride ({ring}, {width})"
                )
        use_hash = "tid2eid" in t
        if use_hash:
            if tokens is None:
                raise ValueError(
                    f"a hash-routed layer needs tokens: int32, one per sample ({self.S})"
                )
            _check_i32("tokens", tokens, (self.S,))
        if x_out is None:
            x_out = torch.empty(*hshape, dtype=torch.bfloat16, device=h.device)
        elif (
            tuple(x_out.shape) != hshape
            or x_out.dtype != torch.bfloat16
            or not x_out.is_contiguous()
        ):
            raise ValueError(
                f"x_out must be contiguous bfloat16 {hshape}, got {tuple(x_out.shape)} {x_out.dtype}"
            )
        tensors = [h, cur_pos, kv_nope, dest_rows, indices, cos, sin, x_out, bt]
        tensors += [v for v in st.values() if isinstance(v, torch.Tensor)]
        tensors += [v for v in t.values() if isinstance(v, torch.Tensor)]
        tensors += [v for v in (kv_rope, tokens) if v is not None]
        if h.device.type != "cuda" or any(
            v.device != self.scratch.device for v in tensors
        ):
            raise ValueError(
                "all raw-pointer inputs must reside on the prepared CUDA device"
            )
        return cfg, kv_nope, kv_rope, t, bt, st, use_hash, x_out

    def _forward_validated(
        self,
        h,
        cur_pos,
        dest_rows,
        indices,
        cos,
        sin,
        *,
        validated,
        layer,
        tokens,
        env_rows,
    ):
        """Launch a resolved contract from this call's complete-stack preflight.

        Private to the runner: never retain contracts across
        calls or expose a serving flag that bypasses input validation.
        """
        cfg, kv_nope, kv_rope, t, bt, st, use_hash, x_out = validated
        p = lambda x: x.data_ptr()
        args = (
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_nope),
            p(kv_rope) if kv_rope is not None else 0,
            p(dest_rows),
            p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["attn_sink"]),
            p(t["ape"]) if "ape" in t else 0,
            p(t["g_ckv"]) if "g_ckv" in t else 0,
            p(st["kv_state"]),
            p(st["score_state"]),
            p(t["i_ape"]) if "i_ape" in t else 0,
            p(t["g_ickv"]) if "g_ickv" in t else 0,
            p(st["i_kv_state"]),
            p(st["i_score_state"]),
            p(st["i_cache"]),
            p(t["hc_attn_fn"]) if "hc_attn_fn" in t else 0,
            p(self.hc_sb["attn"]),
            p(t["hc_ffn_fn"]) if "hc_ffn_fn" in t else 0,
            p(self.hc_sb["ffn"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_qkv_c"]) if "w_qkv_c" in t else 0,
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_i_q_b"]) if "w_i_q_b" in t else 0,
            p(t["s_i_q_b"]) if "s_i_q_b" in t else 0,
            p(t["i_w"]) if "i_w" in t else 0,
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
            p(t["w_sug"]) if "w_sug" in t else 0,
            p(t["s_sug"]) if "s_sug" in t else 0,
            p(t["w_sdn"]) if "w_sdn" in t else 0,
            p(t["s_sdn"]) if "s_sdn" in t else 0,
            p(self.scratch),
            self.sym,
            p(self.peers),
            0 if self.timeline is None else p(self.timeline),
            p(self.step),
            p(self.variant.hang),
            p(st["state_slots"]),
            p(st["state_slots_in"]),
            p(tokens) if use_hash else 0,
            p(t["tid2eid"]) if use_hash else 0,
            p(bt),
            p(st["i_cache_s"]),
            self.rank,
            layer,
            st["st_kv"],
            st["st_sc"],
            st["st_i"],
            st["st_isc"],
            st["st_ic"],
            int(use_hash),
            bt.stride(0),
            env_rows,
            cfg.index_topk_source_layer,
        )
        for launch in self.launches:
            _run_compiled(launch, *args, torch.cuda.current_stream())
        return x_out
