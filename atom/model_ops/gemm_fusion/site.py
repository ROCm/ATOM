# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 SP4 QKV prologue through the neutral routed ``sp_qkv_fused`` op.

#3084's gemm_fusion plan/bucket machinery is gone; this module keeps only the
per-layer site hook, the attention tail and #3084's non-fused fallback.  A
site's forward either
* serves the call from the installed AITER provider (``aiter.sp_qkv_fused``
  after an init/first-use ``aiter.plugin.explain`` gate), or
* takes #3084's ordinary non-fused path (``qkv_proj`` + ``ulysses_attention``
  + head gather) when the provider declines that (mode, M) class.
The choice is made per ``(site, mode, M)`` outside graph capture; there is no
try/except around a fused forward and no ATOM bucket planning -- padding is the
provider's job.
"""

import logging
import types

import torch

from atom.config import get_current_atom_config
from atom.utils import mark_spliting_op
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_OP_NAME = "sp_qkv_fused"
_SP_WORLD = 4

# One output set per (device, mode, geometry, rows), shared by all sites.  The
# provider stages its own padded outputs and copies the real rows into these
# caller buffers; caller pointers are not part of its binding, and the buffer is
# allocated only for a cell explain() actually selects (the provider owns the
# coverage/padding decision, ATOM owns no bucket planning).
_OUTPUTS: dict[tuple, dict] = {}


def rows2d(t):
    """``[max_pos, 1, 1, rot/2]`` rotary caches as ``[max_pos, rot/2]``."""
    return t if t.dim() == 2 else t.reshape(t.shape[0], -1)


def static_weight_reason(lin):
    """Why the neutral contract cannot take the QKV weight as is, or None."""
    w = lin.weight
    if getattr(w, "is_shuffled", False) is not True:
        return "weight_layout"
    n, k = lin.output_size, lin.input_size
    if w.dim() != 2 or w.shape[1] != k:
        return "weight_shape"
    if w.shape[0] < n or (w.shape[0] != n and not lin.is_output_padded):
        return "weight_shape"
    scale = getattr(lin, "weight_scale", None)
    if scale is None or scale.shape[0] < n:
        return "weight_shape"
    return None


def _sp_group():
    """The SP group's torch.distributed ProcessGroup (the op's execution context)."""
    from aiter.dist.parallel_state import get_pcp_group

    return get_pcp_group().device_group


def _aiter():
    import aiter

    return aiter


# Process-local: the site-side enablement is idempotent at the point of use.
_ENSURE_DONE = False


def _ensure_plugin_enabled(aiter, plugin, enabled):
    """Make this process's aiter.plugin configuration match the deployment.

    ModelRunner configures before model build, but the forward path can run in a
    worker/child context that never saw that call (fork/spawn, model
    re-instantiation).  The aiter interface has one process-local config, so the
    site re-asserts it once, at the first routed call, and logs the resulting
    ``enabled``/``routed_ops`` so an empty case is loud.
    """
    global _ENSURE_DONE
    status = None
    try:
        # A lazy-op export registers its @route entry on first attribute access.
        getattr(aiter, _OP_NAME)
        status = plugin.status()
        if not tuple(status.get("enabled", ()) or ()) and enabled:
            from atom.aiter_provider import atom_version

            plugin.configure(source=f"atom {atom_version()}", enable=enabled)
            status = plugin.status()
    except Exception:
        logger.warning(
            "gemm_fusion: could not re-assert aiter provider enablement; "
            "explain() will decide",
            exc_info=True,
        )
    if not _ENSURE_DONE:
        _ENSURE_DONE = True
        if status is None:
            logger.warning(
                "aiter plugin status unavailable; provider explain will decline"
            )
        else:
            logger.info(
                "aiter plugin enabled=%s routed=%s",
                tuple(status.get("enabled", ()) or ()),
                tuple(status.get("routed_ops", {}) or {}),
            )
    return status


def _shared_outputs(device, mode, impl, rows):
    key = (
        str(device),
        mode,
        int(impl.num_heads),
        int(impl.num_kv_heads),
        int(impl.head_dim),
        int(getattr(impl, "num_idx_heads", 0)),
        int(getattr(impl, "index_head_dim", 0)),
        int(rows),
    )
    outs = _OUTPUTS.get(key)
    if outs is None:
        q = torch.empty(
            (rows, impl.num_heads, impl.head_dim), device=device, dtype=torch.bfloat16
        )
        outs = {"q": q}
        if mode in ("dense", "reuse"):
            k = torch.empty(
                (rows, impl.num_kv_heads, impl.head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            outs.update(k=k, v=torch.empty_like(k))
        if mode == "indexer":
            outs["index_q"] = torch.empty(
                (rows, impl.num_idx_heads, impl.index_head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
        _OUTPUTS[key] = outs
    return outs


def _sparse_class():
    from atom.models.minimax_m3.model import MiniMaxM3SparseAttention

    return MiniMaxM3SparseAttention


def _skip_reason(module, config):
    """Why this layer cannot take the neutral prologue, or None.

    These are #3084's static site gates; the provider's own explain() remains
    authoritative for a concrete call.
    """
    from atom.models.minimax_m3.model import _linear_consumes_per_token_fp8

    attn, impl = module.attn, module.attn.impl
    rot = impl.rotary_emb
    if int(getattr(config, "sequence_parallel_size", 1)) != _SP_WORLD:
        return "world"
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
    if isinstance(module, _sparse_class()):
        mode = "reuse" if module.skip_index_topk else "indexer"
        if impl.indexer_cp_group is not None:
            return "indexer_cp"
        if mode == "indexer" and (
            impl.index_cache_dtype != "fp8" or impl.index_head_dim != 128
        ):
            return "index_cache"
    return static_weight_reason(module.qkv_proj)


def _site(layer_name):
    ctx = get_current_atom_config().compilation_config.static_forward_context
    return ctx[layer_name].gemm_fusion_site


def _fake(x, x_scale, positions, layer_name, scale):
    site = _site(layer_name)
    return torch.empty(
        (x.shape[0], site.out_width), device=x.device, dtype=torch.float8_e4m3fn
    )


@mark_spliting_op(is_custom=True, gen_fake=_fake, mutates_args=["scale"])
def sp_qkv_prologue_attention(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _site(layer_name).run(x, x_scale, positions, scale)


def _forward(self, positions, hidden_states, hidden_states_scale=None):
    scale = hidden_states.new_empty((hidden_states.shape[0], 1), dtype=torch.float32)
    out = torch.ops.aiter.sp_qkv_prologue_attention(
        hidden_states, hidden_states_scale, positions, self.attn.layer_name, scale
    )
    return self.o_proj(out, scale)


class QkvSite:
    """The neutral prologue of one attention layer, with a non-fused fallback."""

    def __init__(self, name, attn, linear, mode, config, out_width):
        self.name = name
        self.attn = attn
        self.linear = linear
        self.mode = mode
        self.out_width = out_width
        self.world = int(config.sequence_parallel_size)
        self.provider_enable = tuple(getattr(config, "aiter_provider_enable", ()) or ())
        self.provider_enabled = bool(self.provider_enable)
        self._norm_f32 = {}
        self._routes = {}
        self._group = None
        self._eps = float(attn.impl.q_norm.variance_epsilon)
        self._logged = set()

    def _outputs(self, local_m):
        """Caller outputs for a provider-selected cell (shared, address-stable)."""
        return _shared_outputs(
            self.linear.weight.device, self.mode, self.attn.impl, self.world * local_m
        )

    def _f32(self, key, t):
        cached = self._norm_f32.get(key)
        if cached is None or cached[0] is not t:
            cached = (t, t.float().contiguous())
            self._norm_f32[key] = cached
        return cached[1]

    def build_meta(self, state, positions, slot):
        impl = self.attn.impl
        rot = impl.rotary_emb
        meta = {
            "positions": positions,
            "slot_mapping": slot,
            "k_cache": state["k_cache"],
            "v_cache": state.get("v_cache_shuffle", state["v_cache"]),
            "k_scale": state["k_scale"],
            "v_scale": state["v_scale"],
            "cos_cache": rows2d(rot.cos_cache),
            "sin_cache": rows2d(rot.sin_cache),
            "q_norm_w": self._f32("q", impl.q_norm.weight),
            "k_norm_w": self._f32("k", impl.k_norm.weight),
        }
        if self.mode == "indexer":
            meta["index_cache"] = impl.index_cache
            meta["index_q_norm_w"] = self._f32("iq", impl.index_q_norm.weight)
            meta["index_k_norm_w"] = self._f32("ik", impl.index_k_norm.weight)
        return meta

    def _args(self, x, x_scale, meta, outs, stream, *, meta_outputs=False):
        rows = self.world * x.shape[0]
        if meta_outputs:
            # explain() is non-executing; the large band needs no caller outputs,
            # so a meta q/k/v payload lets the provider classify that cell class
            # without allocating its sized buffers.
            q = torch.empty(
                (rows, self.attn.impl.num_heads, self.attn.impl.head_dim),
                dtype=torch.bfloat16,
                device="meta",
            )
            k = v = index_q = None
            if self.mode in ("dense", "reuse"):
                k = torch.empty(
                    (rows, self.attn.impl.num_kv_heads, self.attn.impl.head_dim),
                    dtype=torch.bfloat16,
                    device="meta",
                )
                v = torch.empty_like(k)
            if self.mode == "indexer":
                index_q = torch.empty(
                    (
                        rows,
                        self.attn.impl.num_idx_heads,
                        self.attn.impl.index_head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device="meta",
                )
        else:
            q = outs["q"][:rows]
            k = outs.get("k")
            v = outs.get("v")
            index_q = outs.get("index_q")
        args = [
            q,
            x,
            x_scale,
            self.linear.weight,
            self.linear.weight_scale,
            meta["q_norm_w"],
            meta["k_norm_w"],
            meta["cos_cache"],
            meta["sin_cache"],
            meta["positions"],
            meta["slot_mapping"],
            meta["k_cache"],
            meta["v_cache"],
            meta["k_scale"],
            meta["v_scale"],
        ]
        kwargs = {
            "mode": self.mode,
            "k": None if k is None else k[:rows],
            "v": None if v is None else v[:rows],
            "index_q": None if index_q is None else index_q[:rows],
            "index_q_norm_w": meta.get("index_q_norm_w"),
            "index_k_norm_w": meta.get("index_k_norm_w"),
            "index_cache": meta.get("index_cache"),
            "eps": self._eps,
            "group": self._group_value(),
            "stream": stream,
        }
        return args, kwargs

    def _group_value(self):
        if self._group is None:
            try:
                self._group = _sp_group()
            except Exception:  # noqa: BLE001 - no interface/group: explain declines
                self._group = False
        return None if self._group is False else self._group

    def _log_once(self, key, msg, *args):
        if key not in self._logged:
            self._logged.add(key)
            logger.info(msg, *args)

    def _layout_ok(self, x, x_scale, positions, slot):
        """#3084's static call-shape checks, without its plan machinery."""
        if x_scale is None or x.dtype != torch.float8_e4m3fn:
            return False
        if (
            x_scale.dtype != torch.float32
            or x_scale.numel() != x.shape[0]
            or not x_scale.is_contiguous()
        ):
            return False
        if x.dim() != 2 or x.shape[1] != self.linear.input_size or x.stride(1) != 1:
            return False
        if positions is None:
            return False
        real = positions.shape[-1]
        if positions.numel() != real or slot.numel() < real:
            return False
        return all(
            t.dtype == torch.int64 and t.is_contiguous() and t.numel() != 0
            for t in (positions, slot)
        )

    def _serves(self, m, args, kwargs):
        """The init/first-use ``aiter.plugin.explain`` gate, cached per local M."""
        if not self.provider_enabled:
            return False
        cached = self._routes.get(m)
        if cached is not None:
            return cached
        served = False
        try:
            aiter = _aiter()
            plugin = getattr(aiter, "plugin", None)
            if plugin is None:
                raise AttributeError("aiter.plugin")
            # Re-assert enablement at the point of use: the process running this
            # forward is not necessarily the process ModelRunner initialized.
            _ensure_plugin_enabled(aiter, plugin, self.provider_enable)
            explanation = plugin.explain(_OP_NAME, *args, **kwargs)
            decision = getattr(explanation, "decision", None)
            served = getattr(decision, "route", None) == "provider"
        except Exception:
            logger.warning(
                "gemm_fusion %s: provider explain() unavailable; non-fused path",
                self.name,
                exc_info=True,
            )
        self._routes[m] = served
        self._log_once(
            "route" if served else f"fallback-{m}",
            "gemm_fusion %s: %s local M=%d",
            self.name,
            "provider sp_qkv_fused" if served else "non-fused fallback",
            m,
        )
        return served

    def warmup_payload(self):
        """A first-call-shaped payload for the provider's framework-owned warmup.

        Only shapes/dtypes are read by the provider's coverage check; the real
        QKV weight/norm weights/rotary caches are passed so a weight/rope layout
        the provider cannot serve declines here.
        """
        impl = self.attn.impl
        t = self.world
        page = 128 if self.mode == "dense" else 16
        fp8 = torch.float8_e4m3fn
        bf16 = torch.bfloat16
        fp32 = torch.float32

        def empty(dtype, *shape):
            return torch.empty(shape, dtype=dtype, device="meta")

        payload = {
            "q": empty(bf16, t, impl.num_heads, impl.head_dim),
            "x": empty(fp8, 1, self.linear.input_size),
            "x_scale": empty(fp32, 1),
            "w": self.linear.weight,
            "w_scale": self.linear.weight_scale,
            "q_norm_w": self._f32("q", impl.q_norm.weight),
            "k_norm_w": self._f32("k", impl.k_norm.weight),
            "cos_cache": rows2d(impl.rotary_emb.cos_cache),
            "sin_cache": rows2d(impl.rotary_emb.sin_cache),
            "positions": empty(torch.int64, t),
            "slot_mapping": empty(torch.int64, t),
            "k_cache": empty(fp8, 2, impl.num_kv_heads, impl.head_dim // 16, page, 16),
            "v_cache": empty(fp8, 2, impl.num_kv_heads, page // 16, impl.head_dim, 16),
            "k_scale": empty(fp32, 2, impl.num_kv_heads, page),
            "v_scale": empty(fp32, 2, impl.num_kv_heads, page),
            # Every contract key is present even when the mode does not use it:
            # the provider's payload validator rejects missing/extra keys.
            "mode": self.mode,
            "k": None,
            "v": None,
            "index_q": None,
            "index_q_norm_w": None,
            "index_k_norm_w": None,
            "index_cache": None,
            "eps": self._eps,
            "group": self._group_value(),
            "stream": None,
        }
        if self.mode == "indexer":
            payload.update(
                index_q=empty(bf16, t, impl.num_idx_heads, impl.index_head_dim),
                index_q_norm_w=self._f32("iq", impl.index_q_norm.weight),
                index_k_norm_w=self._f32("ik", impl.index_k_norm.weight),
                index_cache=empty(fp8, 2, 128, 128),
            )
        else:
            payload.update(
                k=empty(bf16, t, impl.num_kv_heads, impl.head_dim),
                v=empty(bf16, t, impl.num_kv_heads, impl.head_dim),
            )
        return payload

    def _tail(self, fwd_ctx, state, outs, positions):
        real = positions.shape[-1]
        index_q = outs.get("index_q")
        return self.attn.impl.forward_after_prologue(
            outs["q"][:real],
            outs["k"][:real] if "k" in outs else None,
            outs["v"][:real] if "v" in outs else None,
            state["k_cache"],
            state["v_cache"],
            state["k_scale"],
            state["v_scale"],
            fwd_ctx,
            index_q=None if index_q is None else index_q[:real],
        )

    def _native(self, x, x_scale, positions, scale):
        from atom.model_ops.sp_attention import gather_heads, ulysses_attention

        self._log_once("native", "gemm_fusion %s: non-fused path", self.name)
        qkv = self.linear(x, x_scale=x_scale)
        out, rows = ulysses_attention(self.attn, None, None, None, positions, None, qkv)
        return gather_heads(out, rows, self.attn.sp_head_exchange, scale)

    def run(self, x, x_scale, positions, scale):
        """This layer's per-token FP8 attention output; its scales go to ``scale``."""
        from atom.model_ops.sp_attention import gather_heads

        fwd_ctx = get_forward_context()
        if fwd_ctx.context.is_dummy_run:
            return self._native(x, x_scale, positions, scale)
        impl = self.attn.impl
        state = impl.gemm_fusion_prologue_state(fwd_ctx)
        slot = state["slot_mapping"]
        if not self._layout_ok(x, x_scale, positions, slot):
            return self._native(x, x_scale, positions, scale)
        meta = self.build_meta(state, positions, slot)
        stream = torch.cuda.current_stream()
        # The gate is non-executing and sees a meta-output payload, so it can
        # classify any local M without allocating that cell's caller buffers.
        gate_args, gate_kwargs = self._args(
            x, x_scale, meta, None, stream, meta_outputs=True
        )
        if self._serves(x.shape[0], gate_args, gate_kwargs):
            outs = self._outputs(x.shape[0])
            args, kwargs = self._args(x, x_scale, meta, outs, stream)
            aiter = _aiter()
            aiter.sp_qkv_fused(*args, **kwargs)
            out = self._tail(fwd_ctx, state, outs, positions)
            return gather_heads(
                out, self.world * x.shape[0], self.attn.sp_head_exchange, scale
            )
        return self._native(x, x_scale, positions, scale)


def install_minimax_m3(model, config):
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
        reason = _skip_reason(module, config)
        if reason is not None:
            logger.info("gemm_fusion %s: native (%s)", name, reason)
            continue
        site = QkvSite(
            name,
            attn,
            module.qkv_proj,
            mode,
            config,
            out_width=module.num_heads * module.head_dim,
        )
        attn.gemm_fusion_site = site
        module.forward = types.MethodType(_forward, module)
        sites.append(site)
    return sites
