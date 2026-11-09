# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mode-B call site for the neutral aiter ``sp_qkv_fused`` routed op (default OFF).

The op fuses one MiniMax-M3 SP4 QKV step -- the column-parallel QKV+indexer GEMM,
the Ulysses SP head exchange, per-head Gemma-RMSNorm, partial-NeoX RoPE, the
SHUFFLE KV/index cache writes and the per-token FP8 dequant scales -- into one
aiter contract call whose native path is aiter-owned. ATOM keeps only:

* init-time coverage (this file's ``bind``): flag on, op routed by the installed
  aiter, gfx950, FP8 page-128 KV, 16x16-preshuffled FP8 QKV weight, the op's
  SP world 4, and the layer's dense/indexer/reuse mode;
* the tensor mapping: ATOM's bound weights/norm weights/cos-sin cache and the
  per-call activation/positions/slot mapping/KV-cache views;
* the attention tail after the prologue, through the same impl seam #2643 uses
  (``impl.forward_after_prologue`` + the FP8 head gather), so the fused path
  cannot diverge from the native one.

There is deliberately no try/except fallback around the routed call: if the op
declines a call, aiter serves it natively; if a provider's ``run`` raises, the
error propagates (retrying natively could repeat side effects). ATOM's decision
is the flag and the static matrix, both resolved once before the first forward.

The call site is installed on the M3 attention modules after weight loading and
before warm-up/CUDA-graph capture (the same model-adapter seam as
``atom/model_ops/gemm_fusion/model_adapters.py``). The installed forward has no
per-forward Python branch: it is the ``sp_qkv_fused`` custom op plus ``o_proj``.
``atom/models/minimax_m3/model.py`` is not edited, and with the flag off the
module is never imported.
"""

import logging
import types
from types import SimpleNamespace

import torch

from atom.config import get_current_atom_config
from atom.utils import mark_spliting_op

logger = logging.getLogger("atom.sp_qkv_fused")

_OP_NAME = "sp_qkv_fused"
_OP_VERSION = 1
_GFX = "gfx950"
_SP_WORLD = 4
_KV_BLOCK_SIZE = 128
_SUPPORTED_MODES = ("dense", "indexer", "reuse")

# Bind-time refusals are per process, not per layer: one warning each.
_WARNED: set[str] = set()

# sp_qkv_covers() answers are structural (shape/dtype/numel only) and per
# process: cache the per-(mode, local-M cell set) verdict so every layer does
# not re-run aiter's profile match.
_COVERS_CACHE: dict[tuple, tuple[bool | None, str | None]] = {}

# The local-M cells a prologue deployment can present (aiter's small-plan
# buckets), plus the deployment's largest local M when it is larger.
_COVER_CELLS = (1, 2, 4, 8, 12, 16, 32, 64, 128, 256)

# One bucket-sized output set per (device, mode, geometry, max tokens), shared
# by every site: the runtime stamps its own bucket staging into bindings and
# copies the real rows into the caller's buffers at call time, so caller output
# pointers are NOT part of the binding identity. Sharing keeps the per-deployment
# cost to one set instead of one per layer; pointers are allocated once and stay
# fixed for the process lifetime.
_OUTPUTS: dict[tuple, SimpleNamespace] = {}


def _warn_once(key: str, msg: str, *args) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


def get_sp_world_size() -> int:
    """SP width (indirection so CPU tests can patch it)."""
    from atom.distributed.ulysses_sp import get_sp_world_size as _f

    return _f()


def get_sp_group():
    """The SP group's torch.distributed ProcessGroup (the op's execution context)."""
    from atom.distributed.ulysses_sp import get_sp_group as _f

    return _f().device_group


def _get_gfx() -> str | None:
    """Current GPU arch, or None when aiter/ROCm is unavailable (CPU tests)."""
    try:
        from aiter.jit.utils.chip_info import get_gfx as _f

        return _f()
    except Exception:  # noqa: BLE001 - CPU-only test / no aiter install
        return None


def _linear_consumes_per_token_fp8(linear) -> bool:
    from atom.models.minimax_m3.model import _linear_consumes_per_token_fp8 as _f

    return _f(linear)


def _cos_sin_cache(module, rotary_emb, norm_weight):
    from atom.models.minimax_m3.model import _minimax_m3_cos_sin_cache

    return _minimax_m3_cos_sin_cache(rotary_emb, norm_weight)


def _separate_cos_sin(module):
    """Separate ``[max_pos, rotary_dim/2]`` cos/sin views for the neutral op.

    The op contract takes the two halves as their own tensors. The r19 (and
    r17/r18) kernels index both with one baked ``cos_stride_pos``: the host
    requires ``stride(0) % 8 == 0``, ``stride(1) == 1``, equal ``stride(0)`` on
    both halves and 16-byte-aligned base pointers. The halves of M3's contiguous
    concatenated ``[max_pos, rotary_dim]`` cache satisfy those constraints
    zero-copy, so prefer them; fall back to contiguous copies if a rotary module
    ever produces a layout that does not (the driver would otherwise fail at
    prepare time).
    """
    cache = _cos_sin_cache(module, module.rotary_emb, module.q_norm.weight)
    rotary_dim = int(getattr(module.rotary_emb, "rotary_dim", cache.shape[-1]))
    half = rotary_dim // 2
    cos, sin = cache[:, :half], cache[:, half:]
    same_stride = cos.stride(0) == sin.stride(0)
    ok = (
        cos.stride(1) == 1
        and sin.stride(1) == 1
        and same_stride
        and cos.stride(0) % 8 == 0
        and cos.data_ptr() % 16 == 0
        and sin.data_ptr() % 16 == 0
    )
    if ok:
        return cos, sin
    return cos.contiguous(), sin.contiguous()


def _cache_layout_reason(impl, state, *, page, with_index):
    """The #2643 cache-layout check, reused as-is (indirection for CPU tests)."""
    from atom.model_ops.gemm_fusion.site import (
        _cache_layout_reason as _check,
    )

    return _check(impl, state, page=page, with_index=with_index)


def _finish_attention(out, padded_tokens, real_tokens):
    """Re-pad + FP8 head gather (the tail #2643's fused prologue ends with)."""
    from atom.distributed.ulysses_sp import finish_attention
    from atom.model_ops.minimax_m3.attention_fp8 import gather_heads_fp8

    return finish_attention(out, padded_tokens, real_tokens, gather_heads_fp8)


def _aiter_op():
    """The installed aiter's neutral op, or a warning reason."""
    try:
        import aiter
    except Exception as exc:  # noqa: BLE001 - no aiter at all
        return None, f"aiter unavailable ({exc})"
    op = getattr(aiter, _OP_NAME, None)
    if op is None:
        return None, f"this aiter has no {_OP_NAME} op"
    return op, None


def _op_routed():
    """(True, None) when aiter routes ``sp_qkv_fused`` in this build."""
    try:
        import aiter

        plugin = getattr(aiter, "plugin", None)
        if plugin is None:
            return False, "aiter.plugin is absent (no provider interface)"
        status = plugin.status()
        routed = getattr(status, "get", lambda *_: None)("routed_ops") or {}
    except Exception as exc:  # noqa: BLE001 - older/partial interface
        return False, f"aiter.plugin.status() failed ({exc})"
    if _OP_NAME not in routed:
        return False, f"{_OP_NAME} is not in aiter status()['routed_ops']"
    return True, None


def _max_local_rows(atom_config) -> int:
    """Largest local row count one forward can present to the op."""
    budget = int(getattr(atom_config, "max_num_batched_tokens", 0) or 0)
    world = max(1, int(getattr(atom_config, "sequence_parallel_size", 1) or 1))
    return max(1, (budget + world - 1) // world)


def _candidate_local_rows(atom_config) -> tuple[int, ...]:
    """The local-M bucket set this deployment can present to the op.

    Prologue coverage is defined per small-plan bucket; the answer is for
    ``x.shape[0]``, so every M the deployment can schedule must be queried.
    """
    max_m = _max_local_rows(atom_config)
    cells = [m for m in _COVER_CELLS if m <= max_m]
    if max_m not in cells:
        cells.append(max_m)
    return tuple(sorted(cells))


def _op_covers(mode: str, local_ms: tuple[int, ...], world: int, module, atom_config):
    """``(True, None)`` when aiter covers every local-M cell this layer can see.

    ``aiter.sp_qkv_covers(...)`` takes the same flat contract arguments as the op
    and is a cheap, execution-free query: it reads shape/dtype/numel and
    ``w.is_contiguous()`` only, never tensor values/devices, and never raises.
    The only real requirements are a live GPU/ROCm context and a real SP group
    (or the initialized default group), which bind() has on the worker. ATOM
    queries every small-plan bucket this deployment can present plus its largest
    local M, so a recipe whose large band is uncovered (Phase-2; m > 256 today)
    keeps the existing ATOM path even though the small cells are covered.
    Results are cached per (mode, cell set) across layers.
    """
    key = (mode, tuple(local_ms))
    cached = _COVERS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import aiter
    except Exception as exc:  # noqa: BLE001 - no aiter at all
        return None, f"aiter unavailable ({exc})"
    fn = getattr(aiter, "sp_qkv_covers", None)
    if fn is None:
        return None, "this aiter has no sp_qkv_covers() predicate"
    uncovered: list[int] = []
    for local_m in local_ms:
        try:
            payload = _coverage_payload(mode, local_m, world, module, atom_config)
        except Exception as exc:  # noqa: BLE001 - never fatal to the native path
            return None, f"cannot build the sp_qkv_covers() payload ({exc})"
        try:
            covered = bool(fn(*payload.pop("_args"), **payload))
        except Exception as exc:  # noqa: BLE001 - predicate must stay non-fatal
            return None, f"sp_qkv_covers() failed ({exc})"
        if not covered:
            uncovered.append(local_m)
    result = (
        (True, None)
        if not uncovered
        else (False, f"uncovered local_m cells {uncovered}")
    )
    _COVERS_CACHE[key] = result
    return result


def _meta(dtype, *shape):
    return torch.empty(shape, dtype=dtype, device="meta")


def _coverage_payload(mode: str, local_m: int, world: int, module, atom_config):
    """Flat payload for ``aiter.sp_qkv_covers``.

    Real device tensors are used for everything that already exists at bind time
    (weights/scales/norm weights, the separate dense cos/sin caches) and for
    tiny device stand-ins for the KV/index caches (they are allocated only after
    this hook, but the predicate is structural about them). The genuinely
    per-call tensors -- activation/scale, positions/slot mapping and the output
    buffers -- are meta tensors of the exact contract shape/dtype: the coverage
    query must not allocate per-site output buffers before it decides.
    """
    impl = module.attn.impl
    qkv = module.qkv_proj
    heads = int(getattr(impl, "num_heads", 16))
    kv_heads = max(1, int(getattr(impl, "num_kv_heads", 1)))
    head_dim = int(getattr(impl, "head_dim", 128))
    t_rows = int(local_m) * int(world)
    page = _KV_BLOCK_SIZE if mode == "dense" else 16
    fp8 = torch.float8_e4m3fn
    bf16 = torch.bfloat16
    fp32 = torch.float32
    device = module.q_norm.weight.device
    cos_cache, sin_cache = _separate_cos_sin(module)
    q = _meta(bf16, t_rows, heads, head_dim)
    x = _meta(fp8, local_m, 6144)
    x_scale = _meta(fp32, local_m)
    k_cache = torch.empty(
        (2, kv_heads, head_dim // 16, page, 16), dtype=fp8, device=device
    )
    v_cache = torch.empty(
        (2, kv_heads, page // 16, head_dim, 16), dtype=fp8, device=device
    )
    k_scale = torch.empty((2, kv_heads, page), dtype=fp32, device=device)
    v_scale = torch.empty((2, kv_heads, page), dtype=fp32, device=device)
    positions = _meta(torch.int64, t_rows)
    slot_mapping = _meta(torch.int64, t_rows)
    args = [
        q,
        x,
        x_scale,
        qkv.weight,
        qkv.weight_scale,
        module.q_norm.weight,
        module.k_norm.weight,
        cos_cache,
        sin_cache,
        positions,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
    ]
    kwargs = {
        "mode": mode,
        "eps": float(module.q_norm.variance_epsilon),
        "group": get_sp_group(),
        "stream": None,
    }
    if mode == "indexer":
        idx_heads = max(1, int(getattr(impl, "num_idx_heads", 1)))
        idx_dim = int(getattr(impl, "index_head_dim", 128))
        idx_norm = getattr(module, "index_q_norm", None)
        idx_k_norm = getattr(module, "index_k_norm", None)
        kwargs.update(
            k=None,
            v=None,
            index_q=_meta(bf16, t_rows, idx_heads, idx_dim),
            index_q_norm_w=(
                idx_norm.weight if idx_norm is not None else _meta(fp32, 128)
            ),
            index_k_norm_w=(
                idx_k_norm.weight if idx_k_norm is not None else _meta(fp32, 128)
            ),
            index_cache=torch.empty(256, dtype=fp8, device=device),
        )
    else:
        kwargs.update(
            k=_meta(bf16, t_rows, kv_heads, head_dim),
            v=_meta(bf16, t_rows, kv_heads, head_dim),
            index_q=None,
            index_q_norm_w=None,
            index_k_norm_w=None,
            index_cache=None,
        )
    return {"_args": args, **kwargs}


def _shared_outputs(device, mode, impl, index_heads, index_head_dim, max_tokens):
    """One bucket-sized, address-stable output set shared by all sites."""
    key = (
        str(device),
        mode,
        int(impl.num_heads),
        int(impl.num_kv_heads),
        int(impl.head_dim),
        int(index_heads),
        int(index_head_dim),
        int(max_tokens),
    )
    pool = _OUTPUTS.get(key)
    if pool is None:
        q = torch.empty(
            (max_tokens, impl.num_heads, impl.head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        k = v = None
        index_q = None
        if mode in ("dense", "reuse"):
            k = torch.empty(
                (max_tokens, impl.num_kv_heads, impl.head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            v = torch.empty_like(k)
        if mode == "indexer":
            index_q = torch.empty(
                (max_tokens, index_heads, index_head_dim),
                device=device,
                dtype=torch.bfloat16,
            )
        pool = _OUTPUTS[key] = SimpleNamespace(q=q, k=k, v=v, index_q=index_q)
    return pool


def _m3_attention_classes():
    from atom.models.minimax_m3.model import (
        MiniMaxM3Attention,
        MiniMaxM3SparseAttention,
    )

    return (MiniMaxM3Attention, MiniMaxM3SparseAttention)


def _is_m3_attention(module) -> bool:
    return isinstance(module, _m3_attention_classes())


def _mode_of(module) -> str:
    if not getattr(module, "is_indexed_sparse_attention", False):
        return "dense"
    return "reuse" if getattr(module, "skip_index_topk", False) else "indexer"


def bind(module, atom_config, layer_id: int) -> "SpQkvFused | None":
    """Return a bound fused-op callable for one M3 attention module, or None.

    Every check is init-time and rank-uniform (config/weights only): flag on,
    the installed aiter carries a routed ``sp_qkv_fused``, gfx950, FP8 page-128
    KV cache, a 16x16-preshuffled FP8 QKV weight with a weight scale, the op's
    SP world 4, and the layer's mode. ``None`` means "keep the existing ATOM
    composition"; the reason is logged once per process.
    """
    if not getattr(atom_config, "enable_sp_qkv_fused", False):
        return None
    op, why = _aiter_op()
    if op is None:
        _warn_once("op", "sp_qkv_fused requested but %s; native path", why)
        return None
    routed, why = _op_routed()
    if not routed:
        _warn_once(
            "routed",
            "sp_qkv_fused requested but %s; the op stays native",
            why,
        )
        return None

    attn = getattr(module, "attn", None)
    qkv = getattr(module, "qkv_proj", None)
    o_proj = getattr(module, "o_proj", None)
    if attn is None or qkv is None or o_proj is None:
        return None
    impl = getattr(attn, "impl", None)
    if impl is None:
        return None
    layer_id = int(getattr(module, "layer_num", layer_id))

    gfx = _get_gfx()
    if gfx != _GFX:
        _warn_once("gfx", "sp_qkv_fused requires %s, got %r; native", _GFX, gfx)
        return None
    if int(getattr(atom_config, "sequence_parallel_size", 1)) != _SP_WORLD:
        _warn_once(
            "sp",
            "sp_qkv_fused requires --sequence-parallel-size %d, got %s; native",
            _SP_WORLD,
            getattr(atom_config, "sequence_parallel_size", 1),
        )
        return None
    if int(getattr(atom_config, "tensor_parallel_size", 1)) != 1:
        _warn_once("tp", "sp_qkv_fused requires --tensor-parallel-size 1; native")
        return None
    world = get_sp_world_size()
    if world != _SP_WORLD:
        _warn_once(
            "world",
            "sp_qkv_fused requires SP world %d, got %d; native",
            _SP_WORLD,
            world,
        )
        return None
    # Static coverage before any binding: query every local-M cell this
    # deployment can present. aiter's native path is coverage-limited today to
    # the 2-stage prologue cells (local m <= 256); the one-stage large band /
    # r19 16K rows are a Phase-2 native-prologue follow-up and keep the existing
    # ATOM path (which also carries the r19 18-bit tag classification).
    mode = _mode_of(module)
    local_ms = _candidate_local_rows(atom_config)
    covered, why = _op_covers(mode, local_ms, world, module, atom_config)
    if not covered:
        _warn_once(
            "covers",
            "sp_qkv_fused: aiter coverage does not serve mode=%s cells %s "
            "(%s); keeping the existing ATOM path",
            mode,
            local_ms,
            why or "outside the covered cells",
        )
        return None
    if getattr(atom_config, "kv_cache_dtype", None) != "fp8":
        _warn_once(
            "kv_dtype",
            "sp_qkv_fused requires --kv_cache_dtype fp8, got %r; native",
            getattr(atom_config, "kv_cache_dtype", None),
        )
        return None
    if int(getattr(atom_config, "kv_cache_block_size", 0)) != _KV_BLOCK_SIZE:
        _warn_once(
            "kv_page",
            "sp_qkv_fused requires a %d-token KV page (--block-size %d), got %s; native",
            _KV_BLOCK_SIZE,
            _KV_BLOCK_SIZE,
            getattr(atom_config, "kv_cache_block_size", None),
        )
        return None
    if getattr(impl, "kv_cache_dtype", None) != "fp8":
        _warn_once(
            "impl_kv_dtype",
            "sp_qkv_fused: this layer's KV cache is %r, not fp8; native",
            getattr(impl, "kv_cache_dtype", None),
        )
        return None
    if not hasattr(impl, "gemm_fusion_prologue_state") or not hasattr(
        impl, "forward_after_prologue"
    ):
        _warn_once(
            "seam",
            "sp_qkv_fused needs the attention-impl prologue seam (this ATOM "
            "build has no fused-prologue state); native",
        )
        return None

    weight = getattr(qkv, "weight", None)
    weight_scale = getattr(qkv, "weight_scale", None)
    if weight is None or weight_scale is None:
        _warn_once(
            "weight",
            "sp_qkv_fused requires an FP8 QKV weight plus its weight scale; native",
        )
        return None
    if not getattr(weight, "is_shuffled", False):
        _warn_once(
            "preshuffle",
            "sp_qkv_fused requires a 16x16-preshuffled QKV weight; native",
        )
        return None
    if weight.element_size() != 1:
        _warn_once(
            "weight_dtype",
            "sp_qkv_fused requires an FP8 QKV weight (element size 1), got %d; native",
            weight.element_size(),
        )
        return None

    # The tail returns FP8 hidden states with per-token scales, exactly the
    # schema #2643's fused prologue returns; require the same deployment gate.
    if not getattr(attn, "_m3_sp_fp8_output_supported", False) or getattr(
        attn, "_m3_tp_replicated_o_proj", False
    ):
        _warn_once(
            "fp8_out",
            "sp_qkv_fused requires the M3 SP FP8-output path without the "
            "replicated-o_proj variant; native",
        )
        return None
    if int(getattr(o_proj, "input_size", 0)) != 8192 or not (
        _linear_consumes_per_token_fp8(o_proj)
    ):
        _warn_once(
            "o_proj",
            "sp_qkv_fused requires an o_proj over 8192 FP8 inputs with per-token "
            "scales; native",
        )
        return None

    if mode not in _SUPPORTED_MODES:
        return None
    return SpQkvFused(
        module,
        atom_config,
        layer_id,
        mode=mode,
        world=world,
        op=op,
    )


class SpQkvFused:
    """One bound M3 layer: handles bound once, per-call args mapped to the op."""

    def __init__(self, module, atom_config, layer_id: int, *, mode, world, op):
        self.module = module
        self.layer_id = int(layer_id)
        self._attn = module.attn
        self._impl = module.attn.impl
        self._mode = mode
        self._world = int(world)
        self._op = op
        self._layer_name = self._attn.layer_name

        qkv = module.qkv_proj
        self._w = qkv.weight
        self._w_scale = qkv.weight_scale
        self._q_norm_w = module.q_norm.weight
        self._k_norm_w = module.k_norm.weight
        self._eps = float(module.q_norm.variance_epsilon)
        # The neutral op takes separate dense [max_pos, rotary_dim/2] cos/sin
        # tensors, not M3's concatenated cache nor strided half-views: bind the
        # contiguous pair once (the rotary module usually already has it).
        self._cos_cache, self._sin_cache = _separate_cos_sin(module)
        self._index_q_norm_w = getattr(module, "index_q_norm", None)
        self._index_k_norm_w = getattr(module, "index_k_norm", None)
        if self._index_q_norm_w is not None:
            self._index_q_norm_w = self._index_q_norm_w.weight
        if self._index_k_norm_w is not None:
            self._index_k_norm_w = self._index_k_norm_w.weight
        self._index_cache = None
        self._cache_state = None
        self._index_heads = int(
            getattr(self._impl, "num_idx_heads", 0) if self._mode != "dense" else 0
        )
        self._index_head_dim = int(
            getattr(self._impl, "index_head_dim", 0) if self._mode != "dense" else 0
        )
        # Bucket-sized output set shared by every site: the runtime stages its
        # own bucket outputs in the binding and copies the real rows back into
        # the caller's buffers, so caller output pointers are not bound. The set
        # is allocated once (address-stable for the process lifetime) and each
        # call passes its own smaller row views.
        self._max_tokens = _bound_output_tokens(atom_config)
        self._outs = _shared_outputs(
            self._w.device,
            self._mode,
            self._impl,
            self._index_heads,
            self._index_head_dim,
            self._max_tokens,
        )

    # -- bind-time / first-call state -----------------------------------
    def _bind_state(self, fwd_ctx):
        """Bind the KV-cache views once, outside capture; verify pointer stability.

        KV tensors only exist after ``allocate_kv_cache``, which runs after the
        model build where this wrapper is created, so the views are pinned on
        the first call (the pre-capture eager warm-up) rather than in bind().
        Returns the per-call state the op/tail read: the bound cache views plus
        this step's slot mapping.
        """
        state = self._impl.gemm_fusion_prologue_state(fwd_ctx)
        page = 128 if self._mode == "dense" else 16
        reason = _cache_layout_reason(
            self._impl, state, page=page, with_index=self._mode == "indexer"
        )
        if reason is not None:
            raise RuntimeError(
                f"sp_qkv_fused layer {self._layer_name}: bound KV cache does not "
                f"match the op contract ({reason}); refusing to launch"
            )
        k_cache = state["k_cache"]
        v_cache = state.get("v_cache_shuffle", state["v_cache"])
        k_scale = state.get("k_scale_view", state.get("k_scale"))
        v_scale = state.get("v_scale_view", state.get("v_scale"))
        if self._cache_state is None:
            self._cache_state = SimpleNamespace(
                k_cache=k_cache,
                v_cache=v_cache,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            if self._mode == "indexer":
                index_cache = getattr(self._impl, "index_cache", None)
                if index_cache is None:
                    raise RuntimeError(
                        f"sp_qkv_fused layer {self._layer_name}: indexer mode "
                        "needs the bound page-128 index cache; native cache "
                        "binding did not run"
                    )
                self._index_cache = index_cache
        else:
            bound = self._cache_state
            for name, now in (
                ("k_cache", k_cache),
                ("v_cache", v_cache),
                ("k_scale", k_scale),
                ("v_scale", v_scale),
            ):
                was = getattr(bound, name)
                if (was is None) != (now is None) or (
                    was is not None and was.data_ptr() != now.data_ptr()
                ):
                    raise RuntimeError(
                        f"sp_qkv_fused layer {self._layer_name}: the {name} "
                        "view moved after binding; the op keeps raw pointers"
                    )
        bound = self._cache_state
        return SimpleNamespace(
            k_cache=bound.k_cache,
            v_cache=bound.v_cache,
            k_scale=bound.k_scale,
            v_scale=bound.v_scale,
            slot_mapping=state["slot_mapping"],
        )

    def _outputs(self, padded_tokens: int):
        """Per-call row views of the shared address-stable output buffers."""
        n = int(padded_tokens)
        outs = self._outs
        return SimpleNamespace(
            q=outs.q[:n],
            k=None if outs.k is None else outs.k[:n],
            v=None if outs.v is None else outs.v[:n],
            index_q=None if outs.index_q is None else outs.index_q[:n],
        )

    # -- per-call ---------------------------------------------------------
    def __call__(self, hidden_states, hidden_states_scale, positions, fwd_ctx):
        """Run the routed op for one forward and continue with ATOM's tail."""
        state = self._bind_state(fwd_ctx)
        real_tokens = int(positions.shape[-1])
        local_rows = int(hidden_states.shape[0])
        padded_tokens = local_rows * self._world
        if self._max_tokens < padded_tokens:
            raise RuntimeError(
                f"sp_qkv_fused layer {self._layer_name}: {padded_tokens} padded "
                f"tokens exceed the bound output capacity {self._max_tokens}"
            )
        outs = self._outputs(padded_tokens)

        self._op(
            outs.q,
            hidden_states,
            hidden_states_scale,
            self._w,
            self._w_scale,
            self._q_norm_w,
            self._k_norm_w,
            self._cos_cache,
            self._sin_cache,
            positions,
            state.slot_mapping,
            state.k_cache,
            state.v_cache,
            state.k_scale,
            state.v_scale,
            mode=self._mode,
            k=outs.k,
            v=outs.v,
            index_q=outs.index_q,
            index_q_norm_w=self._index_q_norm_w,
            index_k_norm_w=self._index_k_norm_w,
            index_cache=self._index_cache if self._mode == "indexer" else None,
            eps=self._eps,
            group=get_sp_group(),
            stream=torch.cuda.current_stream(),
        )

        k_tail = None if outs.k is None else outs.k[:real_tokens]
        v_tail = None if outs.v is None else outs.v[:real_tokens]
        index_q_tail = outs.index_q[:real_tokens] if outs.index_q is not None else None
        out = self._impl.forward_after_prologue(
            outs.q[:real_tokens],
            k_tail,
            v_tail,
            state.k_cache,
            state.v_cache,
            state.k_scale,
            state.v_scale,
            fwd_ctx,
            index_q=index_q_tail,
        )
        return _finish_attention(out, padded_tokens, real_tokens)


def install_m3(model, atom_config) -> int:
    """Install the fused call site on every eligible M3 attention module.

    Called once after weight loading and before warm-up/compile/capture, the
    same seam #2643 uses (``model_adapters.install_m3_sites``): the module's
    ``forward`` is replaced, so the traced model file needs no branch. Returns
    the number of layers routed through the op.
    """
    if not getattr(atom_config, "enable_sp_qkv_fused", False):
        return 0
    n = 0
    for module in model.modules():
        if not _is_m3_attention(module):
            continue
        attn = module.attn
        if getattr(attn, "_sp_qkv_fused", None) is not None:
            continue
        layer_id = int(getattr(module, "layer_num", 0))
        bound = bind(module, atom_config, layer_id)
        if bound is None:
            continue
        attn._sp_qkv_fused = bound
        module.forward = types.MethodType(_provider_forward, module)
        n += 1
    if n:
        logger.info(
            "sp_qkv_fused: installed the routed QKV prologue on %d M3 layers", n
        )
    return n


def _bound_output_tokens(atom_config) -> int:
    from atom.distributed.ulysses_sp import sp_pad_len

    budget = int(getattr(atom_config, "max_num_batched_tokens", 0) or 0)
    if budget <= 0:
        raise RuntimeError("sp_qkv_fused needs a positive max_num_batched_tokens")
    return int(
        sp_pad_len(budget, int(getattr(atom_config, "sequence_parallel_size", 1)))
    )


def _provider_forward(self, positions, hidden_states, hidden_states_scale=None):
    """Replacement forward for an M3 attention module routed through the op."""
    output, scale = torch.ops.aiter.sp_qkv_fused_attention(
        hidden_states, hidden_states_scale, positions, self.attn.layer_name
    )
    return self.o_proj(output, x_scale=scale)


# -- opaque custom op (Dynamo sees one splitting op, not this Python) --------


def _layer(layer_name):
    return get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]


def run_sp_qkv_fused_attention(attn, x, x_scale, positions):
    wrapper = getattr(attn, "_sp_qkv_fused", None)
    if wrapper is None:
        raise RuntimeError(
            f"sp_qkv_fused: layer {attn.layer_name} has no bound wrapper; the "
            "call site was not installed"
        )
    from atom.utils.forward_context import get_forward_context

    return wrapper(x, x_scale, positions, get_forward_context())


def _fake(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn = _layer(layer_name)
    from aiter import dtypes

    world = get_sp_world_size()
    rows = int(x.shape[0]) * world
    q_cols = attn.impl.num_heads * attn.impl.head_dim * world
    return (
        torch.empty((rows, q_cols), device=x.device, dtype=dtypes.fp8),
        torch.empty((rows, 1), device=x.device, dtype=torch.float32),
    )


@mark_spliting_op(is_custom=True, gen_fake=_fake, mutates_args=[])
def sp_qkv_fused_attention(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    positions: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, _scale = run_sp_qkv_fused_attention(_layer(layer_name), x, x_scale, positions)
    return out, _scale
