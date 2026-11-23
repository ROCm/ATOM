# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 mono MLA kernels on the Kimi-K3 DSpark draft's layers
(``ATOM_KIMI_MONO_MLA_DRAFT``): a ``K3DSparkMLAAttention``'s decode path from
its input norm's FP8 row to the attention's q_out and the step's KV-cache
entries as two resident launches -- ``mla_draft_front_gfx1250``
(fused_qkv_a_proj, fused_qk_rmsnorm_group_quant) and
``mla_absorb_draft_gfx1250`` (q_b_proj with fp32 activation scales, W_UK,
rope + cache) -- production's bits, then ``MLAAttention.forward_decode_q_out``.

A layer the kernels cannot read keeps the production path (``bind``
refuses it, once). A step they cannot take -- a dummy or prefill step, a
width past ``MAX_ROWS``, a slot mapping, cache or input they do not read, a
width first seen while a CUDA graph is being captured -- runs production's
kernels. Where production's ``fused_qkv_a_proj`` dispatch does not split K
in four (wider than 32 rows), its GEMM and norm run as production does and
the absorb takes their outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs

HIDDEN = 7168
Q_LORA = 1536
KV_LORA = 512
ROPE = 64
NOPE = 128
HEAD = NOPE + ROPE
ENTRY = KV_LORA + ROPE
GROUPS_X = HIDDEN // 128
Q_GROUPS = Q_LORA // 128
MAX_ROWS = 64


def _logger():
    import logging

    return logging.getLogger("atom")


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


@dataclass(frozen=True)
class DraftMlaLayer:
    """What the draft kernels read for one draft MLA layer, in place:
    fused_qkv_a_proj and q_b_proj in ``_block_fp8``'s layout, the norm
    weights, W_UK (FP8, one scale), the rope cache; the decode branch's q_out
    shape -- ``heads`` projected (the DCP group's under QREP), ``out_heads``
    (the kernel width the decode runs at: zero heads past ``heads``),
    ``compute_all`` (DCP: every token's q_out), ``prepadded`` (DCP with a
    head pad: the decode is told q_out carries it)."""

    prefix: str
    w_a: torch.Tensor
    s_a: torch.Tensor
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    q_eps: float
    kv_eps: float
    w_qb: torch.Tensor
    s_qb: torch.Tensor
    w_uk: torch.Tensor
    s_uk: torch.Tensor
    heads: int
    out_heads: int
    compute_all: bool
    prepadded: bool
    cos: torch.Tensor
    sin: torch.Tensor
    max_pos: int
    attention: object

    def front_key(self, rows: int, ksplit: int) -> tuple:
        return rows, ksplit, self.q_eps, self.kv_eps

    def absorb_key(self, rows: int, split: int) -> tuple:
        return rows, self.heads, self.out_heads, self.compute_all, split, self.max_pos


def bind(attn) -> DraftMlaLayer:
    """One draft MLA layer's tensors. Refused where the production decode
    path would compute anything else: an input norm that does not hand
    fused_qkv_a_proj the (E4M3, transposed per-1x128 E8M0) pair, a q norm
    that does not quantize q_b_proj's input with fp32 per-1x128 scales,
    projections or layouts other than the kernels', a sparse, segmented or
    shuffled KV cache, a cache or q that is not FP8, static scales other
    than 1, a rotary embedding that is not GPT-J over 64 BF16 dims, the BF16
    or FP4 absorb, DCP without QREP."""
    from aiter import QuantType, dtypes

    from atom.model_ops import attention_mla
    from atom.models.kimi_k3.mono.weights import _block_fp8, _eps, _vec

    name = attn.input_quant_prefix.rsplit(".", 1)[0]
    norm = getattr(attn, "_input_norm", None)
    _need(norm is not None, f"{name}: no input norm bound")
    _need(
        norm.use_fused_quant
        and not norm.fused_allreduce
        and norm.quant_type.value == QuantType.per_1x128.value
        and norm.params_dtype == dtypes.fp8
        and norm._aiter_transpose_scale
        and norm._aiter_e8m0_scale,
        f"{name}: the input norm does not hand fused_qkv_a_proj its E8M0 FP8 row",
    )
    _need(
        (attn.hidden_size, attn.q_lora_rank, attn.kv_lora_rank)
        == (HIDDEN, Q_LORA, KV_LORA)
        and attn.qk_rope_head_dim == ROPE
        and attn.qk_nope_head_dim == NOPE,
        f"{name}: MLA widths",
    )
    _need(
        attn.fuse_qknorm_quant
        and attn.qknorm_dtype == dtypes.fp8
        and attn.qknorm_quant_type_value == QuantType.per_1x128.value,
        f"{name}: the q norm does not quantize q_b_proj's input per 1x128",
    )
    for module, what in (
        (attn.fused_qkv_a_proj, "fused_qkv_a_proj"),
        (attn.q_b_proj, "q_b_proj"),
    ):
        _need(getattr(module, "bias", None) is None, f"{name} {what} has a bias")
    w_a, s_a = _block_fp8(
        attn.fused_qkv_a_proj, f"{name} fused_qkv_a_proj", (Q_LORA + ENTRY, HIDDEN)
    )
    impl = getattr(attn.mla_attn, "impl", None)
    _need(
        callable(getattr(impl, "forward_decode_q_out", None)),
        f"{name}: not an MLAAttention",
    )
    _need(not impl.is_sparse_mla, f"{name}: sparse MLA")
    _need(not impl.use_seg_mla, f"{name}: the segmented KV cache")
    _need(not impl.kv_shuffled, f"{name}: the shuffled KV cache")
    _need(impl.kv_cache_dtype == "fp8", f"{name}: the KV cache is not FP8")
    _need(
        float(impl._k_scale) == 1.0 and float(impl._q_scale) == 1.0,
        f"{name}: static q / k scales other than 1",
    )
    _need(not attention_mla.is_rocm_aiter_fp4bmm_enabled(), f"{name}: the FP4 absorb")
    q_b = attn.q_b_proj
    _need(
        getattr(q_b, "native_a8_group_rows", None) is None
        and q_b.quant_type.value == QuantType.per_1x128.value
        and bool(getattr(q_b, "blockscale_e8m0_scale", False))
        and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE,
        f"{name}: q_b_proj is not the preshuffled per_1x128 E8M0 FP8 GEMM",
    )
    if impl.dcp_world_size > 1:
        _need(impl.qrep_enabled, f"{name}: DCP without QREP")
        w_uk, s_uk, heads = impl.W_K_qrep, impl.W_K_qrep_scale, impl.qrep_num_heads
        out_heads, compute_all = impl.dcp_kernel_num_heads, True
        _need(out_heads >= heads, f"{name}: {out_heads} kernel heads < {heads}")
    else:
        w_uk, s_uk, heads = impl.W_K, impl.W_K_scale, impl.num_heads
        out_heads = impl.padded_num_heads if impl._fused_q_head_pad else heads
        compute_all = False
    _need(s_uk is not None, f"{name}: the BF16 absorb (ATOM_MLA_ABSORB_BF16)")
    _need(heads % 2 == 0, f"{name}: {heads} query heads")
    _need(
        w_uk.dtype == torch.float8_e4m3fn
        and tuple(w_uk.shape) == (heads, KV_LORA, NOPE)
        and w_uk.is_contiguous(),
        f"{name}: W_UK is {w_uk.dtype} {tuple(w_uk.shape)}",
    )
    _need(
        s_uk.dtype == torch.float32
        and s_uk.numel() == 1
        and s_uk.device == w_uk.device,
        f"{name}: W_UK's scale is not one f32 beside it",
    )
    w_qb, s_qb = _block_fp8(q_b, f"{name} q_b_proj", (heads * HEAD, Q_LORA))
    codes = s_qb.view(torch.uint8)
    _need(
        not bool(((codes == 0) | (codes == 255)).any()),
        f"{name}: q_b_proj has an E8M0 scale of 2^-127 or NaN",
    )
    rope = impl.rotary_emb
    cos, sin = getattr(rope, "cos_cache", None), getattr(rope, "sin_cache", None)
    _need(
        getattr(rope, "rotary_dim", None) == ROPE
        and not rope.is_neox_style
        and cos is not None
        and sin is not None
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and cos.shape == sin.shape
        and cos.shape[-1] == ROPE // 2
        and cos.numel() == cos.shape[0] * (ROPE // 2)
        and cos.is_contiguous()
        and sin.is_contiguous(),
        f"{name}: a rotary embedding other than GPT-J over 64 BF16 dims",
    )
    return DraftMlaLayer(
        prefix=name,
        w_a=w_a,
        s_a=s_a,
        q_norm=_vec(attn.q_a_layernorm.weight, Q_LORA, f"{name} q_a_layernorm"),
        kv_norm=_vec(attn.kv_a_layernorm.weight, KV_LORA, f"{name} kv_a_layernorm"),
        q_eps=_eps(attn.q_a_layernorm, f"{name} q_a_layernorm"),
        kv_eps=_eps(attn.kv_a_layernorm, f"{name} kv_a_layernorm"),
        w_qb=w_qb,
        s_qb=s_qb,
        w_uk=w_uk,
        s_uk=s_uk.reshape(1),
        heads=heads,
        out_heads=out_heads,
        compute_all=compute_all,
        prepadded=compute_all and out_heads != heads,
        cos=cos,
        sin=sin,
        max_pos=cos.shape[0],
        attention=impl,
    )


class DraftMlaRunner:
    """The draft kernels of every width a step has used and the buffers they
    share (one ``qkv_c`` hand-off, one split-K scratch, one q exchange):
    the five draft layers run one after another on one stream."""

    def __init__(self, device: torch.device) -> None:
        from atom.models.kimi_k3.mono.kernels.stages import (
            mla_draft_front_layout as flay,
        )

        self.device = device
        self.qkv_c = torch.full(
            (flay.qkv_words(),), flay.UNWRITTEN - (1 << 32), dtype=torch.int32,
            device=device,
        )  # fmt: skip
        self.partials = torch.empty(
            flay.partial_bytes(), dtype=torch.uint8, device=device
        )
        self.q = None
        self.fronts: dict = {}
        self.absorbs: dict = {}
        self.refused: set = set()

    def _refuse(self, why: str) -> None:
        if why not in self.refused:
            self.refused.add(why)
            _logger().warning(
                "Kimi-K3 DSpark draft MLA: production kernels for %s", why
            )

    def _built(self, cache: dict, key: tuple, build):
        """``build()`` compiled, or None where it does not build or compiles
        with a resident hazard; built once a key, never while a CUDA graph is
        captured (None then, until an eager step builds it)."""
        if key in cache:
            return cache[key]
        if torch.cuda.is_current_stream_capturing():
            return None
        from atom.models.kimi_k3.mono.runner_gfx1250 import _resident_hazards
        from atom.mono.runtime.compile import compile_kernels

        try:
            kernel = build()
            (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
            hazards = _resident_hazards(launcher)
        except Exception as why:  # noqa: BLE001 -- any refusal, named
            kernel, hazards = None, (f"{type(why).__name__}: {why}",)
        if hazards:
            self._refuse(f"{key}: {', '.join(hazards)}")
            kernel = None
        cache[key] = kernel
        return kernel

    def front(self, layer: DraftMlaLayer, rows: int):
        """The width's front, or None where production's fused_qkv_a_proj
        does not split K in four (the front's GEMV mirrors that one)."""
        from atom.models.kimi_k3.mono import draft_dispatch
        from atom.models.kimi_k3.mono.kernels.stages import (
            mla_draft_front_gfx1250 as mf,
        )

        try:
            ksplit = draft_dispatch.qkv_a_split_k(rows)
        except MonoUnsupported as why:
            self._refuse(str(why))
            return None
        if ksplit != 4:
            return None
        return self._built(
            self.fronts,
            layer.front_key(rows, ksplit),
            lambda: mf.build_mla_draft_front(rows, ksplit, layer.q_eps, layer.kv_eps),
        )

    def absorb(self, layer: DraftMlaLayer, rows: int):
        from atom.models.kimi_k3.mono import draft_dispatch
        from atom.models.kimi_k3.mono.kernels.stages import (
            mla_absorb_draft_gfx1250 as ma,
        )

        try:
            split = draft_dispatch.q_b_split_groups(rows, layer.heads * HEAD)
        except MonoUnsupported as why:
            self._refuse(str(why))
            return None
        return self._built(
            self.absorbs,
            layer.absorb_key(rows, split),
            lambda: ma.build_mla_absorb_draft(
                rows, layer.heads, layer.out_heads, layer.compute_all, split,
                layer.max_pos,
            ),
        )  # fmt: skip

    def q_exchange(self, heads: int) -> torch.Tensor:
        words = MAX_ROWS * heads * HEAD // 2
        if self.q is None or self.q.numel() < words:
            self.q = torch.empty(words, dtype=torch.int32, device=self.device)
        return self.q


_RUNNERS: dict = {}


def runner_for(device: torch.device) -> DraftMlaRunner:
    key = (device.type, device.index)
    if key not in _RUNNERS:
        _RUNNERS[key] = DraftMlaRunner(device)
    return _RUNNERS[key]


def step_refusal(layer: DraftMlaLayer, rows: int, positions, hidden_states):
    """Why this step's draft decode is not the kernels' (None: it is): a
    dummy or prefill step, a width past MAX_ROWS, an input, slot mapping,
    positions or KV cache they do not read. Else ``(x, x_scale, slots,
    cache)``."""
    from aiter import dtypes

    from atom.utils.forward_context import get_forward_context

    forward = get_forward_context()
    if forward.context.is_dummy_run or forward.context.is_prefill:
        return "not a decode step", None
    if not isinstance(hidden_states, tuple) or len(hidden_states) != 2:
        return "an input norm output that is not an (FP8, scale) pair", None
    x, x_scale = hidden_states
    if not 1 <= rows <= MAX_ROWS:
        return f"{rows} rows", None
    if (
        x.dtype != dtypes.fp8
        or tuple(x.shape) != (rows, HIDDEN)
        or not x.is_contiguous()
        or x_scale.dtype != dtypes.fp8_e8m0
        or x_scale.numel() != rows * GROUPS_X
    ):
        return f"an input row of {x.dtype} {tuple(x.shape)}", None
    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.int64
        or positions.dim() != 1
        or positions.numel() != rows
        or not positions.is_contiguous()
    ):
        return f"positions of {getattr(positions, 'shape', None)}", None
    metadata = forward.attn_metadata
    slots = getattr(metadata, "slot_mapping", None)
    if (
        not isinstance(slots, torch.Tensor)
        or slots.dtype != torch.int64
        or slots.dim() != 1
        or slots.numel() != rows
        or not slots.is_contiguous()
    ):
        return f"a slot mapping of {getattr(slots, 'shape', None)}", None
    if getattr(metadata, "dtype_q", None) != dtypes.fp8:
        return f"q_out as {getattr(metadata, 'dtype_q', None)}", None
    cache = forward.kv_cache_data[f"layer_{layer.attention.layer_num}"].k_cache
    if (
        cache.dtype != dtypes.fp8
        or cache.dim() != 3
        or cache.size(0) == 0
        or tuple(cache.shape[1:]) != (1, ENTRY)
        or cache.stride(2) != 1
    ):
        return f"a KV cache of {cache.dtype} {tuple(cache.shape)}", None
    return None, (x, x_scale, slots, cache)


def _production_front(attn, x, x_scale):
    """``K3DSparkMLAAttention.forward``'s projections and norms, as it runs
    them: ``(q_fp8, q_scale, kv, k_rope)``, k_rope a view of the GEMM's
    output."""
    from atom.models.deepseek_v2 import _fuse_rmsnorm_quant
    from atom.models.dspark_draft import linear_out

    qkv_lora = linear_out(attn.fused_qkv_a_proj(x, x_scale))
    q_lora, kv_c, k_pe = qkv_lora.split(
        [attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
    )
    (q_c, q_scale), _, kv_c, _ = _fuse_rmsnorm_quant(
        q_lora,
        attn.q_a_layernorm.weight,
        attn.q_a_layernorm.eps,
        kv_c,
        attn.kv_a_layernorm.weight,
        attn.kv_a_layernorm.eps,
        None,
        dtype_quant=attn.qknorm_dtype,
        shuffle=False,
        scale_shuffle_padding=False,
        group_size=128,
        quant_type=attn.qknorm_quant_type_value,
        output_unquantized_inp1=False,
        transpose_scale=True,
    )
    return q_c, q_scale, kv_c, k_pe


def forward(attn, positions, hidden_states):
    """``K3DSparkMLAAttention.forward`` on the draft kernels: its output, or
    None where the layer or the step is production's."""
    layer = attn.mla_draft
    if layer is None:
        if torch.cuda.is_current_stream_capturing():
            # binding reads the weights' scales back and allocates the shared
            # buffers: an eager step's business
            return None
        try:
            layer = bind(attn)
            runner_for(layer.w_a.device).q_exchange(layer.heads)
            attn.mla_draft = layer
        except Exception as why:  # noqa: BLE001 -- any refusal, named
            attn._try_mla_draft = False
            _logger().warning(
                "Kimi-K3 DSpark draft MLA: %s keeps the production kernels (%s)",
                attn.input_quant_prefix,
                why,
            )
            return None
    rows = positions.numel() if isinstance(positions, torch.Tensor) else -1
    runner = runner_for(layer.w_a.device)
    why, found = step_refusal(layer, rows, positions, hidden_states)
    if why is not None:
        runner._refuse(why)
        return None
    x, x_scale, slots, cache = found
    absorb = runner.absorb(layer, rows)
    if absorb is None:
        runner._refuse(f"{rows} rows: no absorb built")
        return None
    # None: production's projections and norms (the same bits), where the
    # dispatch does not split K in four or the width's front is not built
    front = runner.front(layer, rows)
    q_out = launch(
        runner, layer, front, absorb, attn, x, x_scale, slots, cache, positions
    )
    return layer.attention.forward_decode_q_out(q_out, prepadded=layer.prepadded)


def launch(runner, layer, front, absorb, attn, x, x_scale, slots, cache, positions):
    """The front (or production's projections and norms where it is None)
    and the absorb: the attention's q_out; the step's KV-cache entries
    written."""
    from aiter import dtypes

    from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_draft_gfx1250 as ma
    from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_gfx1250 as mf

    rows = x.size(0)
    stream = torch.cuda.current_stream()
    p = torch.Tensor.data_ptr
    if front is not None:
        q_fp8 = torch.empty((rows, Q_LORA), dtype=dtypes.fp8, device=x.device)
        q_scale = torch.empty((rows, Q_GROUPS), dtype=torch.float32, device=x.device)
        kv = torch.empty((rows, KV_LORA), dtype=torch.bfloat16, device=x.device)
        k_rope = torch.empty((rows, ROPE), dtype=torch.bfloat16, device=x.device)
        args = mf.MLA_DRAFT_FRONT_ABI.pack({
            "x": p(x), "x_scale": p(x_scale), "w_a": p(layer.w_a), "s_a": p(layer.s_a),
            "q_norm": p(layer.q_norm), "kv_norm": p(layer.kv_norm),
            "qkv_c": p(runner.qkv_c), "q_fp8": p(q_fp8), "q_scale": p(q_scale),
            "kv": p(kv), "k_rope": p(k_rope), "partials": p(runner.partials),
            "qkv_seen": 0,
        })  # fmt: skip
        front.launcher(*args, stream=stream)
    else:
        q_fp8, q_scale, kv, k_rope = _production_front(attn, x, x_scale)
        if k_rope.stride(1) != 1 or k_rope.stride(0) % 2:
            k_rope = k_rope.contiguous()
    q_out = torch.empty(
        (rows, layer.out_heads, ENTRY), dtype=dtypes.fp8, device=x.device
    )
    args = ma.MLA_ABSORB_DRAFT_ABI.pack({
        "q_fp8": p(q_fp8), "q_scale": p(q_scale), "w_qb": p(layer.w_qb),
        "s_qb": p(layer.s_qb), "w_uk": p(layer.w_uk), "s_uk": p(layer.s_uk),
        "kv": p(kv), "k_rope": p(k_rope), "k_rope_stride": k_rope.stride(0),
        "slots": p(slots), "positions": p(positions), "cos": p(layer.cos),
        "sin": p(layer.sin), "kv_cache": p(cache), "kv_stride": cache.stride(0),
        "q_out": p(q_out), "q": p(runner.q_exchange(layer.heads)), "ql_nope": 0,
    })  # fmt: skip
    absorb.launcher(*args, stream=stream)
    return q_out
