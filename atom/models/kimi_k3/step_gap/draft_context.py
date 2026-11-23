# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Kimi-K3 DSpark draft's context path of a decode step on the step-gap
kernels (``ATOM_KIMI_MONO_DRAFT_CONTEXT``): what ``KimiK3DSpark``'s
``write_context_kv`` does after a target forward, production's cache rows in
fewer launches.

Production, per draft layer: ``fused_qkv_a_proj`` quantizes the (same)
projected context to E8M0 FP8 and runs its full GEMM, and the attention writes
the kv_a | rope columns to the layer's cache. Here (part ``kv``) the context
is quantized once, by the same call, and ``ctx_kv_kernels`` computes every
layer's kv_a | rope rows in one launch; then (part ``store``) one launch of
``ctx_kv_store`` writes every layer's cache where each layer's write would be
production's fused kernel on one cache layout (``StorePlan``), else each
layer's cache write is production's own
(``MLAAttention.write_context_kv_latent``) on those rows.

Nothing here waits on another rank (the projection is replicated), so no
decision needs the TP group's agreement: a rank that refuses a layer, a width
or a step runs production's kernels and writes the same rows. Refused: a dummy
or prefill step, a PCP split, a width past ``MAX_ROWS``, a width first seen
while a CUDA graph is captured, a projection the kernel does not read (any
other quant scheme or layout, a bias, a static activation scale), a kernel
that does not build or builds with a resident hazard.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.mono.runtime.consensus import MonoUnsupported

HIDDEN = 7168
Q_LORA = 1536
KV_LORA = 512
ROPE = 64
ENTRY = KV_LORA + ROPE
GROUPS_X = HIDDEN // 128
MAX_ROWS = 64


def _logger():
    import logging

    return logging.getLogger("atom")


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


@dataclass(frozen=True)
class ContextLayer:
    """A draft layer as the context kernels read it: its attention and
    ``fused_qkv_a_proj``'s preshuffled FP8 weight and E8M0 block scales, in
    place."""

    prefix: str
    attn: object
    w_a: torch.Tensor
    s_a: torch.Tensor


def bind_layer(attn, name: str) -> ContextLayer:
    """One draft layer's tensors. Refused where ``write_context_kv`` would
    compute anything but: the dynamic per-1x128 E8M0 quant of the context,
    the preshuffled FP8 GEMM of the replicated ``fused_qkv_a_proj`` (no bias,
    no reduce), and the attention's context write."""
    from aiter import QuantType, dtypes

    from atom.models.kimi_k3.mono.weights import _block_fp8
    from atom.utils import envs

    linear = attn.fused_qkv_a_proj
    _need(
        (attn.hidden_size, attn.q_lora_rank, attn.kv_lora_rank, attn.qk_rope_head_dim)
        == (HIDDEN, Q_LORA, KV_LORA, ROPE),
        f"{name}: MLA widths",
    )
    _need(getattr(linear, "bias", None) is None, f"{name} fused_qkv_a_proj has a bias")
    _need(
        getattr(linear, "native_a8_group_rows", None) is None
        and linear.quant_type.value == QuantType.per_1x128.value
        and bool(getattr(linear, "blockscale_e8m0_scale", False))
        and bool(envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE),
        f"{name}: fused_qkv_a_proj is not the preshuffled per_1x128 E8M0 FP8 GEMM",
    )
    _need(
        getattr(linear, "params_dtype", None) == dtypes.fp8,
        f"{name}: activations as {getattr(linear, 'params_dtype', None)}",
    )
    _need(
        getattr(linear, "input_scale", None) is None,
        f"{name}: fused_qkv_a_proj has a static activation scale",
    )
    _need(
        getattr(linear, "tp_dim", None) != 1,
        f"{name}: fused_qkv_a_proj reduces its output",
    )
    impl = getattr(getattr(attn, "mla_attn", None), "impl", None)
    _need(
        callable(getattr(impl, "write_context_kv_latent", None)),
        f"{name}: not an MLAAttention",
    )
    w_a, s_a = _block_fp8(linear, f"{name} fused_qkv_a_proj", (Q_LORA + ENTRY, HIDDEN))
    return ContextLayer(name, attn, w_a, s_a)


@dataclass(frozen=True)
class StorePlan:
    """The one-launch store's arguments: every layer's (norm weight, cache)
    pointers on the device and what the layers share."""

    tab: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    cache_like: torch.Tensor
    k_scale: torch.Tensor
    eps: float
    is_neox: bool
    apply_scale: bool


def _store_signature(layers) -> tuple:
    """What the store plan depends on, as host values (no device read): a
    change (the engine binding the caches, a layer's rope cache reaching the
    device) rebuilds the plan."""
    sig = []
    for layer in layers:
        attention = layer.attn.mla_attn
        impl, cache = attention.impl, attention.kv_cache
        norm, rope = layer.attn.kv_a_layernorm, impl.rotary_emb
        cos, sin = rope.cos_cache, rope.sin_cache
        sig.append((
            cache.data_ptr(), tuple(cache.shape[1:]), cache.stride(), cache.dtype,
            cache.device, cos.data_ptr(), cos.device, cos.dtype, cos.stride(),
            sin.data_ptr(), norm.weight.data_ptr(), norm.weight.dtype,
            float(norm.eps), impl._k_scale_device.data_ptr(), impl.kv_cache_dtype,
            bool(rope.is_neox_style), bool(impl._ctx_kv_fusion_enabled),
        ))  # fmt: skip
    return tuple(sig)


def build_store_plan(layers) -> StorePlan | None:
    """The plan where every layer's ``write_context_kv_latent`` would take
    its fused kernel (that method's own conditions) with one cache layout,
    one rope cache, one eps and one k scale; else None."""
    from atom.model_ops.layernorm import RMSNorm

    first = layers[0].attn
    impl0, cache0 = first.mla_attn.impl, first.mla_attn.kv_cache
    rope0 = impl0.rotary_emb
    pointers, scales = [], []
    for layer in layers:
        attention = layer.attn.mla_attn
        impl, cache = attention.impl, attention.kv_cache
        norm, rope = layer.attn.kv_a_layernorm, impl.rotary_emb
        fused = (
            impl._ctx_kv_fusion_enabled
            and cache.dim() == 3
            and cache.shape[-1] == impl.kv_lora_rank + impl.qk_rope_head_dim == ENTRY
            and cache.stride(-1) == 1
            and rope.cos_cache.device == cache.device
            and isinstance(norm, RMSNorm)
        )
        alike = fused and (
            cache.numel() > 0
            and cache.dtype == cache0.dtype
            and cache.device == cache0.device
            and cache.shape[1] == cache0.shape[1]
            and cache.stride() == cache0.stride()
            and rope.cos_cache.data_ptr() == rope0.cos_cache.data_ptr()
            and rope.sin_cache.data_ptr() == rope0.sin_cache.data_ptr()
            and rope.cos_cache.stride() == rope0.cos_cache.stride()
            and rope.is_neox_style == rope0.is_neox_style
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.numel() == KV_LORA
            and norm.weight.is_contiguous()
            and norm.weight.device == cache.device
            and norm.eps == first.kv_a_layernorm.eps
            and impl.kv_cache_dtype.startswith("fp8")
            == impl0.kv_cache_dtype.startswith("fp8")
        )
        if not alike:
            return None
        pointers += [norm.weight.data_ptr(), cache.data_ptr()]
        scales.append(impl._k_scale_device)
    if not all(pointers):
        return None
    apply_scale = impl0.kv_cache_dtype.startswith("fp8")
    # read back once a plan (a cache was rebound): the layers' static scales
    if apply_scale and len({float(k) for k in scales}) != 1:
        return None
    tab = torch.tensor(pointers, dtype=torch.int64, device=cache0.device)
    return StorePlan(
        tab, rope0.cos_cache, rope0.sin_cache, cache0, impl0._k_scale_device,
        float(first.kv_a_layernorm.eps), bool(rope0.is_neox_style), apply_scale,
    )  # fmt: skip


class DraftContextRunner:
    """A draft model's context kernels, a build a width, and what they share:
    the layers' pointer table and the split-K scratch."""

    def __init__(self, layers: list[ContextLayer], parts=frozenset(("kv",))) -> None:
        from atom.models.kimi_k3.step_gap import ctx_kv_layout as lay

        self.parts = frozenset(parts)
        self._store_sig = None
        self._store_plan = None

        _need(
            1 <= len(layers) <= lay.MAX_LAYERS,
            f"{len(layers)} draft layers (1..{lay.MAX_LAYERS})",
        )
        self.layers = layers
        self.device = layers[0].w_a.device
        _need(
            all(layer.w_a.device == self.device for layer in layers),
            "draft layers on several devices",
        )
        self.tab = torch.tensor(
            [
                p
                for layer in layers
                for p in (layer.w_a.data_ptr(), layer.s_a.data_ptr())
            ],
            dtype=torch.int64,
            device=self.device,
        )
        _need(bool((self.tab != 0).all()), "a null weight pointer")
        self.partials = torch.empty(
            lay.partial_bytes(len(layers)), dtype=torch.uint8, device=self.device
        )
        self.kernels: dict[int, object] = {}
        self.refused: set[str] = set()

    def refuse(self, why: str) -> None:
        if why not in self.refused:
            self.refused.add(why)
            _logger().warning(
                "Kimi-K3 DSpark draft context: production kernels for %s", why
            )

    def kernel(self, rows: int):
        """The width's projection, or None: a width whose build is refused,
        or one first seen while a CUDA graph is captured (None then, until an
        eager step builds it)."""
        if rows in self.kernels:
            return self.kernels[rows]
        if torch.cuda.is_current_stream_capturing():
            return None
        from atom.models.kimi_k3.mono import draft_dispatch
        from atom.models.kimi_k3.mono.runner_gfx1250 import _resident_hazards
        from atom.models.kimi_k3.step_gap import ctx_kv_kernels as K
        from atom.models.kimi_k3.step_gap import ctx_kv_layout as lay
        from atom.mono.runtime.compile import compile_kernels

        try:
            ksplit = draft_dispatch.qkv_a_split_k(rows)
            why = lay.refusal(rows, ksplit, len(self.layers))
            _need(not why, why)
            kernel = K.build_ctx_kv_proj(rows, ksplit, len(self.layers))
            (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
            hazards = _resident_hazards(launcher)
        except Exception as why:  # noqa: BLE001 -- any refusal, named
            kernel, hazards = None, (f"{type(why).__name__}: {why}",)
        if hazards:
            self.refuse(f"{rows} context rows: {', '.join(hazards)}")
            kernel = None
        else:
            _logger().info(
                "Kimi-K3 DSpark draft context: %d rows on %s", rows, kernel.name
            )
        self.kernels[rows] = kernel
        return kernel

    def write(self, kernel, ctx_hidden, positions, slot_mapping) -> bool:
        """The layers' context rows of ``ctx_hidden`` written through the
        projection launch; False (nothing launched or written) where the
        quantized context is not what the kernel reads."""
        from aiter import dtypes

        from atom.model_ops.linear import per_1x128_e8m0_quant

        rows = ctx_hidden.size(0)
        # fused_qkv_a_proj's own activation quant (LinearBase.forward), once
        x8, x_scale = per_1x128_e8m0_quant(
            ctx_hidden, quant_dtype=dtypes.fp8, transpose_scale=True, scale=None
        )
        if (
            x8.dtype != dtypes.fp8
            or tuple(x8.shape) != (rows, HIDDEN)
            or x_scale.dtype != dtypes.fp8_e8m0
            or x_scale.numel() != rows * GROUPS_X
        ):
            self.refuse(f"a quantized context of {x8.dtype} {tuple(x8.shape)}")
            return False
        if not x8.is_contiguous():
            x8 = x8.contiguous()
        out = torch.empty(
            (len(self.layers), rows, ENTRY), dtype=torch.bfloat16, device=self.device
        )
        p = torch.Tensor.data_ptr
        args = kernel.abi.pack({
            "tab": p(self.tab), "x": p(x8), "x_scale": p(x_scale), "out": p(out),
            "partials": p(self.partials),
        })  # fmt: skip
        if not all(args):
            self.refuse("a null pointer")
            return False
        kernel.launcher(*args, stream=torch.cuda.current_stream())
        plan = self.store_plan()
        if plan is not None:
            from atom.models.kimi_k3.step_gap import ctx_kv_store

            ctx_kv_store.store_layers(
                out, plan.tab, positions, plan.cos, plan.sin, slot_mapping.flatten(),
                plan.cache_like, plan.k_scale, plan.eps, KV_LORA, ROPE, plan.is_neox,
                plan.apply_scale,
            )  # fmt: skip
            return True
        for i, layer in enumerate(self.layers):
            attention = layer.attn.mla_attn
            attention.impl.write_context_kv_latent(
                attention.kv_cache,
                out[i],
                positions,
                slot_mapping,
                layer.attn.kv_a_layernorm,
            )
        return True

    def store_plan(self) -> StorePlan | None:
        """This step's one-launch store, or None (each layer writes through
        its attention): the part is off, or the layers' writes are not one
        fused layout (``build_store_plan``). Rebuilt when what it depends on
        changed (``_store_signature``)."""
        if "store" not in self.parts:
            return None
        signature = _store_signature(self.layers)
        if signature != self._store_sig:
            self._store_plan = build_store_plan(self.layers)
            self._store_sig = signature
            _logger().info(
                "Kimi-K3 DSpark draft context: the cache writes are %s",
                (
                    "one launch"
                    if self._store_plan is not None
                    else "production's, a layer"
                ),
            )
        return self._store_plan


def bind(model) -> DraftContextRunner | None:
    """The model's runner, bound on its first eligible step; a model the
    kernels cannot read keeps production's path for good."""
    try:
        layers = [
            bind_layer(layer.self_attn, f"layers.{i}.self_attn")
            for i, layer in enumerate(model.context_layers)
        ]
        from atom.utils import envs

        runner = DraftContextRunner(layers, envs.ATOM_KIMI_MONO_DRAFT_CONTEXT)
    except Exception as why:  # noqa: BLE001 -- any refusal, named
        model._try_mono_context = False
        _logger().warning(
            "Kimi-K3 DSpark draft context: the production kernels stay (%s)", why
        )
        return None
    model.mono_context = runner
    _logger().info("Kimi-K3 DSpark draft context bound (%d layers)", len(runner.layers))
    return runner


def step_rows(positions) -> int:
    """The step's context rows where the kernels may take it, else 0: a
    decode step of 1..MAX_ROWS rows outside a graph capture."""
    from atom.utils.forward_context import get_forward_context

    context = get_forward_context().context
    if context.is_dummy_run or context.is_prefill:
        return 0
    if not isinstance(positions, torch.Tensor) or positions.dim() != 1:
        return 0
    rows = positions.shape[0]
    if not 1 <= rows <= MAX_ROWS or torch.cuda.is_current_stream_capturing():
        return 0
    return rows


def context_slots(rows: int):
    """The context rows' slots for a tree whose draft layers read them off
    the forward context instead of taking them as an argument: the target
    forward's slot mapping cut to the context rows, the layer's own
    expression."""
    from atom.utils.forward_context import get_forward_context

    return get_forward_context().attn_metadata.slot_mapping[:rows]


def write_context_kv(model, source, positions, *rest) -> bool:
    """``DSparkDraftModel.write_context_kv`` on the context kernels: True
    once the layers' context rows are written; False, with nothing launched,
    where the model or the step is production's.

    For both shapes that method has: ``source`` is whatever the model's
    ``project_context`` takes (the aux taps, or their concatenation), and
    ``rest`` is what the caller hands its layers after ``positions`` -- the
    slot mapping, or nothing where a layer reads it off the forward context
    (``context_slots``). Nothing here depends on which: the projection is the
    model's own, and a refused write is finished by the layers' own method
    with the caller's arguments."""
    rows = step_rows(positions)
    if not rows or len(rest) > 1 or (rest and rest[0] is None):
        return False
    runner = model.mono_context
    if runner is None:
        runner = bind(model)
        if runner is None:
            return False
    kernel = runner.kernel(rows)
    if kernel is None:
        return False
    ctx_hidden = model.project_context(source)
    if (
        isinstance(ctx_hidden, torch.Tensor)
        and ctx_hidden.dtype == torch.bfloat16
        and tuple(ctx_hidden.shape) == (rows, HIDDEN)
    ):
        slot_mapping = rest[0] if rest else context_slots(rows)
        if runner.write(kernel, ctx_hidden, positions, slot_mapping):
            return True
    # the context is projected: finish as production does
    runner.refuse(f"a context of {getattr(ctx_hidden, 'shape', None)}")
    for layer in model.context_layers:
        layer.write_context_kv(ctx_hidden, positions, *rest)
    return True
