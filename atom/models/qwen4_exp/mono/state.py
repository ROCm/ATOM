# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's caches the mono kernels read and write in place, checked to be laid
out as the kernels address them.

- **DeltaNet state.** One pool per kind for all GDN layers
  (``runner.mamba_k_cache`` / ``mamba_v_cache``, ``[gdn layers, slots, ...]``;
  a layer's row is the order it was bound in, ``pool_rows``, not its number).
  The conv state is time-major ``[3 + k, QKV]`` bf16 (B1; k the MTP draft
  tokens, 0 without MTP), the recurrent
  state ``[48, 128, 128]`` f32 stored VK: B3's layout. ATOM
  stores it so for its FlyDSL decode (gfx942 only) and otherwise when asked
  (``Qwen4ExpMetadataBuilder.gdn_state_vk``, which ``install_mono_decode``
  sets); its layers see a KV view either way (a transpose).
- **PLE state.** ``runner.ple_conv_state`` ``[slots, QKV, 9 + k]`` bf16, the
  same slots as the GDN state.
- **QSA caches.** Each QSA layer's ``k_cache`` / ``v_cache`` ``[pages, 16, 2,
  256]``, ``raw_key_cache`` ``[pages, 16, 1, 128]`` and
  ``compressed_key_cache`` ``[pages, 4, 1, 128]`` bf16, paged by the
  request's block table (``qsa.page_block``).

The kernels address a cache through a buffer resource: 32-bit offsets, so a
layer's cache (and a slot's state) must stay below 2 GiB."""

from dataclasses import dataclass

import torch

from atom.model_ops.attentions.gdn_attn import LINEAR_STATE_ROWS
from atom.model_ops.attentions.qwen4_exp_attn import QSA_ROWS
from atom.mono.runtime.consensus import MonoUnsupported

from .kernels.config import DEFAULT as C
from .kernels.stages.qsa import PAGE, PAGE_BLOCKS

BF16, F32 = torch.bfloat16, torch.float32
# a buffer resource's reach: the kernels' byte offsets from a field's base are
# 32-bit, unsigned (offsets up to 4 GiB address correctly; past it they wrap)
MAX_FIELD_BYTES = 2**32
# the conv windows the kernels keep: kernel width - 1, plus one row per MTP
# draft token in ATOM's pools (``step_meta``'s verify layout)
CONV_ROWS = C.delta_conv - 1
# the PLE conv window: (kernel - 1) x dilation (ngram_size); ATOM's pools add
# a column per MTP draft token
PLE_CONV_COLS = 9
# a QSA layer's caches by ``qsa.STATE`` name: (module attribute, page shape)
QSA_CACHES = {
    "key": ("k_cache", (PAGE, C.kv_heads, C.head_dim)),
    "value": ("v_cache", (PAGE, C.kv_heads, C.head_dim)),
    "index_key": ("raw_key_cache", (PAGE, 1, C.index_dim)),
    "index_cmp": ("compressed_key_cache", (PAGE_BLOCKS, 1, C.index_dim)),
}


@dataclass(frozen=True)
class StateCaches:
    """The caches of :mod:`state`'s docstring. ``gdn_rows`` maps a DeltaNet
    layer to its row of ``conv`` / ``rec``; ``qsa`` a QSA layer to its caches
    by ``qsa.STATE`` name."""

    conv: torch.Tensor
    rec: torch.Tensor
    gdn_rows: dict[int, int]
    ple_conv: torch.Tensor
    qsa: dict[int, dict[str, torch.Tensor]]

    @property
    def slots(self) -> int:
        return self.conv.shape[1]


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _check(t, name, shape, dtype, field_bytes) -> None:
    _need(t is not None, f"{name} not allocated")
    _need(
        tuple(t.shape) == tuple(shape) and t.dtype == dtype,
        f"{name} {tuple(t.shape)} {t.dtype}, need {tuple(shape)} {dtype}",
    )
    _need(t.is_contiguous(), f"{name} strides {t.stride()}")
    _need(field_bytes < MAX_FIELD_BYTES, f"{name}: {field_bytes} bytes per field")


def bind_state(builder, layers, kinds, spec_tokens=0) -> StateCaches:
    """The caches of ``layers`` (the decoder layers, ``kinds`` as
    ``Qwen4Config.kind``) from the metadata builder ``builder``'s runner, with
    room for ``spec_tokens`` MTP draft tokens, or ``MonoUnsupported``."""
    runner = builder.model_runner
    rows = builder.pool_rows
    # its kernels address the recurrent state as a contiguous KV pool
    _need(not getattr(builder, "replayssm", False), "ReplaySSM on")
    conv = getattr(runner, "mamba_k_cache", None)
    rec = getattr(runner, "mamba_v_cache", None)
    _need(conv is not None and rec is not None, "GDN state not allocated")
    gdn, slots = conv.shape[0], conv.shape[1]
    conv_rows = CONV_ROWS + spec_tokens
    _check(conv, "GDN conv state", (gdn, slots, conv_rows, C.delta_qkv), BF16, 0)
    rec_shape = (C.delta_v_heads, C.delta_v_dim, C.delta_k_dim)
    _check(rec, "GDN recurrent state", (gdn, slots, *rec_shape), F32, 0)
    for name, t in (("conv", conv), ("recurrent", rec)):
        slot_bytes = t[0, 0].numel() * t.element_size()
        _need(slot_bytes < MAX_FIELD_BYTES, f"GDN {name} slot {slot_bytes} bytes")
    ple_conv = getattr(runner, "ple_conv_state", None)
    ple_cols = PLE_CONV_COLS + spec_tokens
    _check(ple_conv, "PLE conv state", (slots, C.hc_width, ple_cols), BF16, 0)

    gdn_rows, qsa = {}, {}
    for i, layer in enumerate(layers):
        if kinds[i] == "qsa":
            qsa[i] = bind_qsa_caches(builder, layer, i)
        else:
            module = layer.linear_attn.attn
            row = rows.get(LINEAR_STATE_ROWS, {}).get(module)
            _need(row is not None and 0 <= row < gdn, f"layer {i} GDN row {row}")
            # stored VK exactly when the builder transposed the layer's view
            policy = getattr(module.impl, "gdn_flydsl_policy", None)
            _need(
                policy is not None and (policy.decode or builder.gdn_state_vk),
                f"layer {i} GDN recurrent state not stored VK",
            )
            gdn_rows[i] = row
    _need(len(set(gdn_rows.values())) == len(gdn_rows), "GDN rows shared")
    return StateCaches(conv, rec, gdn_rows, ple_conv, qsa)


def bind_qsa_caches(builder, layer, i: int) -> dict[str, torch.Tensor]:
    """QSA decoder layer ``layer``'s caches by ``qsa.STATE`` name (``i`` its
    number, for the refusals), or ``MonoUnsupported``."""
    module = layer.self_attn
    _need(getattr(module, "is_qsa_attention", False), f"layer {i} not QSA")
    _need(
        module in builder.pool_rows.get(QSA_ROWS, {}), f"layer {i} QSA caches unbound"
    )
    _need(
        getattr(module, "rope_position_cache", None) is None,
        f"layer {i} caches mRoPE positions",
    )
    caches = {}
    for name, (attr, page) in QSA_CACHES.items():
        t = getattr(module, attr, None)
        pages = t.shape[0] if t is not None else 0
        field = t.numel() * t.element_size() if t is not None else 0
        _check(t, f"layer {i} {attr}", (pages, *page), BF16, field)
        caches[name] = t
    _need(
        len({t.shape[0] for t in caches.values()}) == 1,
        f"layer {i} QSA caches page counts differ",
    )
    return caches
