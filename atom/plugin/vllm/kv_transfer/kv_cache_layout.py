# SPDX-License-Identifier: MIT
"""Map vLLM's flat KV-cache registration onto ATOM's ``KVCacheTensor`` list.

vLLM hands a connector ``{layer_name: tensor}`` and says nothing about what is
inside each tensor. ATOM's offload codec instead wants, per layer, the movable
tensors named apart (``k_cache`` / ``v_cache`` / scales / ``index_cache``),
because it moves each one as its own contiguous byte segment. This module is
that translation, and nothing else -- no transfer policy, no LMCache.

Why the split matters (MiniMax-M3, the model that forced this):

    dense layers  (nb, 1, bs, 2*hd)       K and V interleaved per token
    sparse layers (nb, 2, bs, nh, hd)     K and V in two SEPARATE regions
    index caches  (nb, bs, hd)            DSA indexer keys, one per sparse layer

A sparse layer's tensor is NOT contiguous as a whole: its ``stride(1)`` jumps
across the entire K region (measured on M3-MXFP4: shape ``(59454, 2, 128, 1,
128)``, stride ``(16384, 974094336, 128, 128, 1)``). ``DenseKVByteCodec``
rejects non-contiguous segments, so the whole tensor cannot be one segment --
but ``t[:, 0]`` and ``t[:, 1]`` each ARE contiguous, and that is exactly the
``k_cache`` / ``v_cache`` pair the codec already expects. Dense layers stay
whole: their K and V interleave inside a block, so the block's bytes are one
opaque run and splitting them would be meaningless.

The codec never interprets these bytes, so "which tensor is K" only has to be
consistent between save and restore -- not semantically right.

What vLLM does NOT hand over is quantisation state. M3's sparse layers keep an
fp32 scale per token per head next to the fp8 cache, on the layer object; the
registration dict has only the caches. Restoring mantissas against the previous
occupant's scales is silent corruption, so the layers are asked for their scales
here and a layer that clearly owns per-block scales without a way to report them
is a hard error rather than a quiet omission.
"""

from typing import Any

import torch

from atom.config import KVCacheTensor

INDEX_CACHE_SUFFIX = ".index_cache"


def _layer_sort_key(layer_name: str) -> tuple:
    """Order layers by their numeric position, falling back to name order.

    Segment order must be identical on save and restore, and dict insertion
    order is whatever vLLM's registration happened to produce.
    """
    parts = []
    for token in layer_name.replace("/", ".").split("."):
        parts.append((0, int(token), "") if token.isdigit() else (1, 0, token))
    return tuple(parts)


def split_kv_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(k_cache, v_cache)`` for one layer's registered KV tensor.

    ``v_cache`` is None when K and V share one opaque per-block run (dense
    layers), in which case the whole tensor travels as ``k_cache``.
    """
    # (nb, 2, ...) -- K/V split across dim 1, each half contiguous on its own.
    if tensor.ndim >= 4 and tensor.shape[1] == 2:
        return tensor[:, 0], tensor[:, 1]
    # Anything else (incl. dense (nb, 1, bs, 2*hd)) is one opaque run.
    return tensor, None


def _transfer_scales(
    name: str, layer: Any, tensor: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Ask one layer for the scales that must travel with its KV bytes.

    Duck-typed on purpose: only the layers that HAVE movable scales implement
    the hook, and the connector must not need a table of which model does.

    The fallback is not "assume none". A layer holding a multi-element
    `k_scale`/`v_scale` has per-block quantisation state, and moving its KV
    without that state restores mantissas against a stale scale -- wrong output
    with nothing logged. Without the hook there is no way to move it, so say so
    here rather than at the far end of a wrong answer. Scalar scales (vLLM's
    usual per-tensor `_k_scale`) are constant and correctly ignored.
    """
    getter = getattr(layer, "get_kv_transfer_scales", None)
    if getter is not None:
        k_scale, v_scale = getter(tensor)
        return k_scale, v_scale

    for attr in ("k_scale", "v_scale", "_k_scale", "_v_scale"):
        scale = getattr(layer, attr, None)
        if isinstance(scale, torch.Tensor) and scale.numel() > 1:
            raise ValueError(
                f"{name}: layer holds a per-block {attr} "
                f"(shape={tuple(scale.shape)}) but has no "
                "get_kv_transfer_scales(); moving its KV without the scales "
                "would restore mantissas against the previous occupant's scale"
            )
    return None, None


def build_kv_cache_tensors(
    kv_caches: dict[str, torch.Tensor],
    layers: dict[str, Any] | None = None,
) -> list[KVCacheTensor]:
    """Translate vLLM's ``{layer_name: tensor}`` into ATOM ``KVCacheTensor``s.

    Layers named ``<layer><INDEX_CACHE_SUFFIX>`` are folded into the owning
    layer's ``index_cache`` rather than becoming layers of their own -- vLLM
    registers M3's DSA indexer keys as separate entries, but they are part of
    the same layer's movable bytes.

    Args:
        kv_caches: vLLM's registration dict.
        layers: the layer modules by name, for the quantisation state that does
            not travel in ``kv_caches``. Omit only when no registered layer has
            per-block scales.

    Raises:
        ValueError: a produced segment is not contiguous (the codec would
            reject it later, with far less context about which layer), or a
            layer owns per-block scales it cannot report.
    """
    index_caches: dict[str, torch.Tensor] = {}
    main: dict[str, torch.Tensor] = {}
    for name, tensor in kv_caches.items():
        if name.endswith(INDEX_CACHE_SUFFIX):
            index_caches[name[: -len(INDEX_CACHE_SUFFIX)]] = tensor
        else:
            main[name] = tensor

    orphans = set(index_caches) - set(main)
    if orphans:
        raise ValueError(
            f"index caches registered without their owning layer: {sorted(orphans)}"
        )

    out: list[KVCacheTensor] = []
    for layer_num, name in enumerate(sorted(main, key=_layer_sort_key)):
        k_cache, v_cache = split_kv_tensor(main[name])
        index_cache = index_caches.get(name)
        k_scale, v_scale = _transfer_scales(
            name, (layers or {}).get(name), main[name]
        )

        for role, seg in (
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
            ("index_cache", index_cache),
        ):
            if seg is not None and not seg.is_contiguous():
                raise ValueError(
                    f"{name}: {role} is not contiguous "
                    f"(shape={tuple(seg.shape)}, stride={seg.stride()}); "
                    "the byte codec can only move contiguous segments"
                )

        # The codec derives each segment's per-block stride as
        # numel // num_blocks, so a scale whose leading axis is not the block
        # axis would be sliced at the wrong granularity -- and, being the same
        # dtype and roughly the right size, would not fail any later check.
        num_blocks = int(k_cache.shape[0])
        for role, seg in (("k_scale", k_scale), ("v_scale", v_scale)):
            if seg is not None and int(seg.shape[0]) != num_blocks:
                raise ValueError(
                    f"{name}: {role} is not block-major "
                    f"(shape={tuple(seg.shape)}, num_blocks={num_blocks})"
                )

        out.append(
            KVCacheTensor(
                layer_num=layer_num,
                k_cache=k_cache,
                v_cache=v_cache if v_cache is not None else torch.tensor([]),
                k_scale=k_scale,
                v_scale=v_scale,
                index_cache=index_cache,
            )
        )
    return out
