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

GLM-5.2 (``GlmMoeDsaForCausalLM``) is the simpler shape of the same idea and
needs no splitting at all:

    MLA layers    (nb, bs, 576)           one latent cache, K and V fused
    index caches  (nb, bs, hd+4)          uint8, fp8 keys with their scale
                                          packed INTO the row

Both of its tensors are block-major and contiguous, so each travels whole. Its
indexer also needs no scale hook: unlike M3, its quantisation state is inside
the bytes being moved.

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

Hybrid models (Kimi-K3: MLA full attention plus KDA recurrent layers) make vLLM
build more than one KV cache group, and the registration dict is still flat --
every group's layers arrive in the same ``{layer_name: tensor}``. A group's
pages have their own geometry and their own block count, so the two groups can
never share one codec. ``split_kv_caches_by_group`` is that separation, done
before ``build_kv_cache_tensors`` rather than inside it, so single-group models
(the M3 / GLM-5.2 path) keep taking the dict exactly as vLLM handed it over.
"""

from collections.abc import Sequence
from typing import Any

import torch

from atom.config import KVCacheTensor

INDEX_CACHE_SUFFIX = ".index_cache"

# A DSA indexer's key cache is registered by vLLM as its own KV-cache entry, but
# it is part of the owning attention layer's movable bytes, not a layer of its
# own. Two spellings exist in-tree and neither side is free to change:
#
#   MiniMax-M3   ``<layer>.index_cache``          -> owner ``<layer>``
#   GLM-5.2 /    ``<p>.indexer.k_cache``          -> owner ``<p>.attn``
#   DeepSeek-V3.2
#
# The GLM pairing is the one ``AiterMlaSparseIndexerMetadataBuilder`` itself
# uses (``attention_prefix = layer_name.removesuffix(".attn")``), so it is the
# model's own convention rather than a guess made here.
_GLM_INDEXER_SUFFIX = ".indexer.k_cache"


def index_cache_owner(name: str) -> str | None:
    """Return the layer this entry's bytes belong to, or None if it is a layer.

    Folding is deliberately never inferred from shape: an entry that merely
    looks indexer-shaped but is a real layer would be attached to a neighbour
    and restored under the wrong key.
    """
    if name.endswith(INDEX_CACHE_SUFFIX):
        return name[: -len(INDEX_CACHE_SUFFIX)]
    if name.endswith(_GLM_INDEXER_SUFFIX):
        return name[: -len(_GLM_INDEXER_SUFFIX)] + ".attn"
    return None


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

    Entries ``index_cache_owner`` recognises as indexer caches are folded into
    the owning layer's ``index_cache`` rather than becoming layers of their own
    -- vLLM registers DSA indexer keys (M3, GLM-5.2, DeepSeek-V3.2) as separate
    entries, but they are part of the same layer's movable bytes.

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
        owner = index_cache_owner(name)
        if owner is not None:
            index_caches[owner] = tensor
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
        k_scale, v_scale = _transfer_scales(name, (layers or {}).get(name), main[name])

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
        leading_dim = int(k_cache.shape[0])
        for role, seg in (("k_scale", k_scale), ("v_scale", v_scale)):
            if seg is not None and int(seg.shape[0]) != leading_dim:
                raise ValueError(
                    f"{name}: {role} does not share k_cache's leading axis "
                    f"(shape={tuple(seg.shape)}, k_cache leading dim="
                    f"{leading_dim})"
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

    # The codec is handed ONE num_blocks and derives every segment's per-block
    # byte stride as ``numel // num_blocks``. That is only meaningful while all
    # segments are paged against the same block table, which in vLLM means one
    # KV cache group. A layer whose leading axis is a DIFFERENT multiple would
    # still divide evenly often enough to pass the codec's own check and then be
    # sliced at the wrong granularity -- bytes restored into the wrong blocks,
    # no error anywhere. Name it here, where the shapes are visible.
    #
    # Scoped to the group rather than to the whole registration: a hybrid model
    # is paged against one block table PER GROUP, and those counts are expected
    # to differ. Checking them together rejected every hybrid model outright,
    # which is the guard this change removes -- the invariant it was protecting
    # is the within-group one, and that is still enforced.
    block_counts: dict[int, list[str]] = {}
    for kvt, name in zip(out, sorted(main, key=_layer_sort_key)):
        for role, seg in (("k_cache", kvt.k_cache), ("index_cache", kvt.index_cache)):
            if seg is None or seg.numel() == 0:
                continue
            block_counts.setdefault(int(seg.shape[0]), []).append(f"{name}.{role}")
    if len(block_counts) > 1:
        detail = "; ".join(
            f"{n}: {names[0]}{f' (+{len(names) - 1} more)' if len(names) > 1 else ''}"
            for n, names in sorted(block_counts.items())
        )
        raise ValueError(
            "ATOM offload connector: KV tensors in one cache group do not share "
            f"a block count ({detail}); the byte codec addresses every segment "
            "in a group with one block table."
        )
    return out


def split_kv_caches_by_group(
    kv_caches: dict[str, torch.Tensor],
    kv_cache_groups: list[Any],
) -> list[dict[str, torch.Tensor]]:
    """Split vLLM's flat registration dict into one dict per KV cache group.

    vLLM registers every group's layers together, but each group has its own
    page geometry and its own block count, so each needs its own codec.

    With a single group the dict is returned unfiltered. That is not an
    optimisation: it keeps the single-group models byte-identical to the
    behaviour that is measured working, and it keeps entries that a group spec
    does not name (a registration vLLM adds later, an auxiliary cache) from
    being dropped on the one path where there is no ambiguity about where they
    belong.

    With more than one group an unmapped entry IS ambiguous, and guessing is
    the failure this function exists to prevent, so it is an error.

    Args:
        kv_caches: vLLM's registration dict.
        kv_cache_groups: ``kv_cache_config.kv_cache_groups``, in group order.
            Each carries ``layer_names``.

    Returns:
        One dict per group, in group order. A group with no registered layers
        yields an empty dict rather than being dropped, so the returned list
        index is always the vLLM group id.

    Raises:
        ValueError: a registered layer belongs to no group (multi-group only).
    """
    if len(kv_cache_groups) <= 1:
        return [dict(kv_caches)]

    owner: dict[str, int] = {}
    for group_id, group in enumerate(kv_cache_groups):
        for layer_name in group.layer_names:
            owner[layer_name] = group_id

    per_group: list[dict[str, torch.Tensor]] = [{} for _ in kv_cache_groups]
    unmapped: list[str] = []
    for name, tensor in kv_caches.items():
        # An index cache rides with the layer that owns it and is named after
        # it; it is not a layer of its own and no group spec lists it.
        base = name.removesuffix(INDEX_CACHE_SUFFIX)
        group_id = owner.get(base)
        if group_id is None:
            unmapped.append(name)
            continue
        per_group[group_id][name] = tensor

    if unmapped:
        raise ValueError(
            f"registered KV caches belong to no KV cache group: {sorted(unmapped)}; "
            "with more than one group there is no safe default -- putting them "
            "in the wrong group moves the wrong bytes under a valid prefix hash"
        )
    return per_group


def gather_group_tensors(
    per_group: list[dict[str, torch.Tensor]],
    kv_cache_groups: list[Any],
    group_ids: Sequence[int],
) -> list[list[torch.Tensor]]:
    """Collect the registered tensors of the named groups, in layer order.

    ``per_group`` is the list ``split_kv_caches_by_group`` returns -- indexed
    by vLLM group id, one entry per group -- and this reads it by that index.
    It is a list and not a mapping on purpose: a group with no registered
    layers still occupies its slot, so the index never shifts.

    The order is the contract, not a convenience. The recurrent store gathers
    and the load scatters through the very list returned here, so a page image
    is readable only by a run that rebuilds the same order: groups in the order
    asked for, and within a group vLLM's own ``layer_names`` order -- never
    ``sorted()``, never dict order.

    Args:
        per_group: ``split_kv_caches_by_group`` output, indexed by group id.
        kv_cache_groups: ``kv_cache_config.kv_cache_groups``, in group order.
        group_ids: the groups to gather, in the order they should be laid out.

    Returns:
        One list of tensors per requested group, in the requested order.

    Raises:
        ValueError: a group id is out of range, or a group names a layer that
            was never registered -- which would make the stored image short by
            exactly that layer, with nothing downstream able to tell.
    """
    gathered: list[list[torch.Tensor]] = []
    for group_id in group_ids:
        if not 0 <= group_id < len(per_group) or group_id >= len(kv_cache_groups):
            raise ValueError(
                f"KV cache group {group_id} is out of range; vLLM registered "
                f"{len(per_group)} group(s)"
            )
        caches = per_group[group_id]
        names = list(kv_cache_groups[group_id].layer_names)
        missing = [name for name in names if name not in caches]
        if missing:
            raise ValueError(
                f"KV cache group {group_id} registered no tensor for {missing}; "
                "a stored page image would be short by those layers"
            )
        gathered.append([caches[name] for name in names])
    return gathered


def resolve_block_count(leading_dim: int, vllm_num_blocks: int, block_size: int) -> int:
    """The block count the codec must stride by, checked against the tensor.

    ``DenseKVByteCodec`` derives every segment's per-block byte stride as
    ``numel // num_blocks``, so this number decides how many bytes one entry in
    a vLLM block table stands for. It must therefore be the count of blocks
    *vLLM's block tables name* -- which is ``KVCacheConfig.num_blocks`` -- and
    not the tensor's leading dimension.

    The two differ whenever a backend asks vLLM for a kernel block size smaller
    than the group's block size. vLLM then allocates the cache at kernel
    granularity and expands manager block id ``b`` into the kernel ids
    ``b * n .. b * n + n - 1`` (``BlockTable.blocks_per_kv_block``), while the
    ids a KV connector receives stay manager ids. ATOM's MLA backend asks for a
    kernel block size of 1, so on Kimi-K3 -- 1536-token manager blocks -- the
    leading dimension is 1536x the block count, and taking it would stride every
    segment 1536x too fine. Nothing would fail: the block ids stay in range, the
    transfers succeed, and the restored bytes come from the wrong rows.

    Folding is sound without any reshape because a manager block's kernel blocks
    are contiguous and consecutive, so its bytes are one unbroken run either way.

    Raises:
        ValueError: vLLM reported no block count, or the tensor's leading
            dimension is not a whole number of blocks of a size that divides the
            block table's own block size -- in which case the two numbers do not
            describe the same cache and guessing which is right is the bug this
            function exists to prevent.
    """
    if vllm_num_blocks <= 0:
        raise ValueError(
            "ATOM offload connector: vLLM reported no KV cache block count "
            f"(num_blocks={vllm_num_blocks}); the byte codec prices one block "
            "table entry by it and cannot fall back to the leading dimension, "
            "which is a token count on token-major MLA"
        )
    if leading_dim <= 0 or leading_dim % vllm_num_blocks != 0:
        raise ValueError(
            "ATOM offload connector: the registered KV tensor's leading "
            f"dimension ({leading_dim}) is not a whole number of vLLM's "
            f"{vllm_num_blocks} KV cache blocks; the two do not describe the "
            "same cache"
        )
    kernel_blocks_per_block = leading_dim // vllm_num_blocks
    if block_size % kernel_blocks_per_block != 0:
        raise ValueError(
            f"ATOM offload connector: {leading_dim} rows over "
            f"{vllm_num_blocks} blocks implies {kernel_blocks_per_block} "
            f"kernel blocks per block, which does not divide the block size "
            f"({block_size}); the leading axis is not the kernel block axis"
        )
    return vllm_num_blocks
