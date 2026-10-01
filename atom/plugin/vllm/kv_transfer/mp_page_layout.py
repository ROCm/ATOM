# SPDX-License-Identifier: MIT
"""Publish vLLM's KV caches as the PAGE layout ATOM's MP offload registers.

ATOM's in-process (``dense``) offload worker takes the layer tensors as they
come -- ``{layer: KVCacheTensor}`` -- and derives each segment's per-block
stride itself. The multiprocess worker cannot: it hands the tensors to a
separate LMCache server process, which addresses them by ``(base_addr,
unit_bytes)`` and copies opaque bytes. So it takes a ``KVTransferTensors``
instead, where every movable plane arrives as an address region paired with a
zero-copy block-major view of exactly those bytes, and it validates the pair
before registering anything (``offload/mp/page_views.py``).

In the ATOM engine that object is published by the attention metadata builder,
which knows its own pool geometry. The vLLM plugin has no such builder -- vLLM
hands it a flat ``{layer_name: tensor}`` -- so this module is the equivalent
producer, built from the same ``KVCacheTensor`` list the dense path already
constructs. It adds no policy: it names the planes, sizes a block, and refuses
anything that cannot be expressed as whole blocks.

Two things here are not free choices:

``num_blocks`` is the *scheduler's* block count, not the tensor's leading
dimension. ATOM's MLA backend asks vLLM for a kernel block size of 1, so vLLM
allocates one row per token and dim 0 comes back ``block_size`` times too large
(1536x on Kimi-K3). ``resolve_block_count`` is what reconciles the two, and the
caller passes its answer in; sizing a unit from ``shape[0]`` instead would
produce a layout that passes every later check and moves 1/1536 of each block.

The plane *order* is the key space. Save and restore both walk ``pages`` in
publication order, so the order below -- layers ascending, and within a layer a
fixed role sequence -- is a wire format, not a formatting preference. The roles
are also named rather than numbered, because ``page_views`` groups planes by
byte geometry and two equal-sized planes are otherwise indistinguishable in a
log.
"""

from typing import Any

import torch

from atom.config import KVCacheTensor
from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import (
    KVTransferTensors,
    RecurrentPageGroup,
)

# Fixed and total: every movable plane a ``KVCacheTensor`` can carry on the
# attention leg. A plane added to that dataclass and not added here would be
# left behind by the save and restored as whatever the previous occupant of the
# block wrote -- a wrong answer with nothing logged -- so the builder checks
# this list against the dataclass's own fields rather than trusting it.
_PAGE_ROLES: tuple[str, ...] = (
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
    "index_cache",
    "index_scale",
)

# The recurrent planes. They are deliberately NOT published through the PAGE
# layout: a recurrent group counts in its own block id space and only its last
# snapshot is meaningful, so ``base + block_id * unit_bytes`` on the attention
# block table addresses someone else's state. ``build_mp_recurrent_groups``
# publishes them instead, as engine groups of their own. Named here so that
# "absent from _PAGE_ROLES" is a decision on record rather than an omission.
_NON_PAGE_ROLES: frozenset[str] = frozenset(
    {
        "layer_num",
        "replay_buf_k",
        "replay_buf_u",
        "replay_buf_g",
        # Not a plane but a verdict on the ones above: True marks this whole
        # entry as per-request recurrent state, addressed by request slot. It
        # is checked rather than merely skipped -- see `build_mp_transfer_tensors`.
        "per_request_state",
    }
)


def _movable_planes(tensor: KVCacheTensor) -> list[tuple[str, torch.Tensor]]:
    """This layer's non-empty planes, in publication order."""
    planes: list[tuple[str, torch.Tensor]] = []
    for role in _PAGE_ROLES:
        plane = getattr(tensor, role, None)
        # ``KVCacheTensor.v_cache`` defaults to a 0-element tensor rather than
        # None (an MLA layer fuses K and V into one run), so emptiness and
        # absence both have to mean "no plane".
        if isinstance(plane, torch.Tensor) and plane.numel():
            planes.append((role, plane))
    return planes


def _check_roles_exhaustive() -> None:
    """Fail if ``KVCacheTensor`` grew a plane this module would leave behind."""
    fields = getattr(KVCacheTensor, "__dataclass_fields__", {})
    unknown = set(fields) - set(_PAGE_ROLES) - _NON_PAGE_ROLES
    if unknown:
        raise ValueError(
            "ATOM LMCache MP: KVCacheTensor carries plane(s) "
            f"{sorted(unknown)} that the PAGE layout neither publishes nor "
            "excludes; a plane left behind is restored as the previous "
            "occupant's bytes. Add it to _PAGE_ROLES or _NON_PAGE_ROLES."
        )


def build_mp_transfer_tensors(
    tensors: list[KVCacheTensor],
    *,
    num_blocks: int,
    tp_replication_factor: int = 1,
) -> KVTransferTensors:
    """Describe one attention group's KV caches as a PAGE-only layout.

    Args:
        tensors: the group's layers, as ``build_kv_cache_tensors`` produced
            them. Publication order follows this list.
        num_blocks: the scheduler's block count for the group -- what
            ``resolve_block_count`` returns, NOT ``k_cache.shape[0]``.
        tp_replication_factor: how many TP ranks hold byte-identical copies of
            this whole layout. Defaults to 1, the sharded answer, because vLLM
            splits KV heads across TP and claiming otherwise would let LMCache
            serve one rank's KV to another. The caller raises it only where the
            cache is the MLA latent, which every rank computes identically --
            see ``AtomLMCacheOffloadConnector._page_tp_replication_factor``.

    Returns:
        A ``KVTransferTensors`` carrying only ``pages``. Every SLOT-side field
        is left empty, which is what makes the MP worker select its PAGE-only
        transport rather than refusing the layout.
    """
    _check_roles_exhaustive()
    if type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError(
            f"ATOM LMCache MP: num_blocks must be a positive int, got {num_blocks!r}"
        )
    if not tensors:
        raise ValueError("ATOM LMCache MP: no KV cache tensors to publish")

    pages = []
    for tensor in tensors:
        layer_num = int(tensor.layer_num)
        if getattr(tensor, "per_request_state", False):
            # These bytes live in ``kv_cache_data`` like paged KV but are
            # indexed in the recurrent group's own block space, so ``base +
            # block_id * unit_bytes`` on the attention block table addresses
            # someone else's state. The plugin routes recurrent groups to
            # ``build_mp_recurrent_groups`` long before here; reaching it means
            # that routing has a hole, and continuing would publish a layout
            # whose every later check passes.
            raise ValueError(
                f"ATOM LMCache MP: layer {layer_num} is per-request recurrent "
                "state, not paged KV; no block-addressed transport may move it"
            )
        for role, plane in _movable_planes(tensor):
            total_bytes = plane.numel() * plane.element_size()
            # Checked on the shape as well as on the byte count, because the
            # byte count alone does not catch it: a plane with one row too many
            # can still divide evenly into ``num_blocks`` units (1025 rows of
            # 1152 B over 64 blocks gives a whole 18450 B unit), and every later
            # check -- `set_block_count`, `validate_page_views` -- is expressed
            # in those same bytes and would agree. The stride would simply be
            # wrong, by less than one row per block.
            if int(plane.shape[0]) % num_blocks:
                raise ValueError(
                    f"ATOM LMCache MP: layer {layer_num} {role} has leading "
                    f"dimension {int(plane.shape[0])}, which is not a whole "
                    f"number of {num_blocks} blocks; its leading axis is not "
                    "the block axis"
                )
            if total_bytes % num_blocks:
                raise ValueError(
                    f"ATOM LMCache MP: layer {layer_num} {role} holds "
                    f"{total_bytes} B, which is not a whole number of "
                    f"{num_blocks} blocks; the server addresses this plane as "
                    "base + block_id * unit_bytes and would move a fraction of "
                    "each block"
                )
            pages.append(
                page_region(
                    plane,
                    # Stable across code versions and across save/restore:
                    # positions shift when a model gains a plane, names do not.
                    semantic_role=f"L{layer_num:04d}.{role}",
                    unit_bytes=total_bytes // num_blocks,
                    total_bytes=total_bytes,
                )
            )

    transfer_tensors = KVTransferTensors(
        pages=pages,
        tp_replication_factor=int(tp_replication_factor),
    )
    # Re-checks every region divides into exactly ``num_blocks`` units, and is
    # what tells the worker the block id space it was handed.
    transfer_tensors.set_block_count(num_blocks)
    return transfer_tensors


def build_mp_recurrent_groups(
    views: Any,
    *,
    tokens_per_block: int,
) -> tuple[RecurrentPageGroup, ...]:
    """Describe the mamba groups as recurrent engine groups of their own.

    ``views`` is the ``KdaPageViews`` the recurrent leg already builds, so the
    plane order here is the order its codec gathers in -- group order, and
    within a group vLLM's canonical layer order. That order is the key space,
    same as for PAGE.

    ``tokens_per_block`` is the mamba block size, which vLLM has already forced
    to equal the attention block size and the LMCache chunk size. It is passed
    rather than read off the tensors because a snapshot's row count says
    nothing about how many tokens it covers.
    """
    if type(tokens_per_block) is not int or tokens_per_block <= 0:
        raise ValueError(
            "ATOM LMCache MP: recurrent tokens_per_block must be a positive "
            f"int, got {tokens_per_block!r}"
        )
    groups: list[RecurrentPageGroup] = []
    for ordinal, (tensors, num_blocks) in enumerate(
        zip(views.groups, views.num_blocks, strict=True)
    ):
        pages = []
        for layer, plane in enumerate(tensors):
            total_bytes = plane.numel() * plane.element_size()
            if int(plane.shape[0]) != num_blocks or total_bytes % num_blocks:
                raise ValueError(
                    f"ATOM LMCache MP: recurrent group {ordinal} layer {layer} "
                    f"has leading dimension {int(plane.shape[0])} and "
                    f"{total_bytes} B over {num_blocks} blocks; its leading "
                    "axis is not the block axis"
                )
            pages.append(
                page_region(
                    plane,
                    semantic_role=f"R{ordinal:02d}.L{layer:04d}.state",
                    unit_bytes=total_bytes // num_blocks,
                    total_bytes=total_bytes,
                )
            )
        groups.append(
            RecurrentPageGroup(
                pages=tuple(pages),
                num_blocks=int(num_blocks),
                tokens_per_block=tokens_per_block,
            )
        )
    return tuple(groups)


def summarize_layout(transfer_tensors: Any) -> str:
    """One line naming the plane count and the bytes one block costs."""
    regions = transfer_tensors.block_regions
    bytes_per_block = sum(int(region.unit_bytes) for region in regions)
    line = (
        f"{len(regions)} planes, {bytes_per_block} B/block, "
        f"{transfer_tensors.num_blocks} blocks"
    )
    recurrent = tuple(getattr(transfer_tensors, "recurrent_page_groups", None) or ())
    if recurrent:
        snapshot = sum(
            int(page.region.unit_bytes) for group in recurrent for page in group.pages
        )
        line += (
            f"; {len(recurrent)} recurrent group(s), {snapshot} B/snapshot, "
            f"{recurrent[0].tokens_per_block} tokens/snapshot"
        )
    return line


__all__ = [
    "build_mp_recurrent_groups",
    "build_mp_transfer_tensors",
    "summarize_layout",
]
