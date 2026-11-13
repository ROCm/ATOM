# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA state, carried inside the PAGE object (MP backend).

A restored MLA prefix is correct only with the KDA state at the same token
boundary. In-process, `kda_state` moves that state on a leg of its own and joins
the two halves before a load is reported. Under the LMCache MP server it rides
the PAGE object instead, the way DeepSeek-V4.1's checkpoints do
(`ride_mp_scheduler`): a transfer names the state as a native-state image, and
the KV and the state commit and restore as one LMCache object. There is no
second engine, no second pool, and no join.

No copy is needed because of where vLLM puts the state. Every group of a hybrid
model aliases one allocation from byte 0, layer-outer, at one page size
(`get_kv_cache_config_from_groups`), so layer ``j`` of a mamba group shares its
bytes with layer ``j`` of the attention group. On K3 that is three mamba groups
of 23 KDA layers over the first 23 of the 29 MLA planes, an 832,512 B state in
each 884,736 B page. The state a mamba group holds in block ``b`` is therefore
unit ``b`` of the leading MLA planes, and the state at a boundary is an image of
one such unit per mamba group. `publish_kda_ride_state` checks that aliasing on
the tensors vLLM registered before it declares anything; a vLLM that lays the
groups out differently is refused at startup rather than served someone else's
bytes.

What differs from V4.1 is where the boundaries come from. V4.1 writes
checkpoints of its own; K3's are vLLM's. In ``--mamba-cache-mode align`` the
scheduler ends every prefill chunk on a mamba block, and the block holding the
state after it is cached under the boundary's ``BlockHash`` -- the map a local
mamba hit is served from. So a save's source is that cached block, looked up by
hash (`PageRideMixin.source_blocks_for`), and a load's destination is the row
``preprocess_mamba`` reads a resumed request's state from
(`BoundaryPlanner._boundary_blocks`). The load writes the state where the
forward reads it, so nothing is restored afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .boundary_planner import BoundaryPlanner
from .page_ride import PageRideMixin

logger = logging.getLogger("atom")

#: Bumped whenever the image's byte order changes for the same vLLM geometry:
#: the namespace hashes the layout id, so a new order starts a new key space
#: instead of reading back objects written under the old one.
_RIDE_LAYOUT_VERSION = 1


def refuse_kda_ride_without_prerequisites(vllm_config) -> None:
    """The settings the KDA ride needs, refused at startup when absent.

    Checked in both halves: each refusal is about bytes a worker would move or
    a hit the scheduler would serve, and a half that started without it would
    wait for the other to fail.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    cache_mode = getattr(cache_config, "mamba_cache_mode", None)
    if cache_mode != "align":
        # Any other mode keeps one running state per request and caches no
        # boundary, so there is no block whose state a save could carry.
        raise ValueError(
            "ATOM offload connector: Kimi-K3 offload over the LMCache MP server "
            f"needs --mamba-cache-mode align, got {cache_mode!r}; only that mode "
            "keeps the KDA state of a boundary in a block a transfer can name"
        )
    if not getattr(cache_config, "enable_prefix_caching", False):
        # A save's source is the state block vLLM caches under the boundary's
        # hash; with prefix caching off nothing is cached and every save waits
        # for a source that never comes.
        raise ValueError(
            "ATOM offload connector: Kimi-K3 offload over the LMCache MP server "
            "needs --enable-prefix-caching; a save carries the KDA state vLLM "
            "caches under the boundary's block hash, and without prefix caching "
            "there is none"
        )
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    policy = getattr(kv_transfer_config, "kv_load_failure_policy", "fail")
    if policy != "recompute":
        raise ValueError(
            "ATOM offload connector: Kimi-K3 offload over the LMCache MP server "
            "needs kv_load_failure_policy='recompute'. Its KV groups report every "
            "load that does not land through `failed_recving` -- a declined "
            "state ride or a lost tier prefix as well as a fault -- and "
            f"{policy!r} turns each into a failed request."
        )


def _attention_layer_bytes(group) -> list[int]:
    """Bytes per block each attention layer publishes, in vLLM's order.

    The order vLLM allocates the group in, which is the order it aliases mamba
    layer ``j`` onto attention layer ``j``, and the order `kda_ride_page_order`
    publishes the aliased layers in. A DSA indexer's cache is folded into the
    layer that owns it (`index_cache_owner`), as `build_kv_cache_tensors`
    publishes it: a layer may publish several planes, so this sums them per
    layer, and the worker reads off the regions how many planes that is.
    """
    from .kv_cache_layout import index_cache_owner

    spec = group.kv_cache_spec
    members = getattr(spec, "kv_cache_specs", None)
    names = list(members) if isinstance(members, dict) and members else []
    names += [name for name in group.layer_names if name not in names]
    per_layer: dict[str, int] = {}
    for name in names:
        member = members[name] if name in (members or {}) else spec
        owner = index_cache_owner(name) or name
        per_layer[owner] = per_layer.get(owner, 0) + int(member.page_size_bytes)
    return list(per_layer.values())


def kda_ride_page_order(tensors, tensors_by_group):
    """The attention layers in PAGE order for the KDA ride.

    The layers the mamba layers alias come first, in mamba-layer order, then
    the rest in the order given. ATOM publishes PAGE planes by layer name
    (`build_kv_cache_tensors`) while vLLM aliases mamba layer ``j`` onto the
    ``j``-th attention layer it allocates, and the two orders differ: on
    Kimi-K3 with DSpark, five of the 29 attention layers sort ahead of the
    first one a mamba layer aliases. The image covers a leading run of planes,
    so the aliased layers are moved to the front, matched on addresses, and
    renumbered so the region names follow the new order.
    """
    import dataclasses

    at = {int(tensor.k_cache.data_ptr()): i for i, tensor in enumerate(tensors)}
    aliased: list[int] = []
    for j in range(max(len(group) for group in tensors_by_group)):
        hits = {
            at.get(int(group[j].data_ptr()))
            for group in tensors_by_group
            if j < len(group)
        }
        if len(hits) != 1 or None in hits or next(iter(hits)) in aliased:
            raise ValueError(
                f"ATOM offload connector: mamba layer {j} of the KDA groups does "
                "not start where exactly one attention layer's cache starts "
                f"(matched {sorted(map(str, hits))}); its state is not in the "
                "PAGE pool a transfer moves"
            )
        aliased.append(hits.pop())
    order = aliased + [i for i in range(len(tensors)) if i not in set(aliased)]
    return [dataclasses.replace(tensors[i], layer_num=n) for n, i in enumerate(order)]


def _published_layer(region) -> int:
    """The attention layer a published PAGE region belongs to.

    `build_mp_transfer_tensors` names each region ``L<layer>.<role>``.
    """
    return int(str(region.semantic_role).split(".", 1)[0].removeprefix("L"))


def _describe_spec(spec) -> str:
    """The fields of a mamba group's spec that change the bytes of its state."""
    mamba_type = getattr(spec, "mamba_type", None)
    shapes = ";".join(str(tuple(shape)) for shape in getattr(spec, "shapes", ()))
    dtypes = ",".join(str(dtype) for dtype in getattr(spec, "dtypes", ()))
    return "|".join(
        (
            str(getattr(mamba_type, "name", mamba_type)),
            f"bs={int(getattr(spec, 'block_size', 0))}",
            f"page={int(getattr(spec, 'page_size_bytes', 0))}",
            f"shapes={shapes}",
            f"dtypes={dtypes}",
            f"spec_blocks={int(getattr(spec, 'num_speculative_blocks', 0) or 0)}",
        )
    )


@dataclass(frozen=True)
class KdaRideGeometry:
    """Where K3's KDA state sits in the PAGE unit, and the image it makes.

    ``state_layers`` is how many leading attention layers the mamba groups
    alias: the most layers any of them has. Every PAGE plane those layers
    publish -- the MLA cache, and an index cache where a layer has one -- makes
    up the checkpoint's PAGE unit (`paged_state_region_count`), so an image
    moves the state groups' bytes and not the layers no KDA layer shares.
    """

    group_ids: tuple[int, ...]
    layers_per_group: tuple[int, ...]
    state_layers: int
    spec: Any


def kda_ride_geometry(
    kv_cache_config, attention_group_id: int, mamba_groups
) -> KdaRideGeometry:
    """The KDA image geometry, from vLLM's KV cache config alone.

    Both halves build it -- the scheduler names its LMCache namespace by the
    spec, and each worker registers under the same one -- and the scheduler
    has no tensors to read. So it is predicted here from the group specs, and
    the worker holds that prediction against the registered tensors
    (`publish_kda_ride_state`), which is where a wrong one would show.

    One unit per mamba group, each every plane of the leading
    ``state_layers`` attention layers whole: on K3 (no index cache),
    3 x 23 x 884,736 B, of which 3 x 23 x 832,512 B are state and the rest the
    page padding behind it.
    """
    from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec

    groups = list(getattr(kv_cache_config, "kv_cache_groups", None) or ())
    if not mamba_groups:
        raise ValueError(
            "ATOM offload connector: Kimi-K3's KDA state cannot ride the PAGE "
            "object with no mamba group to name it"
        )
    group_ids = tuple(int(group_id) for group_id, _ in mamba_groups)
    layers_per_group = tuple(
        len(groups[group_id].layer_names) for group_id in group_ids
    )
    layer_bytes = _attention_layer_bytes(groups[attention_group_id])
    state_layers = max(layers_per_group)
    if not state_layers or state_layers > len(layer_bytes):
        raise ValueError(
            "ATOM offload connector: a mamba group has "
            f"{state_layers} layers but the attention group has "
            f"{len(layer_bytes)}; vLLM aliases mamba layer j onto attention "
            "layer j, so the KDA state would sit outside the PAGE units a "
            "transfer moves"
        )
    page_unit_bytes = sum(layer_bytes[:state_layers])
    image_bytes = len(group_ids) * page_unit_bytes
    layout_id = "|".join(
        (
            f"vllm-kda-ride-v{_RIDE_LAYOUT_VERSION}",
            "groups=" + ",".join(str(n) for n in layers_per_group),
            f"state_layers={state_layers}",
            "layers=" + ",".join(str(b) for b in layer_bytes[:state_layers]),
            *(_describe_spec(spec) for _, spec in mamba_groups),
        )
    )
    spec = PagedStateCheckpointSpec(
        page_unit_bytes=page_unit_bytes,
        # No Active SLOT: the state is restored into vLLM's own mamba block,
        # so the image is the whole of what a resumer reads.
        slot_bytes=image_bytes,
        layout_id=layout_id,
        image_bytes=image_bytes,
    )
    return KdaRideGeometry(
        group_ids=group_ids,
        layers_per_group=layers_per_group,
        state_layers=state_layers,
        spec=spec,
    )


def _no_state_copies(stores, restores, descriptor_slot: int = 0) -> None:
    """The native layout's copy callback, which this layout never calls for.

    The callback copies between an Active SLOT and the PAGE units an image is
    kept in. K3's state has no SLOT on the plugin: vLLM's mamba blocks are the
    units, and a load writes the state where the forward reads it.
    """
    raise RuntimeError(
        "Kimi-K3's KDA state rides vLLM's own mamba blocks; there is no Active "
        "SLOT for a native-state copy to read or fill"
    )


def _block_extent_bytes(tensor) -> int:
    """Bytes one block of *tensor* spans from its first element to its last."""
    span = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(tensor.shape[1:], tensor.stride()[1:])
    )
    return span * tensor.element_size()


def publish_kda_ride_state(transfer_tensors, tensors_by_group, geometry) -> None:
    """Declare the KDA image on a PAGE layout, after proving the state is in it.

    *tensors_by_group* is every mamba group's registered layers, in
    ``geometry.group_ids`` order (`gather_group_tensors`). The image covers
    every plane the leading ``geometry.state_layers`` attention layers publish,
    which PAGE publication puts first (layer-major). Each mamba layer has to
    start where one of those planes starts, stride by that plane's unit, and
    keep a block's state inside one unit -- or a transfer of unit ``b`` would
    move somebody else's bytes for block ``b``. Checked on addresses, not
    shapes: two planes of one size are indistinguishable by shape, and the
    aliasing is a property of vLLM's allocator that nothing in the specs
    states.

    The four fields are the ones native K3 publishes
    (`KimiMLAGDNBackend.get_kv_transfer_tensors`). The state holds TP-sharded
    KDA heads, so it is declared per rank even where the MLA KV beside it is
    replicated.
    """
    pages = list(getattr(transfer_tensors, "pages", None) or ())
    num_blocks = int(getattr(transfer_tensors, "num_blocks", 0) or 0)
    state_layers = int(geometry.state_layers)
    spec = geometry.spec
    layers = [_published_layer(page.region) for page in pages]
    state_planes = sum(1 for layer in layers if layer < state_layers)
    if layers[:state_planes] != sorted(layers[:state_planes]) or any(
        layer < state_layers for layer in layers[state_planes:]
    ):
        raise ValueError(
            "ATOM offload connector: the PAGE planes of the leading "
            f"{state_layers} attention layers are not the leading planes "
            f"published ({layers}); the KDA image covers a leading run of them"
        )
    unit_bytes = sum(int(page.region.unit_bytes) for page in pages[:state_planes])
    if unit_bytes != int(spec.page_unit_bytes):
        raise ValueError(
            f"ATOM offload connector: the {state_planes} PAGE planes of the "
            f"leading {state_layers} attention layers hold {unit_bytes} B per "
            f"block, but the KDA image was sized from vLLM's specs at "
            f"{spec.page_unit_bytes} B; the scheduler and the workers would "
            "name different namespaces"
        )
    plane_at = {int(page.region.base_addr): index for index, page in enumerate(pages)}
    if len(tensors_by_group) != len(geometry.group_ids):
        raise ValueError(
            f"ATOM offload connector: {len(tensors_by_group)} mamba group(s) "
            f"registered, {len(geometry.group_ids)} expected"
        )
    for group_id, expected, tensors in zip(
        geometry.group_ids, geometry.layers_per_group, tensors_by_group, strict=True
    ):
        if len(tensors) != expected:
            raise ValueError(
                f"ATOM offload connector: mamba group {group_id} registered "
                f"{len(tensors)} layers, its spec names {expected}"
            )
        for layer, tensor in enumerate(tensors):
            where = f"mamba group {group_id} layer {layer}"
            index = plane_at.get(int(tensor.data_ptr()))
            if index is None or index >= state_planes:
                raise ValueError(
                    f"ATOM offload connector: {where} does not start where one "
                    f"of the leading {state_planes} PAGE planes does (found "
                    f"plane {index}); its KDA state is not in the units a "
                    "transfer moves, so a load would restore the KV over "
                    "another prefix's state"
                )
            unit = int(pages[index].region.unit_bytes)
            stride = int(tensor.stride(0)) * tensor.element_size()
            if int(tensor.shape[0]) != num_blocks or stride != unit:
                raise ValueError(
                    f"ATOM offload connector: {where} is {int(tensor.shape[0])} "
                    f"blocks of {stride} B, PAGE plane {index} is {num_blocks} "
                    f"blocks of {unit} B; a block id would address different "
                    "bytes in the two"
                )
            if _block_extent_bytes(tensor) > unit:
                raise ValueError(
                    f"ATOM offload connector: one block of {where} spans "
                    f"{_block_extent_bytes(tensor)} B, past the {unit} B unit "
                    "a transfer moves for it"
                )
    transfer_tensors.paged_state_checkpoint_spec = spec
    transfer_tensors.execute_paged_state_copies = _no_state_copies
    transfer_tensors.paged_state_region_count = state_planes
    transfer_tensors.native_state_tp_replication_factor = 1
    state = sum(
        int(tensor[0].numel()) * tensor.element_size()
        for tensors in tensors_by_group
        for tensor in tensors
    )
    logger.info(
        "ATOM LMCache offload: KDA state rides the PAGE object as %d unit(s) "
        "over the %d planes of the leading %d attention layers (mamba groups "
        "of %s layers), %d B of state per boundary in a %d B image",
        len(geometry.group_ids),
        state_planes,
        state_layers,
        ",".join(str(n) for n in geometry.layers_per_group),
        state,
        int(spec.image_bytes),
    )


class KdaRidePlanner(BoundaryPlanner, PageRideMixin):
    """Scheduler half of K3's KDA state when it rides the PAGE object.

    One invariant: a request vLLM is told has ``n`` computed tokens must start
    its forward from the KDA state after exactly ``n`` tokens, in every mamba
    group. A hit off the chunk grid is shortened to it (`cap_hit`), and a load
    is admitted only with the state blocks it restores named (`resolve_load`).
    """

    leg_name = "KDA state ride"

    def __init__(
        self,
        *,
        group_ids,
        mamba_block_size: int,
        hash_block_size: int,
        chunk_size: int,
        prefill_tail: int = 1,
    ) -> None:
        # A boundary is one mamba block in every mamba group; the groups are in
        # the order the worker registered them, which is image unit order and
        # therefore the key space.
        super().__init__(
            boundary_grid=int(mamba_block_size), boundary_group_ids=group_ids
        )
        if not self.boundary_group_ids:
            raise ValueError(
                "KDA state ride: no mamba group to name the state in; a transfer "
                "would carry KV whose state is someone else's"
            )
        self.hash_block_size = int(hash_block_size)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0 or self.chunk_size % self.boundary_grid:
            raise ValueError(
                f"KDA state ride: LMCache chunk {self.chunk_size} is not a whole "
                f"number of mamba blocks ({self.boundary_grid} tokens). A save "
                "carries the state at the chunk it ends on, and vLLM caches a "
                "state only on a mamba block; set the LMCache chunk size to a "
                "multiple of the mamba block"
            )
        if self.boundary_grid % self.hash_block_size:
            raise ValueError(
                f"KDA state ride: mamba block {self.boundary_grid} is not a "
                f"multiple of the hash block {self.hash_block_size}; a boundary "
                "on it could not be looked up by hash"
            )
        # Tokens a hit leaves to compute. Two under speculative decoding: a hit
        # leaving one is admitted by vLLM as a decode row padded with
        # placeholder drafts (`pad_spec_decode`) rather than as a prefill, and
        # nothing has exercised a restored KDA state through that path. The
        # cost is a chunk of recompute on prompts one token past a chunk.
        self.prefill_tail = int(prefill_tail)
        # `PageRideMixin`'s names for the same geometry.
        self.ride_group_ids = self.boundary_group_ids
        self.ride_boundary_grid = self.boundary_grid
        self._init_page_ride()
        # One warning per planner, not per hit.
        self._warned_unarmed = False

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config,
        kv_cache_config,
        mamba_groups,
        *,
        scheduler,
        request_of,
    ) -> KdaRidePlanner:
        """Stand up the scheduler half of the KDA ride, bound to *scheduler*.

        ``request_of(seq)`` is the vLLM request a scheduler ``SeqView`` stands
        for, or None.
        """
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        refuse_kda_ride_without_prerequisites(vllm_config)
        _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        spec = mamba_groups[0][1]
        speculative = getattr(vllm_config, "speculative_config", None) is not None
        planner = cls(
            group_ids=tuple(group_id for group_id, _ in mamba_groups),
            mamba_block_size=int(spec.block_size),
            hash_block_size=int(hash_block_size),
            chunk_size=int(scheduler.chunk_size),
            prefill_tail=2 if speculative else 1,
        )
        scheduler.bind_planner(planner, request_of)
        logger.info(
            "ATOM LMCache offload: KDA state rides the PAGE object on group(s) "
            "%s (mamba_block=%d, hash_block=%d, chunk=%d, prefill_tail=%d)",
            ",".join(str(group_id) for group_id in planner.boundary_group_ids),
            planner.boundary_grid,
            planner.hash_block_size,
            planner.chunk_size,
            planner.prefill_tail,
        )
        return planner

    # -- lookup ----------------------------------------------------------
    def _cap_unarmed(self, seq, hit: int) -> int:
        """Decline a hit reported outside any lookup.

        A hit that never reached the cap could restore KV past the last state
        the tier holds, onto the previous occupant's recurrence; declining it
        costs a recompute.
        """
        if not self._warned_unarmed:
            self._warned_unarmed = True
            logger.warning(
                "%s: the hit cap fired outside a lookup (seq=%s, hit=%d); "
                "declining the external hit",
                self.leg_name,
                getattr(seq, "id", None),
                hit,
            )
        return 0

    def _cap_armed(self, request, hit: int) -> int:
        """Shorten *hit* to the chunk grid, where a stored state can sit.

        No index to probe: the KV and the state that continues it are one
        object, and the server reports a hit only as far as every image group
        holds the hit's last chunk. What still has to hold is the shape -- a
        save carries the state of the chunk edge it ends on -- and the
        speculative tail.
        """
        boundary = (hit // self.chunk_size) * self.chunk_size
        total = int(getattr(request, "num_tokens", 0) or 0)
        if total and total - boundary < self.prefill_tail:
            # One chunk back keeps it a prefill. The lookup already stopped
            # short of the tail (`StateRideMPScheduler._lookup_len`), so this
            # is the guard for a hit that did not come through it.
            boundary -= self.chunk_size
        if boundary <= 0:
            return 0
        return min(hit, boundary)

    # -- load ------------------------------------------------------------
    def admit_load(
        self,
        request,
        *,
        group_blocks,
        num_total_computed: int,
        num_external_tokens: int,
        attention_group_id: int,
        attention_block_size: int,
    ) -> None:
        """Name the state blocks an admitted PAGE load restores.

        When they cannot be named, the scheduler finds no destination in
        `take_ride_units` and declines the load, and the connector withdraws
        the promise (`_settle_promised_loads`): the request recomputes rather
        than resuming on a state nobody restored.
        """
        self.resolve_load(request, num_total_computed, group_blocks=group_blocks)

    def resolve_load(self, request, num_total_computed: int, group_blocks=()) -> bool:
        """Record the state blocks a restored PAGE prefix ending here fills.

        False when they cannot be resolved. Every admission reallocates, so
        what an earlier one recorded is dropped first: a failed or withdrawn
        load leaves blocks vLLM has since freed.
        """
        self.forget_ride(str(request.request_id))
        block_ids = self._boundary_blocks(group_blocks, num_total_computed)
        if not block_ids:
            return False
        self._record_ride_boundary(
            str(request.request_id), int(num_total_computed), block_ids
        )
        return True

    def forget_request(self, req_id: str) -> None:
        """Drop a finished or preempted request's recorded destinations.

        A preempted request reallocates, so its old blocks name someone else's
        state.
        """
        self.forget_ride(req_id)

    def has_pending_work(self) -> bool:
        """A ride pin is given back by a step's reconciliation
        (`StateRideMPScheduler.build_connector_meta`), so one still held needs
        the engine to keep stepping."""
        return self.has_ride_pins()

    def has_pending_frees(self, horizon_s: float) -> bool:
        """Only a recent pin that is a block's last holder is a block about to
        come back; retiring any other drops a share and frees nothing."""
        return self.has_ride_pins(max_age_s=horizon_s, sole_holder=True)


__all__ = [
    "KdaRideGeometry",
    "KdaRidePlanner",
    "kda_ride_geometry",
    "kda_ride_page_order",
    "publish_kda_ride_state",
    "refuse_kda_ride_without_prerequisites",
]
