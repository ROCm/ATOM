# SPDX-License-Identifier: MIT
"""Unit coverage for the Kimi-K3 (hybrid) leg of the vLLM-plugin KV offload.

These are the pieces whose failure mode is silent wrong output rather than a
crash, so "it booted" proves nothing about them:

* ``split_kv_caches_by_group`` -- put a layer in the wrong group and its bytes
  move under another group's prefix hash.
* ``KdaBoundaryPlanner.collect_stores`` -- every filter it applies exists
  because the rejected hand-off would otherwise be persisted as if it were a
  committed boundary state.
* ``KdaBoundaryPlanner.cap_hit`` -- the entire joint-correctness argument for
  the two groups. If it ever returns more than the last boundary the index
  claims, an MLA prefix is served with somebody else's recurrent state.
* ``absorb_reports`` -- the quorum that decides both when a pinned block goes
  back to the pool and whether a hash may be advertised at all.

They deliberately avoid vLLM: both modules under test are importable without it
(``find_mamba_groups`` degrades to an empty list), and the CI unit job has
no vLLM.
"""

import pytest
import torch

from atom.plugin.vllm.kv_transfer.kda_state import (
    _MAX_CAP_DESCENT,
    KdaBoundaryPlanner,
    KdaPageViews,
    boundary_prefix_hash,
    build_layout_id,
    step_boundary_offloads,
    summarize_layout_id,
    unwrap_kv_cache_spec,
)
from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    gather_group_tensors,
    resolve_block_count,
    split_kv_caches_by_group,
)

MAMBA_GROUP = 1
#: K3 really has three; vLLM splits the recurrent layers into equal-sized
#: groups and takes the size from the smallest family. The planner treats them
#: as one state, so the multi-group tests below use both of these.
MAMBA_GROUP_2 = 2
ATTENTION_GROUP = 0
MAMBA_BLOCK = 64
HASH_BLOCK = 64
CHUNK = 256
WORLD = 2


class FakeGroup:
    def __init__(self, layer_names, spec=None):
        self.layer_names = list(layer_names)
        self.kv_cache_spec = spec


class FakeRequest:
    """Only the two attributes the planner reads off a vLLM ``Request``."""

    def __init__(self, request_id, num_hashes=16):
        self.request_id = request_id
        self.block_hashes = [f"{request_id}:{i}".encode() for i in range(num_hashes)]


class FakeSeq:
    """ATOM's ``SeqView`` as far as the hit-cap hook is concerned."""

    def __init__(self, sid):
        self.id = sid


class FakeBlock:
    """A ``KVCacheBlock`` as far as ``get_cached_block``'s caller is concerned."""

    def __init__(self, block_id):
        self.block_id = block_id


class FakePool:
    """vLLM's ``BlockPool``, reduced to the surface the planner uses: the pin
    pair, and the content-addressed lookup the boundary sweep resolves through.
    """

    def __init__(self, num_blocks=64):
        self.blocks = [f"blk{i}" for i in range(num_blocks)]
        self.touched = []
        self.freed = []
        # {block_hash: {group_id: block_id}}, mirroring
        # ``cached_block_hash_to_block`` keyed by (hash, group).
        self.cached: dict = {}

    def publish(self, block_hash, group_blocks):
        self.cached[block_hash] = dict(group_blocks)

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        entry = self.cached.get(block_hash)
        if entry is None:
            return None
        out = []
        for group_id in kv_cache_group_ids:
            if group_id not in entry:
                return None
            out.append(FakeBlock(entry[group_id]))
        return out

    def touch(self, blocks):
        self.touched.extend(blocks)

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


def make_planner(**kwargs):
    params = {
        "group_ids": (MAMBA_GROUP,),
        "mamba_block_size": MAMBA_BLOCK,
        "hash_block_size": HASH_BLOCK,
        "chunk_size": CHUNK,
        "world_size": WORLD,
    }
    params.update(kwargs)
    return KdaBoundaryPlanner(**params)


def store_one_two_groups(planner, request, *, boundary=CHUNK):
    """The same, for a planner that owns two mamba groups."""
    stores = planner.collect_stores(
        {
            request.request_id: [
                (MAMBA_GROUP, 7, boundary),
                (MAMBA_GROUP_2, 9, boundary),
            ]
        },
        {request.request_id: request},
    )
    for store in stores:
        planner.absorb_reports({store.op_id: WORLD}, {})
    return stores


def store_one(planner, request, *, block_id=7, boundary=CHUNK, group=MAMBA_GROUP):
    """Run one boundary hand-off all the way to indexed, as a step would."""
    stores = planner.collect_stores(
        {request.request_id: [(group, block_id, boundary)]},
        {request.request_id: request},
    )
    for store in stores:
        planner.absorb_reports({store.op_id: WORLD}, {})
    return stores


# --------------------------------------------------------------------------
# split_kv_caches_by_group
# --------------------------------------------------------------------------
def test_two_groups_split_into_two_dicts_indexed_by_group_id():
    caches = {
        "model.layers.0.attn": torch.zeros(4, 8),
        "model.layers.1.attn": torch.zeros(4, 8),
        "model.layers.2.kda": torch.zeros(3, 8),
    }
    groups = [
        FakeGroup(["model.layers.0.attn", "model.layers.1.attn"]),
        FakeGroup(["model.layers.2.kda"]),
    ]

    per_group = split_kv_caches_by_group(caches, groups)

    assert len(per_group) == 2
    assert set(per_group[0]) == {"model.layers.0.attn", "model.layers.1.attn"}
    assert set(per_group[1]) == {"model.layers.2.kda"}


def test_index_cache_follows_the_layer_that_owns_it():
    caches = {
        "model.layers.0.attn": torch.zeros(4, 8),
        "model.layers.0.attn.index_cache": torch.zeros(4, 2),
        "model.layers.1.kda": torch.zeros(3, 8),
    }
    groups = [FakeGroup(["model.layers.0.attn"]), FakeGroup(["model.layers.1.kda"])]

    per_group = split_kv_caches_by_group(caches, groups)

    assert "model.layers.0.attn.index_cache" in per_group[0]
    assert set(per_group[1]) == {"model.layers.1.kda"}


def test_layer_belonging_to_no_group_is_an_error_not_a_guess():
    caches = {
        "model.layers.0.attn": torch.zeros(4, 8),
        "model.layers.9.stray": torch.zeros(4, 8),
    }
    groups = [FakeGroup(["model.layers.0.attn"]), FakeGroup(["model.layers.1.kda"])]

    with pytest.raises(ValueError, match="model.layers.9.stray"):
        split_kv_caches_by_group(caches, groups)


def test_single_group_passes_the_dict_through_unfiltered():
    """The M3 / GLM-5.2 path must stay byte-identical, unclaimed names included."""
    caches = {
        "model.layers.0.attn": torch.zeros(4, 8),
        "not.named.by.any.spec": torch.zeros(4, 8),
    }
    groups = [FakeGroup(["model.layers.0.attn"])]

    per_group = split_kv_caches_by_group(caches, groups)

    assert per_group == [caches]


def test_group_with_no_registered_layers_keeps_its_index():
    caches = {"model.layers.0.attn": torch.zeros(4, 8)}
    groups = [
        FakeGroup(["model.layers.0.attn"]),
        FakeGroup(["model.layers.1.kda"]),
        FakeGroup(["model.layers.2.kda"]),
    ]

    per_group = split_kv_caches_by_group(caches, groups)

    assert len(per_group) == 3
    assert per_group[1] == {} and per_group[2] == {}


# --------------------------------------------------------------------------
# build_kv_cache_tensors: the block-count guard is per group, not global
# --------------------------------------------------------------------------
def test_groups_may_disagree_on_block_count_across_groups():
    """The whole reason the guard had to be scoped: K3's two groups differ."""
    attention = {f"layers.{i}.attn": torch.zeros(1024, 4, 16) for i in range(2)}
    mamba = {f"layers.{i}.kda": torch.zeros(9, 4, 16) for i in range(2, 4)}

    assert len(build_kv_cache_tensors(attention)) == 2
    assert len(build_kv_cache_tensors(mamba)) == 2


def test_block_count_disagreement_inside_one_group_still_raises():
    caches = {
        "layers.0.attn": torch.zeros(1024, 4, 16),
        "layers.1.attn": torch.zeros(512, 4, 16),
    }

    with pytest.raises(ValueError, match="do not share a block count"):
        build_kv_cache_tensors(caches)


# --------------------------------------------------------------------------
# boundary_hash / boundary_prefix_hash
# --------------------------------------------------------------------------
def test_boundary_hash_indexes_the_block_that_ends_the_boundary():
    planner = make_planner()
    request = FakeRequest("r0")

    assert planner.boundary_hash(request.block_hashes, CHUNK) == boundary_prefix_hash(
        request.block_hashes[CHUNK // HASH_BLOCK - 1]
    )


def test_boundary_hash_is_none_when_unaligned_or_beyond_the_hashed_prefix():
    planner = make_planner()
    request = FakeRequest("r0", num_hashes=4)

    assert planner.boundary_hash(request.block_hashes, 0) is None
    assert planner.boundary_hash(request.block_hashes, HASH_BLOCK - 1) is None
    # 4 hashes cover 256 tokens; 320 would index row 4.
    assert planner.boundary_hash(request.block_hashes, 320) is None


def test_boundary_prefix_hash_survives_a_restart():
    """Python salts ``hash`` per process; a salted key orphans every entry."""
    assert boundary_prefix_hash(b"atom-kda-boundary") == 1452988423931930918


# --------------------------------------------------------------------------
# step_boundary_offloads: the hand-off is spelled two ways across vLLM versions
# --------------------------------------------------------------------------
HANDOFF = {"r0": [(1, 7, 256)]}


class FlatStep:
    """vLLM 0.28: ``SchedulerOutput.partial_tail_offloads``."""

    partial_tail_offloads = HANDOFF


class NestedStep:
    """vLLM 0.29: the same payload under ``kv_connector_block_state``."""

    class kv_connector_block_state:
        boundary_state_offloads = HANDOFF


class BothStep(FlatStep, NestedStep):
    pass


class EmptyStep:
    partial_tail_offloads = None
    kv_connector_block_state = None


def test_flat_handoff_is_read_on_the_pinned_vllm():
    assert step_boundary_offloads(FlatStep()) == HANDOFF


def test_nested_handoff_is_read_on_the_newer_vllm():
    assert step_boundary_offloads(NestedStep()) == HANDOFF


def test_either_spelling_alone_is_enough():
    """Neither attribute may be assumed present: the step object that carries
    one carries no placeholder for the other, and a missing attribute must read
    as "no hand-off this step", not as an AttributeError mid-step."""
    assert step_boundary_offloads(BothStep()) == HANDOFF
    assert step_boundary_offloads(object()) is None


def test_a_step_with_no_handoff_yields_nothing():
    assert step_boundary_offloads(EmptyStep()) is None


# --------------------------------------------------------------------------
# collect_stores: every filter
# --------------------------------------------------------------------------
def test_accepted_boundary_is_pinned_and_keyed_by_its_prefix_hash():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")

    stores = planner.collect_stores({"r0": [(MAMBA_GROUP, 7, CHUNK)]}, {"r0": request})

    assert len(stores) == 1
    assert stores[0].block_ids == (7,)
    assert stores[0].prefix_hash == planner.boundary_hash(request.block_hashes, CHUNK)
    assert pool.touched == ["blk7"]
    assert planner.has_pending_work()


@pytest.mark.parametrize(
    "entry,why",
    [
        ((ATTENTION_GROUP, 7, CHUNK), "another group is saved positionally"),
        ((MAMBA_GROUP, 0, CHUNK), "NULL_BLOCK_ID is a placeholder"),
        ((MAMBA_GROUP, 7, CHUNK + 1), "not a whole mamba block"),
        ((MAMBA_GROUP, 7, MAMBA_BLOCK), "not chunk-aligned, cap_hit cannot select it"),
        ((MAMBA_GROUP, 7, 64 * 64), "past the hashed prefix, no key"),
    ],
)
def test_boundary_handoffs_that_must_be_dropped(entry, why):
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)

    stores = planner.collect_stores({"r0": [entry]}, {"r0": FakeRequest("r0")})

    assert stores == [], why
    assert pool.touched == [], why
    assert not planner.has_pending_work()


def test_unknown_and_skipped_requests_are_dropped():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")
    handoff = {"r0": [(MAMBA_GROUP, 7, CHUNK)]}

    assert planner.collect_stores(handoff, {}) == []
    assert planner.collect_stores(handoff, {"r0": request}, skip_req_ids={"r0"}) == []
    assert pool.touched == []


# --------------------------------------------------------------------------
# absorb_reports: quorum, indexing, unpinning
# --------------------------------------------------------------------------
def test_pin_is_held_until_every_rank_reports():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")
    (store,) = planner.collect_stores(
        {"r0": [(MAMBA_GROUP, 7, CHUNK)]}, {"r0": request}
    )

    planner.absorb_reports({store.op_id: 1}, {})
    assert pool.freed == []
    assert planner.has_pending_work()

    planner.absorb_reports({store.op_id: 1}, {})
    assert pool.freed == ["blk7"]
    assert not planner.has_pending_work()


def test_a_failed_rank_still_completes_the_quorum_but_blocks_indexing():
    """A rank that could not write never reports twice; waiting pins forever.

    And a partially stored state must not be advertised: restoring it would be
    the exact half-restore the whole design exists to prevent.
    """
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")
    (store,) = planner.collect_stores(
        {"r0": [(MAMBA_GROUP, 7, CHUNK)]}, {"r0": request}
    )

    planner.absorb_reports({store.op_id: 1}, {store.op_id: 1})

    assert pool.freed == ["blk7"]
    assert not planner.has_pending_work()

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), CHUNK + 10) == 0


# --------------------------------------------------------------------------
# cap_hit: the joint gate
# --------------------------------------------------------------------------
def test_hit_is_capped_to_the_boundary_the_index_claims():
    planner = make_planner()
    request = FakeRequest("r0")
    store_one(planner, request, boundary=CHUNK)

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), CHUNK + 100) == CHUNK
    assert planner.cap_hit(FakeSeq("r0"), CHUNK) == CHUNK


def test_cap_descends_in_chunk_steps_to_an_older_stored_boundary():
    planner = make_planner()
    request = FakeRequest("r0")
    store_one(planner, request, boundary=CHUNK)

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), 3 * CHUNK + 5) == CHUNK


def test_cap_gives_up_after_the_descent_bound():
    planner = make_planner()
    request = FakeRequest("r0", num_hashes=256)
    store_one(planner, request, boundary=CHUNK)

    planner.begin_lookup(request)
    unreachable = CHUNK * (_MAX_CAP_DESCENT + 2)
    assert planner.cap_hit(FakeSeq("r0"), unreachable) == 0


def test_no_stored_boundary_declines_the_hit_entirely():
    planner = make_planner()
    request = FakeRequest("r0")

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), 2 * CHUNK) == 0


def test_cap_declines_when_armed_for_a_different_sequence():
    planner = make_planner()
    request = FakeRequest("r0")
    store_one(planner, request, boundary=CHUNK)

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("someone-else"), CHUNK) == 0


def test_cap_is_inert_outside_a_lookup():
    """The hook stays installed on ATOM's scheduler for every model."""
    planner = make_planner()
    request = FakeRequest("r0")
    store_one(planner, request, boundary=CHUNK)
    planner.end_lookup()

    assert planner.cap_hit(FakeSeq("r0"), 999) == 999

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), 0) == 0
    assert planner.cap_hit(FakeSeq("r0"), -1) == -1


# --------------------------------------------------------------------------
# resolve_load
# --------------------------------------------------------------------------
def test_load_destination_is_the_block_row_that_ends_the_hit():
    planner = make_planner()
    request = FakeRequest("r0")
    store_one(planner, request, boundary=CHUNK)
    mamba_blocks = [11, 12, 13, 14]
    attention_blocks = [21, 22]

    planner.resolve_load(
        request,
        (attention_blocks, mamba_blocks),
        CHUNK,
        ATTENTION_GROUP,
        128,
        128,
    )

    (load,) = planner.take_loads()
    assert load.req_id == "r0"
    assert load.block_ids == (mamba_blocks[CHUNK // MAMBA_BLOCK - 1],)
    assert load.prefix_hash == planner.boundary_hash(request.block_hashes, CHUNK)
    # Exactly the attention blocks the dense leg is filling: [128, 256).
    assert load.error_block_ids == (22,)


def test_unresolvable_destination_is_queued_as_a_failing_load():
    """Dropping it would let the dense leg report success on its own."""
    planner = make_planner()
    request = FakeRequest("r0")

    planner.resolve_load(request, ([21, 22], []), CHUNK, ATTENTION_GROUP, 128, 128)

    (load,) = planner.take_loads()
    assert load.block_ids == ()
    assert load.error_block_ids == (22,)


def test_unaligned_hit_fails_closed_instead_of_writing_the_wrong_row():
    """``n // block - 1`` and ``(n - 1) // block`` disagree when n is not a
    multiple of the mamba block. The forward reads the second. Writing the
    first stores a successful load into a block the forward never reads.

    The hash block divides 70, so this is the alignment guard and not the
    missing-key guard.
    """
    planner = make_planner(hash_block_size=10)
    request = FakeRequest("r0")

    planner.resolve_load(
        request,
        ([21, 22, 23], [11, 12, 13, 14]),
        70,
        ATTENTION_GROUP,
        10,
        MAMBA_BLOCK,
    )

    (load,) = planner.take_loads()
    assert load.block_ids == ()
    assert load.error_block_ids


def test_a_missed_attention_slice_still_names_real_blocks():
    """An empty invalid set lets vLLM cache the MLA prefix."""
    planner = make_planner()
    request = FakeRequest("r0")

    planner.resolve_load(
        request,
        ([21], []),
        CHUNK,
        ATTENTION_GROUP,
        64,
        64,
    )

    (load,) = planner.take_loads()
    assert load.block_ids == ()
    assert load.error_block_ids == (21,)


def test_no_external_tokens_queues_nothing():
    planner = make_planner()
    planner.resolve_load(
        FakeRequest("r0"), ([21], [11]), CHUNK, ATTENTION_GROUP, 0, 128
    )
    assert planner.take_loads() == []


# --------------------------------------------------------------------------
# several mamba groups -- K3's real shape
# --------------------------------------------------------------------------
def make_two_group_planner(**kwargs):
    return make_planner(group_ids=(MAMBA_GROUP, MAMBA_GROUP_2), **kwargs)


def test_a_boundary_is_stored_only_when_every_mamba_group_reported():
    """One store, carrying one block per group, pinned in every group.

    vLLM commits the same boundary in each mamba group and hands each one off
    separately. Storing them as separate entries would put images under the
    same prefix hash that no lookup could reconcile; storing one of them under
    that hash would be an image that cannot be restored.
    """
    planner = make_two_group_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")

    stores = planner.collect_stores(
        {"r0": [(MAMBA_GROUP, 7, CHUNK), (MAMBA_GROUP_2, 9, CHUNK)]},
        {"r0": request},
    )

    assert len(stores) == 1
    # Group order, not hand-off order -- the byte stream is built that way.
    assert stores[0].block_ids == (7, 9)
    assert sorted(pool.touched) == ["blk7", "blk9"]

    planner.absorb_reports({stores[0].op_id: WORLD}, {})
    assert sorted(pool.freed) == ["blk7", "blk9"]
    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), CHUNK) == CHUNK


def test_a_boundary_missing_one_group_is_dropped_whole():
    planner = make_two_group_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r0")

    stores = planner.collect_stores({"r0": [(MAMBA_GROUP, 7, CHUNK)]}, {"r0": request})

    assert stores == []
    assert pool.touched == []
    assert not planner.has_pending_work()
    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r0"), CHUNK) == 0


def test_one_group_rejecting_a_boundary_takes_the_other_group_with_it():
    """The null block is rejected per group, and that leaves the boundary
    incomplete -- it must not be stored from the surviving group alone."""
    planner = make_two_group_planner()
    stores = planner.collect_stores(
        {"r0": [(MAMBA_GROUP, 7, CHUNK), (MAMBA_GROUP_2, 0, CHUNK)]},
        {"r0": FakeRequest("r0")},
    )
    assert stores == []


def test_each_boundary_is_joined_independently():
    planner = make_two_group_planner()
    planner.bind_gpu_block_pool(FakePool())
    request = FakeRequest("r0")

    stores = planner.collect_stores(
        {
            "r0": [
                (MAMBA_GROUP, 7, CHUNK),
                (MAMBA_GROUP_2, 9, CHUNK),
                (MAMBA_GROUP, 8, 2 * CHUNK),
            ]
        },
        {"r0": request},
    )

    assert [store.block_ids for store in stores] == [(7, 9)]


def test_load_names_one_destination_block_per_mamba_group():
    planner = make_two_group_planner()
    request = FakeRequest("r0")
    store_one_two_groups(planner, request)
    group_blocks = ([21, 22], [11, 12, 13, 14], [31, 32, 33, 34])

    planner.resolve_load(request, group_blocks, CHUNK, ATTENTION_GROUP, 128, 128)

    (load,) = planner.take_loads()
    row = CHUNK // MAMBA_BLOCK - 1
    assert load.block_ids == (
        group_blocks[MAMBA_GROUP][row],
        group_blocks[MAMBA_GROUP_2][row],
    )


def test_a_load_whose_second_group_has_no_destination_fails_whole():
    """Scattering into one group would leave the other holding the previous
    occupant's recurrence -- the half restore, with nothing to report it."""
    planner = make_two_group_planner()
    request = FakeRequest("r0")
    store_one_two_groups(planner, request)

    planner.resolve_load(
        request, ([21, 22], [11, 12, 13, 14], []), CHUNK, ATTENTION_GROUP, 128, 128
    )

    (load,) = planner.take_loads()
    assert load.block_ids == ()
    assert load.error_block_ids == (22,)


def test_mamba_groups_must_agree_on_block_size():
    """A boundary is one block in every group at the same token count."""
    pytest.importorskip("vllm")
    from vllm.v1.kv_cache_interface import MambaSpec

    from atom.plugin.vllm.kv_transfer.kda_state import find_mamba_groups

    class Group:
        def __init__(self, spec):
            self.kv_cache_spec = spec

    specs = [
        MambaSpec.__new__(MambaSpec),
        MambaSpec.__new__(MambaSpec),
    ]
    object.__setattr__(specs[0], "block_size", 64)
    object.__setattr__(specs[1], "block_size", 128)
    with pytest.raises(ValueError, match="disagree on block size"):
        find_mamba_groups([Group(specs[0]), Group(specs[1])])


# --------------------------------------------------------------------------
# construction guard
# --------------------------------------------------------------------------
def test_chunk_size_must_be_a_multiple_of_the_mamba_block_size():
    with pytest.raises(ValueError, match="not a multiple of the mamba block size"):
        make_planner(chunk_size=CHUNK + 1)


# --------------------------------------------------------------------------
# KdaPageViews / build_layout_id
# --------------------------------------------------------------------------
def test_page_views_address_one_block_across_every_layer():
    tensors = [torch.arange(4 * 6, dtype=torch.uint8).reshape(4, 6) for _ in range(3)]
    views = KdaPageViews([tensors], layout_id="x")

    assert views.num_blocks == [4]
    assert views.entry_bytes == 3 * 6
    assert len(views.page_unit_views([2])) == 3
    assert all(
        v.data_ptr() == t[2].data_ptr()
        for v, t in zip(views.page_unit_views([2]), tensors)
    )
    assert [v.data_ptr() for v in views.state_entry_views([2])] == [
        v.data_ptr() for v in views.page_unit_views([2])
    ]


def test_page_views_span_every_mamba_group_in_group_order():
    """The image is the groups concatenated, each at its own block id: vLLM
    gives every mamba group its own block table, and a state restored into
    only some of them is the half restore this module exists to prevent."""
    g0 = [torch.arange(4 * 6, dtype=torch.uint8).reshape(4, 6) for _ in range(2)]
    g1 = [torch.arange(4 * 6, dtype=torch.uint8).reshape(4, 6) for _ in range(3)]
    views = KdaPageViews([g0, g1], layout_id="x")

    assert views.num_blocks == [4, 4]
    assert views.entry_bytes == (2 + 3) * 6

    got = views.page_unit_views([1, 3])
    assert [v.data_ptr() for v in got] == [t[1].data_ptr() for t in g0] + [
        t[3].data_ptr() for t in g1
    ]
    # Store and load must walk the identical stream, or the bytes land
    # transposed across groups with no error anywhere.
    assert [v.data_ptr() for v in views.state_entry_views([1, 3])] == [
        v.data_ptr() for v in got
    ]


def test_page_views_reject_layers_that_disagree_on_block_count():
    with pytest.raises(ValueError, match="disagree on block count"):
        KdaPageViews([[torch.zeros(4, 6), torch.zeros(5, 6)]], layout_id="x")


def test_a_boundary_is_one_block_per_mamba_group():
    views = KdaPageViews([[torch.zeros(4, 6)]], layout_id="x")
    with pytest.raises(ValueError, match="one block per mamba group"):
        views.page_unit_views([1, 2])
    with pytest.raises(IndexError):
        views.state_entry_views([4])


def test_layout_id_separates_geometries_that_share_a_pool():
    class Spec:
        def __init__(self, block_size, page_size_bytes):
            self.mamba_type = "kda"
            self.block_size = block_size
            self.page_size_bytes = page_size_bytes
            self.num_speculative_blocks = 0
            self.tp_replicated = False

    tensors = [[torch.zeros(4, 6), torch.zeros(4, 6)]]
    base = build_layout_id([Spec(64, 1024)], tensors)

    assert build_layout_id([Spec(128, 1024)], tensors) != base
    assert build_layout_id([Spec(64, 2048)], tensors) != base
    assert build_layout_id([Spec(64, 1024)], [[torch.zeros(4, 7)]]) != base
    assert build_layout_id([Spec(64, 1024)], tensors * 2) != base
    # Same layers, different split: the stream is the groups concatenated, so
    # re-splitting reorders it and must not read back under the old key.
    assert (
        build_layout_id([Spec(64, 1024)], [[torch.zeros(4, 6)], [torch.zeros(4, 6)]])
        != base
    )


def test_uniform_type_wrapper_is_unwrapped_before_classification():
    """``isinstance(wrapper, MambaSpec)`` is False -- a mamba group would read
    as attention and be saved positionally."""

    class Inner:
        pass

    class Wrapper:
        def __init__(self, inner):
            self.kv_cache_specs = {"layers.0": inner}

    inner = Inner()
    assert unwrap_kv_cache_spec(Wrapper(inner)) is inner
    assert unwrap_kv_cache_spec(inner) is inner


# ---------------------------------------------------------------------------
# Gathering the recurrent groups' tensors out of vLLM's registration
#
# This seam is where the worker half of the recurrent leg is wired up, and it
# reads `split_kv_caches_by_group`'s output. That output is a LIST indexed by
# group id, not a mapping -- a boot failure that no unit test saw, because
# nothing fed the two functions to each other.
# ---------------------------------------------------------------------------


def test_group_tensors_are_gathered_through_the_split_output():
    """The two functions compose: whatever the split returns, the gather reads."""
    groups = [
        FakeGroup(["kda.0", "kda.1"]),
        FakeGroup(["mla.0"]),
        FakeGroup(["kda.2"]),
    ]
    caches = {
        "kda.0": torch.zeros(2),
        "kda.1": torch.ones(2),
        "mla.0": torch.full((2,), 7.0),
        "kda.2": torch.full((2,), 9.0),
    }
    per_group = split_kv_caches_by_group(caches, groups)
    gathered = gather_group_tensors(per_group, groups, [0, 2])
    assert [[float(t[0]) for t in g] for g in gathered] == [[0.0, 1.0], [9.0]]


def test_gathered_layers_keep_vllm_order_not_sorted_order():
    """Sorting here would silently reorder the byte stream between runs."""
    groups = [FakeGroup(["kda.10", "kda.2"])]
    caches = {"kda.2": torch.zeros(1), "kda.10": torch.ones(1)}
    gathered = gather_group_tensors(
        split_kv_caches_by_group(caches, groups), groups, [0]
    )
    assert [float(t[0]) for t in gathered[0]] == [1.0, 0.0]


def test_a_group_missing_one_layer_is_refused_not_shortened():
    groups = [FakeGroup(["kda.0", "kda.1"]), FakeGroup(["mla.0"])]
    caches = {"kda.0": torch.zeros(1), "mla.0": torch.zeros(1)}
    per_group = split_kv_caches_by_group(caches, groups)
    with pytest.raises(ValueError, match="kda.1"):
        gather_group_tensors(per_group, groups, [0])


def test_an_out_of_range_group_id_is_refused():
    groups = [FakeGroup(["kda.0"])]
    per_group = split_kv_caches_by_group({"kda.0": torch.zeros(1)}, groups)
    with pytest.raises(ValueError, match="out of range"):
        gather_group_tensors(per_group, groups, [1])


# --- the block count the codec strides by -----------------------------------
#
# The dense leg has no bounds check that can catch a wrong block count: the ids
# stay in range and the transfers succeed, so a stride that is a whole power of
# the block size too fine restores bytes from the wrong rows in silence. The
# recurrent leg does bounds-check (`KdaPageViews` refuses an id past its own
# tensors), which is why that leg would have failed loudly and this one did not.


def test_a_token_major_leading_dim_folds_to_the_block_count():
    # Kimi-K3: ATOM's MLA backend asks for a kernel block size of 1, so vLLM
    # allocates one row per token -- 1584 blocks of 1536 tokens.
    assert resolve_block_count(1584 * 1536, 1584, 1536) == 1584


def test_a_block_major_leading_dim_is_already_the_block_count():
    # GLM-5.2: kernel block size equals the block size, so nothing folds.
    assert resolve_block_count(8192, 8192, 64) == 8192


def test_a_missing_vllm_block_count_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no KV cache block count"):
        resolve_block_count(1584 * 1536, 0, 1536)


def test_a_leading_dim_that_is_not_whole_blocks_is_refused():
    with pytest.raises(ValueError, match="not a whole number"):
        resolve_block_count(1000, 7, 1536)


def test_a_fold_that_does_not_divide_the_block_size_is_refused():
    # 1584 rows over 792 blocks implies 2 kernel blocks per block, which cannot
    # be right when the block size is odd -- the leading axis is something else.
    with pytest.raises(ValueError, match="does not divide the block size"):
        resolve_block_count(1584, 792, 1535)


# --- the layout id a human reads vs the one the key folds in -----------------
#
# `build_layout_id` ends with one `shape:dtype` per layer: 69 identical entries
# on K3, 1,877 characters on one line per worker. `summarize_layout_id` is for
# the log only. It has to stay lossless, because its whole job is letting a
# reader see that two boots' layouts differ -- a summary that could collapse a
# real difference would read as "same layout" while the cache refuses to hit.


def test_identical_layers_collapse_to_one_run():
    layout = "vllm-kda|bs=1536|" + ";".join(["(1, 1, 884736):torch.int8"] * 69)
    assert summarize_layout_id(layout) == (
        "vllm-kda|bs=1536|69x(1, 1, 884736):torch.int8"
    )


def test_a_differing_layer_survives_the_collapse():
    same = "(1, 1, 884736):torch.int8"
    odd = "(1, 1, 884736):torch.float32"
    a = summarize_layout_id("head|" + ";".join([same] * 69))
    b = summarize_layout_id("head|" + ";".join([same] * 68 + [odd]))
    assert a != b
    assert b == f"head|68x{same};{odd}"


def test_runs_keep_their_order():
    a = "(1,):torch.int8"
    b = "(2,):torch.int8"
    # Same multiset, different order: the byte stream is the groups concatenated
    # in order, so these are different layouts and must not summarize alike.
    first = summarize_layout_id(f"head|{a};{a};{b}")
    second = summarize_layout_id(f"head|{b};{a};{a}")
    assert first == "head|2x(1,):torch.int8;(2,):torch.int8"
    assert second == "head|(2,):torch.int8;2x(1,):torch.int8"
    assert first != second


def test_a_layout_id_with_no_tail_is_returned_unchanged():
    assert summarize_layout_id("vllm-kda") == "vllm-kda"


def test_the_summary_is_not_what_the_key_folds_in():
    # If these ever became the same call, shortening the log would shorten the
    # discriminator and two layouts could share a key -- silent wrong bytes, not
    # a miss. Pin them apart.
    class Spec:
        mamba_type = "kda"
        block_size = 1536
        page_size_bytes = 884736
        num_speculative_blocks = 0
        tp_replicated = False

    spec = Spec()
    tensors = [[torch.zeros(4, 1, 1, 884736, dtype=torch.int8)] * 2]
    full = build_layout_id([spec], tensors)
    assert summarize_layout_id(full) != full
    assert full.endswith("(1, 1, 884736):torch.int8;(1, 1, 884736):torch.int8")


# --------------------------------------------------------------------------
# collect_cached_boundary_stores -- the source that makes a joint hit possible
#
# The hand-off alone produced exactly zero joint hits on Kimi-K3 over 900s and
# 3.96M external queries, because vLLM emits a boundary only when it is *not* a
# whole block while this planner can only use one that *is* (cap_hit descends in
# chunk steps). The two acceptance domains are disjoint, so the tests below are
# about the second source: boundaries resolved by hash out of the block pool.
# --------------------------------------------------------------------------
def publish_boundary(pool, request, boundary, group_blocks):
    """Register *boundary*'s block in each group, as vLLM's own caching does."""
    pool.publish(request.block_hashes[boundary // HASH_BLOCK - 1], group_blocks)


def test_whole_block_boundary_is_stored_although_no_handoff_names_it():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})

    # No hand-off at all -- exactly the K3 shape.
    assert planner.collect_stores({}, {"r1": request}) == []
    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert [store.block_ids for store in stores] == [(11,)]
    assert pool.touched == ["blk11"]
    assert stores[0].prefix_hash == planner.boundary_hash(request.block_hashes, CHUNK)


def test_sweep_offers_every_chunk_aligned_boundary_below_the_frontier():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    for n in (1, 2, 3):
        publish_boundary(pool, request, n * CHUNK, {MAMBA_GROUP: 20 + n})

    stores = planner.collect_cached_boundary_stores(
        {"r1": 3 * CHUNK + 5}, {"r1": request}
    )

    assert [store.block_ids for store in stores] == [(21,), (22,), (23,)]


def test_an_uncached_hole_is_retried_without_restoring_a_later_boundary():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, 2 * CHUNK, {MAMBA_GROUP: 22})

    first = planner.collect_cached_boundary_stores({"r1": 2 * CHUNK}, {"r1": request})
    assert [store.block_ids for store in first] == [(22,)]

    second = planner.collect_cached_boundary_stores({"r1": 2 * CHUNK}, {"r1": request})
    assert second == []

    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})
    third = planner.collect_cached_boundary_stores({"r1": 2 * CHUNK}, {"r1": request})
    assert [store.block_ids for store in third] == [(11,)]


def test_a_boundary_is_offered_once_across_steps():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})

    first = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})
    second = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert len(first) == 1
    assert second == []


def test_a_boundary_the_pool_no_longer_holds_is_counted_not_guessed():
    """The whole safety argument: a superseded, freed or relocated state block
    is not in the pool under that hash, so the sweep declines rather than
    persisting whatever now occupies the row it would have indexed."""
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")

    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert stores == []
    assert pool.touched == []
    assert planner.stats()["sweep_uncached"] == 1


def test_a_boundary_missing_one_mamba_group_is_not_stored_at_all():
    planner = make_two_group_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})  # group 2 absent

    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert stores == []
    assert pool.touched == []


def test_both_mamba_groups_travel_as_one_boundary():
    planner = make_two_group_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11, MAMBA_GROUP_2: 12})

    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert [store.block_ids for store in stores] == [(11, 12)]


def test_swept_boundary_makes_the_hit_cap_pass():
    """End to end over the gate that reported zero: store by sweep, then ask
    ``cap_hit`` the question the scheduler asks."""
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})

    for store in planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request}):
        planner.absorb_reports({store.op_id: WORLD}, {})

    planner.begin_lookup(request)
    assert planner.cap_hit(FakeSeq("r1"), CHUNK) == CHUNK


def test_an_already_stored_boundary_is_not_stored_again():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})
    for store in planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request}):
        planner.absorb_reports({store.op_id: WORLD}, {})

    planner.forget_request("r1")  # as a preemption or a second request would
    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    assert stores == []
    assert planner.stats()["sweep_known"] == 1


def test_preemption_rewinds_the_sweep_cursor():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})
    planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    planner.forget_request("r1")
    stores = planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request})

    # Offered again (the index has not indexed it -- no rank reported yet), so
    # a request that comes back from preemption does not silently lose the
    # boundaries it is about to recompute.
    assert [store.block_ids for store in stores] == [(11,)]


def test_finished_and_preempted_requests_are_skipped_by_the_sweep():
    planner = make_planner()
    pool = FakePool()
    planner.bind_gpu_block_pool(pool)
    request = FakeRequest("r1")
    publish_boundary(pool, request, CHUNK, {MAMBA_GROUP: 11})

    stores = planner.collect_cached_boundary_stores(
        {"r1": CHUNK}, {"r1": request}, skip_req_ids={"r1"}
    )

    assert stores == []


def test_sweep_is_inert_until_the_pool_is_bound():
    planner = make_planner()
    request = FakeRequest("r1")

    assert planner.collect_cached_boundary_stores({"r1": CHUNK}, {"r1": request}) == []


def test_stats_name_every_reason_a_boundary_was_not_stored():
    """The run this fixes was silent because every explanation was ``debug``.
    Whatever else changes, the counters have to keep saying why."""
    planner = make_planner()
    planner.bind_gpu_block_pool(FakePool())
    stats = planner.stats()

    for name in (
        "handoff_entries",
        "sweep_offered",
        "sweep_stores",
        "sweep_uncached",
        "sweep_no_hash",
        "sweep_known",
        "cap_kept",
        "cap_declined",
    ):
        assert name in stats
