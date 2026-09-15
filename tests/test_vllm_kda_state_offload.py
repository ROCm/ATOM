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
(``find_mamba_group`` degrades to None), and the CI unit job has no vLLM.
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
    unwrap_kv_cache_spec,
)
from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    split_kv_caches_by_group,
)

MAMBA_GROUP = 1
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


class FakePool:
    """vLLM's ``BlockPool``, reduced to the pin surface the planner uses."""

    def __init__(self, num_blocks=64):
        self.blocks = [f"blk{i}" for i in range(num_blocks)]
        self.touched = []
        self.freed = []

    def touch(self, blocks):
        self.touched.extend(blocks)

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


def make_planner(**kwargs):
    params = {
        "group_id": MAMBA_GROUP,
        "mamba_block_size": MAMBA_BLOCK,
        "hash_block_size": HASH_BLOCK,
        "chunk_size": CHUNK,
        "world_size": WORLD,
    }
    params.update(kwargs)
    return KdaBoundaryPlanner(**params)


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
    assert stores[0].block_id == 7
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
    assert load.block_id == mamba_blocks[CHUNK // MAMBA_BLOCK - 1]
    assert load.prefix_hash == planner.boundary_hash(request.block_hashes, CHUNK)
    # Exactly the attention blocks the dense leg is filling: [128, 256).
    assert load.error_block_ids == (22,)


def test_unresolvable_destination_is_queued_as_a_failing_load():
    """Dropping it would let the dense leg report success on its own."""
    planner = make_planner()
    request = FakeRequest("r0")

    planner.resolve_load(request, ([21, 22], []), CHUNK, ATTENTION_GROUP, 128, 128)

    (load,) = planner.take_loads()
    assert load.block_id == 0
    assert load.error_block_ids == (22,)


def test_no_external_tokens_queues_nothing():
    planner = make_planner()
    planner.resolve_load(
        FakeRequest("r0"), ([21], [11]), CHUNK, ATTENTION_GROUP, 0, 128
    )
    assert planner.take_loads() == []


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
    views = KdaPageViews(tensors, layout_id="x")

    assert views.num_blocks == 4
    assert views.entry_bytes == 3 * 6
    assert len(views.page_unit_views([2])) == 3
    assert all(
        v.data_ptr() == t[2].data_ptr()
        for v, t in zip(views.page_unit_views([2]), tensors)
    )
    assert [v.data_ptr() for v in views.state_entry_views(2)] == [
        v.data_ptr() for v in views.page_unit_views([2])
    ]


def test_page_views_reject_layers_that_disagree_on_block_count():
    with pytest.raises(ValueError, match="disagree on block count"):
        KdaPageViews([torch.zeros(4, 6), torch.zeros(5, 6)], layout_id="x")


def test_a_boundary_is_exactly_one_block():
    views = KdaPageViews([torch.zeros(4, 6)], layout_id="x")
    with pytest.raises(ValueError, match="a boundary is one block"):
        views.page_unit_views([1, 2])
    with pytest.raises(IndexError):
        views.state_entry_views(4)


def test_layout_id_separates_geometries_that_share_a_pool():
    class Spec:
        def __init__(self, block_size, page_size_bytes):
            self.mamba_type = "kda"
            self.block_size = block_size
            self.page_size_bytes = page_size_bytes
            self.num_speculative_blocks = 0
            self.tp_replicated = False

    tensors = [torch.zeros(4, 6)]
    base = build_layout_id(Spec(64, 1024), tensors)

    assert build_layout_id(Spec(128, 1024), tensors) != base
    assert build_layout_id(Spec(64, 2048), tensors) != base
    assert build_layout_id(Spec(64, 1024), [torch.zeros(4, 7)]) != base
    assert build_layout_id(Spec(64, 1024), tensors * 2) != base


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
