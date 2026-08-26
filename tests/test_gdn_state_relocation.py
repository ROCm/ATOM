# SPDX-License-Identifier: MIT
# Tests for relocating a GDN state slot's bytes.
#
# GDN checkpoints by forking, so this path is not about checkpoints: moving the
# state pool's boundary has to be able to shift a slot out of the way, and that
# is a byte move whatever mechanism the class uses to checkpoint.
#
# The unit that moves is ONE slot. A request under speculative decoding holds
# `1 + num_spec` of them, but they are allocated one at a time and need not be
# adjacent, so relocating a whole request is several of these calls and the
# caller names each slot. `num_spec` therefore does not appear in this path at
# all -- which is the property most of these cases are pinning down.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.model_ops.attentions.gdn_attn import GDNStateMixin, slot_indexed_caches
from atom.model_ops.attentions.kimi_mla_gdn_attn import _KimiMLAGDNCommon

LAYERS = 3
SLOTS = 12
SHAPE_K = (2, 5)
SHAPE_V = (2, 3, 4)


def build(num_spec: int = 0, replayssm: bool = False):
    """Caches whose every (layer, slot) plane carries a distinct value."""
    k = torch.zeros((LAYERS, SLOTS) + SHAPE_K)
    v = torch.zeros((LAYERS, SLOTS) + SHAPE_V)
    for layer in range(LAYERS):
        for slot in range(SLOTS):
            k[layer, slot] = layer * 100 + slot
            v[layer, slot] = -(layer * 100 + slot)
    runner = SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v)
    if replayssm:
        # The record buffers share the (layer, slot) axes, so they carry the
        # same distinct values; the cursor is per-slot and one-dimensional.
        for name, shape in (
            ("replayssm_buf_k", SHAPE_K),
            ("replayssm_buf_u", SHAPE_V),
            ("replayssm_buf_g", (2,)),
        ):
            buf = torch.zeros((LAYERS, SLOTS) + shape)
            for layer in range(LAYERS):
                for slot in range(SLOTS):
                    buf[layer, slot] = layer * 100 + slot
            setattr(runner, name, buf)
        runner.replayssm_write_pos = torch.arange(SLOTS, dtype=torch.int32)
    stub = SimpleNamespace(
        num_spec=num_spec,
        replayssm=replayssm,
        model_runner=runner,
    )
    return stub, k, v


@pytest.mark.parametrize("num_spec", [0, 2])
def test_relocation_moves_every_layer_of_the_slot(num_spec):
    """And moves exactly the one slot, whatever `num_spec` says.

    Parametrized over `num_spec` precisely because the answer must not depend
    on it: the slot is the unit, and a wider request is more calls, not a wider
    call.
    """
    stub, k, v = build(num_spec)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    assert torch.equal(k[:, 3], before_k[:, 1])
    assert torch.equal(v[:, 3], before_v[:, 1])
    # The source is untouched: relocation duplicates, the caller retires the
    # old index afterwards.
    assert torch.equal(k[:, 1], before_k[:, 1])


def test_relocation_leaves_every_other_slot_alone():
    stub, k, v = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    for slot in range(SLOTS):
        if slot == 3:
            continue
        assert torch.equal(k[:, slot], before_k[:, slot])
        assert torch.equal(v[:, slot], before_v[:, slot])


def test_several_pairs_in_one_call():
    """Including a scattered set, which is what a whole request looks like now.

    (0, 2) and (1, 7) are one request's two slots going to two destinations
    that are neither adjacent to each other nor at a fixed offset from the
    sources. Nothing in this path may assume otherwise.
    """
    stub, k, _ = build(num_spec=1)
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(0, 2), (1, 7)])

    assert torch.equal(k[:, 2], before_k[:, 0])
    assert torch.equal(k[:, 7], before_k[:, 1])


def test_no_pairs_is_a_no_op():
    stub, k, v = build(num_spec=2)
    before_k, before_v = k.clone(), v.clone()

    GDNStateMixin.relocate_state_slots(stub, [])

    assert torch.equal(k, before_k)
    assert torch.equal(v, before_v)


# --- ReplaySSM -------------------------------------------------------------
# Under ReplaySSM a slot's state is not the two caches alone: the (k, u, g)
# records written since the checkpoint, and the cursor saying how many there
# are, are as much a part of it. A relocation that moved only the caches would
# leave the destination reading the previous tenant's decode history -- which
# is silent, because every tensor involved is validly shaped and populated.


def test_relocation_moves_the_record_buffers_too():
    stub, _, _ = build(num_spec=2, replayssm=True)
    runner = stub.model_runner
    before = {
        name: getattr(runner, name).clone()
        for name in ("replayssm_buf_k", "replayssm_buf_u", "replayssm_buf_g")
    }

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    for name, was in before.items():
        now = getattr(runner, name)
        assert torch.equal(now[:, 3], was[:, 1])
        # Every other slot untouched, same as the caches.
        for slot in range(SLOTS):
            if slot != 3:
                assert torch.equal(now[:, slot], was[:, slot])


def test_relocation_moves_the_write_cursor():
    """The cursor is what makes the records mean anything.

    It is indexed by slot rather than by (layer, slot) -- one cursor serves
    every linear-attention layer -- so it moves by a different call than the
    buffers and is worth its own case.
    """
    stub, _, _ = build(num_spec=2, replayssm=True)
    write_pos = stub.model_runner.replayssm_write_pos
    before = write_pos.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3), (0, 7)])

    assert write_pos[3] == before[1]
    assert write_pos[7] == before[0]
    for slot in range(SLOTS):
        if slot not in (3, 7):
            assert write_pos[slot] == before[slot]


def test_relocation_ignores_replay_buffers_when_disabled():
    """A stub with no ReplaySSM attributes at all must still relocate.

    ReplaySSM is off by default, so the baseline path may not so much as look
    at `replayssm_buf_*` -- reading them would turn the default configuration
    into an AttributeError.
    """
    stub, k, _ = build(num_spec=2)
    assert not hasattr(stub.model_runner, "replayssm_buf_k")
    before_k = k.clone()

    GDNStateMixin.relocate_state_slots(stub, [(1, 3)])

    assert torch.equal(k[:, 3], before_k[:, 1])


# --- Transfer regions ------------------------------------------------------
# `state_slot_regions` names the same bytes for a third caller, which unlike
# the other two holds no tensor: a peer's NIC, writing from an address it
# computed itself. A region that resolved a slot to the wrong offset would not
# raise -- it would land on another slot's plane and be transferred happily --
# so the arithmetic is pinned here rather than left to the first deployment.


def test_a_region_resolves_a_slot_to_that_slot_s_own_bytes():
    """For every layer, not just layer 0.

    The slot sits on axis 1, so a layer's planes are one contiguous run and
    the layers are a stride apart. Getting that backwards still yields a
    plausible address inside the cache.
    """
    stub, k, v = build()

    regions, num_slots = GDNStateMixin.state_slot_regions(stub)

    assert num_slots == SLOTS
    assert len(regions) == 2 * LAYERS
    for cache, own in ((k, regions[:LAYERS]), (v, regions[LAYERS:])):
        for layer, region in enumerate(own):
            for slot in range(SLOTS):
                assert region.unit_addr(slot) == cache[layer, slot].data_ptr()


def test_a_region_stops_at_the_end_of_its_own_layer():
    """`total_bytes` is what gets registered with the RDMA engine and what
    bounds the peer's addressing. A region running to the end of the cache
    would claim the following layers' bytes as its own slots.
    """
    stub, _, _ = build()

    regions, _ = GDNStateMixin.state_slot_regions(stub)

    for region in regions:
        assert region.total_bytes == SLOTS * region.unit_bytes
        last_slot_end = region.unit_addr(SLOTS - 1) + region.unit_bytes
        assert last_slot_end == region.base_addr + region.total_bytes


@pytest.mark.parametrize("replayssm", [False, True])
def test_the_regions_describe_every_byte_the_relocation_moves(replayssm):
    """The drift guard between the two.

    Both read `_slot_indexed_caches`, so a cache added to one is added to the
    other; this pins that the region builder also describes each of them
    *whole*, which the shared list alone does not say.
    """
    stub, _, _ = build(replayssm=replayssm)

    regions, _ = GDNStateMixin.state_slot_regions(stub)

    described = sum(r.total_bytes for r in regions)
    expected = sum(
        c.numel() * c.element_size()
        for c in slot_indexed_caches(stub.model_runner, replayssm)
    )
    if replayssm:
        pos = stub.model_runner.replayssm_write_pos
        expected += pos.numel() * pos.element_size()
    assert described == expected


def test_the_write_cursor_gets_a_region_of_its_own():
    """It has no layer axis -- one int32 per slot -- so it is one region with
    an element-sized unit, not one per layer like everything else. Omitting it
    would send records the peer cannot tell the extent of.
    """
    plain, _, _ = build()
    replay, _, _ = build(replayssm=True)

    plain_regions, _ = GDNStateMixin.state_slot_regions(plain)
    replay_regions, _ = GDNStateMixin.state_slot_regions(replay)

    # Three more (layer, slot) buffers, plus the cursor.
    assert len(replay_regions) == len(plain_regions) + 3 * LAYERS + 1

    cursor = replay_regions[-1]
    write_pos = replay.model_runner.replayssm_write_pos
    assert cursor.unit_bytes == write_pos.element_size()
    for slot in range(SLOTS):
        assert cursor.unit_addr(slot) == write_pos[slot].data_ptr()


# --- Joining the two halves (K3) -------------------------------------------
# K3's cache is half paged and half slot-indexed, and the MLA base class
# describes only the paged half. Everything above pins the slot half in
# isolation; these pin that it reaches the connector attached to the other
# one, and only when a peer is actually going to read it.


class _MLABase:
    """Stands in for `AiterMLAMetadataBuilder`: block regions, no slot half."""

    BLOCKS = 7

    def get_kv_transfer_tensors(self):
        return KVTransferTensors(
            block_regions=[KVTransferRegion(0x1000, 64 * self.BLOCKS, 64)],
            slot_regions=[],
            num_blocks=self.BLOCKS,
        )


def kimi_builder(kv_transfer_config, pp_size=1, replayssm=False):
    """A K3 builder over the relocation fixture's caches.

    Concrete rather than a `SimpleNamespace` because the method under test
    calls `super()`, which needs a real MRO to walk -- the MLA half of the
    answer is the base class's to give.
    """

    class _Builder(_KimiMLAGDNCommon, _MLABase):
        def __init__(self):  # the real one wants a live ModelRunner
            pass

    stub, _, _ = build(replayssm=replayssm)
    builder = _Builder()
    builder.model_runner = stub.model_runner
    builder.replayssm = replayssm
    builder.mla_idx_by_layer = dict.fromkeys(range(24), 0)
    builder.kda_idx_by_layer = dict.fromkeys(range(LAYERS), 0)
    builder.model_runner.config = SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        pipeline_parallel_size=pp_size,
        hf_config=SimpleNamespace(model_type="kimi_linear"),
    )
    return builder


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        pytest.param(None, id="none"),
        pytest.param({}, id="empty"),
        pytest.param({"kv_connector": "lmcache_offload"}, id="offload"),
    ],
)
def test_an_aggregated_engine_is_left_exactly_as_the_base_class_answered(
    kv_transfer_config,
):
    """No peer, no slot regions -- including for the offload tier.

    The aggregated LMCache configuration is a populated `kv_transfer_config`
    that nobody transfers from, and it reads the state pool through
    `state_backend` rather than through regions. Describing the pool twice
    would be harmless; treating this as disaggregated and rejecting it would
    take down a configuration that works today, which is why `lmcache_offload`
    is a case here rather than folded into the truthiness of the dict.
    """
    builder = kimi_builder(kv_transfer_config)

    tensors = builder.get_kv_transfer_tensors()

    assert tensors.slot_regions == []
    assert tensors.num_slots == 0
    assert tensors.num_blocks == _MLABase.BLOCKS


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        pytest.param({"kv_connector": "mooncake"}, id="mooncake"),
        pytest.param(
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload"},
                    {"kv_connector": "mooncake"},
                ],
            },
            id="multi",
        ),
    ],
)
def test_a_mooncake_peer_gets_both_halves(kv_transfer_config):
    """Including behind `multi`, which is how a prefill node that also offloads
    is configured -- the slot half must not depend on mooncake being named at
    the top level.
    """
    builder = kimi_builder(kv_transfer_config)

    tensors = builder.get_kv_transfer_tensors()

    assert tensors.num_slots == SLOTS
    assert len(tensors.slot_regions) == 2 * LAYERS
    # The paged half survives the join.
    assert len(tensors.block_regions) == 1
    assert tensors.num_blocks == _MLABase.BLOCKS


def test_a_connector_that_ignores_slot_regions_is_refused():
    """MoRIIO transfers block regions only. Handing it this region map would
    move the full-attention layers, report success and leave decode running
    the linear-attention ones on a zeroed state.
    """
    builder = kimi_builder({"kv_connector": "moriio"})

    with pytest.raises(NotImplementedError, match="mooncake"):
        builder.get_kv_transfer_tensors()


def test_pipeline_parallelism_is_refused():
    """`_consumer_region_map` aligns a stage's regions by shifting over
    `num_hidden_layers`; the slot regions are indexed by KDA layer, a shorter
    axis. The misalignment writes one layer's state onto another's rather than
    failing, so it is refused where it is knowable.
    """
    builder = kimi_builder({"kv_connector": "mooncake"}, pp_size=2)

    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        builder.get_kv_transfer_tensors()
