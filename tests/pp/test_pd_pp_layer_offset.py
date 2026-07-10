# SPDX-License-Identifier: MIT
# CPP+PD J1 — producer per-layer region mapping for PP-prefill push (GPU-free).

from atom.kv_transfer.disaggregation.port_offset import consumer_region_indices


def _starts(partitions):
    """Global start layer of each stage, given a per-stage layer-count list."""
    starts, acc = [], 0
    for p in partitions:
        starts.append(acc)
        acc += p
    return starts


def test_identity_when_pp1():
    # pp_size=1: local layers == all layers → identity map.
    assert consumer_region_indices(156, 78, 0, 78, 1) == list(range(156))


def test_identity_when_empty():
    assert consumer_region_indices(0, 0, 5, 78, 4) == []


def test_group_major_single_group():
    # 1 group (1 region/layer). Stage of 20 layers starting at global 18 →
    # consumer indices 18..37.
    assert consumer_region_indices(20, 20, 18, 78, 4) == list(range(18, 38))


def test_group_major_two_groups_mla():
    # MLA layout: 2 groups [kv..., index...], stage=20 layers @ start 18, N=78.
    # kv group -> consumer 18..37 ; index group -> consumer 78+18..78+37.
    got = consumer_region_indices(40, 20, 18, 78, 4)
    assert got[:20] == list(range(18, 38))
    assert got[20:] == list(range(78 + 18, 78 + 38))


def test_undefined_when_not_multiple():
    assert consumer_region_indices(41, 20, 18, 78, 4) is None


def test_stage_maps_tile_consumer_group_major_no_overlap():
    # GLM-5.2: 78 layers, PP4 partition [18,20,20,20], 2 groups (kv + index).
    # Every stage's regions must map onto distinct consumer indices that,
    # unioned, exactly cover the consumer's 156 regions with no overlap.
    partitions, num_hidden, groups = [18, 20, 20, 20], 78, 2
    covered = []
    for start, n_local in zip(_starts(partitions), partitions):
        cmap = consumer_region_indices(n_local * groups, n_local, start, num_hidden, 4)
        covered.extend(cmap)
    total = num_hidden * groups
    assert sorted(covered) == list(range(total))
    assert len(covered) == len(set(covered))  # no overlap


def test_group_major_beats_naive_offset():
    # Regression guard for the bug found on GPU: a naive additive offset
    # (start_layer*groups + i) would misroute group-major layouts. Stage1's
    # index-group region 0 (local idx 20) must land in the consumer's index
    # group (>=78), NOT at 36+20=56 inside the kv group.
    cmap = consumer_region_indices(40, 20, 18, 78, 4)
    assert cmap[20] == 78 + 18  # first index-group region
    assert cmap[20] != 56  # what the old additive-offset formula produced
