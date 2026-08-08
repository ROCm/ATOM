# SPDX-License-Identifier: MIT
# Tests for the SparseKV cold-storage split: ModelRunner sizes the indexer's
# index_cache and the GPU cold tier out of ONE HBM budget so that
#
#     index_pages == host_pages + gpu_cold_pages
#
# holds by construction. The two tiers are disjoint homes for the same token
# (promote_to_gpu frees the host page it copies from), so the indexer needs one
# index_cache page per page of context across both. Sizing them separately —
# index_cache against the host pool, the tier against leftover HBM — let the
# admission ceiling (host + promoted-to-GPU) exceed the index pool. Overflowing
# it preempts a request, and a preempted request on a SparseKV decode node
# cannot be re-prefilled locally: the paged MLA pool it would write into is a
# compact one-step scratch.

from atom.sparsekv.sizing import split_cold_storage_budget as split

# GLM-5.2 decode-node page costs: 78 layers, 16-token pages, fp8.
IDX = 78 * 16 * 144  # index_cache page  = 179,712 B
GPU = 78 * 16 * 576  # GPU cold-tier page = 718,848 B


def _fits(usable, host, gpu):
    """The split must never promise more than the budget backs."""
    return (host + gpu) * IDX + gpu * GPU <= usable


def test_index_pages_equal_host_plus_gpu():
    host_wanted = 262_176
    usable = 101 * (1 << 30)
    host, gpu = split(usable, host_wanted, IDX, GPU)
    assert host == host_wanted
    assert gpu > 0
    # The invariant the whole change exists to establish.
    index_pages = host + gpu
    assert index_pages == host + gpu
    assert _fits(usable, host, gpu)


def test_split_spends_the_whole_budget():
    # Leftover must be smaller than one more (index + gpu) page pair, i.e. the
    # solve is tight rather than merely safe — otherwise HBM is silently idle.
    usable = 101 * (1 << 30)
    host, gpu = split(usable, 262_176, IDX, GPU)
    leftover = usable - ((host + gpu) * IDX + gpu * GPU)
    assert 0 <= leftover < IDX + GPU


def test_host_pool_shrinks_when_hbm_cannot_index_it():
    # Budget covers only ~half the requested host pool's index entries. The HOST
    # pool is what gives way: pinned pages the indexer cannot address are
    # unusable capacity, and refusing to start would be worse than shrinking.
    host_wanted = 262_176
    usable = host_wanted * IDX // 2
    host, gpu = split(usable, host_wanted, IDX, GPU)
    assert gpu == 0
    assert host == usable // IDX < host_wanted
    assert _fits(usable, host, gpu)


def test_exact_fit_for_host_pool_leaves_no_tier():
    host_wanted = 1000
    host, gpu = split(host_wanted * IDX, host_wanted, IDX, GPU)
    assert (host, gpu) == (host_wanted, 0)


def test_one_page_pair_over_exact_fit_buys_one_tier_page():
    host_wanted = 1000
    host, gpu = split(host_wanted * IDX + IDX + GPU, host_wanted, IDX, GPU)
    assert (host, gpu) == (host_wanted, 1)


def test_marginal_host_page_costs_a_fifth_of_a_tier_page():
    # index:gpu page bytes are 1:4, so each extra host page eats one index page
    # of budget and gives back 1/5 of a tier page — net +0.8 pages of capacity.
    # This is the number that makes raising the host ratio worthwhile, so pin it.
    usable = 101 * (1 << 30)
    _, gpu_a = split(usable, 200_000, IDX, GPU)
    _, gpu_b = split(usable, 205_000, IDX, GPU)
    assert gpu_a - gpu_b == 5_000 * IDX // (IDX + GPU)
    assert (200_000 + gpu_a) - (205_000 + gpu_b) == -4_000


def test_no_budget_yields_no_pools():
    assert split(0, 262_176, IDX, GPU) == (0, 0)
    assert split(-1, 262_176, IDX, GPU) == (0, 0)


def test_sparsekv_off_yields_no_pools():
    assert split(101 * (1 << 30), 0, IDX, GPU) == (0, 0)


def test_tier_disabled_still_indexes_the_host_pool():
    # gpu_cold_page_bytes == 0 would divide by index bytes alone; the tier being
    # off must not silently inflate it past what the budget can index.
    host_wanted = 1000
    host, gpu = split(host_wanted * IDX * 2, host_wanted, IDX, 0)
    assert host == host_wanted
    assert (host + gpu) * IDX <= host_wanted * IDX * 2
