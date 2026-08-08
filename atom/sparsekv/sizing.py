# SPDX-License-Identifier: MIT
"""Sizing math for SparseKV's two-tier cold storage.

Kept apart from the coordinator and the model runner because it is pure
arithmetic over byte counts: the runner owns the HBM budget, the coordinator
owns the tensors, and this decides how the budget splits between them.
"""


def split_cold_storage_budget(
    usable_bytes: int,
    host_pages_wanted: int,
    index_page_bytes: int,
    gpu_cold_page_bytes: int,
) -> tuple[int, int]:
    """Split the cold-storage HBM budget so that
    ``index_pages == host_pages + gpu_cold_pages``.

    The indexer needs one index_cache page per page of context regardless of
    where that context's KV lives, and the two cold tiers are disjoint homes for
    the same token (``promote_to_gpu`` frees the host page it copies from), so
    their page counts add. Sizing index_cache against the host pool alone — while
    admission discounts promoted pages and therefore admits up to host+GPU — is
    what let the index pool overflow. Overflowing it preempts a request, and a
    preempted request on a SparseKV decode node cannot be re-prefilled locally:
    the paged MLA pool it would write into is a compact one-step scratch.

    Solving ``(host + gpu) * index_page + gpu * gpu_page <= usable`` for ``gpu``
    gives the closed form below — one budget, one solve, no predict-then-measure
    round trip for the two pools to drift apart in.

    When the budget cannot even index the requested host pool, the HOST pool is
    what shrinks: pinned pages the indexer can never address are unusable
    capacity, not a reason to refuse to start.

    Returns ``(host_pages, gpu_cold_pages)``.
    """
    if usable_bytes <= 0 or index_page_bytes <= 0 or host_pages_wanted <= 0:
        return 0, 0
    index_only_pages = usable_bytes // index_page_bytes
    if index_only_pages <= host_pages_wanted:
        return int(max(1, index_only_pages)), 0
    gpu_pages = (usable_bytes - host_pages_wanted * index_page_bytes) // (
        index_page_bytes + gpu_cold_page_bytes
    )
    return int(host_pages_wanted), int(max(0, gpu_pages))
