# SPDX-License-Identifier: MIT
"""Let vLLM recompute KV-load failures on a model with more than one KV group.

When a connector reports blocks it could not load, vLLM truncates each affected
request to its longest valid prefix and reschedules the rest -- the behaviour
``kv_load_failure_policy=recompute`` asks for. That path reads a request's
blocks like this (``v1/core/sched/scheduler.py``)::

    # TODO (davidb): add support for hybrid memory allocator
    (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)

``get_block_ids`` returns one list per KV cache group, so the unpack holds only
for models with exactly one. Kimi-K3 is hybrid -- MLA attention beside the KDA
recurrent state, four groups -- and the unpack raises ``ValueError: too many
values to unpack (expected 1)``. It raises inside ``update_from_output``, which
is not a per-request failure: the engine core dies and every in-flight request
gets a connection error. It also fires only when a load actually fails, so a
run can offload correctly for hours and then lose the engine to the first
preemption that makes LMCache return ``result=False``.

The generalisation is small because all four of K3's groups share one block id
space (``HybridKVCacheCoordinator`` allocates every group from a single
``BlockPool``) and one token-per-block figure. This wrapper scans every
attention group, truncates at the earliest invalid block across them, and
evicts each group's tail from the block that contains that token. On a
single-group model it defers to vLLM's own implementation rather than
reimplementing it, so the common path keeps whatever upstream does with it.

Recurrent groups are the part that is not a generalisation. A KDA group's
"blocks" are state slots, not prefix blocks: slot *i* holds the recurrent state
after every token the request has computed so far, and there is no state for an
arbitrary earlier prefix to rewind to. Truncating such a request to token *N*
would resume it with the state of a longer prefix -- silently wrong at best,
and on K3 it takes the engine down with an asynchronous GPU fault that first
surfaces at the next sync inside the KDA chunk kernel. vLLM's own rewind of a
running request, preemption, therefore always goes to zero
(``_preempt_request``), and its prefix-block clipping skips non-attention
groups outright (``get_block_ids_for_computed_tokens``). This patch follows
both: with any recurrent group present an affected request rewinds to token 0,
and recurrent groups take no part in the truncation arithmetic.

Rewinding to zero is not something a running request can be told by hand,
though. Under async scheduling a running request carries ``num_output_placeholders``
for the tokens its in-flight steps will sample; move its ``num_computed_tokens``
backwards without touching them and the next delivered token decrements a count
that was never incremented for it, tripping ``assert
request.num_output_placeholders >= 0`` in ``async_scheduler.py`` and taking the
engine core down just as surely as the GPU fault did. All of that bookkeeping --
freeing blocks, clearing spec tokens, zeroing the placeholders and marking the
in-flight output stale -- is exactly what ``Scheduler._preempt_request`` does, so
a running request rewound here is put through it, having first been removed from
``scheduler.running`` as that method requires. Requests rewound on the async
branch (``WAITING_FOR_REMOTE_KVS``, ``evict_blocks=False``) are not running, have
no placeholders, and keep the plain rewind.
"""

import functools
import logging
import time

logger = logging.getLogger("atom")


def _group_layout(scheduler) -> tuple[list[int], list[bool]]:
    """Tokens per block, and recurrent-ness, per KV cache group.

    Both lists are in ``get_block_ids`` order. A group is recurrent when its
    spec is not an ``AttentionSpec`` -- the same test vLLM's own
    ``get_block_ids_for_computed_tokens`` uses to decide that a group's blocks
    cannot be clipped by a token count.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    config = scheduler.kv_cache_manager.kv_cache_config
    sizes = []
    recurrent = []
    for group in config.kv_cache_groups:
        is_recurrent = not isinstance(group.kv_cache_spec, AttentionSpec)
        recurrent.append(is_recurrent)
        block_size = getattr(group.kv_cache_spec, "block_size", None)
        if is_recurrent:
            # Never used: a recurrent group takes no part in the truncation
            # arithmetic, so its page size need not even be expressible in
            # tokens of the prefix.
            sizes.append(0)
            continue
        if not block_size:
            # Every attention spec carries one; a group without it cannot be
            # addressed in tokens, and guessing the scheduler's own block size
            # here would truncate requests at the wrong token.
            raise ValueError(
                "ATOM hybrid invalid-blocks patch: KV cache group "
                f"{group.kv_cache_spec!r} reports no block_size, so the token "
                "offset of an invalid block in it cannot be computed"
            )
        sizes.append(int(block_size))
    return sizes, recurrent


def update_requests_with_invalid_blocks(
    scheduler,
    requests,
    invalid_block_ids,
    num_scheduled_tokens,
    block_sizes,
    evict_blocks=True,
    recurrent_groups=None,
    on_full_rewind=None,
):
    """The multi-group body of the patched method.

    Module-level, and taking the group layout rather than reading it off the
    scheduler, so the truncation arithmetic can be tested against a stub --
    importing vLLM's Scheduler is not possible in the unit test environment,
    and this is the part with a wrong answer to get wrong.

    ``recurrent_groups`` marks, per group, the ones whose blocks are recurrent
    state slots rather than prefix blocks. They are skipped by the scan, and
    their presence makes an affected request rewind to token 0 instead of to a
    prefix boundary: see the module docstring for why a partial rewind is not
    available on such a model. ``None`` means every group is an attention
    group.

    ``on_full_rewind`` is called with each request that was rewound to token 0,
    so the caller can put a running one through the scheduler's own preemption;
    see the module docstring. The rewind itself is done here either way.
    """
    if recurrent_groups is None:
        recurrent_groups = [False] * len(block_sizes)
    has_recurrent = any(recurrent_groups)
    affected_req_ids: set[str] = set()
    total_affected_tokens = 0
    blocks_to_evict: set[int] = set()
    # An invalid block shared by several requests in the batch is
    # recomputed by the first of them only; the others may still treat it
    # as computed. Same rule as upstream, one set for all groups because
    # the groups share a block id space.
    marked_invalid_block_ids: set[int] = set()

    for request in requests:
        req_id = request.request_id
        group_block_ids = scheduler.kv_cache_manager.get_block_ids(req_id)
        req_num_computed_tokens = (
            request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
        )

        is_affected = False
        newly_marked: list[int] = []
        # Token offset of the earliest not-yet-marked invalid block, over
        # every group. Upstream truncates at the first such block in its
        # single group; with several groups the prefix is only valid up to
        # the earliest of them, because a request reads all groups.
        truncate_at = None

        for blocks, block_size, is_recurrent in zip(
            group_block_ids, block_sizes, recurrent_groups
        ):
            if is_recurrent:
                # A state slot is not a prefix block: it is never one of the
                # connector's load destinations, and its index carries no
                # token offset to truncate at.
                continue
            num_computed_blocks = (
                req_num_computed_tokens + block_size - 1
            ) // block_size
            for idx, block_id in zip(range(num_computed_blocks), blocks):
                if block_id not in invalid_block_ids:
                    continue
                is_affected = True
                if block_id in marked_invalid_block_ids:
                    continue
                newly_marked.append(block_id)
                # With a recurrent group in the model the only prefix whose
                # state exists is the empty one.
                offset = 0 if has_recurrent else idx * block_size
                if truncate_at is None or offset < truncate_at:
                    truncate_at = offset

        if not is_affected:
            continue

        if truncate_at is None:
            # Every invalid block of this request is already being
            # recomputed by an earlier request in the batch. Fall back to
            # counting only its cached tokens as computed, as upstream does.
            # Sharing does not rescue the recurrent half: whoever recomputes
            # the block, this request's own state slot still holds the state
            # of the prefix it can no longer claim.
            if has_recurrent:
                total_affected_tokens += request.num_computed_tokens
                request.num_computed_tokens = 0
                if on_full_rewind is not None:
                    on_full_rewind(request)
            else:
                total_affected_tokens += (
                    request.num_computed_tokens - req_num_computed_tokens
                )
                request.num_computed_tokens = req_num_computed_tokens
        else:
            marked_invalid_block_ids.update(newly_marked)
            total_affected_tokens += req_num_computed_tokens - truncate_at
            request.num_computed_tokens = truncate_at
            if has_recurrent and on_full_rewind is not None:
                # ``truncate_at`` is 0 on such a model -- see above.
                on_full_rewind(request)
            if evict_blocks:
                # The invalid block and everything after it, in every
                # group: a block that holds tokens past the truncation
                # point is no longer a valid cache entry for this prefix,
                # whichever group it belongs to.
                for blocks, block_size, is_recurrent in zip(
                    group_block_ids, block_sizes, recurrent_groups
                ):
                    if is_recurrent:
                        # Not a cache entry keyed by a prefix; eviction is not
                        # the mechanism that reclaims it.
                        continue
                    blocks_to_evict.update(blocks[truncate_at // block_size :])

        affected_req_ids.add(req_id)

    return affected_req_ids, total_affected_tokens, blocks_to_evict


def preempt_rewound_requests(scheduler, rewound) -> int:
    """Put every rewound *running* request through vLLM's own preemption.

    Returns the number preempted. Requests that are not running -- the async
    branch's ``WAITING_FOR_REMOTE_KVS`` ones -- are left alone: they hold no
    output placeholders and ``_preempt_request`` asserts on their status.
    """
    from vllm.v1.request import RequestStatus

    victims = [r for r in rewound if r.status == RequestStatus.RUNNING]
    if not victims:
        return 0
    victim_ids = {r.request_id for r in victims}
    # _preempt_request requires the request to have been popped from the
    # running queue already, and re-adds it to the waiting queue itself.
    scheduler.running = [r for r in scheduler.running if r.request_id not in victim_ids]
    timestamp = time.monotonic()
    for request in victims:
        scheduler._preempt_request(request, timestamp, drop_stale_output=True)
    return len(victims)


def apply_vllm_hybrid_invalid_blocks_patch() -> None:
    """Make ``_update_requests_with_invalid_blocks`` multi-group aware."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler._update_requests_with_invalid_blocks
    if getattr(original, "_atom_hybrid_invalid_blocks", False):
        return

    @functools.wraps(original)
    def wrapped(
        self,
        requests,
        invalid_block_ids,
        num_scheduled_tokens,
        evict_blocks=True,
    ):
        block_sizes, recurrent_groups = _group_layout(self)
        if len(block_sizes) <= 1:
            return original(
                self,
                requests,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=evict_blocks,
            )

        rewound: list = []
        result = update_requests_with_invalid_blocks(
            self,
            requests,
            invalid_block_ids,
            num_scheduled_tokens,
            block_sizes,
            evict_blocks=evict_blocks,
            recurrent_groups=recurrent_groups,
            on_full_rewind=rewound.append,
        )
        if rewound:
            # After the scan, not during it: ``requests`` may be
            # ``scheduler.running`` itself, which preemption rewrites.
            num_preempted = preempt_rewound_requests(self, rewound)
            if num_preempted:
                logger.warning(
                    "ATOM plugin: preempted %d request(s) rewound to token 0 "
                    "by a KV load failure (recurrent state has no partial "
                    "prefix to resume from)",
                    num_preempted,
                )
        return result

    wrapped._atom_hybrid_invalid_blocks = True
    Scheduler._update_requests_with_invalid_blocks = wrapped
    logger.info(
        "ATOM plugin: KV-load-failure recompute made multi-group aware "
        "(vLLM's own path unpacks a single KV cache group)"
    )


__all__ = [
    "apply_vllm_hybrid_invalid_blocks_patch",
    "preempt_rewound_requests",
    "update_requests_with_invalid_blocks",
]
