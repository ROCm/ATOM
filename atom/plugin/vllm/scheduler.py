"""Scheduler subclasses that make vLLM's KV-load-failure recovery hybrid-aware.

vLLM's ``Scheduler._update_requests_with_invalid_blocks`` assumes a single KV
cache group and unpacks ``get_block_ids(req_id)`` as a 1-tuple, under its own
``# TODO (davidb): add support for hybrid memory allocator``.  A hybrid model
(Kimi-K3: MLA attention layers plus KDA recurrent layers) has more than one
group, so the first KV load that actually fails raises ValueError inside
EngineCore: EngineDeadError, every in-flight request turned into a 500, server
down.  That path is reached only with ``kv_load_failure_policy=recompute``,
which is why synthetic workloads never saw it.

WHY A SUBCLASS AND NOT A MONKEYPATCH.  vLLM offers ``scheduler_cls`` as a
supported way to supply the scheduler (``SchedulerConfig.get_scheduler_cls``),
so we hand it a subclass instead of rewriting a method on vLLM's own class.
The blast radius is then exactly the engines that selected it, and a version
skew fails loudly at startup rather than silently shadowing upstream code.

Be honest about what this does NOT buy: ``_update_requests_with_invalid_blocks``
is private either way, and vLLM warns that the scheduler interface is not
public.  The coupling to a vLLM version is the same as a monkeypatch's; only
the blast radius is smaller.

WHY TWO SUBCLASSES.  ``get_scheduler_cls`` consults ``async_scheduling`` ONLY
when ``scheduler_cls`` is None; once set, our class is returned as-is.  So
setting a ``Scheduler`` subclass would silently disable async scheduling --
vLLM says as much in its own warning.  ``select_scheduler_cls`` therefore picks
the subclass matching the already-resolved ``async_scheduling`` value.
"""

import inspect
import logging

logger = logging.getLogger("atom")

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

# The exact expression this override exists to replace.  If it is gone, vLLM
# has grown its own hybrid support and we must stand down rather than shadow
# it with a copy that is now a version behind.
_SINGLE_GROUP_UNPACK = "(req_block_ids,) = self.kv_cache_manager.get_block_ids("


class _HybridKVLoadFailureMixin:
    """Group-aware ``_update_requests_with_invalid_blocks``."""

    def _update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """Hybrid-aware replacement for the KV-load-failure recovery scan.

        WHY THIS EXISTS.  vLLM's own version assumes a single KV cache group and
        unpacks ``get_block_ids(req_id)`` as a 1-tuple, under a TODO admitting the
        gap.  A hybrid model (Kimi-K3: MLA attention layers plus KDA recurrent
        layers) gets more than one group, so the unpack raises ValueError.  It
        raises inside EngineCore, so the blast radius is not the one request that
        missed -- it is EngineDeadError, every in-flight request turned into a 500,
        and the server shutting down.  The path is reached only with
        ``kv_load_failure_policy=recompute`` and a KV load that actually fails,
        which is why synthetic workloads never saw it and the customer's
        cache-validation profile hit it within minutes.

        WHAT CHANGES.  Scan every group instead of assuming one, mirroring the
        idiom vLLM already uses in
        ``KVCacheManager.get_block_ids_for_computed_tokens``: skip non-attention
        groups and use each group's OWN ``spec.block_size``.

        WHY SKIPPING NON-ATTENTION GROUPS IS SAFE HERE.  Two independent reasons,
        and the second is a property of ATOM's connector rather than of this code:

          1. A mamba group's "block" is a recurrent state page.  Its index does not
             convert to a token offset by multiplying by a block size, so it cannot
             name a truncation point.  Using it would corrupt num_computed_tokens
             silently -- no crash, no log, just wrong output.

          2. ``AtomLMCacheOffloadConnector`` never reports a recurrent-state block
             as invalid.  A KDA boundary that fails to arrive is reported as the
             ATTENTION blocks whose prefix it invalidates (see
             ``KdaLoad.error_block_ids`` in kv_transfer/kda_state.py and the merge
             in ``connector.py``), precisely because an MLA prefix is valid only if
             the recurrent state at the same boundary was restored.  So the ids
             this function is handed are attention-group ids, which are exactly the
             ones scanned below.

        If a future connector ever reports non-attention block ids here, reason (2)
        lapses and those failures would go undetected -- that is the assumption to
        re-check before changing what the connector puts in ``invalid_block_ids``.

        TRUNCATION ACROSS GROUPS.  When more than one group names an invalid block,
        the truncation point is the EARLIEST across groups: a prefix is valid only
        where it is valid in every group.
        """
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            CrossAttentionSpec,
            EncoderOnlyAttentionSpec,
        )

        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()

        kv_cache_groups = self.kv_cache_manager.kv_cache_config.kv_cache_groups

        for request in requests:
            req_id = request.request_id
            all_block_ids = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens.
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            is_affected = False
            truncate_at: int | None = None

            for kv_group, req_block_ids in zip(kv_cache_groups, all_block_ids):
                spec = kv_group.kv_cache_spec
                if not isinstance(spec, AttentionSpec) or isinstance(
                    spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
                ):
                    continue

                group_block_size = spec.block_size
                req_num_computed_blocks = (
                    req_num_computed_tokens + group_block_size - 1
                ) // group_block_size

                group_marked = False
                for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                    if block_id not in invalid_block_ids:
                        continue

                    is_affected = True

                    if block_id in marked_invalid_block_ids:
                        # Shared with a previous request, which already marked it
                        # for recomputation; this request may still treat it as
                        # computed when rescheduled.
                        continue

                    marked_invalid_block_ids.add(block_id)

                    if group_marked:
                        continue
                    group_marked = True

                    candidate = idx * group_block_size
                    if truncate_at is None or candidate < truncate_at:
                        truncate_at = candidate

                    # collect invalid block and all downstream dependent blocks
                    if evict_blocks:
                        blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if truncate_at is None:
                    # Every invalid block of this request is shared with a previous
                    # request and will be recomputed by it.  Revert to considering
                    # only cached tokens as computed.
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens
                else:
                    request.num_computed_tokens = truncate_at
                    total_affected_tokens += req_num_computed_tokens - truncate_at

                affected_req_ids.add(req_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict


class VllmAtomScheduler(_HybridKVLoadFailureMixin, Scheduler):
    """Synchronous scheduler with hybrid-aware KV-load-failure recovery."""


class VllmAtomAsyncScheduler(_HybridKVLoadFailureMixin, AsyncScheduler):
    """Async scheduler with hybrid-aware KV-load-failure recovery."""


def vllm_needs_hybrid_kv_load_fix() -> bool:
    """True when vLLM's own recovery path still assumes a single group.

    Ask the live method, not a file: the tree we import from is not necessarily
    the tree we think we are reading.  If the source cannot be read at all,
    answer True -- an unnecessary subclass is a no-op, while skipping a needed
    one is a dead engine.
    """
    original = getattr(Scheduler, "_update_requests_with_invalid_blocks", None)
    if original is None:
        return False
    try:
        return _SINGLE_GROUP_UNPACK in inspect.getsource(original)
    except (OSError, TypeError):
        return True


def select_scheduler_cls(scheduler_config) -> str | None:
    """Return the qualified name of the scheduler to use, or None to leave vLLM's.

    Returns None whenever the caller already chose a scheduler, or vLLM no
    longer needs the fix.  The choice between the two subclasses mirrors
    ``get_scheduler_cls``: by the time a platform's ``check_and_update_config``
    runs, ``async_scheduling`` has already been resolved from None to a bool
    (``VllmConfig.__post_init__`` settles it well before the platform hook), so
    reading it here is reading the final value.
    """
    if getattr(scheduler_config, "scheduler_cls", None) is not None:
        return None
    if not vllm_needs_hybrid_kv_load_fix():
        logger.info(
            "ATOM: vLLM's KV-load-failure recovery already handles hybrid KV "
            "cache groups; leaving the scheduler class alone."
        )
        return None
    name = (
        "VllmAtomAsyncScheduler"
        if getattr(scheduler_config, "async_scheduling", False)
        else "VllmAtomScheduler"
    )
    # Keep this message free of words that boot-time log scanners treat as
    # crash signatures ("ValueError", "Traceback", ...).  A line announcing a
    # successful fix is not a failure, and a scanner cannot tell the
    # difference -- our own arm scripts killed a server on that once.
    logger.info(
        "ATOM: selecting %s so vLLM's KV-load-failure recovery handles this "
        "model's multiple KV cache groups; upstream assumes a single group and "
        "would abort the engine on the first failed load.",
        name,
    )
    return f"atom.plugin.vllm.scheduler.{name}"
