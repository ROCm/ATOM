# SPDX-License-Identifier: MIT
"""vLLM KV connector that drives ATOM's byte-level LMCache offload.

Why this exists rather than pointing vLLM at LMCache's own connector:

LMCache's GPU connectors accept only the clean NHD/HND family and pick ONE
format for the whole model. MiniMax-M3 registers three physical layouts at once
(dense K/V interleaved, sparse K/V in separate regions, plus a DSA index cache),
so no single format describes it -- and on ROCm the paths that could describe it
are unavailable anyway: the per-layer-format connector (V3) is off by default
and hangs on M3, and the multi-process path needs cupy, which LMCache's
``platform/rocm`` does not provide.

ATOM already solved this for its native engine by not asking LMCache to
understand the layout at all: ``DenseKVByteCodec`` gathers whole paged blocks
into a chunk-major uint8 blob, and LMCache only ever stores opaque bytes. That
codec is reused verbatim here; this module is the adapter that lets vLLM drive
it, so the plugin path gets the same guarantee the native path already tests
(byte-identical round-trip).

Layer-granular hooks are deliberately inert: ATOM moves a whole request's blocks
per transfer, not one layer at a time.
"""

import logging
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from atom.plugin.vllm.kv_transfer.kv_cache_layout import build_kv_cache_tensors
from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config
from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = logging.getLogger("atom")


class AtomOffloadMetadata(KVConnectorMetadata):
    """Carries ATOM's offload metadata through vLLM's connector plumbing.

    ATOM's ``LMCacheOffloadMetadata`` derives from ATOM's own ConnectorMetadata,
    not vLLM's, so it cannot be returned directly from ``build_connector_meta``.
    Wrapping keeps ATOM's descriptors intact instead of flattening them into a
    vLLM-shaped copy that would then have to be kept in sync.
    """

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner


class AtomLMCacheOffloadConnector(KVConnectorBase_V1):
    """Drives ``atom.kv_transfer.offload`` from vLLM's connector API."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        # kv_cache_config is required of out-of-tree v1 connectors: the factory
        # rejects the 2-argument signature outright, and the base class stores
        # it for the group-aware paths.
        super().__init__(vllm_config, role, kv_cache_config)
        self._config = build_offload_config(vllm_config)
        self._worker = None
        self._scheduler = None

        self._seqs = SeqViewRegistry()
        # finished_sending is only legal for a request vLLM has already
        # finished AND whose save has landed; the two events arrive in either
        # order, so both sides are accumulated until they meet.
        self._saved_awaiting_finish: set[str] = set()
        self._finished_awaiting_save: set[str] = set()

        if role == KVConnectorRole.WORKER:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

            self._worker = DenseOffloadConnector(self._config)
        else:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

            self._scheduler = DenseOffloadScheduler(self._config)

    # ---- worker side --------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Translate vLLM's flat registration and hand it to ATOM's codec."""
        tensors = build_kv_cache_tensors(kv_caches)
        if not tensors:
            raise ValueError("ATOM offload connector: vLLM registered no KV caches")

        # Every segment's per-block stride is derived from num_blocks, so it has
        # to be the physical block count -- not a token count. Block-major KV
        # carries it in dim 0; taking it from the first mapped k_cache keeps the
        # value consistent with the very tensors the codec will slice.
        num_blocks = int(tensors[0].k_cache.shape[0])

        self._worker.register_kv_caches(
            {str(t.layer_num): t for t in tensors},
            num_blocks=num_blocks,
        )
        logger.info(
            "ATOM LMCache offload: registered %d layers, num_blocks=%d",
            len(tensors),
            num_blocks,
        )
        # Layout/dtype census. The codec moves opaque bytes, so a wrong dtype
        # never surfaces here -- it surfaces much later inside an attention
        # kernel ("Both operands must be same dtype"), with nothing pointing
        # back at registration. One line here makes that diagnosable.
        census: dict[tuple, int] = {}
        for name, tensor in sorted(kv_caches.items()):
            key = (
                "index" if name.endswith(".index_cache") else "kv",
                tuple(tensor.shape[1:]),
                str(tensor.dtype),
            )
            census[key] = census.get(key, 0) + 1
        for (kind, shape, dtype), count in sorted(census.items(), key=str):
            logger.info(
                "ATOM LMCache offload:   %d x %s tail_shape=%s dtype=%s",
                count,
                kind,
                shape,
                dtype,
            )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        inner = getattr(metadata, "inner", None)
        if inner is not None:
            self._worker.start_load_kv(inner)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Inert: transfers are per-request, not per-layer.

        A load is published to the scheduler through ``get_finished`` only once
        every block of the request has landed, so there is no partially-loaded
        layer for a forward to wait on.
        """

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        """Inert: saves are issued per request from ``build_connector_meta``."""

    def wait_for_save(self) -> None:
        """Inert: saves are fire-and-forget on ATOM's save executor.

        Blocking the forward on them would put offload on the critical path,
        which is the opposite of what the tier is for. Completion still reaches
        the scheduler via ``get_finished``.
        """

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Translate ATOM's four completion sets into vLLM's two.

        ``finished_saving`` surfaces as vLLM's ``finished_sending``, but only
        once the request has ALSO finished. vLLM's scheduler asserts both
        ``req_id in self.requests`` and ``request.is_finished()`` before
        freeing, while ATOM's saves are fire-and-forget and routinely land
        while the request is still decoding -- reporting those crashed the
        engine on ``assert request.is_finished()``. The two events arrive in
        either order, so each side is held until its counterpart shows up.

        ATOM's own worker deliberately reports an empty ``finished_sending``
        because ITS scheduler reads that as a P/D producer handoff. Same name,
        two contracts; the translation lives here rather than in either side.

        A failed load is reported as finished too: the request is parked
        waiting on it, and the alternative to waking it is a hang. vLLM then
        recomputes the tokens it had counted as externally supplied.
        """
        out = self._worker.get_finished()

        finished_recving = {_req_id_of(c) for c in out.finished_loading}
        failed = {_req_id_of(c) for c in out.failed_loading}
        if failed:
            logger.warning(
                "ATOM LMCache offload: load failed for %s; recomputing", sorted(failed)
            )
            finished_recving |= failed

        self._saved_awaiting_finish |= {_req_id_of(c) for c in out.finished_saving}
        self._finished_awaiting_save |= set(finished_req_ids or ())
        finished_sending = self._saved_awaiting_finish & self._finished_awaiting_save
        self._saved_awaiting_finish -= finished_sending
        self._finished_awaiting_save -= finished_sending

        return finished_sending, finished_recving

    def shutdown(self) -> None:
        for side in (self._worker, self._scheduler):
            close = getattr(side, "close", None) or getattr(side, "shutdown", None)
            if close is not None:
                close()

    # ---- scheduler side -----------------------------------------------

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """How many extra prompt tokens the offload tier can supply.

        ``num_computed_tokens`` is vLLM's HBM-prefix-cache frontier; ATOM reads
        the same quantity off the seq to avoid re-loading what is already
        resident, so it has to be pushed in before the lookup runs.
        """
        seq = self._seqs.get_or_create(request)
        seq.set_num_cached_tokens(num_computed_tokens)
        return self._scheduler.get_num_new_matched_tokens(seq)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        seq = self._seqs.get_or_create(request)
        seq.set_block_table(_block_ids(blocks))
        self._scheduler.update_state_after_alloc(seq)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        """Snapshot this step's transfers.

        The frontier of every scheduled request is refreshed first: ATOM decides
        which chunks are safe to save by comparing against it, and a stale value
        would either skip chunks or offer up tokens that are not computed yet.
        """
        for req_id, num_tokens in _scheduled_frontiers(scheduler_output):
            seq = self._seqs.get(req_id)
            if seq is not None:
                seq.set_num_cached_tokens(num_tokens)
        return AtomOffloadMetadata(self._scheduler.build_connector_meta())

    def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
        seq = self._seqs.get(request.request_id)
        if seq is not None:
            self._scheduler.request_finished(seq)
            # Blocks may still be pinned by an in-flight save; ATOM says when.
            if self._scheduler.should_defer_free(seq):
                return True, None
            self._seqs.drop(request.request_id)
        return False, None


def _req_id_of(completion_id) -> str:
    """Completion ids are a bare request id, or one tagged with a generation."""
    return str(getattr(completion_id, "req_id", completion_id))


def _block_ids(blocks) -> list[int]:
    """Flatten vLLM's allocated-block structure to plain ids.

    vLLM has spelled this several ways across versions (KVCacheBlocks with
    ``get_block_ids()``, a per-group tuple of lists, or already-flat ids), and
    the offload codec only ever needs the ids.
    """
    getter = getattr(blocks, "get_block_ids", None)
    if getter is not None:
        blocks = getter()
    if (
        isinstance(blocks, (list, tuple))
        and blocks
        and isinstance(blocks[0], (list, tuple))
    ):
        return [int(b) for group in blocks for b in group]
    return [int(b) for b in (blocks or [])]


def _scheduled_frontiers(scheduler_output):
    """Yield ``(request_id, num_computed_tokens)`` for this step's requests."""
    for req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        yield req.req_id, getattr(req, "num_computed_tokens", 0)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = getattr(cached, "req_ids", None) or []
    computed = getattr(cached, "num_computed_tokens", None) or []
    yield from zip(req_ids, computed)
