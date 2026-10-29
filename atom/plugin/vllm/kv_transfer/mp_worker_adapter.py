# SPDX-License-Identifier: MIT
"""Give ATOM's multiprocess offload worker the two hooks vLLM's connector needs.

The in-process (``dense``) worker inherits ``OffloadWorkerMixin``, which owns a
thread pool and therefore knows, for any request, whether work for it is still
running and which GPU blocks a failed load left unwritten. The multiprocess
worker owns neither: its transfers run in the LMCache server process and it
tracks them as futures keyed by *operation*, not by request. It is a plain
``KVConnectorBase`` and so has neither ``wait_for_requests`` nor
``take_load_error_blocks``.

ATOM's own engine never calls those two. vLLM's connector calls both on every
step, and neither has a safe default:

``wait_for_requests`` is the preemption fence. vLLM frees a preempted request's
blocks inside ``schedule()`` and hands them straight to the next allocation, so
a save still reading them would store the next occupant's bytes under the
preempted request's token ids. A no-op fence turns that into a silent wrong
answer on a later hit.

``take_load_error_blocks`` names the GPU blocks a failed load never wrote.
vLLM truncates ``num_computed_tokens`` to the first block named and recomputes
from there; handed an empty set it takes the request's appearance in
``finished_recving`` at face value and serves never-written blocks as KV. So
returning ``set()`` is not a conservative default -- it is the corrupting one.

Both are answered here rather than by extending ATOM's MP worker, because both
are vLLM scheduler contracts rather than properties of the transport, and the
MP worker is shared with ATOM's own engine, which has its own answers.

The block ids come from the metadata the connector itself dispatched, not from
inside the transport: a load's destination blocks are exactly the ones the
scheduler named for it, which is the only place both this wrapper and the
transport agree on. The set is deliberately the request's whole span rather
than the failing chunk's -- over-naming costs recompute, under-naming is the
silent-corruption case above, and vLLM truncates to the first block regardless.
"""

from typing import Any

from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata
from atom.kv_transfer.offload.mp.transfer import _UnprovableSubmission

# The preemption fence waits as long as the worker's own transfer deadline
# (`_transfer_deadline_s`): a transfer still running before that is healthy, so
# preempting its request must not stop the engine sooner. Past it the fence
# raises -- fail-stop, the verdict the worker reaches for the transfer itself.
# This is the bound only for a worker that carries no deadline.
_FENCE_TIMEOUT_S = 30.0
_FENCE_POLL_S = 0.0005


def _req_id_of(completion: Any) -> str:
    """The request id inside a completion id, which may be bare or structured."""
    return str(getattr(completion, "req_id", completion))


class MPOffloadWorkerAdapter:
    """``LMCacheMPConnector`` plus the two hooks the vLLM connector calls.

    Everything else is forwarded unchanged, so the wrapped worker stays the
    single implementation of the transport.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # Destination blocks per in-flight load, by request id. Kept until the
        # load is reported terminal, because that report is what consumes them.
        self._load_blocks: dict[str, set[int]] = {}
        self._error_blocks: set[int] = set()

    # -- forwarding -----------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        self._inner.register_kv_caches(kv_caches, transfer_tensors, num_blocks)

    # -- the two hooks --------------------------------------------------

    def start_load_kv(self, metadata: Any) -> None:
        """Record each load's destination blocks, then dispatch the step.

        Only the blocks the load writes, from its HBM frontier to its end:
        those below are valid and may be shared with a running request, which
        vLLM's invalid-block handling would fail or recompute as well.
        """
        if isinstance(metadata, LMCacheOffloadMetadata):
            block_size = int(getattr(self._inner, "block_size", 0) or 0)
            for req in metadata.requests:
                spec = req.load_spec
                if spec is None:
                    continue
                blocks = list(req.block_ids)
                if block_size > 0:
                    end = (
                        spec.lmcache_cached_tokens
                        if spec.transfer_end_tokens is None
                        else spec.transfer_end_tokens
                    )
                    blocks = blocks[
                        int(spec.hbm_cached_tokens)
                        // block_size : -(-int(end) // block_size)
                    ]
                self._load_blocks[str(req.req_id)] = {int(b) for b in blocks}
        self._inner.start_load_kv(metadata)

    def get_finished(self) -> KVConnectorOutput:
        """Drain the transport, converting failed loads into error blocks."""
        out = self._inner.get_finished()
        for completion in out.failed_loading:
            blocks = self._load_blocks.pop(_req_id_of(completion), None)
            if blocks:
                self._error_blocks |= blocks
        for completion in out.finished_loading:
            self._load_blocks.pop(_req_id_of(completion), None)
        return out

    def take_load_error_blocks(self) -> set[int]:
        blocks = self._error_blocks
        self._error_blocks = set()
        return blocks

    def wait_for_requests(self, req_ids) -> None:
        """Block until no transfer for these requests is still reading blocks.

        Reads the transport's pending maps rather than draining them: the
        completions they carry belong to ``get_finished``, and consuming one
        here would lose the block lease it releases. Waiting on the futures in
        place leaves that report intact.
        """
        wanted = {str(req_id) for req_id in req_ids}
        if not wanted:
            return
        inner = self._inner
        lock = getattr(inner, "_lock", None)
        if lock is None:
            return

        import time

        bound_s = float(getattr(inner, "_transfer_deadline_s", _FENCE_TIMEOUT_S))
        deadline = time.monotonic() + bound_s
        while True:
            with lock:
                pending = [
                    (operation_id, entry.future)
                    for source in ("_pending_saves", "_pending_loads")
                    for operation_id, entry in getattr(inner, source, {}).items()
                    if _req_id_of(entry.completion) in wanted
                ]
                # Queued or mid-submit: no future yet, but the drain thread
                # will hand it to the server, which then reads these blocks.
                submitting = [
                    operation_id
                    for operation_id, owner in getattr(
                        inner, "_submitting_owner", {}
                    ).items()
                    if owner in wanted
                ]
            # Not polled here: the drain thread polls these futures and drops
            # terminal ones from the pending maps, and a second poller would
            # race it on LMCache's future, which imports the server's event.
            outstanding = submitting + [operation_id for operation_id, _ in pending]
            if not outstanding:
                return
            # Causes that waiting cannot clear, named now rather than as a
            # timeout: the server may still write these blocks either way.
            error = getattr(inner, "_drain_error", None)
            if error is not None:
                raise RuntimeError(
                    "ATOM LMCache MP: preemption fence found the transfer drain "
                    f"failed with {len(outstanding)} transfer(s) in flight for "
                    f"{sorted(wanted)}"
                ) from error
            unprovable = [
                operation_id
                for operation_id, future in pending
                if isinstance(future, _UnprovableSubmission)
            ]
            if unprovable:
                raise RuntimeError(
                    "ATOM LMCache MP: preemption fence cannot release "
                    f"{sorted(wanted)}: submit raised for {unprovable}, so the "
                    "server may have the request and still write its blocks"
                )
            drain = getattr(inner, "_drain_thread", None)
            if drain is None or not drain.is_alive():
                raise RuntimeError(
                    "ATOM LMCache MP: preemption fence found no drain thread to "
                    f"submit or poll {outstanding} for {sorted(wanted)}"
                )
            wake = getattr(inner, "_drain_wake", None)
            if wake is not None:
                wake.set()
            if time.monotonic() >= deadline:
                # Fail-stop: returning would let the forward overwrite blocks
                # a transfer still reads, caching one wrong prefix silently.
                raise RuntimeError(
                    f"ATOM LMCache MP: preemption fence timed out after "
                    f"{bound_s:.0f}s, the transfer deadline, with "
                    f"{len(outstanding)} transfer(s) still in flight for "
                    f"{sorted(wanted)}"
                )
            time.sleep(_FENCE_POLL_S)


__all__ = ["MPOffloadWorkerAdapter"]
