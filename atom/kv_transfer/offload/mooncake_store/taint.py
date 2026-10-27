# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The mark of a Store a put may still write after Mooncake gave up on it.

Mooncake revokes a put whose transfer it gave up on before the call returns,
and the master frees the owner memory it had given the put at once, for the
next put that needs as much (``FinalizeBatchPut``, ``PutRevoke``). The
abandoned RDMA write can still land there later, with that put's bytes, inside
another key's object of the same size, and nothing in a load would tell. The
owners' memory is out of every client's reach, so no client of that Store may
load from it again: not this worker, not the other workers and engines that
share the Store, and not one restarted against it.

The worker whose put it was creates :data:`STORE_TAINT_KEY` in its Store. The
object is hard-pinned, so eviction never takes it, carries a replica on every
segment it can (a segment that unmounts takes its own objects with it, and the
marker lives on in the others), and is created by the master alone, with no
transfer, so it gets there while the fabric is stalled. Every worker looks for
it at startup, before every get and after a load's last get
(``MooncakeStoreOffloadConnector``), and
every scheduler lookup asks for it with the chunks (``_StoreLookup``); a
process that has seen it never trusts that Store again, and one that cannot
tell does not load. The mark goes only with the Store: a restart of its master
and owners. A remove of the key, or of every object, takes it too, but not the
late write; a tainted Store is restarted, not emptied.
"""

from __future__ import annotations

import logging
import threading

from atom.kv_transfer.offload.mooncake_store import client as store_client
from atom.kv_transfer.offload.mooncake_store.keys import STORE_TAINT_KEY

logger = logging.getLogger("atom")

# The marker's size; nothing ever reads its bytes.
MARKER_BYTES = 4096
# Replicas asked for, one per segment: the master's random allocator tries at
# most 100 segments, and places fewer where there are fewer.
MARKER_REPLICAS = 100
# Seconds before the first retry of a marker the Store did not take, doubled
# after each failed retry up to `_RETRY_MAX_S`.
_RETRY_BASE_S = 1.0
_RETRY_MAX_S = 30.0
# Seconds `TaintMarker.close` waits for a retry in a Store call to return.
_CLOSE_WAIT_S = 5.0


class TaintMarker:
    """Creates :data:`STORE_TAINT_KEY` in one Store until the Store holds it."""

    def __init__(self, client: store_client.MooncakeStoreClient) -> None:
        self._client = client
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.placed = False

    def place(self, *, retry: bool = True) -> bool:
        """Create the marker now; if the Store does not take it, keep trying.

        Retries run on a daemon thread, with backoff, until the Store holds the
        marker or :meth:`close`; ``retry=False`` tries once. Returns whether
        the Store holds it now.
        """
        with self._lock:
            if self.placed:
                return True
            if self._thread is not None:
                return False
            if self._try():
                return True
            if retry:
                self._thread = threading.Thread(
                    target=self._retry, name="mooncake-store-taint", daemon=True
                )
                self._thread.start()
            return False

    def close(self) -> bool:
        """Stop retrying. Returns whether no retry is still inside a Store call."""
        self._stop.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(_CLOSE_WAIT_S)
        return not thread.is_alive()

    def _retry(self) -> None:
        delay = _RETRY_BASE_S
        while not self._stop.wait(delay):
            if self._try():
                return
            delay = min(2 * delay, _RETRY_MAX_S)

    def _try(self) -> bool:
        """One attempt; True once the Store holds the marker."""
        master = self._client.master_server_addr
        try:
            if self._client.exists([STORE_TAINT_KEY]) != [1]:
                code = self._client.put_marker(
                    STORE_TAINT_KEY, MARKER_BYTES, replicas=MARKER_REPLICAS
                )
                if self._client.exists([STORE_TAINT_KEY]) != [1]:
                    logger.error(
                        "Mooncake Store offload: the Store at %s did not take the "
                        "taint marker %s (%s); other clients of it keep loading "
                        "from it until it does, and this worker keeps trying",
                        master,
                        STORE_TAINT_KEY,
                        store_client.describe(code),
                    )
                    return False
        except Exception:
            logger.exception(
                "Mooncake Store offload: placing the taint marker %s in the Store "
                "at %s failed; other clients of it keep loading from it until it "
                "lands, and this worker keeps trying",
                STORE_TAINT_KEY,
                master,
            )
            return False
        self.placed = True
        logger.warning(
            "Mooncake Store offload: the Store at %s holds the taint marker %s: "
            "no client of it loads from it again until it restarts",
            master,
            STORE_TAINT_KEY,
        )
        return True


__all__ = ["MARKER_BYTES", "MARKER_REPLICAS", "STORE_TAINT_KEY", "TaintMarker"]
