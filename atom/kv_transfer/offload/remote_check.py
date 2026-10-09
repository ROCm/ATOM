# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Startup check of the in-process LMCache offload's remote tier.

LMCache logs, and then carries on through, every way its remote tier
(``LMCACHE_REMOTE_URL``) can fail to come up:

* a connector that raised while connecting leaves ``RemoteBackend.connection``
  None, and nothing retries it -- ATOM builds the engine without LMCache's
  health monitor, the only caller of the reconnect path;
* a failed registration of the CPU pool with the transfer engine is a
  warning, after which no put can move a byte;
* a put whose transfer failed completes as a success, and is not counted in
  ``put_failed_count``.

Each leaves a server that runs and answers but whose L2 never holds anything.
:func:`verify_remote_backend` runs once per worker after the engine is built
and fails startup instead, unless one chunk written through the remote tier
reads back byte for byte. One chunk takes one path: in a Store pooled from
several owners it lands on one of them, so whether every owner is up is for
whoever starts them to check (the atomesh launcher waits for their capacity).
"""

from __future__ import annotations

import logging
import secrets
import threading
from typing import Any

logger = logging.getLogger("atom")

# Generous on purpose: a healthy round trip takes milliseconds, and a put that
# is still running after this long will not serve real traffic either.
DEFAULT_PUT_TIMEOUT_S = 120.0
_PROBE_NAMESPACE_SUFFIX = "::atom-remote-check"


def verify_remote_backend(
    engine: Any, metadata: Any, *, put_timeout_s: float = DEFAULT_PUT_TIMEOUT_S
) -> None:
    """Fail unless the engine's remote tier stores one chunk and returns it.

    Args:
        engine: A post-initialized ``LMCacheEngine`` whose config sets a
            remote URL.
        metadata: The engine's metadata; the probe chunk has its full-chunk
            shape and a key in a namespace no real chunk uses.
        put_timeout_s: How long to wait for the probe put to finish.

    Raises:
        RuntimeError: The remote tier is missing, unconnected, unregistered,
            or loses or alters the probe chunk.
    """
    storage_manager = getattr(engine, "storage_manager", None)
    backends = getattr(storage_manager, "storage_backends", None) or {}
    remote = backends.get("RemoteBackend")
    if remote is None:
        raise RuntimeError(
            "LMCACHE_REMOTE_URL is set but LMCache built no RemoteBackend "
            f"(backends: {', '.join(backends) or 'none'})"
        )
    connection = getattr(remote, "connection", None)
    if connection is None:
        raise RuntimeError(
            "the LMCache remote tier did not connect (see 'Failed to "
            "initialize/re-establish remote connection' above) and is never "
            "retried; is the remote store up?"
        )
    connector = connection
    unwrap = getattr(connection, "getWrappedConnector", None)
    if callable(unwrap):
        connector = unwrap()
    if (
        hasattr(connector, "registered_buffer_ptr")
        and connector.registered_buffer_ptr is None
    ):
        raise RuntimeError(
            "the LMCache remote tier could not register the CPU pool with its "
            "transfer engine, so every put would fail without an error (see "
            "'Buffer registration' above; LMCache also gets here when the store "
            "setup itself failed, e.g. with an unreachable master)"
        )
    local = backends.get("LocalCPUBackend")
    if local is None:
        raise RuntimeError(
            "the LMCache remote tier needs the LocalCPUBackend pool to stage "
            "chunks through, and LMCache built none"
        )
    _round_trip(remote, local, metadata, put_timeout_s)
    logger.info(
        "LMCache remote tier %s: one-chunk round trip verified",
        getattr(remote, "remote_url", ""),
    )


def _probe_key(metadata: Any):
    from lmcache.utils import CacheEngineKey

    return CacheEngineKey(
        f"{metadata.model_name}{_PROBE_NAMESPACE_SUFFIX}",
        int(metadata.world_size),
        int(metadata.worker_id),
        secrets.randbits(63),
        metadata.kv_dtype,
    )


def _round_trip(remote: Any, local: Any, metadata: Any, put_timeout_s: float) -> None:
    import torch
    from lmcache.v1.memory_management import MemoryFormat

    key = _probe_key(metadata)
    source = local.allocate(
        metadata.get_shapes(), metadata.get_dtypes(), fmt=MemoryFormat.KV_2LTD
    )
    if source is None:
        raise RuntimeError("could not allocate a probe chunk in the LMCache CPU pool")
    try:
        payload = source.tensor.view(torch.uint8).reshape(-1)
        # On the CPU like the chunks it is compared with, whatever the caller's
        # default device.
        expected = (
            torch.arange(payload.numel(), dtype=torch.int64, device="cpu")
            .add(key.chunk_hash % 251)
            .remainder(251)
            .to(torch.uint8)
        )
        payload.copy_(expected)
        put_done = threading.Event()
        remote.batched_submit_put_task(
            [key], [source], on_complete_callback=lambda _key: put_done.set()
        )
        if not put_done.wait(put_timeout_s):
            raise RuntimeError(
                f"a one-chunk put to the LMCache remote tier did not finish in "
                f"{put_timeout_s:g} s"
            )
    finally:
        source.ref_count_down()
    if remote.batched_contains([key]) != 1:
        raise RuntimeError(
            "a one-chunk put to the LMCache remote tier reported success but the "
            "store does not hold the chunk (a failed put is not counted in "
            "put_failed_count; look for transfer errors such as "
            "AddressNotRegistered in the log)"
        )
    (returned,) = remote.batched_get_blocking([key])
    if returned is None:
        raise RuntimeError(
            "the LMCache remote tier holds the probe chunk but get failed"
        )
    try:
        if not torch.equal(returned.tensor.view(torch.uint8).reshape(-1), expected):
            raise RuntimeError(
                "the LMCache remote tier returned the probe chunk with different "
                "bytes"
            )
    finally:
        returned.ref_count_down()
    _remove_probe(remote, key)


def _remove_probe(remote: Any, key: Any) -> None:
    # Best effort: not every connector removes, and the store evicts the probe
    # like any other chunk.
    try:
        remote.connection.remove_sync(key)
    except Exception:  # optional third-party cleanup boundary
        logger.debug("LMCache remote tier: probe chunk not removed", exc_info=True)
