# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A mono runner's construction, collective over the TP group: every rank binds
and checks its shard, all agree, and only then the peer handshake -- a rank that
refuses turns mono off on every rank instead of leaving the others waiting in a
collective it never reaches."""

import weakref

from atom.mono.runtime.consensus import MonoUnsupported, tp_agree
from atom.mono.runtime.peer_memory import PeerBuffer


def bind_agreed(bind, group) -> None:
    """``bind()`` on this rank (raising ``MonoUnsupported`` to refuse), then every
    rank's verdict agreed: a rank that raised re-raises, the others raise too."""
    try:
        bind()
    except Exception:
        tp_agree(False, group)
        raise
    if not tp_agree(True, group):
        raise MonoUnsupported("another TP rank refused mono")


def owned_peer_buffer(owner, nbytes, group, rank, npes, device, *, debug=False):
    """``(PeerBuffer, finalizer)``: the buffer is freed when ``owner`` is dropped
    (a model reload), not at interpreter exit, where the HIP runtime may already
    be gone; calling the finalizer frees it now. Collective (``PeerBuffer``)."""
    peers = PeerBuffer(nbytes, group, rank, npes, device, debug=debug)
    finalizer = weakref.finalize(owner, peers.close)
    finalizer.atexit = False
    return peers, finalizer
