# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A mono runner's peer buffer, owned for its lifetime. The buffer's handshake is
collective: a runner reaches it only after every rank bound and agreed
(``consensus.bind_agreed``), so a rank that refuses turns mono off on every rank
instead of leaving the others waiting in a collective it never reaches."""

import weakref

from atom.mono.runtime.peer_memory import PeerBuffer


def closed_with(owner, resource):
    """``resource``'s ``close`` (its peer mappings) run when ``owner`` is
    dropped (a model reload), not at interpreter exit, where the HIP runtime
    may already be gone: the finalizer, which closes them now if called."""
    finalizer = weakref.finalize(owner, resource.close)
    finalizer.atexit = False
    return finalizer


def owned_peer_buffer(owner, nbytes, group, rank, npes, device, plain_bytes=0):
    """``(PeerBuffer, finalizer)``: the buffer's peer mappings closed with
    ``owner`` (``closed_with``). The memory itself is never freed
    (``uncached``). Collective (``PeerBuffer``)."""
    peers = PeerBuffer(nbytes, group, rank, npes, device, plain_bytes)
    return peers, closed_with(owner, peers)
