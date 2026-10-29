# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run a long-lived loop on one large CPython data-stack chunk (ATOM_DEEP_FRAME_STACK).

CPython 3.11 and 3.12 keep Python frames in 16 KiB chunks of a per-thread data
stack and free a chunk as soon as the frame at its base returns. A hot path
whose call depth straddles the end of a chunk so maps a chunk on every call that
crosses it and unmaps it on the return: an mmap and a munmap, the munmap a TLB
shootdown on every CPU the process has run on. In an ATOM worker that is
FlyDSL's launch path -- dozens of `inspect` property calls a launch -- at
whatever depth the eager prefill puts it: at c2048 DEP4, 16-23% of a worker's
main thread went to mmap / munmap, and a launch cost 350-470 us of host time
against ~50 us at a depth clear of a chunk end.

A frame whose code reserves a large value stack is placed at the base of a chunk
of its own size, rounded up to a power of two, and every frame called beneath it
lands in the rest of that chunk, so the loop maps and frees no chunk at all.
"""

import sys

# 1 MiB of value stack: a 2 MiB chunk, ~1 MiB of it for the frames below
_RESERVED_SLOTS = 1 << 17


def _anchored(fn, args, kwargs):
    return fn(*args, **kwargs)


if sys.version_info[:2] in ((3, 11), (3, 12)):
    _anchored.__code__ = _anchored.__code__.replace(co_stacksize=_RESERVED_SLOTS)


def run_on_deep_stack(fn, *args, **kwargs):
    """`fn(*args, **kwargs)`, under a frame that holds a chunk of its own."""
    from atom.utils import envs

    if not envs.ATOM_DEEP_FRAME_STACK:
        return fn(*args, **kwargs)
    return _anchored(fn, args, kwargs)
