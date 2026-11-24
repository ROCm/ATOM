# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A kernel's mailbox regions laid out in its scratch or peer buffer."""

PAIR_BYTES = 8  # a (value, tag) pair
# MI355X homes each 8 KiB run of a 2 MiB page on one of its two I/O dies,
# alternating: a poll of the other die's memory takes a longer round trip
NUMA_PAGE = 1 << 21
NUMA_RUN = 8192
# a debug build's wait record, one a CTA: flag, region id, pair index, tag,
# seen tag, CTA, clock, pad (``atom.mono.device.sync._spin_bounded``)
DIAG_WORDS = 8


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


def dual_layout(items, start: int = 0) -> tuple[tuple, int]:
    """``items`` (name, bytes) of regions kept in two copies, one on each I/O
    die (``atom.mono.device.sync.dreg``), laid out in order from ``start`` (a
    page boundary): ((name, byte offset, page of the area), ...). A region's
    copy 0 lies in the even runs from its offset, copy 1 in the odd ones
    (``dual_word``), so it takes twice its bytes rounded to runs, from an
    even run; none crosses a page (a page's run parity is its own). -> (the
    table, the area's bytes)."""
    assert start % NUMA_PAGE == 0, start
    out, off = [], 0
    for name, nbytes in items:
        need = 2 * -(-nbytes // NUMA_RUN) * NUMA_RUN
        assert need <= NUMA_PAGE, (name, need)
        if off % NUMA_PAGE + need > NUMA_PAGE:
            off = -(-off // NUMA_PAGE) * NUMA_PAGE
        out.append((name, start + off, off // NUMA_PAGE))
        off += need
    return tuple(out), off


def dual_spec(table) -> str:
    """``dual_layout``'s table as a build key's scalar: "name:offset:page,..."."""
    return ",".join(f"{name}:{off}:{page}" for name, off, page in table)


def dual_places(spec: str) -> dict:
    """``dual_spec``'s regions: name -> (byte offset, page)."""
    out = {}
    for item in filter(None, spec.split(",")):
        name, off, page = item.split(":")
        out[name] = (int(off), int(page))
    return out


def dual_word(w):
    """Copy 0's word of word ``w`` of a dual region (``dual_layout``): run k
    of the region at run 2 k. Copy 1 is ``NUMA_RUN`` bytes on."""
    run = NUMA_RUN // 4
    return (w - (w & (run - 1))) * 2 + (w & (run - 1))


def byte_layout(items, align: int = 16, start: int = 0) -> dict:
    """``items`` (name, bytes) of plain words laid out in order: name -> (byte
    offset, bytes), each region from an ``align``-byte boundary at or past
    ``start``.

    Plain words never share a buffer with tagged pairs: one launch's stale
    plain word in a later launch's tag position can equal that launch's tag,
    and its poll takes the pair as landed."""
    out, off = {}, start
    for name, nbytes in items:
        off = (off + align - 1) // align * align
        out[name] = (off, nbytes)
        off += nbytes
    return out
