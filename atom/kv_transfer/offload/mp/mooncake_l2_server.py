# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run an LMCache MP server whose L2 is a Mooncake Store pool.

    python3 -m atom.kv_transfer.offload.mp.mooncake_l2_server --gpu G --numa N
        --local-hostname IP [--master HOST:PORT --metadata URL]
        -- <lmcache.v1.multiprocess.server arguments>

One server serves one PP stage, and so one GPU: LMCache keeps one layout per
(model, world size), so a server serving stages with different layer counts
would size its L2 prefetches wrongly (``lmcache.mp.server_per_rank_layouts``).
The server reads and writes the Store over RDMA on one NIC, chosen as the
in-process Store L2 chooses a worker's (``mooncake_store_l2``): the GPU's NIC
in the PCI tree, or ``ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES``. With per-NIC
pools (``ATOM_LMCACHE_MOONCAKE_POOLS``) it joins the pool of that NIC, whose
owners share it; otherwise the pool of ``--master``/``--metadata``.

This module appends the ``mooncake_store`` ``--l2-adapter`` for that NIC and
pool, then runs LMCache's server in this process.

The adapter registers the whole L1 with the NIC at startup, as one MR, which an
ionic NIC refuses if a single 4 KiB page is in it. The launcher therefore
starts the server with glibc's THP malloc and ``--l1-align-bytes 2097152``.
LMCache aligns every L1 object to that value too, which puts each ~3 MB chunk
in a 4 MiB slot and wastes a quarter of the L1, so this module keeps the base
on 2 MiB and puts the objects back on 4 KiB (``keep_l1_objects_page_aligned``).

A lookup reads its whole hit from the Store into L1 before it answers, and the
part the GPU already holds is never loaded, so ATOM asks from past it and this
module has the server read from there; it still looks the whole prompt up,
which keeps the Store from evicting the GPU's prefix
(``prefetch_from_lookup_start``).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import runpy
import sys

from atom.kv_transfer.offload.mooncake_store_l2 import (
    _collapse_into_huge_pages,
    _touch_every_huge_page,
    requester_rdma_device,
    store_pool_of,
)

logger = logging.getLogger("atom")

LMCACHE_SERVER_MODULE = "lmcache.v1.multiprocess.server"
# LMCache's default L1 object alignment (``--l1-align-bytes``).
L1_OBJECT_ALIGN_BYTES = 4096


def keep_l1_objects_page_aligned() -> None:
    """Align only the L1's base to ``--l1-align-bytes``, not each object in it.

    LMCache's lazy L1 aligns the buffer base and every object to one value. An
    L1 pinned whole at startup (``--l1-init-size-gb`` >= ``--l1-size-gb``) never
    grows, so its object allocator can be rebuilt with page alignment once the
    buffer exists. A lazily growing L1 is left alone: its expansion thread owns
    the address space.
    """
    from lmcache.v1.memory_allocators import lazy_memory_allocator as lazy
    from lmcache.v1.memory_allocators.tensor_memory_allocator import (
        TensorMemoryAllocator,
    )

    allocator_init = lazy.LazyMemoryAllocator.__init__
    if getattr(allocator_init, "_atom_page_aligned_objects", False):
        return

    def __init__(self, *args, **kwargs):
        allocator_init(self, *args, **kwargs)
        try:
            pinned_whole = self._curr_size >= self._final_size
            buffer = self._buffer
        except AttributeError:
            logger.warning(
                "LMCache L1 objects keep the base alignment: this LMCache's "
                "LazyMemoryAllocator has no _curr_size/_final_size/_buffer"
            )
            return
        if not pinned_whole:
            return
        self._allocator = TensorMemoryAllocator(
            tensor=buffer,
            align_bytes=L1_OBJECT_ALIGN_BYTES,
            init_address_space=self._curr_size,
        )
        self._address_manager = self._allocator.address_manager

    __init__._atom_page_aligned_objects = True
    lazy.LazyMemoryAllocator.__init__ = __init__


def make_l1_huge_before_pinning(numa_node: int) -> None:
    """Fault the lazy L1 as huge pages, and collapse the rest, before pinning.

    glibc's THP malloc only advises huge pages; on a fragmented node the first
    fault of a 2 MiB stretch falls back to 4 KiB pages, and the adapter's one
    MR for the whole L1 then fails (two half nodes, pit2-p03-g52: compaction
    reached 99.6% and three of four 96 GiB L1s failed to register with
    ENOMEM). Each chunk LMCache pins is touched once per 2 MiB and collapsed
    (MADV_COLLAPSE compacts synchronously, retried on ENOMEM/EAGAIN); a chunk
    the collapse cannot make all huge pages fails the server's start here,
    with the shortfall named, instead of in the adapter. Pinned pages cannot
    move, so this has to run before the pin.
    """
    from lmcache.v1.memory_allocators import lazy_memory_allocator as lazy

    pin_chunk = lazy.LazyMemoryAllocator._pin_memory_chunk
    if getattr(pin_chunk, "_atom_huge_before_pinning", False):
        return

    def _pin_memory_chunk(self, offset: int, size: int) -> None:
        address = self._buffer.data_ptr() + offset
        _touch_every_huge_page(address, size, numa_node)
        # LMCache's L1 is an aligned slice of a larger mapping, so smaps cannot
        # count its huge pages; MADV_COLLAPSE succeeding over the aligned
        # range is the check.
        if not _collapse_into_huge_pages(address, size):
            raise RuntimeError(
                f"LMCache MP L1 chunk at {address:#x} ({size / 2**30:.2f} GiB) "
                f"is not all huge pages after MADV_COLLAPSE on NUMA node "
                f"{numa_node}; the Store adapter cannot register it as one MR. "
                "Free or compact the node's memory (the launcher's "
                "numa_memory_budget.py --compact) and restart"
            )
        return pin_chunk(self, offset, size)

    _pin_memory_chunk._atom_huge_before_pinning = True
    lazy.LazyMemoryAllocator._pin_memory_chunk = _pin_memory_chunk


def prefetch_from_lookup_start() -> None:
    """Have a LOOKUP read into L1 only the chunks from its key's start on.

    LMCache's LOOKUP reads every chunk of a hit into L1 before it answers,
    whatever ``key.start`` says. ATOM's scheduler asks from past the prompt's
    HBM prefix (``lmcache.mp.lookup_past_hbm_prefix``), which it never loads;
    with the Store as L2 those reads were 33-56% of the prefetched tokens
    (GLM-5.2 cpp4/dcp4, one and two nodes, c96 and c112).

    Every chunk is still looked up from 0. The answer stays a prefix length,
    and looking a chunk up renews its Store lease: the Store evicts the
    objects whose lease is oldest, so a prefix only the GPU serves would
    otherwise go first, and fall out of the tier before the GPU lets go of it
    (reading only from the start, two-node c112 recomputed 21% more tokens
    per request). Only the reads change. Every full-attention row gets a
    window of the chunks from ``key.start`` to the end -- LMCache's sliding-
    window prefetch -- so a hit of ``L`` chunks reads and locks only its last
    ``window`` chunks: those from the start on, or a few below it when the hit
    stops short of the end. A narrower sliding window is kept.

    Releases are clamped to the chunks LMCache locked, since a release from 0
    must not drop another request's lock on a shared chunk; for the same
    reason a release for a session this server never looked up is dropped
    (LMCache would release the whole range). A key starting at 0 behaves as
    before.
    """
    from lmcache.v1.multiprocess.modules import lookup as lookup_module

    module = lookup_module.LookupModule
    if getattr(module.free_lookup_locks, "_atom_from_lookup_start", False):
        return
    # One LookupModule per server, so one chunk size.
    chunk_tokens: dict[str, int] = {}

    def read_window(key, size: int) -> int:
        """Chunks from the key's start to the last whole chunk it hashes."""
        return min(key.end, len(key.token_ids)) // size - key.start // size

    init = module.__init__

    @functools.wraps(init)
    def __init__(self, ctx, *args, **kwargs):
        init(self, ctx, *args, **kwargs)
        chunk_tokens["size"] = int(ctx.chunk_size)

    group_keys = lookup_module.ipc_key_to_grouped_object_keys

    def keys_read_from_lookup_start(ipc_key, chunk_hashes, *args, **kwargs):
        rows = group_keys(ipc_key, chunk_hashes, *args, **kwargs)
        size = chunk_tokens.get("size")
        if not size or ipc_key.start <= 0:
            return rows
        window = read_window(ipc_key, size)
        # A window below 1 would read as full attention; ATOM's adapter
        # always leaves a chunk past the start.
        if window < 1:
            return rows
        return [
            dataclasses.replace(
                row,
                sliding_window_size=(
                    window
                    if row.sliding_window_size < 1
                    else min(row.sliding_window_size, window)
                ),
            )
            for row in rows
        ]

    free = module.free_lookup_locks

    @functools.wraps(free)
    def free_lookup_locks(self, key, tp_size):
        session = self._ctx.session_manager.get_or_create(key.request_id)
        lookup_key = session.lookup_ipc_key
        if lookup_key is None:
            return None
        locked_from = 0
        if lookup_key.start > 0:
            size = self._ctx.chunk_size
            hit = session.prefetch_hit_chunks
            # Before the answer is read nothing says how far below the start
            # the window reached; leave that to the read-lock TTL.
            locked_from = (
                lookup_key.start
                if hit < 0
                else max(0, hit - read_window(lookup_key, size)) * size
            )
        if key.start < locked_from:
            if key.end <= locked_from:
                return None
            key = dataclasses.replace(key, start=locked_from)
        return free(self, key, tp_size)

    free_lookup_locks._atom_from_lookup_start = True
    module.__init__ = __init__
    lookup_module.ipc_key_to_grouped_object_keys = keys_read_from_lookup_start
    module.free_lookup_locks = free_lookup_locks


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit(
            "usage: mooncake_l2_server --gpu G --numa N --local-hostname IP "
            "[--master HOST:PORT --metadata URL] -- <LMCache server arguments>"
        )
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="mooncake_l2_server")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--numa", type=int, required=True)
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--master", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args(argv[:split])
    server_args = argv[split + 1 :]
    for arg in server_args:
        if arg == "--l2-adapter" or arg.startswith("--l2-adapter="):
            raise SystemExit(
                "mooncake_l2_server: the server's --l2-adapter is this module's"
            )
    return args, server_args


def l2_adapter_spec(
    *,
    device: str,
    master: str,
    metadata: str,
    local_hostname: str,
    num_workers: int,
) -> dict:
    """The ``mooncake_store`` adapter of one server on one NIC.

    LMCache forwards every key but ``type``, ``num_workers``, ``eviction`` and
    ``per_op_workers`` to Mooncake as a string and fills in no default, so the
    Mooncake defaults that would be wrong here are spelled out: no global
    segment (the owners hold the memory) and no local buffer (transfers go
    straight from the registered L1).
    """
    return {
        "type": "mooncake_store",
        "num_workers": num_workers,
        "master_server_addr": master,
        "metadata_server": metadata,
        "local_hostname": local_hostname,
        "protocol": "rdma",
        "rdma_devices": device,
        "global_segment_size": "0",
        "local_buffer_size": "0",
    }


def main(argv: list[str] | None = None) -> None:
    args, server_args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    device = requester_rdma_device(args.gpu)
    pool = store_pool_of(device)
    if pool is not None:
        master, metadata = pool.master, pool.metadata
    elif args.master and args.metadata:
        master, metadata = args.master, args.metadata
    else:
        raise SystemExit(
            "mooncake_l2_server: name the pool with --master and --metadata, "
            "or set ATOM_LMCACHE_MOONCAKE_POOLS"
        )
    spec = l2_adapter_spec(
        device=device,
        master=master,
        metadata=metadata,
        local_hostname=args.local_hostname,
        num_workers=args.num_workers,
    )
    logger.info(
        "LMCache MP server for GPU %d: Mooncake Store L2 on %s, master %s",
        args.gpu,
        device,
        master,
    )
    keep_l1_objects_page_aligned()
    make_l1_huge_before_pinning(args.numa)
    prefetch_from_lookup_start()
    logger.info("LMCache MP server reads a LOOKUP into L1 from its key's start")
    # The server parses sys.argv itself; JSON with no spaces survives any
    # later whitespace split of the logged command line.
    sys.argv = [
        LMCACHE_SERVER_MODULE,
        *server_args,
        "--l2-adapter",
        json.dumps(spec, separators=(",", ":")),
    ]
    runpy.run_module(LMCACHE_SERVER_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
