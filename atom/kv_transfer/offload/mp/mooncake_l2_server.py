# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run an LMCache MP server whose L2 is a Mooncake Store pool.

    python3 -m atom.kv_transfer.offload.mp.mooncake_l2_server --gpu G
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
"""

from __future__ import annotations

import argparse
import json
import logging
import runpy
import sys

from atom.kv_transfer.offload.mooncake_store_l2 import (
    requester_rdma_device,
    store_pool_of,
)

logger = logging.getLogger("atom")

LMCACHE_SERVER_MODULE = "lmcache.v1.multiprocess.server"


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" not in argv:
        raise SystemExit(
            "usage: mooncake_l2_server --gpu G --local-hostname IP "
            "[--master HOST:PORT --metadata URL] -- <LMCache server arguments>"
        )
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="mooncake_l2_server")
    parser.add_argument("--gpu", type=int, required=True)
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
