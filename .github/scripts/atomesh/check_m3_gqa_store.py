#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate the two approved GQA/Store experiment cells before GPU allocation."""

import argparse
import json
from pathlib import Path

DRAFT = "/share_nfs/models/Inferact/MiniMax-M3-EAGLE3-GQA"
AITER = "4fe2800a5c252adff8ee28d199bea66f8b18efac"
CASES = {
    "minimax-m3-gluon-gqa-mcstore-tp4-c32": (4, 32, 1),
    "minimax-m3-gluon-gqa-mcstore-tp8-c72": (8, 72, 2),
}


def validate(cell):
    def require(condition, detail):
        if not condition:
            raise ValueError(f"GQA Store experiment: {detail}")

    tp, concurrency, nodes = CASES[cell["name"]]
    require(
        cell["concurrency"] == [concurrency] and cell["num_nodes"] == nodes,
        "concurrency/nodes",
    )
    require(
        cell["pd_worker_layout"] == ("single_node" if nodes == 1 else "multi_node"),
        "node layout",
    )
    for role in ("prefill", "decode"):
        service, env = cell["service"][role], cell["env"][role]
        require(service["tp"] == tp and service["workers"] == 1, f"{role} TP/workers")
        require(
            not service.get("extra_args"), f"{role} has unexpected PP/eager/extra flags"
        )
        require(
            service["cudagraph_mode"] == "FULL"
            and max(json.loads(service["cudagraph"])) == 80,
            f"{role} graph ladder",
        )
        require(env["ATOM_DRAFT_CUDAGRAPH"] == "1", f"{role} draft graph disabled")
        require(env["ATOM_AITER_SCATTER_REF"] == AITER, f"{role} AITER pin")
        require(
            env["ATOM_FORCE_ATTN_TRITON"] == "1"
            and env.get("ATOM_USE_UNIFIED_ATTN", "0") == "0",
            f"{role} native Gluon path",
        )
        require(
            env["ATOM_M3_INDEXER_CP"] == ("1" if tp == 4 else "0"), f"{role} indexer CP"
        )
        require(not any(k.startswith("LMCACHE_") for k in env), "unexpected LMCache")
        require(env["MC_NUM_QP_PER_EP"] == "1", "RDMA QP mismatch")
        ids = env["HIP_VISIBLE_DEVICES"]
        require(
            ids
            == (
                "0,1,2,3,4,5,6,7"
                if tp == 8
                else ("0,1,2,3" if role == "prefill" else "4,5,6,7")
            ),
            f"{role} GPU IDs",
        )
    args = cell["server_args"]
    for k, v in {
        "draft_model": DRAFT,
        "num_speculative_tokens": 3,
        "spec_decode_acceptance_length": 2.78,
        "kv_cache_dtype": "fp8",
        "block_size": 128,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 80,
        "decode_max_num_seqs": 80,
        "enable_prefix_caching": True,
    }.items():
        require(args[k] == v, f"server {k}")
    require(
        "--enforce-eager" not in args["extra_args"]
        and "workspace" not in args["extra_args"],
        "eager/gather budget",
    )
    b = cell["benchmark"]
    for k, v in {
        "max_context_length": 1048576,
        "num_dataset_entries": 393,
        "benchmark_duration": 3600,
        "warmup_requests_per_lane": 10,
        "prepare_lmcache_host_memory": False,
        "prepare_m3_gqa_draft": True,
        "mesh_source": "image",
    }.items():
        require(b[k] == v, f"benchmark {k}")
    require(not cell["run_eval"], "synthetic acceptance must not run accuracy scoring")
    p = cell["env"]["prefill"]
    require(p["MOONCAKE_STORE"] == "1", "Store disabled")
    connector = json.loads(
        p["PREFILL_KV_TRANSFER_CONFIG"].replace("${HANDSHAKE_PORT}", "6301")
    )
    require(
        connector["kv_connector"] == "mooncake" and connector["protocol"] == "rdma",
        "P/D connector",
    )
    owners = [entry.split(":") for entry in p["MOONCAKE_STORE_OWNERS"].split(";")]
    require(sum(int(owner[1]) for owner in owners) == 1536, "Store must total 1536 GiB")
    if tp == 8:
        require(
            p["MOONCAKE_STORE_POOLS"] == "per_nic", "TP8 needs independent NIC pools"
        )
        require(
            len(owners) == 8
            and {o[2] for o in owners} == {f"rdma{i}" for i in range(8)},
            "every P NIC needs a reachable pool",
        )
        require(all(o[1] == "192" for o in owners), "uneven Store pools")
    else:
        require(p.get("MOONCAKE_STORE_POOLS", "shared") == "shared", "C32 shared Store")
        require(
            p["MOONCAKE_STORE_OWNERS"] == "0:768:rdma4,rdma5;1:768:rdma6,rdma7",
            "C32 Store owners",
        )
    print(
        f"[gqa-store-config] {cell['name']}: P/D TP{tp}, {nodes * 8} GPUs, C{concurrency}, mean AL2.78, Store1536GiB"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    cells = json.loads(args.matrix.read_text())["include"]
    if not cells:
        raise ValueError("Empty GQA experiment matrix")
    for cell in cells:
        validate(cell)


if __name__ == "__main__":
    main()
