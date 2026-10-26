# SPDX-License-Identifier: MIT
"""Time and measure page-locking one rank's Engram UVA shard, as the server does.

Loads the checkpoint's Engram tables (the same safetensors mmap the server
uses), page-locks this rank's head range of each, and reports per-rank time,
memory and decode-sized gather latency as one JSON line. Run one process per
rank at once to reproduce a TP job's contention:

    for r in 0 1; do HIP_VISIBLE_DEVICES=$r ATOM_ENGRAM_UVA_COPY=1 \\
        python tools/engram/bench_uva_shard.py --model DIR --tp 2 --rank $r & done

`ATOM_ENGRAM_UVA_COPY` picks copy (1) or in-place registration (0). `--cold`
drops the engram shards from the page cache first (posix_fadvise; effective
only when no other process maps them).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from atom.model_ops.engram.mapping import EngramConfig, NgramHashMapping
from atom.model_ops.engram.tables import HostEmbeddingTable


def memory() -> dict:
    """MiB: RSS, its peak, locked and pinned pages, and PSS split anon/file."""
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "VmLck", "VmPin"):
            out[key] = int(value.split()[0]) >> 10
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Pss", "Pss_Anon", "Pss_File", "Rss_Anon"):
            out[key] = int(value.split()[0]) >> 10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--tokens", type=int, default=192, help="rows per decode gather")
    args = ap.parse_args()
    model = Path(args.model)
    cfg_json = json.loads((model / "config.json").read_text())
    cfg = EngramConfig.from_hf(cfg_json) or EngramConfig.from_hf(
        cfg_json.get("text_config", {})
    )
    if cfg is None:
        raise SystemExit("no engram in this checkpoint's config")
    mapping = NgramHashMapping.__new__(NgramHashMapping)  # row layout only
    mapping.config = cfg
    sizes = mapping._derive_head_vocab_sizes()
    weight_map = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = {layer: f"layers.{layer}.engram.embed" for layer in cfg.layer_ids}
    files = sorted(
        {weight_map[n + s] for n in names.values() for s in (".weight", ".scale")}
    )
    if args.cold:
        for f in files:
            fd = os.open(model / f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    torch.cuda.init()
    torch.empty(1, device="cuda")
    handles = {f: safe_open(model / f, framework="pt", device="cpu") for f in files}
    tables = {}
    for layer, rows in zip(cfg.layer_ids, cfg.num_embeddings):
        n = names[layer]
        w = handles[weight_map[n + ".weight"]].get_tensor(n + ".weight")
        s = handles[weight_map[n + ".scale"]].get_tensor(n + ".scale")
        tables[layer] = HostEmbeddingTable(
            w, num_rows=rows, head_dim=cfg.head_dim, scale=s
        )

    heads = cfg.num_hash_heads
    per = -(-heads // args.tp)
    head_start = args.rank * per
    local = min(per, heads - head_start)
    before = memory()
    t0 = time.perf_counter()
    pinned = 0
    for layer, table in tables.items():
        offsets = np.concatenate([[0], np.cumsum(sizes[layer][:-1])])
        start = int(offsets[head_start])
        end = int(
            offsets[head_start + local - 1] + sizes[layer][head_start + local - 1]
        )
        assert table.enable_uva(start, end), "page-locking refused"
        pinned += (end - start) * (table.head_dim + table._scale.shape[1])
    seconds = time.perf_counter() - t0
    after = memory()

    from atom.model_ops.engram.device.uva import uva_gather_into

    rng = np.random.default_rng(args.rank)
    gathers = []
    for layer, table in tables.items():
        offsets = np.concatenate([[0], np.cumsum(sizes[layer][:-1])])
        ids = torch.from_numpy(
            offsets[None, :]
            + rng.integers(0, sizes[layer][None, :], (args.tokens, heads))
        ).cuda()
        out = torch.empty(
            args.tokens, local, cfg.head_dim, dtype=torch.bfloat16, device="cuda"
        )
        kw = {"head_start": head_start, "local_heads": local, "total_heads": heads}
        for _ in range(20):
            uva_gather_into(table, ids, out, **kw)
        start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        start_ev.record()
        for _ in range(200):
            uva_gather_into(table, ids, out, **kw)
        end_ev.record()
        torch.cuda.synchronize()
        gathers.append(start_ev.elapsed_time(end_ev) / 200)
    print(
        json.dumps(
            {
                "mode": (
                    "copy"
                    if os.environ.get("ATOM_ENGRAM_UVA_COPY", "1") == "1"
                    else "register"
                ),
                "tp": args.tp,
                "rank": args.rank,
                "cold": args.cold,
                "pinned_gib": round(pinned / 2**30, 2),
                "pin_seconds": round(seconds, 2),
                "gather_ms_per_layer": [round(g, 3) for g in gathers],
                "before_mib": before,
                "after_mib": after,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
