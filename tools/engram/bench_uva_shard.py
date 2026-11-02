# SPDX-License-Identifier: MIT
"""Time and measure page-locking one rank's Engram UVA shard, as the server does.

Loads the checkpoint's Engram tables (the same shared read-only mapping the
server uses, `weights.shared_tensor`), page-locks this rank's head range of
each, and reports per-rank time, memory, decode-sized gather latency and
whether the gathered rows match the host path, as one JSON line. Run one
process per rank at once to reproduce a TP job's contention:

    for r in 0 1; do HIP_VISIBLE_DEVICES=$r ATOM_ENGRAM_UVA_COPY=1 \\
        python tools/engram/bench_uva_shard.py --model DIR --tp 2 --rank $r & done

or a DP-attention job's, where every rank page-locks the whole tables:

    for r in 0 1 2 3; do HIP_VISIBLE_DEVICES=$r \\
        python tools/engram/bench_uva_shard.py --model DIR --dp 4 --rank $r & done

`ATOM_ENGRAM_UVA_COPY` picks copy (1) or in-place registration (0); unset, a
DP rank registers in place and a TP rank copies, as the server does
(`host_memory.copies`). `--cold` drops the engram shards from the page cache
first (posix_fadvise; effective only when no other process maps them).
`--hold` keeps the shard page-locked that many seconds after reporting, so the
node's memory can be read while every rank holds its shard.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from atom.model_ops.engram import host_memory
from atom.model_ops.engram.mapping import EngramConfig, NgramHashMapping
from atom.model_ops.engram.tables import HostEmbeddingTable
from atom.models.deepseek_v41.weights import shared_tensor


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
    ap.add_argument("--dp", type=int, default=1, help="DP ranks (whole tables)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--tokens", type=int, default=192, help="rows per decode gather")
    ap.add_argument("--hold", type=float, default=0, help="seconds to hold after")
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
    tables = {}
    for layer, rows in zip(cfg.layer_ids, cfg.num_embeddings):
        n = names[layer]
        w = shared_tensor(model / weight_map[n + ".weight"], n + ".weight")
        s = shared_tensor(model / weight_map[n + ".scale"], n + ".scale")
        tables[layer] = HostEmbeddingTable(
            w, num_rows=rows, head_dim=cfg.head_dim, scale=s
        )

    heads = cfg.num_hash_heads
    if args.dp > 1 and args.tp > 1:
        raise SystemExit("--dp is one GPU per TP group: use --tp 1")
    shared = args.dp > 1  # as `EngramHost._enable_uva` decides
    per = -(-heads // args.tp)
    # A DP rank holds every head; a TP rank its own share.
    head_start = (0 if args.dp > 1 else args.rank) * per
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
        assert table.enable_uva(start, end, shared=shared), "page-locking refused"
        pinned += (end - start) * (table.head_dim + table._scale.shape[1])
    seconds = time.perf_counter() - t0
    after = memory()

    from atom.model_ops.engram.device.uva import uva_gather_into

    rng = np.random.default_rng(args.rank)
    gathers, exact = [], True
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
        # The rows the host path gathers for the same ids (this rank's heads).
        local_ids = ids[:, head_start : head_start + local].cpu().numpy()
        want = table.gather(local_ids, out_dtype=torch.bfloat16)
        exact &= torch.equal(out.cpu(), want)
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
                "mode": "copy" if host_memory.copies(shared) else "register",
                "tp": args.tp,
                "dp": args.dp,
                "rank": args.rank,
                "cold": args.cold,
                "pinned_gib": round(pinned / 2**30, 2),
                "pin_seconds": round(seconds, 2),
                "gather_ms_per_layer": [round(g, 3) for g in gathers],
                "matches_host_path": exact,
                "before_mib": before,
                "after_mib": after,
            }
        ),
        flush=True,
    )
    time.sleep(args.hold)


if __name__ == "__main__":
    main()
