# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay an aiperf trace through the real `BlockPool` at a model's shape.

The question: with superblocks in play, how often does the pool end up holding
superblocks it cannot reclaim whole because one block is still shared? A
superblock with a live block cannot become a contiguous state slot however much
of it is free, so the risk is not the per-event rate but the **standing count
over time**.

This drives the actual `BlockPool` and `SuperblockMap` rather than modelling
them, so the vacant/cached order, ref counting, sharing and eviction are the
production ones. What is still a model is the *workload*: block counts come
from per-request prompt/cache-read token counts, and a turn is assumed to reuse
its conversation's prefix plus any globally shared prefix.

Cross-conversation sharing IS modelled here (`--shared-prefix-blocks`), because
it is what creates pinning — a shared prefix ending mid-superblock leaves the
shared half live and the rest dead. An earlier version of this harness omitted
it and its pinning numbers were a lower bound.

Even so: a replay is not a server. A previous harness in this project reported
a flat curve where hardware lost 12 points at conc-4. Read the extremes, not
the third decimal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.superblock import SuperblockMap


@dataclass
class SimConfig:
    block_tokens: int = 128
    blocks_per_super: int = 32
    num_superblocks: int = 658
    max_num_seqs: int = 32
    #: Blocks of system prompt every conversation shares. This is the pinning
    #: driver: where it ends mid-superblock, that superblock stays live.
    shared_prefix_blocks: int = 4


@dataclass
class SimResult:
    requests: int = 0
    evictions: int = 0
    state_claims: int = 0
    state_denied: int = 0
    #: Standing count of superblocks holding a live block but not full, sampled
    #: once per request. The accumulation curve, which is the real risk.
    pinned_samples: list[int] = field(default_factory=list)
    reclaimable_samples: list[int] = field(default_factory=list)

    @property
    def pinned_peak(self) -> int:
        return max(self.pinned_samples) if self.pinned_samples else 0

    @property
    def pinned_final(self) -> int:
        return self.pinned_samples[-1] if self.pinned_samples else 0

    @property
    def pinned_mean(self) -> float:
        n = len(self.pinned_samples)
        return sum(self.pinned_samples) / n if n else 0.0


def load_trace(path: str) -> list[dict]:
    with open(path) as fh:
        lines = fh.readlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        metrics = rec.get("metrics", {})
        if "usage_prompt_tokens" not in metrics:
            continue
        md = rec.get("metadata", {})
        out.append(
            {
                "conv": md.get("conversation_id"),
                "start": md.get("request_start_ns") or 0,
                "end": md.get("request_end_ns") or 0,
                "prompt": int(metrics["usage_prompt_tokens"]["value"]),
                "cached": int(
                    metrics.get("usage_prompt_cache_read_tokens", {}).get("value", 0)
                ),
            }
        )
    out.sort(key=lambda r: r["start"])
    return out


def simulate(trace: list[dict], cfg: SimConfig) -> SimResult:
    num_blocks = cfg.num_superblocks * cfg.blocks_per_super
    sb = SuperblockMap(num_blocks, cfg.blocks_per_super)
    pool = BlockPool(num_blocks, superblocks=sb)
    res = SimResult()

    conv_blocks: dict[str, list[int]] = defaultdict(list)
    shared: list[int] = []
    live_state: list[int] = []
    inflight: list[tuple[int, list[int], int]] = []
    next_hash = [1]

    def retire(now: int) -> None:
        while inflight and inflight[0][0] <= now:
            _end, owned, slot = inflight.pop(0)
            for b in owned:
                pool.free(b)
            if slot >= 0:
                pool.release_superblock(slot)
                live_state.remove(slot)

    def fresh_block() -> int:
        if pool.num_free == 0:
            return -1
        b = pool.pop()
        had = pool.blocks[b].hash != -1
        pool.allocate(b)
        if had:
            res.evictions += 1
        h = next_hash[0]
        next_hash[0] += 1
        pool.publish(b, h, [h])
        return b

    # The globally shared system prompt, built once and never released, so it
    # is genuinely pinned for the whole run.
    for _ in range(cfg.shared_prefix_blocks):
        b = fresh_block()
        if b >= 0:
            shared.append(b)

    for req in trace:
        res.requests += 1
        retire(req["start"])

        slot = -1
        if len(live_state) < cfg.max_num_seqs:
            slot = pool.claim_superblock()
            if slot < 0:
                res.state_denied += 1
            else:
                res.state_claims += 1
                live_state.append(slot)

        total = max(1, req["prompt"] // cfg.block_tokens)
        hit = min(req["cached"] // cfg.block_tokens, total)
        prior = conv_blocks[req["conv"]]
        reuse = (shared + prior)[:hit]

        owned: list[int] = []
        for b in reuse:
            # Reachable by hash is the whole test, and it is the same one
            # `can_allocate` applies: a block spent for its space, or recycled
            # when its superblock became a state slot, has lost the hash the
            # request would have hit. The trace's own cache-read count cannot
            # see those misses — it was recorded against a pool with no
            # superblocks — so replaying it as written would claim blocks that
            # no longer hold the content.
            if pool.lookup(pool.blocks[b].hash) != b:
                continue
            pool.claim(b)
            owned.append(b)
        for _ in range(total - len(owned)):
            b = fresh_block()
            if b < 0:
                break
            owned.append(b)

        conv_blocks[req["conv"]] = [b for b in owned if b not in shared]
        inflight.append((req["end"], owned, slot))
        inflight.sort(key=lambda x: x[0])

        occ = sb.occupancy()
        res.pinned_samples.append(occ["supers_partially_pinned"])
        res.reclaimable_samples.append(occ["supers_reclaimable"])

    return res


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--blocks-per-super", type=int, default=32)
    ap.add_argument("--num-superblocks", type=int, default=658)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--block-tokens", type=int, default=128)
    ap.add_argument("--shared-prefix-blocks", type=int, default=4)
    args = ap.parse_args()

    cfg = SimConfig(
        block_tokens=args.block_tokens,
        blocks_per_super=args.blocks_per_super,
        num_superblocks=args.num_superblocks,
        max_num_seqs=args.max_num_seqs,
        shared_prefix_blocks=args.shared_prefix_blocks,
    )
    res = simulate(load_trace(args.trace), cfg)

    print(f"requests                 {res.requests}")
    print(f"superblocks              {cfg.num_superblocks} x {cfg.blocks_per_super}")
    print(f"evictions                {res.evictions}")
    print(f"state claims             {res.state_claims}")
    print(f"state DENIED             {res.state_denied}")
    print()
    print("partially-pinned superblocks (the accumulation curve):")
    print(f"  mean                   {res.pinned_mean:.1f}")
    print(f"  peak                   {res.pinned_peak}")
    print(f"  final                  {res.pinned_final}")
    print(
        f"  as % of pool (peak)    {100 * res.pinned_peak / cfg.num_superblocks:.2f}%"
    )
    if res.reclaimable_samples:
        print(f"reclaimable at end       {res.reclaimable_samples[-1]}")


if __name__ == "__main__":
    main()
