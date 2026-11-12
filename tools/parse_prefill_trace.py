#!/usr/bin/env python3
"""Per-layer breakdown of the ``prefill[`` forwards in a rank-0 trace.

Reads the trace ``tools/k31_prefill_profile.py`` writes. A forward is the
main-stream GPU work (kernels, memcpy, memset) launched inside its ``prefill[``
CPU span: kineto does not project that span onto the GPU reliably. Each op goes
to a layer and to its attention or MLP half, by module tag:

- a tag with a layer index names the layer, and ``warmup_item_owner_layers``
  gives an unindexed tag between two ops of one layer to that layer;
- the remaining ops sit between layers: an ``apply_attn_res`` opens the next
  layer, anything else (the all-reduces after a MoE) closes the previous one;
- the MLP half starts at a layer's first ``apply_attn_res`` after its first op.

Per forward it prints the span, the share of it the per-layer sums of op time
cover, and the mean per-layer time of each half of the DSA and KDA layers,
which it tells apart by the model's ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from parse_trace import (
    event_end,
    gpu_tag_for_kernel,
    load_events,
    primary_replay_stream,
    warmup_item_owner_layers,
)

GPU_OPS = {"kernel", "gpu_memcpy", "gpu_memset"}
ATTN_RES = "apply_attn_res"


def correlation(event: dict[str, Any]) -> Any:
    return (event.get("args") or {}).get("correlation")


def prefill_forwards(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("prefill[")
        ),
        key=lambda e: float(e["ts"]),
    )


def forward_ops(
    events: list[dict[str, Any]], forward: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """The forward's main-stream GPU ops in time order, and their module tags."""
    start, end = float(forward["ts"]), event_end(forward)
    launches = {
        correlation(e): e
        for e in events
        if e.get("cat") == "cuda_runtime"
        and start <= float(e["ts"]) < end
        and correlation(e) is not None
    }
    ops = [e for e in events if e.get("cat") in GPU_OPS and correlation(e) in launches]
    stream = primary_replay_stream(ops)
    ops = sorted(
        (e for e in ops if (e.get("args") or {}).get("stream") == stream),
        key=lambda e: float(e["ts"]),
    )
    gpu_start, gpu_end = float(ops[0]["ts"]), event_end(ops[-1])
    gpu_tags = [
        e
        for e in events
        if e.get("cat") == "gpu_user_annotation"
        and gpu_start <= float(e["ts"]) < gpu_end
    ]
    cpu_tags = [
        e
        for e in events
        if e.get("cat") in ("user_annotation", "cpu_op")
        and start <= float(e["ts"]) < end
    ]
    return ops, [gpu_tag_for_kernel(op, gpu_tags, launches, cpu_tags) for op in ops]


def attribute(tags: list[str]) -> list[tuple[int | None, bool]]:
    """``(layer, in_mlp_half)`` for each op's tag, by the rules above."""
    owners = warmup_item_owner_layers([{"module": tag} for tag in tags])
    following: list[int | None] = []
    nxt = None
    for owner in reversed(owners):
        nxt = owner if owner is not None else nxt
        following.append(nxt)
    following.reverse()

    out: list[tuple[int | None, bool]] = []
    layer, mlp = None, False
    for tag, owner, after in zip(tags, owners, following, strict=True):
        if owner is None:
            opens = layer is None or (tag == ATTN_RES and after is not None)
            owner = after if opens else layer
        if owner != layer:
            layer, mlp = owner, False
        elif tag == ATTN_RES:
            mlp = True
        out.append((layer, mlp))
    return out


def layer_kinds(model: str) -> tuple[set[int], int]:
    """The KDA layer indices and the layer count; every other layer is DSA."""
    with open(os.path.join(model, "config.json")) as f:
        config = json.load(f)
    config = config.get("text_config", config)
    kda = {i - 1 for i in config["linear_attn_config"]["kda_layers"]}  # 1-based
    return kda, config["num_hidden_layers"]


def breakdown(trace: str, model: str) -> list[dict[str, Any]]:
    kda, num_layers = layer_kinds(model)
    dsa = set(range(num_layers)) - kda
    events = [e for e in load_events(trace) if e.get("ph") == "X"]
    rows = []
    for forward in prefill_forwards(events):
        ops, tags = forward_ops(events, forward)
        times: dict[tuple[int | None, bool], float] = {}
        for op, key in zip(ops, attribute(tags), strict=True):
            times[key] = times.get(key, 0.0) + float(op.get("dur", 0.0))
        seen = {layer for layer, _ in times}
        if seen != set(range(num_layers)):
            raise SystemExit(
                f"{forward['name']}: ops attributed to layers {sorted(seen, key=str)}, "
                f"expected 0..{num_layers - 1}"
            )
        span = event_end(ops[-1]) - float(ops[0]["ts"])
        halves = {
            kind: tuple(
                sum(times.get((i, mlp), 0.0) for i in layers) / len(layers)
                for mlp in (False, True)
            )
            for kind, layers in (("dsa", dsa), ("kda", kda))
        }
        rows.append(
            {
                "forward": str(forward["name"])[len("prefill[") : -1],
                "span_us": span,
                "covered": sum(times.values()) / span,
                **halves,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("trace", help="Rank 0's *.pt.trace.json.gz.")
    parser.add_argument(
        "--model",
        default="/shared/data/amd_int/models/Kimi-K3.1-Pro",
        help="Model directory whose config.json names the KDA layers.",
    )
    args = parser.parse_args()
    print(
        "| forward | span (ms) | layer sums / span | DSA attn (µs) | DSA MLP (µs) "
        "| DSA layer (µs) | KDA attn (µs) | KDA MLP (µs) | KDA layer (µs) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in breakdown(args.trace, args.model):
        cells = [
            f"{t:.0f}" for kind in ("dsa", "kda") for t in (*row[kind], sum(row[kind]))
        ]
        print(
            f"| {row['forward']} | {row['span_us'] / 1e3:.1f} | {row['covered']:.2%} | "
            + " | ".join(cells)
            + " |"
        )


if __name__ == "__main__":
    main()
