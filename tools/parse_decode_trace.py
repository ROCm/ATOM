#!/usr/bin/env python3
"""Per-layer breakdown of the ``decode[`` graph replays in a rank-0 trace.

Reads the trace ``tools/k31_decode_profile.py`` writes. A forward is the
kernels of the graph a ``decode[`` span launched. Replayed kernels carry
no module tags, so ``match_replay_to_warmup`` matches them to the eager warmup
forward recorded before that graph's capture (the ``capture_traces/`` file next
to the run trace), and each warmup kernel goes to a layer and to its attention
or MLP half by its module tag, as ``parse_prefill_trace.attribute`` does for a
prefill.

Per forward it prints the span of the main-stream kernels, the share of it the
per-layer sums of their time cover, and the mean per-layer time of each half
of the DSA and KDA layers; then, for each layer type, the main-stream kernels
of its median layer with their mean time over the forwards.
"""

from __future__ import annotations

import argparse
from typing import Any

from parse_prefill_trace import attribute, layer_kinds
from parse_trace import (
    build_warmup_mapping,
    decode_batch_sizes,
    decode_query_len,
    event_end,
    find_capture_graph_for_bs,
    load_events,
    match_replay_to_warmup,
    primary_replay_stream,
    resolve_capture_trace,
    short,
    warmup_window_for_graph,
)


def decode_forwards(
    events: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    return sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("decode[")
            and decode_batch_sizes(e)[0] == batch_size
        ),
        key=lambda e: float(e["ts"]),
    )


def graph_kernels(
    events: list[dict[str, Any]], forward: dict[str, Any]
) -> list[dict[str, Any]]:
    """The kernels of the graph launched in *forward*'s CPU span.

    At TP > 1 the head's GEMM and vocab all-gather run eagerly after the graph.
    HIP runs the graph's memset and memcpy nodes as ``__amd_rocclr_*`` kernels;
    the warmup recorded those ops as memory operations, not kernels.
    """
    launches = {
        e["args"].get("correlation")
        for e in events
        if e.get("cat") == "cuda_runtime"
        and e.get("name") == "hipGraphLaunch"
        and forward["ts"] <= e["ts"] < event_end(forward)
    }
    return sorted(
        (
            e
            for e in events
            if e.get("cat") == "kernel"
            and (e.get("args") or {}).get("correlation") in launches
            and not str(e.get("name")).startswith("__amd_rocclr_")
        ),
        key=lambda e: float(e["ts"]),
    )


def warmup_template(
    trace: str, forward: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """The capture trace of the graph *forward* replayed, and its warmup kernels."""
    scheduled, graph_bs = decode_batch_sizes(forward)
    path = resolve_capture_trace(trace, graph_bs, decode_query_len(forward, scheduled))
    capture = load_events(path)
    graph = find_capture_graph_for_bs(capture, graph_bs)
    return path, build_warmup_mapping(capture, *warmup_window_for_graph(capture, graph))


def breakdown(
    trace: str, model: str, batch_size: int, skip: int, count: int
) -> dict[str, Any]:
    kda, num_layers = layer_kinds(model)
    kinds = {"dsa": sorted(set(range(num_layers)) - kda), "kda": sorted(kda)}
    events = [e for e in load_events(trace) if e.get("ph") == "X"]
    forwards = decode_forwards(events, batch_size)[skip : skip + count]
    if len(forwards) < count:
        raise SystemExit(
            f"{trace}: {len(forwards)} decode[bs={batch_size} forwards after "
            f"skipping {skip}, {count} wanted"
        )
    capture, mapping = warmup_template(trace, forwards[0])
    keys = attribute([row["module"] for row in mapping])
    kernel_us = [0.0] * len(mapping)
    replayed = [row["kernel"] for row in mapping]
    rows = []
    for forward in forwards:
        if decode_batch_sizes(forward)[1] != decode_batch_sizes(forwards[0])[1]:
            raise SystemExit(f"{forward['name']} replayed another graph")
        kernels = graph_kernels(events, forward)
        matched = match_replay_to_warmup(kernels, mapping)
        unmatched = [r["kernel_name"] for r in matched if r["warmup_index"] < 0]
        if unmatched:
            raise SystemExit(
                f"{forward['name']}: {len(unmatched)} kernels not in {capture}, "
                f"first {unmatched[0]}"
            )
        stream = primary_replay_stream(kernels)
        main = [k for k in kernels if (k.get("args") or {}).get("stream") == stream]
        times: dict[tuple[int | None, bool], float] = {}
        for r in matched:
            if r["stream"] == stream:
                key = keys[r["warmup_index"]]
                times[key] = times.get(key, 0.0) + r["duration_us"]
                kernel_us[r["warmup_index"]] += r["duration_us"] / count
                replayed[r["warmup_index"]] = r["kernel_name"]
        span = event_end(main[-1]) - float(main[0]["ts"])
        rows.append(
            {
                "forward": str(forward["name"])[len("decode[") : -1],
                "span_us": span,
                "covered": sum(times.values()) / span,
                "times": times,
            }
        )

    def mean(layer: int, mlp: bool) -> float:
        return sum(row["times"].get((layer, mlp), 0.0) for row in rows) / count

    medians = {}
    for kind, layers in kinds.items():
        layer = sorted(layers, key=lambda i: mean(i, False) + mean(i, True))[
            len(layers) // 2
        ]
        medians[kind] = (
            layer,
            [
                (mlp, row["module"], replayed[i], kernel_us[i])
                for i, (row, (owner, mlp)) in enumerate(zip(mapping, keys, strict=True))
                if owner == layer and kernel_us[i] > 0
            ],
        )
    for row in rows:
        row.update(
            {
                kind: tuple(
                    sum(row["times"].get((i, mlp), 0.0) for i in layers) / len(layers)
                    for mlp in (False, True)
                )
                for kind, layers in kinds.items()
            }
        )
    return {"capture": capture, "rows": rows, "medians": medians}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("trace", help="Rank 0's *.pt.trace.json.gz.")
    parser.add_argument(
        "--model",
        default="/shared/data/amd_int/models/Kimi-K3.1-Pro",
        help="Model directory whose config.json names the KDA layers.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--skip", type=int, default=2, help="Leading decode forwards to drop."
    )
    parser.add_argument("--forwards", type=int, default=20)
    args = parser.parse_args()
    result = breakdown(
        args.trace, args.model, args.batch_size, args.skip, args.forwards
    )
    rows = result["rows"]
    print(f"Capture trace: {result['capture']}\n")
    print(
        "| forward | span (µs) | layer sums / span | DSA attn (µs) | DSA MLP (µs) "
        "| DSA layer (µs) | KDA attn (µs) | KDA MLP (µs) | KDA layer (µs) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    mean_row = {
        "forward": f"mean of {len(rows)}",
        **{
            key: sum(row[key] for row in rows) / len(rows)
            for key in ("span_us", "covered")
        },
        **{
            kind: tuple(sum(row[kind][h] for row in rows) / len(rows) for h in (0, 1))
            for kind in ("dsa", "kda")
        },
    }
    for row in [*rows, mean_row]:
        cells = [
            f"{t:.1f}" for kind in ("dsa", "kda") for t in (*row[kind], sum(row[kind]))
        ]
        print(
            f"| {row['forward']} | {row['span_us']:.0f} | {row['covered']:.2%} | "
            + " | ".join(cells)
            + " |"
        )
    for kind, (layer, kernels) in result["medians"].items():
        print(f"\n{kind.upper()} median layer {layer}:\n")
        print("| half | module | kernel | mean (µs) |")
        print("|---|---|---|---:|")
        for mlp, module, kernel, us in kernels:
            print(
                f"| {'MLP' if mlp else 'attn'} | `{short(module, 60)}` "
                f"| `{short(kernel, 90)}` | {us:.1f} |"
            )


if __name__ == "__main__":
    main()
