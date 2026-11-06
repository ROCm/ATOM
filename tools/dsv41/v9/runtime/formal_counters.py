# ruff: noqa: S102
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Opt-in, host-only diagnostics. Install after the existing owner bootstrap.

Snapshots are requested while the client has drained its warmup/formal requests.
The worker lock excludes an in-progress ModelRunner.forward from a snapshot.
No GPU operations or file writes are added to a graph or per-step hot path.
"""

import ast
import collections
import copy
import functools
import inspect
import json
import os
import re
import textwrap
import threading
import time
from pathlib import Path

LOCK = threading.RLock()
STATE = None


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def bind_replay_observer(original, observe):
    source = textwrap.dedent(inspect.getsource(original))
    return bind_replay_source(source, original.__globals__, original.__name__, observe)


def bind_replay_source(source, namespace, name, observe):
    tree = ast.parse(source)
    before = copy.deepcopy(tree)
    body = tree.body[0].body
    matches = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and (node.value.func.value.id, node.value.func.attr) == ("graph", "replay")
    ]
    if len(matches) != 1:
        raise ValueError("OwnerServing.run replay location changed")
    index = matches[0] + 1
    body.insert(
        index,
        ast.parse("_p0_replayed(self, rows, metadata, original, key, capture)").body[0],
    )
    restored = copy.deepcopy(tree)
    del restored.body[0].body[index]
    assert ast.dump(restored) == ast.dump(before), "Observer changed execution"
    replay_call = body[matches[0]].value
    replay_call.func = ast.Name(id="_p2_trace_replay", ctx=ast.Load())
    replay_call.args = [
        ast.Name(id=n, ctx=ast.Load()) for n in ("self", "graph", "rows", "metadata")
    ]
    ns = dict(namespace, _p0_replayed=observe, _p2_trace_replay=tagged_replay)
    exec(compile(ast.fix_missing_locations(tree), "<p0-owner-observer>", "exec"), ns)
    return ns[name]


def tagged_replay(serving, graph, rows, metadata):
    if serving.runner.profiler is None:
        return graph.replay()
    import torch

    with torch.profiler.record_function(
        f"p2_owner_graph[capacity={rows.capacity} actual={metadata.step.scheduled}]"
    ):
        return graph.replay()


class Counters:
    def __init__(self, serving):
        self.serving = serving
        self.counts = collections.Counter()
        self.buckets = collections.Counter()
        self.contracts = {}

    def replayed(self, rows, metadata, original, key, capture):
        capacity = int(rows.capacity)
        actual = sum(int(span.length) for span in metadata.step.requests)
        global_actual = int(original.step.scheduled)
        assert 0 <= actual <= capacity and global_actual <= 4 * capacity
        self.counts.update(
            owner_graph_replays_submitted=1,
            owner_local_actual_rows=actual,
            owner_local_capacity_rows=capacity,
            owner_local_padding_rows=capacity - actual,
            owner_global_actual_rows=global_actual,
            owner_global_capacity_rows=4 * capacity,
            owner_capture_on_call=int(capture),
        )
        self.buckets[f"owner_local{capacity}_q{metadata.step.max_q_len}"] += 1
        if str(capacity) not in self.contracts:
            layers = []
            for layer in self.serving.model.layers:
                experts = layer.ffn.experts
                backend = experts._comm_fused_moe
                layers.append(
                    {
                        "layer": layer.attn.spec.layer_id,
                        "attention_layer_mode": str(layer.attn.spec.mode),
                        "ffn_fused_supported": bool(
                            backend is not None and backend.supports(4 * capacity)
                        ),
                        "prequantized_eligible": experts.supports_prequantized_input(),
                    }
                )
            self.contracts[str(capacity)] = {
                "layers": layers,
                "attention_environment": os.environ.get(
                    "ATOM_DSV41_OWNER_FP8_ATTENTION", "native"
                ),
                "scope": "static loaded/dispatch contract attached to real graph replay bucket; not per-kernel GPU counters",
            }

    def snapshot(self, request):
        return {
            "request": request,
            "pid": os.getpid(),
            "rank": self.serving.rank,
            "time_ns": time.time_ns(),
            "counts": dict(self.counts),
            "buckets": dict(self.buckets),
            "owner_counts": dict(self.serving.counters),
            "contracts": self.contracts,
            "scope": "cumulative CPU scheduling/submission counts; snapshot excludes active forward; not GPU-completion timestamps",
        }


def watch(root, stop):
    seen = None
    while not stop.wait(0.05):
        try:
            request = json.loads((root / "request.json").read_text())
            key = request["id"]
            if key == seen or re.fullmatch(r"[A-Za-z0-9_-]+", key) is None:
                continue
            with LOCK:
                if STATE is None:
                    continue
                data = STATE.snapshot(request)
            atomic_json(root / f"{key}_rank{data['rank']}.json", data)
            seen = key
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            atomic_json(
                root / f"observer_error_{os.getpid()}.json", {"error": repr(exc)}
            )
            return


def install():
    root_value = os.environ.get("P0_COUNTER_DIR")
    if root_value is None:
        return
    from atom.model_engine.model_runner import ModelRunner
    from atom.models.deepseek_v41.owner_service.owner_serving import OwnerServing
    from atom.spec_decode.draft_graph import DraftGraph

    if getattr(OwnerServing, "_p0_counter_installed", False):
        return
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    old_init, old_forward = OwnerServing.__init__, ModelRunner.forward
    old_draft = DraftGraph.run

    @functools.wraps(old_init)
    def init(self, *args, **kwargs):
        global STATE
        old_init(self, *args, **kwargs)
        if STATE is not None:
            raise RuntimeError("One P0 observer per worker")
        STATE = Counters(self)

    @functools.wraps(old_forward)
    def forward(self, batch):
        with LOCK:
            output = old_forward(self, batch)
            if STATE is not None and not batch.is_dummy_run:
                STATE.counts.update(
                    completed_runner_forwards=1,
                    scheduled_global_rows=int(batch.total_tokens_num),
                    scheduled_sequences=int(batch.total_seqs_num),
                    prefill_sequences=int(batch.total_seqs_num_prefill),
                )
            return output

    @functools.wraps(old_draft)
    def draft(self, running_bs, **staged):
        captured = running_bs in self._graph
        result = old_draft(self, running_bs, **staged)
        if STATE is not None:
            STATE.counts.update(
                draft_graph_calls=int(captured), draft_eager_calls=int(not captured)
            )
            STATE.buckets[f"draft_{self.name}_bs{running_bs}_graph{int(captured)}"] += 1
        return result

    def replayed(serving, rows, metadata, original, key, capture):
        assert STATE is not None and STATE.serving is serving
        STATE.replayed(rows, metadata, original, key, capture)

    OwnerServing.run = bind_replay_observer(OwnerServing.run, replayed)
    OwnerServing.__init__, ModelRunner.forward, DraftGraph.run = init, forward, draft
    OwnerServing._p0_counter_installed = True
    stop = threading.Event()
    threading.Thread(
        target=watch, args=(root, stop), daemon=True, name="p0-snapshots"
    ).start()


def counter_delta(before, after):
    result = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in before.keys() | after.keys()
    }
    if any(value < 0 for value in result.values()):
        raise ValueError("Counter reset/restart inside measurement interval")
    return result


def summarize(before, after):
    # MTP API aggregates DP, not duplicated TP worker counters. Derive AL from sums.
    a, b = before["mtp"], after["mtp"]
    if not a.get("enabled") or not b.get("enabled"):
        raise ValueError("MTP statistics unavailable")
    distribution = counter_delta(a["distribution"], b["distribution"])
    accepted = int(b["total_accepted_tokens"]) - int(a["total_accepted_tokens"])
    drafted = int(b["total_draft_tokens"]) - int(a["total_draft_tokens"])
    steps = sum(distribution.values())
    if (
        accepted < 0
        or drafted < 0
        or sum(int(k) * v for k, v in distribution.items()) != accepted
    ):
        raise ValueError("Inconsistent/reset MTP counters")
    workers = []
    for rank in range(4):
        x, y = before["workers"][rank], after["workers"][rank]
        if (x["pid"], x["rank"]) != (y["pid"], y["rank"]):
            raise ValueError("Worker restarted or ranks differ")
        workers.append(
            {
                "rank": rank,
                **{
                    field: counter_delta(x[field], y[field])
                    for field in ("counts", "buckets", "owner_counts")
                },
            }
        )
    global_rows = [w["counts"].get("owner_global_actual_rows", 0) for w in workers]
    if len(set(global_rows)) != 1:
        raise ValueError("TP ranks disagree about global rows")
    if (
        sum(w["counts"].get("owner_local_actual_rows", 0) for w in workers)
        != global_rows[0]
    ):
        raise ValueError("Owner rows do not partition the global rows")
    return {
        "formal_only": True,
        "accepted_bonus_tokens": accepted,
        "engine_reported_draft_tokens": drafted,
        "sequence_decode_steps": steps,
        "accept_length": 1 + accepted / steps if steps else None,
        "acceptance_rate": accepted / drafted if drafted else None,
        "distribution": distribution,
        "owner_global_actual_rows": global_rows[0],
        "workers": workers,
        "scope": "boundary deltas; TP global counters validated, not summed four times; AL is per sequence decode step",
    }
