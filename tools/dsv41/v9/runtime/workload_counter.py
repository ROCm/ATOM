# ruff: noqa: I001, S102
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Diagnostic GPU ring: one scalar store per layer, one step advance.

No per-step CPU readback. The formal boundary snapshot waits for completion.
Only owner target FFNs are observed; prefill/draft are excluded.
"""

import ast
import copy
import functools
import importlib
import inspect
import textwrap

import torch
import triton as tr
import triton.language as tl

SIZE = 16384
CURRENT = None


@tr.jit
def write_valid(valid, ring, cursor, layer: tl.constexpr, size: tl.constexpr):
    step = tl.load(cursor) % size
    tl.store(ring + step * 40 + layer, tl.load(valid))


@tr.jit
def advance(cursor):
    tl.store(cursor, tl.load(cursor) + 1)


class WorkloadRing:
    def __init__(self):
        self.buffers = {}
        self.bindings = {}

    def ensure(self, capacity):
        if capacity not in self.buffers:
            assert not torch.cuda.is_current_stream_capturing()
            self.buffers[capacity] = (
                torch.full((SIZE, 40), -1, device="cuda", dtype=torch.int32),
                torch.zeros(1, device="cuda", dtype=torch.int64),
            )
        return self.buffers[capacity]

    def record(self, layer, capacity, scope):
        ring, cursor = self.ensure(capacity)
        meta = scope["metadata"]
        binding = {
            "dispatch": {
                k: repr(getattr(meta, k))
                for k in ("stage1", "stage2", "block_m", "ksplit", "prequant")
            },
            "shapes": {
                k: list(scope[k].shape) for k in ("hidden_states", "a1", "w1", "w2")
            },
            "dimensions": {
                k: int(scope[k])
                for k in (
                    "token_num",
                    "model_dim",
                    "inter_dim",
                    "topk",
                    "hidden_pad",
                    "intermediate_pad",
                )
            },
        }
        key = f"{capacity}/{layer}"
        if key in self.bindings:
            assert self.bindings[key] == binding, (key, self.bindings[key], binding)
        else:
            self.bindings[key] = binding
        write_valid[(1,)](
            scope["num_valid_ids"], ring, cursor, layer, SIZE, num_warps=1
        )

    def finish(self, capacity):
        _, cursor = self.ensure(capacity)
        advance[(1,)](cursor, num_warps=1)

    def snapshot(self):
        if self.buffers:
            device = next(iter(self.buffers.values()))[0].device
            torch.cuda.synchronize(device=device)
        buffers = {}
        for cap, (ring, cursor) in self.buffers.items():
            index = int(cursor.item())
            rows = ring[: min(index, SIZE)].cpu().tolist()
            buffers[str(cap)] = {"cursor": index, "rows": rows, "size": SIZE}
        return {
            "buffers": buffers,
            "bindings": self.bindings,
            "scope": "Actual owner target FFN GPU executions, including eager capture priming; boundary deltas exclude warmup. All static packet rows included; ring scalar stores perturb timing.",
        }


def observe(scope):
    if CURRENT is not None:
        state, layer, capacity = CURRENT
        assert 0 <= layer < 40
        state.record(layer, capacity, scope)


def set_layer(serving, layer, capacity):
    global CURRENT
    CURRENT = (serving._workload_ring, layer, capacity)


def instrument(original, call, where):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    old = copy.deepcopy(tree)
    inserted = []

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            n = node.value
            match = (
                where == "target"
                and isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and (n.func.value.id, n.func.attr) == ("Block", "prepare_ffn")
            )
            match = match or (
                where == "moe"
                and any(
                    isinstance(t, ast.Name) and t.id == "_stage1_call"
                    for t in node.targets
                )
            )
            if match:
                add = ast.parse(call).body[0]
                inserted.append(ast.dump(add))
                return [add, node]
            return node

    tree = Insert().visit(tree)
    assert len(inserted) == 1

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if ast.dump(node) in inserted else node

    assert ast.dump(Strip().visit(copy.deepcopy(tree))) == ast.dump(old)
    ns = dict(
        original.__globals__, _workload_observe=observe, _workload_layer=set_layer
    )
    exec(compile(ast.fix_missing_locations(tree), "<owner-workload-ring>", "exec"), ns)
    return functools.wraps(original)(ns[original.__name__])


def install():
    from atom.models.deepseek_v41.owner_service.owner_serving import (
        OwnerServing,
    )
    from formal_counters import Counters

    if getattr(OwnerServing, "_workload_installed", False):
        return
    mod = importlib.import_module("aiter.fused_moe")
    mod.fused_moe_2stages = instrument(
        mod.fused_moe_2stages, "_workload_observe(locals())", "moe"
    )
    target = instrument(
        OwnerServing.target, "_workload_layer(self, i, capacity)", "target"
    )

    @functools.wraps(target)
    def wrapped(self, *args, **kwargs):
        global CURRENT
        previous = CURRENT
        if not hasattr(self, "_workload_ring"):
            self._workload_ring = WorkloadRing()
        try:
            result = target(self, *args, **kwargs)
            self._workload_ring.finish(args[1].step.width)
            return result
        finally:
            CURRENT = previous

    OwnerServing.target = wrapped
    old = Counters.snapshot

    def snapshot(self, request):
        data = old(self, request)
        data["expert_workload"] = (
            self.serving._workload_ring.snapshot()
            if hasattr(self.serving, "_workload_ring")
            else {"buffers": {}, "bindings": {}}
        )
        return data

    Counters.snapshot = snapshot
    OwnerServing._workload_installed = True


def extract_interval(before, after):
    result = {}
    for cap, end in after["buffers"].items():
        start = before["buffers"].get(cap, {"cursor": 0})["cursor"]
        stop = end["cursor"]
        size = end["size"]
        if not 0 <= stop - start <= size:
            raise ValueError("Ring reset/overflow in formal interval")
        rows = [end["rows"][i % size] for i in range(start, stop)]
        if any(
            len(r) != 40
            or any(
                v < 0 or v % int(after["bindings"][f"{cap}/{i}"]["dispatch"]["block_m"])
                for i, v in enumerate(r)
            )
            for r in rows
        ):
            raise ValueError("Incomplete or invalid expert row counters")
        result[cap] = {
            "first_cursor": start,
            "after_cursor": stop,
            "steps": len(rows),
            "rows": rows,
            "layer_sums": [sum(r[i] for r in rows) for i in range(40)],
        }
    return result
