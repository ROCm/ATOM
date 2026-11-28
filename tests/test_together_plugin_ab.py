# SPDX-License-Identifier: MIT
"""Together-style acceptance of the actual host, on a model-free layer fixture.

CPU: configure a plan and switch its arms in one process on the real route.
GPU opt-in: capture two real ROCm graphs (one per arm) and replay on fresh
values. No timing or full serving accuracy claim is made here.
"""

import json
import os
import types

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes

OUT = "v41.attn.out"
OWNER = "together-test"


def plan(seam, *, mode="eager", on_decline="error"):
    graph = {"mode": mode}
    if mode == "capture":
        graph["buckets"] = [4]
    return recipes.resolve(
        [
            {
                "schema": recipes.RECIPE_SCHEMA,
                "arch": "DeepseekV41ForCausalLM",
                "strategy": "tp4-fixture",
                "graph": graph,
                "plugins": [{"name": OWNER, "kind": "layer", "serves": [seam]}],
                "seams": {seam: [OWNER]},
                "defaults": {"on_decline": on_decline},
            }
        ]
    )


@pytest.fixture
def install(monkeypatch):
    monkeypatch.setattr(host, "_configure_aiter", lambda names: None)
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    host._reset_for_testing()

    def configure(p, register):
        entry = types.SimpleNamespace(load=lambda: register)
        monkeypatch.setattr(host, "_entry_points", lambda: {OWNER: entry})
        host.configure(p, kill_switch=False)

    yield configure
    host._reset_for_testing()


def record(name, **items):
    entry = {"case": name, "pid": os.getpid(), "result": "pass", **items}
    print("TOGETHER_EVIDENCE " + json.dumps(entry, sort_keys=True), flush=True)


def bind_out(install, mode):
    def provider(h):
        def candidate(x, scale, weight, weight_scale, *, dtype, capability, valid_rows):
            return torch.add(x, x) + weight, scale

        h.register(
            seam=OUT, contract=1, owner=OWNER, impl=candidate, graph="capture_safe"
        )

    p = plan(OUT, mode=mode)
    install(p, provider)
    native = lambda x, scale, weight, weight_scale, *, dtype: (x * 2 + weight, scale)
    return host.bind(OUT, native), p.plan_id


def test_actual_host_same_input_abba_and_aa(install):
    bound, pin = bind_out(install, "eager")
    args = (
        torch.arange(32, dtype=torch.float32).reshape(4, 8),
        None,
        torch.ones(8),
        None,
    )
    outputs = []
    for arm in ("native", "plugin", "plugin", "native"):
        host.set_arm(arm)
        y, _ = bound(*args, dtype=torch.float32)
        outputs.append(y)
    assert all(torch.equal(outputs[0], x) for x in outputs[1:])
    assert torch.equal(args[0], torch.arange(32).reshape(4, 8).float())
    assert host.route_proof(OUT, OWNER, 2) == 2
    record("host-abba", plan=pin, plugin_hits=2, correct=True, timing="not collected")


@pytest.mark.skipif(
    os.getenv("ATOM_TOGETHER_GPU_TEST") != "1" or not torch.cuda.is_available(),
    reason="explicit ROCm/CUDA graph test opt-in required",
)
def test_real_graph_capture_is_per_arm_and_replay_does_not_redispatch(
    install, monkeypatch
):
    bound, pin = bind_out(install, "capture")
    monkeypatch.setattr(
        host, "_graph_state", lambda: (torch.cuda.is_current_stream_capturing(), 4)
    )
    x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
    weight = torch.ones(8, device="cuda")
    graphs = {}
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            bound(x, None, weight, None, dtype=x.dtype)
    torch.cuda.current_stream().wait_stream(warm)
    for arm in ("native", "plugin"):
        host.set_arm(arm)
        static = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, _ = bound(static, None, weight, None, dtype=static.dtype)
        graphs[arm] = (graph, static, out)
    assert host.route_proof(OUT, OWNER, 1) == 1
    before = host.decision_counts()
    # Change host arm and both graphs' inputs. Replay must run the recorded arm,
    # not call Python or select a different provider.
    host.set_arm("native")
    for delta in (1, 7):
        gold = (x + delta) * 2 + weight
        for graph, static, out in graphs.values():
            static.copy_(x + delta)
            graph.replay()
            assert torch.equal(out, gold)
    assert host.decision_counts() == before
    record(
        "real-graph-ab",
        plan=pin,
        captured_arms=list(graphs),
        plugin_capture_hits=1,
        replay_steps=4,
        replay_redispatch=0,
        hip=torch.version.hip,
        correct=True,
    )
