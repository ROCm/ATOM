# SPDX-License-Identifier: MIT
"""Together-style acceptance of the actual host, on a model-free layer fixture.

CPU: configure/install plans in one process, exercise the real route and the
cross-layer pair lifecycle. GPU opt-in: capture two real ROCm graphs and replay
on fresh values. No timing or full serving accuracy claim is made here.
"""

import json
import os
import types

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes
from atom import plugin_transition as trans

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
                "model_arch": "DeepseekV41ForCausalLM",
                "strategy": "tp4-fixture",
                "graph": graph,
                "plugins": [{"name": OWNER, "kind": "layer", "serves": [seam]}],
                "plan": {"seams": {seam: [OWNER]}},
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


def test_same_process_hotplug_install_rejection_and_rollback(install, monkeypatch):
    p = plan(OUT)
    host.configure(None, hotplug=True, kill_switch=False)
    args = (torch.arange(32).reshape(4, 8).float(), None, torch.ones(8), None)
    bound = host.bind(OUT, lambda x, s, w, ws, *, dtype: (x * 2 + w, s))
    native, _ = bound(*args, dtype=torch.float32)

    def register(h):
        h.register(
            seam=OUT,
            contract=1,
            owner=OWNER,
            impl=lambda x, s, w, ws, **kw: (torch.add(x, x) + w, s),
        )

    entry = types.SimpleNamespace(load=lambda: register)
    monkeypatch.setattr(host, "_entry_points", lambda: {OWNER: entry})
    assert host.hotplug_plan(p)["plan_id"] == p.plan_id
    assert torch.equal(bound(*args, dtype=torch.float32)[0], native)
    assert host.route_proof(OUT, OWNER, 1) == 1
    with pytest.raises(ValueError, match="unknown arm"):
        host.hotplug_plan(p, arm="not-a-real-arm")
    assert torch.equal(bound(*args, dtype=torch.float32)[0], native)
    assert host.route_proof(OUT, OWNER, 2) == 2
    host.hotplug_plan(None)
    assert torch.equal(bound(*args, dtype=torch.float32)[0], native)
    assert host.route_proof(OUT, OWNER, 2) == 2
    record(
        "same-process-hotplug",
        plan=p.plan_id,
        plugin_hits=2,
        bad_plan_restored=True,
        removed_plan_native=True,
    )


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


def install_pair(
    install, *, decline="error", bad=False, mode="eager", graph="eager", marker=None
):
    def provider(h):
        def begin(x, shared, *, handle):
            handle["shape"] = x.shape
            return x.clone(), None if shared is None else shared.clone()

        def end(hidden, scale, *, project, output_sizes, handle):
            assert hidden.shape == handle["shape"]
            if marker is not None:
                marker.fill_(11)
            y, k = project(hidden, scale)
            return (y, k[:, :-1]) if bad else (y.clone(), k.clone())

        h.register_window(
            window=trans.WINDOW,
            contract=1,
            owner=OWNER,
            pairs=[(5, 6)],
            begin=begin,
            end=end,
            graph=graph,
        )

    p = plan(trans.SEAM, on_decline=decline, mode=mode)
    install(p, provider)
    return p.plan_id


def pair_forward(
    x, shared, *, agree=lambda token: [token] * 4, collective=lambda x: x * 4
):
    with trans.scope(
        eligible=lambda pairs: pairs == ((5, 6),), rows=len(x), agree=agree
    ):
        r, s = trans.begin(5, x, shared)
        combined = collective(r if s is None else r + s)
        # A mutable, observable next-layer state (mHC stand-in, not model output).
        next_state = combined.clone().add_(3)
        q, k = trans.end(
            6, lambda h, _: (h[:, :3] * 2, h[:, 3:] + 1), next_state, None, (3, 5)
        )
        return q, k, next_state


@pytest.mark.parametrize("shared_add", [False, True])
def test_window_pair_ab_state_and_uniform_decline(install, shared_add):
    pin = install_pair(install, decline="native")
    x = torch.arange(32).reshape(4, 8).float()
    shared = x + 1 if shared_add else None
    host.set_arm("native")
    ref = pair_forward(x, shared)
    host.set_arm("plugin")
    got = pair_forward(x, shared)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    assert host.route_proof(trans.SEAM + ".begin", OWNER) == 1
    assert host.route_proof(trans.SEAM + ".end", OWNER) == 1
    denied = pair_forward(x, shared, agree=lambda token: [token, token, None, token])
    assert all(torch.equal(a, b) for a, b in zip(ref, denied))
    assert host.route_proof(trans.SEAM + ".begin", OWNER) == 1
    assert host.last_decision().reason == "rank-plan-mismatch"
    record(
        "transition-pair-ab",
        plan=pin,
        shared_add=shared_add,
        correct=True,
        both_phases_hit=True,
        whole_pair_decline=True,
        collective="CPU stub",
    )


def test_window_strict_peer_refusal_stops_even_rank_without_active_route(install):
    pin = install_pair(install)
    host.set_arm("native")
    peer = ((pin, "plugin", OWNER, ((5, 6),), 4), True)
    with pytest.raises(host.PluginDeclined, match="rank-plan-mismatch"):
        with trans.scope(
            eligible=lambda _: False, rows=4, agree=lambda local: [local, peer]
        ):
            pytest.fail(
                "this rank must not enter native collective when a strict peer refuses"
            )
    assert trans._step.get() is None


def test_window_contract_failure_cleans_scope(install):
    install_pair(install, bad=True)
    x = torch.ones(4, 8)
    with pytest.raises(host.PluginContractError, match="two projected"):
        pair_forward(x, None)
    assert trans._step.get() is None
    host.set_arm("native")
    pair_forward(x, None)


def test_transition_capture_recipe_refuses_unsupported_runtime(monkeypatch):
    p = plan(trans.SEAM, mode="capture")
    monkeypatch.setenv("ATOM_V41_TRANSITION_AB", "1")
    cfg = types.SimpleNamespace(
        tensor_parallel_size=4,
        compilation_config=types.SimpleNamespace(level=0, cudagraph_mode="FULL"),
    )
    recipes.check_runtime(p, cfg)
    cfg.compilation_config.cudagraph_mode = "TR_HYBRID"
    with pytest.raises(recipes.RecipeRefused, match="recipe-window-topology"):
        recipes.check_runtime(p, cfg)
    cfg.compilation_config.cudagraph_mode = "FULL"
    cfg.enable_dp_attention = True
    with pytest.raises(recipes.RecipeRefused, match="recipe-window-topology"):
        recipes.check_runtime(p, cfg)


def test_capture_window_agrees_once_and_detects_stale_plan(install, monkeypatch):
    install_pair(install, mode="capture", graph="capture_safe")
    monkeypatch.setattr(host, "_graph_state", lambda: (True, 4))
    monkeypatch.setattr(host, "_stream_capturing", lambda: False)
    x = torch.ones(4, 8)
    votes = []

    def agree(vote):
        votes.append(vote)
        return [vote] * 4

    with trans.capture_transaction(4):
        warmup = pair_forward(x, None, agree=agree)
        monkeypatch.setattr(host, "_stream_capturing", lambda: True)
        captured = pair_forward(x, None, agree=agree)
        assert all(torch.equal(a, b) for a, b in zip(warmup, captured))
        assert host.last_decision().captured and host.last_decision().bucket == 4
        assert len(votes) == 1
        host.set_arm("native")
        with pytest.raises(host.PluginContractError, match="plan changed"):
            pair_forward(x, None, agree=agree)
    assert len(votes) == 1
    with (
        trans.capture_transaction(4),
        pytest.raises(host.PluginContractError, match="requires a warmup vote"),
    ):
        pair_forward(x, None, agree=agree)


def test_capture_window_eager_only_provider_falls_back_as_pair(install, monkeypatch):
    install_pair(install, mode="capture", graph="eager", decline="native")
    monkeypatch.setattr(host, "_graph_state", lambda: (True, 4))
    x = torch.ones(4, 8)
    with trans.capture_transaction(4):
        pair_forward(x, x)
    decision = host.last_decision()
    assert (
        decision.seam == trans.SEAM and decision.route == "native" and decision.captured
    )


@pytest.mark.skipif(
    os.getenv("ATOM_TOGETHER_WORLD") != "4", reason="requires torchrun on four GPUs"
)
def test_real_tp4_window_same_outputs_and_one_collective(install):
    from datetime import timedelta

    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=75))
    cpu_group = None
    try:
        assert dist.get_world_size() == torch.cuda.device_count() == 4
        cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=50))
        pin = install_pair(install, decline="native")
        x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) + rank
        counts = 0

        def collective(inp):
            nonlocal counts
            out = inp.clone()
            dist.all_reduce(out)
            counts += 1
            return out

        def agreement(token):
            got = [None] * 4
            dist.all_gather_object(got, token, group=cpu_group)
            return got

        for shared in (None, x + 1):
            host.set_arm("native")
            ref = pair_forward(x, shared, agree=agreement, collective=collective)
            host.set_arm("plugin")
            got = pair_forward(x, shared, agree=agreement, collective=collective)
            assert all(torch.equal(a, b) for a, b in zip(ref, got))
            denied = pair_forward(
                x,
                shared,
                agree=lambda token: agreement(None if rank == 2 else token),
                collective=collective,
            )
            assert all(torch.equal(a, b) for a, b in zip(ref, denied))
        assert counts == 6
        assert host.route_proof(trans.SEAM + ".begin", OWNER, 2) == 2
        assert host.route_proof(trans.SEAM + ".end", OWNER, 2) == 2
        record(
            "real-tp4-transition",
            plan=pin,
            rank=rank,
            matched=True,
            collective_calls=counts,
            whole_pair_decline=True,
            both_phases_hit=2,
            scope="actual host + RCCL; fixture mHC/project",
        )
    finally:
        if cpu_group is not None:
            dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("ATOM_TOGETHER_WORLD") != "4", reason="requires torchrun on four GPUs"
)
def test_real_tp4_window_capture(install, monkeypatch):
    from datetime import timedelta

    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=75))
    cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=50))
    graphs = []
    try:
        marker = torch.zeros(1, device="cuda", dtype=torch.int32)
        pin = install_pair(install, mode="capture", graph="capture_safe", marker=marker)
        monkeypatch.setattr(host, "_graph_state", lambda: (True, 4))
        static = torch.ones(4, 8, device="cuda")
        p = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8)
        votes = 0

        def agreement(vote):
            nonlocal votes
            assert not torch.cuda.is_current_stream_capturing()
            seen = [None] * 4
            dist.all_gather_object(seen, vote, group=cpu)
            votes += 1
            return seen

        def collective(x):
            out = x.clone()
            dist.all_reduce(out)
            return out

        captures = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for arm in ("native", "plugin"):
                host.set_arm(arm)
                g = torch.cuda.CUDAGraph()
                graphs.append(g)
                with trans.capture_transaction(4):
                    pair_forward(static, None, agree=agreement, collective=collective)
                    # Keep the watchdog on other threads from invalidating the graph.
                    with torch.cuda.graph(
                        g, stream=stream, capture_error_mode="thread_local"
                    ):
                        out = pair_forward(
                            static, None, agree=agreement, collective=collective
                        )
                captures[arm] = g, out
        torch.cuda.current_stream().wait_stream(stream)
        assert votes == 2
        assert host.route_proof(trans.SEAM + ".begin", OWNER, 2) == 2
        assert host.route_proof(trans.SEAM + ".end", OWNER, 2) == 2
        hits = host.decision_counts()
        host.set_arm("native")
        for delta in (1, 7):
            static.copy_(p + rank + delta)
            expected = 4 * (p + delta) + 6 + 3
            for arm, (g, (q, kv, state)) in captures.items():
                marker.zero_()
                g.replay()
                assert torch.equal(state, expected)
                assert torch.equal(q, expected[:, :3] * 2)
                assert torch.equal(kv, expected[:, 3:] + 1)
                assert marker.item() == (11 if arm == "plugin" else 0)
        assert votes == 2 and host.decision_counts() == hits
        record(
            "real-tp4-window-capture",
            plan=pin,
            rank=rank,
            host_votes=votes,
            recorded_arms=["native", "plugin"],
            both_phases_captured=True,
            replay_steps=4,
            replay_redispatch=0,
            replay_device_marker=True,
        )
    finally:
        # A graph owns references to its RCCL communicator. Release the graph
        # first or ProcessGroupNCCL shutdown can wait for its own live graph.
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group(cpu)
        dist.destroy_process_group()
