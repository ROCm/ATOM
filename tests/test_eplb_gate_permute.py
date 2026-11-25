# SPDX-License-Identifier: MIT
# Tests for gate-permute EPLB (ATOM_EPLB_GATE_PERMUTE,
# atom/model_ops/eplb_gate_permute.py) and for what stays as it was with the
# switches unset.

import ast
import json
import logging
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.config import EPLBConfig
    from atom.model_ops import eplb
    from atom.model_ops import eplb_gate_permute as gp
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

ROOT = Path(__file__).resolve().parent.parent
SWITCHES = (
    "ATOM_EPLB_GATE_PERMUTE",
    "ATOM_EPLB_ALL_LAYERS",
    "ATOM_EPLB_DECODE_SAMPLE_EVERY",
    "ATOM_EPLB_FIRST_REBALANCE_STEPS",
    "ATOM_EPLB_MAX_REBALANCES",
    "ATOM_EPLB_MIN_PICKS",
)
LAYERS, EXPERTS, RANKS, TOP_K = 3, 16, 4, 3
PER = EXPERTS // RANKS


@pytest.fixture(autouse=True)
def clean_switches(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(eplb, "_MANAGER", None)
    monkeypatch.setattr(eplb, "_MONITOR", None)


def method_source(path: str, cls: str, name: str) -> str:
    """The source of ``cls.name`` in ``path``: a module that needs a GPU to
    import still has its method run here, on stand-ins."""
    text = (ROOT / path).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    start = (
                        item.decorator_list[0].lineno
                        if item.decorator_list
                        else item.lineno
                    )
                    lines = text.splitlines()[start - 1 : item.end_lineno]
                    return textwrap.dedent("\n".join(lines))
    raise AssertionError(f"{cls}.{name} not found in {path}")


# ---------------------------------------------------------------- stand-ins
class _Gate:
    def __init__(self, experts=EXPERTS, hidden=8, seed=0, bias_vl=True):
        g = torch.Generator().manual_seed(seed)
        # Quarters, against integer rows: a logit is exact whatever the order
        # its products are summed in, so a moved row scores bit for bit alike.
        self.weight = torch.nn.Parameter(
            torch.randint(-8, 9, (experts, hidden), generator=g).float() / 4
        )
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.randn(experts, generator=g)
        )
        if bias_vl:
            self.bias_vl = torch.nn.Parameter(torch.randn(experts, generator=g))


class _Method:
    eplb_gate_permute_capable = True


def _layer(layer_id, gate=None, experts=EXPERTS):
    return SimpleNamespace(
        layer_id=layer_id,
        prefix=f"layers.{layer_id}.ffn.experts",
        top_k=TOP_K,
        global_num_experts=experts,
        num_redundant_experts=0,
        quant_method=_Method(),
        expert_layout=SimpleNamespace(
            shared_is_routed=False, num_fused_shared_experts=0
        ),
        moe_parallel_config=SimpleNamespace(pcp_merge=False, use_all2all_kernels=True),
        dp_size=RANKS,
        e_score_correction_bias=None if gate is None else gate.e_score_correction_bias,
    )


class _Model:
    def __init__(self, blocks):
        self._blocks = blocks

    def modules(self):
        return list(self._blocks)


def _setup(monkeypatch, *, rank=1, layer_ids=(0, 2), **env):
    """A gate-permute runtime over layers ``layer_ids`` of a 3-layer model on
    EP rank ``rank`` of 4, counting with torch ops (as without Triton)."""
    monkeypatch.setattr(gp, "_HAS_TRITON", False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=LAYERS, num_logical_experts=EXPERTS, ep_size=RANKS, ep_rank=rank
    )
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=4)
    monitor.initialize_for_metadata(meta)
    monitor.set_load_group(SimpleNamespace(world_size=1))
    gates = {i: _Gate(seed=i) for i in layer_ids}
    layers = {i: _layer(i, gates[i]) for i in layer_ids}
    blocks = [SimpleNamespace(gate=gates[i], experts=layers[i]) for i in layer_ids]
    manager = SimpleNamespace(rebalancing_stopped=False)
    runtime = gp.GatePermuteRuntime(
        manager=manager,
        monitor=monitor,
        layers=layers,
        gates=gp.collect_gates(_Model(blocks), layers),
        meta=meta,
    )
    monitor.use_logical_window(runtime.fold_index, runtime.pf_total)
    return SimpleNamespace(
        meta=meta,
        monitor=monitor,
        gates=gates,
        layers=layers,
        manager=manager,
        gp=runtime,
    )


def _commit(env, layer_id, new_p2l):
    """Move ``layer_id`` to ``new_p2l`` as a rebalance does: maps, then gate."""
    meta = env.meta
    old = meta.physical_to_logical_map[layer_id].clone()
    p2l = meta.physical_to_logical_map.clone()
    p2l[layer_id] = new_p2l.to(torch.int32)
    l2p = torch.empty_like(p2l)
    for i in range(p2l.shape[0]):
        l2p[i, p2l[i].long()] = torch.arange(EXPERTS, dtype=torch.int32)
    new_meta = eplb.ExpertLocationMetadata.from_rebalance_result(
        physical_to_logical_map=p2l,
        logical_to_physical_map=l2p.unsqueeze(-1),
        logical_replica_count=torch.ones_like(p2l),
        ep_size=meta.ep_size,
        ep_rank=meta.ep_rank,
        max_num_replicas=1,
    )
    staged = env.gp.stage_commit(layer_id, old, p2l[layer_id])
    meta.update(new_meta, [layer_id])
    env.gp.apply_commit(layer_id, staged)


def _route(gate, x, bias="e_score_correction_bias"):
    """Top-k ids and weights of rows ``x`` as the V4.1 router picks them: the
    bias picks, the unbiased score weighs."""
    score = torch.sqrt(torch.nn.functional.softplus(x @ gate.weight.data.T))
    ids = (score + getattr(gate, bias).data).topk(TOP_K, dim=-1).indices
    weights = score.gather(1, ids)
    return ids, weights / weights.sum(-1, keepdim=True)


def _logical_counts(ids, experts=EXPERTS):
    flat = ids.reshape(-1)
    return torch.bincount(flat[flat >= 0], minlength=experts)


ROLL = torch.arange(EXPERTS).roll(1)  # every slot moves on by one


def _cycle3():
    """Slots 2, 7, 11 trade experts in a 3-cycle, which is not its own
    inverse (slot -> logical differs from logical -> slot); the rest stay."""
    p2l = torch.arange(EXPERTS)
    p2l[2], p2l[7], p2l[11] = 7, 11, 2
    l2p = torch.empty_like(p2l)
    l2p[p2l] = torch.arange(EXPERTS)
    assert not torch.equal(l2p, p2l)
    return p2l


# ---------------------------------------------------------------- gate rows
def _permute_rows(tensors, old_p2l, new_p2l):
    """`stage_rows`, written in place as a rebalance writes it."""
    staged = gp.stage_rows(tensors, old_p2l, new_p2l)
    for tensor, rows in zip(tensors, staged or ()):
        tensor.copy_(rows)
    return staged is not None


def test_permuted_gate_routes_to_the_slots_of_the_same_experts():
    gate = _Gate()
    x = torch.randint(
        -8, 9, (64, 8), generator=torch.Generator().manual_seed(1)
    ).float()
    want = {b: _route(gate, x, b) for b in ("e_score_correction_bias", "bias_vl")}
    tensors = gp.gate_row_tensors(gate)
    assert len(tensors) == 3
    addresses = [t.data_ptr() for t in tensors]
    params = [gate.weight, gate.e_score_correction_bias, gate.bias_vl]

    new_p2l = torch.randperm(EXPERTS, generator=torch.Generator().manual_seed(2))
    l2p = torch.empty_like(new_p2l)
    l2p[new_p2l] = torch.arange(EXPERTS)
    assert _permute_rows(tensors, torch.arange(EXPERTS), new_p2l)

    for bias, (ids, weights) in want.items():
        got_ids, got_weights = _route(gate, x, bias)
        assert torch.equal(got_ids, l2p[ids])  # physical = l2p[logical]
        assert torch.equal(new_p2l[got_ids], ids)
        assert torch.equal(got_weights, weights)
    assert [t.data_ptr() for t in gp.gate_row_tensors(gate)] == addresses
    assert params == [gate.weight, gate.e_score_correction_bias, gate.bias_vl]


def test_two_permutations_compose():
    gen = torch.Generator().manual_seed(3)
    first = torch.randperm(EXPERTS, generator=gen)
    second = torch.randperm(EXPERTS, generator=gen)
    stepwise, direct, pristine = _Gate(seed=5), _Gate(seed=5), _Gate(seed=5)
    identity = torch.arange(EXPERTS)
    _permute_rows(gp.gate_row_tensors(stepwise), identity, first)
    # from the live placement, with no pristine copy at hand
    _permute_rows(gp.gate_row_tensors(stepwise), first, second)
    _permute_rows(gp.gate_row_tensors(direct), identity, second)
    for a, b, c in zip(
        gp.gate_row_tensors(stepwise),
        gp.gate_row_tensors(direct),
        gp.gate_row_tensors(pristine),
    ):
        assert torch.equal(a, b)
        assert torch.equal(a, c[second])  # row p is logical second[p]'s
    # and back
    _permute_rows(gp.gate_row_tensors(stepwise), second, identity)
    assert torch.equal(stepwise.weight.data, pristine.weight.data)


def test_same_placement_writes_nothing():
    gate = _Gate(bias_vl=False)
    before = gate.weight.data.clone()
    p2l = torch.arange(EXPERTS).flip(0)
    assert len(gp.gate_row_tensors(gate)) == 2
    assert gp.stage_rows(gp.gate_row_tensors(gate), p2l, p2l.clone()) is None
    assert not _permute_rows(gp.gate_row_tensors(gate), p2l, p2l.clone())
    assert torch.equal(gate.weight.data, before)
    with pytest.raises(ValueError, match="permutation"):
        gp.stage_rows(gp.gate_row_tensors(gate), p2l, torch.zeros(EXPERTS))
    assert torch.equal(gate.weight.data, before)


# ---------------------------------------------------------------- routing
def test_place_counts_a_prefill_pass_and_launches_nothing_in_a_decode_capture(
    monkeypatch,
):
    env = _setup(monkeypatch)
    layer = env.layers[2]
    ids = torch.tensor([[0, 5, 15], [5, 5, 9], [-1, -1, -1]], dtype=torch.int32)
    env.monitor.on_forward_start()
    env.gp.place(layer, ids)
    count = env.monitor._cur_pass_count
    assert count[2].tolist() == _logical_counts(ids).tolist()
    assert int(count[0].sum()) == 0

    # inside a decode request-graph recording: kept, not counted
    recorded = []
    monkeypatch.setattr(env.monitor, "record", lambda **kw: recorded.append(kw))
    with env.gp.decode_capture(("bucket", 1)):
        env.gp.place(layer, ids)
    assert recorded == []
    held = env.gp._captures[("bucket", 1)]
    assert len(held) == 1 and held[0][0] == 2 and held[0][1] is ids
    # outside it again: counted
    env.gp.place(layer, ids)
    assert len(recorded) == 1 and recorded[0]["topk_physical"] is ids


def test_place_ignores_a_layer_eplb_does_not_place(monkeypatch):
    env = _setup(monkeypatch)
    ids = torch.zeros((2, TOP_K), dtype=torch.int32)
    env.monitor.on_forward_start()
    # a drafter layer past the maps, and another module under a placed id
    env.gp.place(_layer(LAYERS + 4), ids)
    env.gp.place(_layer(2), ids)
    env.gp.place(SimpleNamespace(), ids)
    with env.gp.decode_capture("k"):
        env.gp.place(_layer(2), ids)
    assert int(env.monitor._cur_pass_count.sum()) == 0
    assert env.gp._captures["k"] == []


# ---------------------------------------------------------------- statistics
def test_prefill_load_is_folded_by_the_placement_of_its_pass(monkeypatch):
    env = _setup(monkeypatch)
    layer = env.layers[0]
    gen = torch.Generator().manual_seed(7)
    routes = [torch.randint(0, EXPERTS, (20, TOP_K), generator=gen) for _ in range(3)]
    placements = [torch.arange(EXPERTS), _cycle3(), ROLL]
    for logical, p2l in zip(routes, placements):
        if not torch.equal(env.meta.physical_to_logical_map[0].long(), p2l):
            _commit(env, 0, p2l)  # between two passes
        l2p = env.meta.logical_to_physical_map[0, :, 0].long()
        env.monitor.on_forward_start()
        # the router emits physical ids: the slots of the experts it picked
        env.gp.place(layer, l2p[logical].to(torch.int32))
        env.monitor.on_forward_end(is_dummy_run=False, is_pure_prefill=True)
    want = sum(_logical_counts(r) for r in routes)
    dec, pf_total, *_ = env.gp.dump_global(None)
    assert pf_total[0].tolist() == want.tolist()
    assert int(pf_total[1].sum()) == 0 and int(pf_total[2].sum()) == 0
    assert int(dec.sum()) == 0
    # the placement in force is the last one, and the fold index follows it
    assert torch.equal(env.meta.physical_to_logical_map[0].long(), ROLL)
    assert torch.equal(env.gp.fold_index[0], env.meta.logical_to_physical_map[0, :, 0])


def test_only_a_real_pure_prefill_pass_is_committed(monkeypatch):
    env = _setup(monkeypatch)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    for dummy, pure in ((True, True), (False, False)):
        env.monitor.on_forward_start()
        env.gp.place(env.layers[0], ids)
        env.monitor.on_forward_end(is_dummy_run=dummy, is_pure_prefill=pure)
    assert int(env.gp.pf_total.sum()) == 0
    env.monitor.on_forward_start()
    env.gp.place(env.layers[0], ids)
    env.monitor.on_forward_end(is_dummy_run=False, is_pure_prefill=True)
    assert int(env.gp.pf_total.sum()) == 3


def test_decode_replays_are_sampled_from_the_held_ids(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=2)
    key = (4, 5)
    bucket_rows = 8
    held = {i: torch.full((bucket_rows, TOP_K), -1, dtype=torch.int32) for i in (0, 2)}
    with env.gp.decode_capture(key):
        for i in (0, 2):
            env.gp.place(env.layers[i], held[i])
    assert int(env.monitor._cur_pass_count.sum()) == 0

    gen = torch.Generator().manual_seed(11)
    want = torch.zeros((LAYERS, EXPERTS), dtype=torch.int64)
    live_rows = sampled = 0
    placements = [None, _cycle3(), None, ROLL, None, None]
    for step, p2l in enumerate(placements):
        if p2l is not None:
            _commit(env, 2, p2l)
        rows = 3 + step  # this rank's live rows of the step
        for i in (0, 2):
            # what a replay leaves in the recording's id tensor: the live rows'
            # slots, pad rows at -1, and (never read) stale rows past them
            l2p = env.meta.logical_to_physical_map[i, :, 0].long()
            logical = torch.randint(0, EXPERTS, (rows, TOP_K), generator=gen)
            held[i].fill_(-1)
            held[i][:rows] = l2p[logical].to(torch.int32)
            held[i][rows - 1, 1] = -1  # an invalid pick inside a live row
            logical[rows - 1, 1] = -1
            if rows < bucket_rows - 1:
                held[i][rows + 1 :] = 5  # garbage the row count must cut off
            if step % 2 == 0:  # the first replay, then every second
                want[i] += _logical_counts(logical)
        env.gp.after_decode_replay(key, rows)
        if step % 2 == 0:  # the first replay, then every second
            live_rows += rows
            sampled += 1
    # the samples sit by slot until they are read or a placement changes
    assert env.gp.dec_raw[:, 0].sum() > 0  # the pad picks' column
    env.gp.fold_decode()
    assert int(env.gp.dec_raw.sum()) == 0
    assert env.gp.dec_load.tolist() == want.tolist()
    assert (env.gp.dec_steps, env.gp.dec_rows) == (sampled, live_rows)
    dec, _, rows_per_step, steps, *_ = env.gp.dump_global(None)
    assert dec.tolist() == want.tolist()
    assert steps == sampled and rows_per_step == pytest.approx(live_rows / sampled)


def test_the_sampling_phase_restarts_with_a_prefill_step(monkeypatch):
    """`sync_phase` (a prefill-bearing step ended, on every rank alike): the
    next decode replay is sampled wherever the count stood, so ranks whose
    counts differ sample the same steps from there."""
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=4)
    ids = torch.full((4, TOP_K), 1, dtype=torch.int32)
    with env.gp.decode_capture("k"):
        env.gp.place(env.layers[0], ids)
    sampled = []
    for step in range(11):
        if step in (2, 5):
            env.gp.sync_phase()
        before = env.gp.dec_steps
        env.gp.after_decode_replay("k", 2)
        if env.gp.dec_steps != before:
            sampled.append(step)
    assert sampled == [0, 2, 5, 9]


def test_decode_step_without_live_rows_counts_as_a_step(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    ids = torch.full((4, TOP_K), 3, dtype=torch.int32)
    with env.gp.decode_capture("k"):
        env.gp.place(env.layers[0], ids)
    env.gp.after_decode_replay("k", 0)  # a rank with no request
    env.gp.after_decode_replay("k", 2)
    env.gp.after_decode_replay("other", 2)  # a bucket recorded without ids
    assert (env.gp.dec_steps, env.gp.dec_rows) == (2, 2)
    env.gp.fold_decode()
    assert int(env.gp.dec_load.sum()) == 2 * TOP_K


def test_micro_batched_recording_is_counted_tensor_by_tensor(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    first = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    second = torch.tensor([[7, 8, 9], [-1, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    with env.gp.decode_capture("k"):
        env.gp.place(env.layers[0], first)
        env.gp.place(env.layers[0], second)
    env.gp.after_decode_replay("k", 3)
    assert (
        env.gp.dec_load[0].tolist()
        == _logical_counts(torch.cat([first, second])).tolist()
    )


def test_nothing_is_counted_once_rebalancing_stopped(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    ids = torch.full((2, TOP_K), 1, dtype=torch.int32)
    with env.gp.decode_capture("k"):
        env.gp.place(env.layers[0], ids)
    env.manager.rebalancing_stopped = True
    env.monitor.on_forward_start()
    env.gp.place(env.layers[0], ids)
    env.gp.after_decode_replay("k", 2)
    assert int(env.monitor._cur_pass_count.sum()) == 0
    assert int(env.gp.dec_load.sum()) == 0 and env.gp.dec_steps == 0


def test_released_graphs_drop_the_held_ids(monkeypatch):
    env = _setup(monkeypatch)
    with env.gp.decode_capture("k"):
        env.gp.place(env.layers[0], torch.zeros((1, TOP_K), dtype=torch.int32))
    assert env.gp.release_graphs() == 1
    assert env.gp._captures == {}


# ---------------------------------------------------------------- end to end
def test_swapped_experts_are_reached_through_the_permuted_gate(monkeypatch):
    """Every rank plans the same swaps, moves its weights the way the existing
    migration plans them and permutes its gate: a row then reaches, by the
    physical id its router emits, the very expert it was routed to before."""
    dim = 4
    gen = torch.Generator().manual_seed(13)
    envs = [_setup(monkeypatch, rank=r, layer_ids=(0,)) for r in range(RANKS)]
    pristine = _Gate(seed=0)
    # weight rows: expert e's is e + 1 everywhere, so a slot shows who sits there
    weights = [
        torch.arange(r * PER, (r + 1) * PER, dtype=torch.float32)[:, None].repeat(
            1, dim
        )
        + 1
        for r in range(RANKS)
    ]
    pf = (torch.rand(LAYERS, EXPERTS, generator=gen) ** 4 * 1e5).long()
    dec = (torch.rand(LAYERS, EXPERTS, generator=gen) ** 4 * 1e4).long()
    x = torch.randint(-8, 9, (32, 8), generator=gen).float()
    want_ids, want_weights = _route(pristine, x)

    for round_ in range(2):  # the second rebalance starts from the first's
        plans = [
            gp.swap_rebalance(
                pf,
                dec,
                24.0,
                e.meta.physical_to_logical_map,
                num_gpus=RANKS,
                top_k=TOP_K,
                budget=3,
                eps=0.0,
                layer_ids=[0],
            )
            for e in envs
        ]
        p2l, l2p, logcnt, swaps = plans[0]
        assert swaps[0] > 0 or round_
        assert all(torch.equal(p[0], p2l) for p in plans)  # the group's plan
        old = envs[0].meta.physical_to_logical_map.clone()
        snapshot = [w.clone() for w in weights]
        for r, e in enumerate(envs):
            plan, local, _sends, recvs = eplb._plan_single_layer_migration(
                old_p2l_layer=old[0],
                new_p2l_layer=p2l[0],
                num_local_physical_experts=PER,
                num_gpu_per_node=RANKS,
                rank=r,
                world_size=RANKS,
            )
            temp = torch.zeros_like(weights[r])
            for action in local:
                temp[action.dst_slot] = snapshot[r][action.src_slot]
            for action in recvs:  # what the peer sends: its slot of that expert
                peer_slot = old[
                    0, action.peer_rank * PER : (action.peer_rank + 1) * PER
                ]
                src = int((peer_slot == action.logical_expert_id).nonzero()[0])
                temp[action.local_slot] = snapshot[action.peer_rank][src]
            new_meta = eplb.ExpertLocationMetadata.from_rebalance_result(
                physical_to_logical_map=p2l,
                logical_to_physical_map=l2p,
                logical_replica_count=logcnt,
                ep_size=RANKS,
                ep_rank=r,
                max_num_replicas=1,
            )
            staged = e.gp.stage_commit(0, old[0], p2l[0])
            eplb.commit_layer(plan, [temp], [weights[r]], e.meta, new_meta, 0)
            assert e.gp.apply_commit(0, staged) == (swaps[0] > 0)

        for r, e in enumerate(envs):
            ids, got_weights = _route(e.gates[0], x)
            assert torch.equal(got_weights, want_weights)
            # dispatch: rank = id // per, local slot = id % per
            reached = torch.stack(
                [weights[int(p) // PER][int(p) % PER, 0] for p in ids.reshape(-1)]
            ).reshape(ids.shape)
            assert torch.equal(reached.long() - 1, want_ids)
            assert torch.equal(e.meta.physical_to_logical_map[0].long()[ids], want_ids)


# ---------------------------------------------------------------- refusals
def test_runtime_refuses_redundant_experts():
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=1,
        num_logical_experts=12,
        num_physical_experts=16,
        ep_size=RANKS,
        ep_rank=0,
    )
    with pytest.raises(RuntimeError, match="num_redundant_experts == 0"):
        gp.GatePermuteRuntime(
            manager=None, monitor=None, layers={}, gates={}, meta=meta
        )


@pytest.mark.parametrize(
    "spoil,why",
    [
        (lambda layer, gate: setattr(gate, "tid2eid", torch.zeros(1)), "hash-routed"),
        (
            lambda layer, gate: setattr(layer.expert_layout, "shared_is_routed", True),
            "shared expert",
        ),
        (
            lambda layer, gate: setattr(
                layer.expert_layout, "num_fused_shared_experts", 1
            ),
            "shared expert",
        ),
        (
            lambda layer, gate: setattr(layer, "quant_method", object()),
            "does not place",
        ),
        (
            lambda layer, gate: setattr(
                layer.moe_parallel_config, "use_all2all_kernels", False
            ),
            "gathered",
        ),
        (lambda layer, gate: setattr(layer, "global_num_experts", EXPERTS + 4), "rows"),
        (
            lambda layer, gate: setattr(
                layer, "e_score_correction_bias", gate.e_score_correction_bias.clone()
            ),
            "copy of the gate",
        ),
    ],
)
def test_collect_gates_refuses_what_it_cannot_serve(spoil, why):
    gate = _Gate()
    layer = _layer(0, gate)
    model = _Model([SimpleNamespace(gate=gate, experts=layer)])
    assert gp.collect_gates(model, {0: layer}) == {0: gate}
    spoil(layer, gate)
    with pytest.raises(RuntimeError, match=why):
        gp.collect_gates(model, {0: layer})


def test_collect_gates_refuses_a_layer_without_a_gate():
    layer = _layer(0)
    with pytest.raises(RuntimeError, match="no module holds it"):
        gp.collect_gates(
            _Model([SimpleNamespace(gate=None, experts=layer)]), {0: layer}
        )


def test_gate_permute_without_eplb_is_refused(monkeypatch):
    monkeypatch.setattr(eplb, "_get_configured_eplb_manager", lambda: None)
    assert eplb.initialize_eplb_runtime(object()) is None
    monkeypatch.setenv("ATOM_EPLB_GATE_PERMUTE", "1")
    with pytest.raises(RuntimeError, match="needs EPLB"):
        eplb.initialize_eplb_runtime(object())


# ---------------------------------------------------------------- manager
def _manager(policy="naive", interval=8, window=4):
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=window)
    return eplb.EPLBManager(
        enabled=True,
        monitor=monitor,
        rebalance_interval=interval,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy=policy,
    )


def _bound_manager(monkeypatch, policy="swap", **env):
    """A manager on EP rank 0 of 4 over a 3-layer model, bound without a
    process group (nothing is migrated over one here)."""
    monkeypatch.setattr(gp, "_HAS_TRITON", False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    manager = _manager(policy)
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=LAYERS, num_logical_experts=EXPERTS, ep_size=RANKS, ep_rank=0
    )
    gates = {i: _Gate(seed=i) for i in range(LAYERS)}
    layers = {i: _layer(i, gates[i]) for i in range(LAYERS)}
    manager.live_metadata = meta
    manager._moe_layers = layers
    manager._ep_group = object()
    manager.monitor.initialize_for_metadata(meta)
    manager.monitor.set_load_group(SimpleNamespace(world_size=1))
    model = _Model([SimpleNamespace(gate=gates[i], experts=layers[i]) for i in layers])
    manager._init_gate_permute(model)
    return manager, gates, layers


def _skewed(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(LAYERS, EXPERTS, generator=gen) ** 4 * 1e5).long()


def _drain(generator, limit=50):
    steps = 0
    for _ in generator:
        steps += 1
        assert steps < limit
    return steps


def test_swap_rebalance_moves_weights_maps_and_gate_as_one_plan(monkeypatch, caplog):
    monkeypatch.setattr(eplb, "SWAP_BUDGET", 2)
    monkeypatch.setattr(eplb, "SWAP_EPS", 0)
    manager, gates, _layers = _bound_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=1)
    gpr = manager.gate_permute
    assert gpr is not None
    pristine = {i: _Gate(seed=i) for i in gates}
    pf, dec = _skewed(1), _skewed(2)
    pf[1], dec[1] = 100, 50  # balanced: layer 1 keeps its placement
    gpr.pf_total.copy_(pf)
    gpr.dec_load.copy_(dec)
    gpr.dec_rows, gpr.dec_steps = 480, 20

    switched = []
    monkeypatch.setattr(
        eplb, "migrate_experts_chunk", lambda **kw: {kw["layer_ids"][0]: []}
    )

    def commit(**kw):
        (layer_id,) = kw["layer_ids"]
        switched.append(layer_id)
        kw["live_meta"].update(kw["new_meta"], [layer_id])

    monkeypatch.setattr(eplb, "commit_experts_chunk", commit)
    manager._rebalance_layers_per_chunk = 1
    with caplog.at_level(logging.INFO, logger="atom"):
        steps = _drain(manager._rebalance())

    assert switched == [0, 2] and steps == 2  # a chunk a changed layer
    assert manager.rebalance_count == 1 and not manager.rebalancing_stopped
    p2l = manager.live_metadata.physical_to_logical_map
    assert torch.equal(p2l[1], torch.arange(EXPERTS, dtype=torch.int32))
    for i in (0, 2):
        assert int((p2l[i] != torch.arange(EXPERTS)).sum()) == 4  # two swaps
        assert torch.equal(gates[i].weight.data, pristine[i].weight.data[p2l[i].long()])
        assert torch.equal(
            gpr.fold_index[i], manager.live_metadata.logical_to_physical_map[i, :, 0]
        )
    assert torch.equal(gates[1].weight.data, pristine[1].weight.data)
    # cumulative statistics survive the rebalance
    assert int(gpr.dec_load.sum()) > 0 and gpr.dec_steps == 20

    lines = [r.getMessage() for r in caplog.records if gp.TAG in r.getMessage()]
    text = "\n".join(lines)
    assert "rebalance #1: policy=swap swaps sum=4 max=2 layers_changed=2/3" in text
    assert "rows/step=24.0" in text and "decode samples/rank=20" in text
    assert "rebalance #1 swaps/layer: 2 0 2" in text
    for name in ("before", "after", "identity"):
        assert f"rebalance #1 metrics {name}" in text
    assert text.count("migrate=") == 2
    assert "rebalance #1 done: 2 layers migrated in 2 chunks" in text
    assert "FATAL" not in text


def test_mode_line_names_the_switches(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="atom"):
        _bound_manager(
            monkeypatch, ATOM_EPLB_GATE_PERMUTE=1, ATOM_EPLB_FIRST_REBALANCE_STEPS=7
        )
    (line,) = [r.getMessage() for r in caplog.records if "mode:" in r.getMessage()]
    assert line.startswith(gp.TAG)
    for part in (
        "gate_permute=1",
        "count=torch",
        "decode_sample_every=64",
        "policy=swap",
        "layers=3 (0..2)",
        "first=7",
        "swap_budget=16",
        "swap_eps=0.005",
    ):
        assert part in line


def test_swap_policy_is_not_held_back_by_the_balancedness_gate(monkeypatch):
    manager, _, _ = _bound_manager(monkeypatch, policy="swap")
    assert manager.gate_permute is None  # the policy alone, ids still remapped
    monkeypatch.setattr(
        manager, "_need_rebalance", lambda load: pytest.fail("gate consulted")
    )
    ran = []
    monkeypatch.setattr(
        manager, "_execute_runtime_rebalance", lambda: ran.append(1) or iter(())
    )
    _drain(manager._rebalance())
    assert ran == [1]


def test_failed_policy_stops_rebalancing_with_nothing_moved(monkeypatch, caplog):
    manager, gates, _ = _bound_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=1)
    manager.gate_permute.pf_total.copy_(_skewed(3))
    before = {i: g.weight.data.clone() for i, g in gates.items()}
    live = manager.live_metadata.physical_to_logical_map.clone()

    def broken(*args, **kwargs):
        raise FloatingPointError("boom")

    monkeypatch.setattr(gp, "swap_rebalance", broken)
    monkeypatch.setattr(
        eplb, "migrate_experts_chunk", lambda **kw: pytest.fail("migrated")
    )
    with caplog.at_level(logging.ERROR, logger="atom"):
        assert _drain(manager._rebalance()) == 0
    assert manager.rebalancing_stopped
    assert torch.equal(manager.live_metadata.physical_to_logical_map, live)
    assert all(torch.equal(g.weight.data, before[i]) for i, g in gates.items())
    assert any(f"{gp.TAG} FATAL" in r.getMessage() for r in caplog.records)
    # and no later forward schedules anything
    manager.on_forward_pass_end(True, True)


@pytest.mark.parametrize("stage", ["prepare", "p2p", "commit"])
def test_failed_layer_switch_is_fatal_and_never_half_applied_silently(
    monkeypatch, caplog, stage
):
    monkeypatch.setattr(eplb, "SWAP_EPS", 0)
    manager, gates, _ = _bound_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=1)
    gpr = manager.gate_permute
    gpr.pf_total.copy_(_skewed(3))
    before = {i: g.weight.data.clone() for i, g in gates.items()}
    live = manager.live_metadata.physical_to_logical_map.clone()

    def fail(**kw):
        raise OSError(stage)

    monkeypatch.setattr(
        eplb, "migrate_experts_chunk", lambda **kw: {kw["layer_ids"][0]: []}
    )
    monkeypatch.setattr(
        eplb,
        "commit_experts_chunk",
        lambda **kw: kw["live_meta"].update(kw["new_meta"], kw["layer_ids"]),
    )
    if stage == "prepare":
        monkeypatch.setattr(gpr, "stage_commit", lambda *a: fail())
    elif stage == "p2p":
        monkeypatch.setattr(eplb, "migrate_experts_chunk", fail)
    else:
        monkeypatch.setattr(eplb, "commit_experts_chunk", fail)
    with (
        caplog.at_level(logging.ERROR, logger="atom"),
        pytest.raises(OSError, match=stage),
    ):
        _drain(manager._rebalance())
    assert manager.rebalancing_stopped
    (fatal,) = [r.getMessage() for r in caplog.records if "FATAL" in r.getMessage()]
    assert fatal.startswith(gp.TAG) and f"failed at '{stage}'" in fatal
    # the gate was not touched in any of the three, and before the commit
    # neither were the maps
    assert all(torch.equal(g.weight.data, before[i]) for i, g in gates.items())
    assert torch.equal(manager.live_metadata.physical_to_logical_map, live)
    assert ("half switched" in fatal) == (stage == "commit")


def test_first_rebalance_steps(monkeypatch):
    def steps_before_first(manager):
        calls = []
        monkeypatch.setattr(manager, "_rebalance", lambda: calls.append(1) or iter(()))
        gen = manager._entrypoint()
        count = 0
        while not calls:
            next(gen)
            count += 1
        return count - 1

    assert steps_before_first(_manager(interval=40, window=4)) == 10
    monkeypatch.setenv("ATOM_EPLB_FIRST_REBALANCE_STEPS", "3")
    assert steps_before_first(_manager(interval=40, window=4)) == 3


def _steps_taken(monkeypatch, manager, flags):
    """How often the schedule advances over forwards of the given dummy flags,
    each a prefill-bearing step of the group."""
    taken = []
    monkeypatch.setattr(manager, "_gen", iter(lambda: taken.append(1), object()))
    manager._dp_is_migration_group = True
    for dummy in flags:
        manager.on_forward_pass_end(True, True, is_dummy_run=dummy)
    return len(taken)


def test_dummy_forwards_are_not_rebalance_steps(monkeypatch):
    flags = [True] * 5 + [False, True, False]
    swap, _, _ = _bound_manager(monkeypatch, policy="swap")
    assert _steps_taken(monkeypatch, swap, flags) == 2
    permute, _, _ = _bound_manager(
        monkeypatch, policy="naive", ATOM_EPLB_GATE_PERMUTE=1
    )
    assert _steps_taken(monkeypatch, permute, flags) == 2
    # the default EPLB keeps counting every prefill-bearing forward
    monkeypatch.delenv("ATOM_EPLB_GATE_PERMUTE")
    default, _, _ = _bound_manager(monkeypatch, policy="naive")
    assert _steps_taken(monkeypatch, default, flags) == len(flags)


def test_forward_monitor_tells_the_schedule_a_dummy_forward(monkeypatch):
    manager, _, _ = _bound_manager(monkeypatch, policy="swap")
    seen = []
    monkeypatch.setattr(
        manager, "on_forward_pass_end", lambda *a, **kw: seen.append((a, kw))
    )
    monkeypatch.setattr(eplb, "_MANAGER", manager)
    forward = eplb.with_eplb_forward_monitor(lambda self, batch: None)
    runner = SimpleNamespace(_eplb_any_rank_has_prefill=True)
    warmup = SimpleNamespace(
        total_tokens_num_prefill=9, total_tokens_num_decode=0, is_dummy_run=True
    )
    forward(runner, warmup)
    forward(runner, PREFILL)
    assert seen == [
        ((True, True), {"is_dummy_run": True}),
        ((True, True), {"is_dummy_run": False}),
    ]


def test_rebalance_with_too_little_load_is_skipped_and_not_counted(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_EPLB_MAX_REBALANCES", "1")
    monkeypatch.setattr(eplb, "SWAP_EPS", 0)
    manager, gates, _ = _bound_manager(
        monkeypatch, ATOM_EPLB_GATE_PERMUTE=1, ATOM_EPLB_MIN_PICKS=1000
    )
    assert manager.max_rebalances == 1
    gpr = manager.gate_permute
    before = {i: g.weight.data.clone() for i, g in gates.items()}
    switched = []
    monkeypatch.setattr(
        eplb, "migrate_experts_chunk", lambda **kw: {kw["layer_ids"][0]: []}
    )
    monkeypatch.setattr(
        eplb,
        "commit_experts_chunk",
        lambda **kw: switched.append(kw["layer_ids"][0])
        or kw["live_meta"].update(kw["new_meta"], kw["layer_ids"]),
    )
    # 999 picks, prefill and decode together: one short
    gpr.pf_total[0, :3] = torch.tensor([600, 300, 90])
    gpr.dec_load[2, 5] = 9
    with caplog.at_level(logging.INFO, logger="atom"):
        assert _drain(manager._rebalance()) == 0
    assert manager.rebalance_count == 0 and not manager.rebalancing_stopped
    assert not manager._max_rebalances_reached()
    assert switched == []
    assert all(torch.equal(g.weight.data, before[i]) for i, g in gates.items())
    (line,) = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert line.startswith(gp.TAG) and "999 picks < ATOM_EPLB_MIN_PICKS=1000" in line
    # one more pick and the rebalance runs, and is the one MAX_REBALANCES allows
    gpr.dec_load[2, 5] = 10
    _drain(manager._rebalance())
    assert manager.rebalance_count == 1 and switched
    assert manager._max_rebalances_reached()


def test_min_picks_does_not_touch_the_default_policy(monkeypatch):
    manager, _, _ = _bound_manager(
        monkeypatch, policy="naive", ATOM_EPLB_MIN_PICKS=10**9
    )
    planned = []
    monkeypatch.setattr(
        eplb, "rebalance_experts", lambda *a, **kw: planned.append(1) or 1 / 0
    )
    with pytest.raises(ZeroDivisionError):
        _drain(manager._execute_rebalance())
    assert planned == [1] and manager.rebalance_count == 1


# ---------------------------------------------------------------- default off
def test_switches_default_off():
    assert eplb.eplb_gate_permute() is None
    assert isinstance(eplb.eplb_decode_capture(("k", 1)), nullcontext)


def test_default_manager_has_no_gate_permute_and_keeps_the_gate(monkeypatch):
    manager, _gates, _ = _bound_manager(monkeypatch, policy="naive")
    assert manager.gate_permute is None
    assert manager.monitor._logical_fold is None
    monkeypatch.setattr(eplb, "_MANAGER", manager)
    assert eplb.eplb_gate_permute() is None
    # the balancedness gate still decides
    asked = []
    monkeypatch.setattr(
        manager, "_need_rebalance", lambda load: asked.append(1) and False
    )
    monkeypatch.setattr(
        manager, "_execute_runtime_rebalance", lambda: pytest.fail("rebalanced")
    )
    _drain(manager._rebalance())
    assert asked == [1]


def test_default_monitor_keeps_physical_counts_and_folds_on_dump(monkeypatch):
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=2, num_logical_experts=EXPERTS, ep_size=RANKS, ep_rank=0
    )
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    monitor.initialize_for_metadata(meta)
    monitor.set_load_group(SimpleNamespace(world_size=1))
    ids = torch.tensor([[0, 5, 5], [7, 7, 7]], dtype=torch.int32)
    monitor.on_forward_start()
    monitor.record(layer_id=1, topk_physical=ids, num_physical=EXPERTS)
    monitor.on_forward_end(is_dummy_run=False, is_pure_prefill=True)
    assert torch.equal(monitor._expert_load_window[0], monitor._cur_pass_count)
    physical = monitor.dump_global_physical_load()
    assert physical[1].tolist() == _logical_counts(ids).tolist()
    # a moved placement re-reads the window, as before
    moved = meta.physical_to_logical_map.clone()
    moved[1] = ROLL.to(torch.int32)
    meta.physical_to_logical_map.copy_(moved)
    monkeypatch.setattr(eplb, "get_live_expert_location_metadata", lambda: meta)
    logical = monitor.dump_global_logical_load()
    want = eplb.physical_load_to_logical_load(physical, moved, EXPERTS)
    assert torch.equal(logical, want) and not torch.equal(logical, physical)


def _forward_monitor_calls(monkeypatch, manager, batch):
    calls = []
    monitor = manager.monitor
    monkeypatch.setattr(monitor, "on_forward_start", lambda: calls.append("start"))
    monkeypatch.setattr(monitor, "on_forward_end", lambda *a: calls.append(("end", *a)))
    monkeypatch.setattr(
        manager,
        "on_forward_pass_end",
        lambda *a, is_dummy_run: calls.append(("step", *a)),
    )
    monkeypatch.setattr(eplb, "_MANAGER", manager)
    runner = SimpleNamespace(_eplb_any_rank_has_prefill=True)
    forward = eplb.with_eplb_forward_monitor(lambda self, batch: "out")
    assert forward(runner, batch) == "out"
    return calls


PREFILL = SimpleNamespace(
    total_tokens_num_prefill=9, total_tokens_num_decode=0, is_dummy_run=False
)
DECODE = SimpleNamespace(
    total_tokens_num_prefill=0, total_tokens_num_decode=6, is_dummy_run=False
)


@pytest.mark.parametrize("batch,pure", [(PREFILL, True), (DECODE, False)])
def test_default_forward_monitor_brackets_every_forward(monkeypatch, batch, pure):
    manager, _, _ = _bound_manager(monkeypatch, policy="naive")
    calls = _forward_monitor_calls(monkeypatch, manager, batch)
    assert calls == ["start", ("end", False, pure), ("step", pure, True)]


def test_gate_permute_forward_monitor_touches_the_pass_buffer_in_prefill_only(
    monkeypatch,
):
    manager, _, _ = _bound_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=1)
    assert _forward_monitor_calls(monkeypatch, manager, DECODE) == [
        ("step", False, True)
    ]
    manager2, _, _ = _bound_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=1)
    assert _forward_monitor_calls(monkeypatch, manager2, PREFILL) == [
        "start",
        ("end", False, True),
        ("step", True, True),
    ]
    manager2._rebalancing_stopped = True
    assert _forward_monitor_calls(monkeypatch, manager2, PREFILL) == [
        ("step", True, True)
    ]


def _place_experts(monkeypatch, gate_permute):
    """`Mxfp4MoEMethod.place_experts` as written in moe.py, on stand-ins."""
    calls = []
    namespace = {
        "torch": torch,
        "eplb_gate_permute": lambda: gate_permute,
        "eplb_map_and_record_fused": lambda layer, ids, gathered=False: (
            calls.append(("map", layer, ids, gathered)) or ids + 100
        ),
        "eplb_append_shared_map_and_record_fused": lambda *a: (
            calls.append(("shared", *a)) or ("w", "ids")
        ),
    }
    exec(  # noqa: S102 - the source is this repository's own, read above
        method_source("atom/model_ops/moe.py", "Mxfp4MoEMethod", "place_experts"),
        namespace,
    )
    return namespace["place_experts"], calls


def test_default_place_experts_remaps_as_before(monkeypatch):
    place, calls = _place_experts(monkeypatch, None)
    ids = torch.tensor([[1, 2, 3]])
    layer = SimpleNamespace(
        expert_layout=SimpleNamespace(shared_is_routed=False),
        to_dispatch_space=lambda w, i: (w, i),
    )
    weights, out = place(None, layer, "weights", ids, True)
    assert weights == "weights" and torch.equal(out, ids + 100)
    assert calls == [("map", layer, ids, True)]


def test_gate_permute_place_experts_hands_the_ids_through(monkeypatch):
    seen = []
    runtime = SimpleNamespace(place=lambda layer, ids: seen.append((layer, ids)))
    place, calls = _place_experts(monkeypatch, runtime)
    ids = torch.tensor([[1, 2, 3]])
    layer = SimpleNamespace(
        expert_layout=SimpleNamespace(shared_is_routed=False),
        to_dispatch_space=lambda w, i: (w, i),
    )
    weights, out = place(None, layer, "weights", ids, False)
    assert out is ids and weights == "weights"
    assert seen == [(layer, ids)] and calls == []


def _eplb_layer_end(monkeypatch, replay):
    from atom.utils import envs

    namespace = {"envs": envs}
    source = method_source(
        "atom/models/deepseek_v41/runtime.py",
        "DeepseekV41RuntimeModel",
        "eplb_layer_end",
    )
    exec(  # noqa: S102 - the source is this repository's own, read above
        source.replace("@property\n", ""), namespace
    )
    return namespace["eplb_layer_end"](SimpleNamespace(replay=replay, decoder_start=20))


def test_eplb_layer_end(monkeypatch):
    assert _eplb_layer_end(monkeypatch, True) == 20
    assert _eplb_layer_end(monkeypatch, False) is None
    monkeypatch.setenv("ATOM_EPLB_ALL_LAYERS", "1")
    assert _eplb_layer_end(monkeypatch, True) is None
    assert _eplb_layer_end(monkeypatch, False) is None


def test_eplb_layer_end_on_the_model(monkeypatch):
    from conftest import gpu_stack_import

    runtime = gpu_stack_import("atom.models.deepseek_v41.runtime")
    model = runtime.DeepseekV41RuntimeModel.__new__(runtime.DeepseekV41RuntimeModel)
    model.__dict__.update(replay=True, decoder_start=20)
    assert model.eplb_layer_end == 20
    monkeypatch.setenv("ATOM_EPLB_ALL_LAYERS", "1")
    assert model.eplb_layer_end is None


# ---------------------------------------------------------------- config
def test_eplb_config_from_a_file(tmp_path):
    from atom.model_engine.arg_utils import json_or_file

    inline = (
        '{"placement_policy": "swap", "rebalance_interval": 28, "load_window_size": 28}'
    )
    path = tmp_path / "eplb.json"
    path.write_text(inline)
    want = json.loads(inline)
    assert json_or_file(inline) == want
    assert json_or_file(str(path)) == want
    with pytest.raises(json.JSONDecodeError):
        json_or_file(str(tmp_path / "missing.json"))  # neither a file nor JSON
    config = EPLBConfig.from_dict(json_or_file(str(path)))
    assert (config.placement_policy, config.rebalance_interval) == ("swap", 28)


def test_swap_policy_needs_no_redundant_experts():
    assert EPLBConfig(placement_policy="swap").placement_policy == "swap"
    with pytest.raises(AssertionError, match="swap"):
        EPLBConfig(placement_policy="swap", num_redundant_experts=8)
    with pytest.raises(AssertionError):
        EPLBConfig(placement_policy="greedy")


# ---------------------------------------------------------------- logging
def test_metrics_line_reads_the_placement(monkeypatch):
    pf = np.zeros((4, EXPERTS))
    dec = np.zeros((4, EXPERTS))
    pf[:, :PER] = 1.0  # every prefill pick on the first rank's experts
    dec[:, ::PER] = 1.0  # decode picks spread evenly over the ranks
    identity = np.tile(np.arange(EXPERTS) // PER, (4, 1))
    line = gp.metrics_line(pf, dec, 8.0, identity, [0, 1, 2, 3], RANKS, TOP_K)
    assert "prefill max/mean L0-1 4.000 L2-2 4.000" in line
    assert "decode-row max/mean 1.000" in line
    spread = np.tile(np.arange(EXPERTS) % RANKS, (4, 1))
    line = gp.metrics_line(pf, dec, 8.0, spread, [0, 1, 2, 3], RANKS, TOP_K)
    assert "prefill max/mean L0-1 1.000" in line
    assert "decode-row max/mean 4.000" in line
    empty = gp.metrics_line(pf * 0, dec * 0, 0.0, identity, [0, 1], RANKS, TOP_K)
    assert "nan" in empty


def test_count_stride_reads_about_the_asked_blocks():
    """`count_stride`: a pass of at most ``blocks`` blocks is read whole, a
    larger one every stride-th block, about ``blocks`` of them."""
    assert gp.count_stride(768, 256, 32) == 1
    assert gp.count_stride(32 * 256, 256, 32) == 1
    assert gp.count_stride(98304, 256, 32) == 12  # 384 blocks -> 32 read
    assert gp.count_stride(1, 256, 32) == 1


def test_the_count_flag_follows_counting(monkeypatch):
    """The device flag the recorded count launches read: on while loads are
    counted, off once the rebalances stopped."""
    env = _setup(monkeypatch)
    flag = env.gp._count_flag
    assert flag.tolist() == [1]
    env.gp.sync_count_flag()
    assert flag.tolist() == [1]
    env.manager.rebalancing_stopped = True
    assert flag.tolist() == [1]  # until the next forward syncs it
    env.gp.sync_count_flag()
    assert flag.tolist() == [0] and env.gp._count_flag is flag
