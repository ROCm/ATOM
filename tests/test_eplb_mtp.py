# SPDX-License-Identifier: MIT
# Tests for gate-permute EPLB's placement of the drafter's MoE layers
# (ATOM_EPLB_MTP): ownership, decode sampling of the draft pass, the plan,
# the migration through the host transport and the gate rows that follow.

import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.model_ops import eplb
    from atom.model_ops import eplb_gate_permute as gp
    from atom.spec_decode import draft_graph
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

from test_eplb_gate_permute import (
    EXPERTS,
    LAYERS,
    RANKS,
    TOP_K,
    _Gate,
    _layer,
    _Model,
    _route,
    _skewed,
    clean_switches,  # noqa: F401  (autouse: every switch unset)
)
from test_eplb_window_and_freeze import _lines, _manager, _scripted_policy

ROOT = Path(__file__).resolve().parent.parent
# The draft layers have an expert count of their own: 8 over 4 ranks, where
# the target layers of these tests have 16.
DRAFT_IDS = (40, 41, 42)
DRAFT_EXPERTS = 8
DRAFT_PER = DRAFT_EXPERTS // RANKS
KEY = ("Drafter._block_backbone", 4)


@pytest.fixture(autouse=True)
def no_record_scope():
    draft_graph.record_scope = None
    yield
    draft_graph.record_scope = None


def _full_weights(seed, experts=DRAFT_EXPERTS):
    """One draft layer's expert weights as the checkpoint has them: four
    parts, expert-major, a recognisable value per expert and part."""
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 255, (experts, 6, 4), generator=gen, dtype=torch.uint8),
        torch.randint(0, 255, (experts, 4, 3), generator=gen, dtype=torch.uint8),
        torch.randn(experts, 6, generator=gen),
        torch.randn(experts, 4, generator=gen),
    ]


NAMES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def _drafter(rank=0, seed=0):
    """A drafter whose model holds three EP MoE layers of 8 experts with
    their gates, as EP rank ``rank`` of 4 holds them."""
    gates, layers, modules, full = {}, {}, [], {}
    for n, layer_id in enumerate(DRAFT_IDS):
        gate = _Gate(experts=DRAFT_EXPERTS, seed=seed + 100 + n)
        layer = _layer(layer_id, gate, experts=DRAFT_EXPERTS)
        layer.prefix = f"mtp.{n}.ffn.experts"
        layer.use_ep, layer.ep_size, layer.ep_rank = True, RANKS, rank
        full[layer_id] = _full_weights(seed + n)
        for name, part in zip(NAMES, full[layer_id]):
            setattr(
                layer, name, part[rank * DRAFT_PER : (rank + 1) * DRAFT_PER].clone()
            )
        gates[layer_id], layers[layer_id] = gate, layer
        modules += [SimpleNamespace(gate=gate, experts=layer), layer]
    drafter = SimpleNamespace(model=_Model(modules))
    return SimpleNamespace(drafter=drafter, gates=gates, layers=layers, full=full)


def _mtp_manager(monkeypatch, *, rank=0, seed=0, **env):
    env = {"ATOM_EPLB_MTP": 1, **env}
    manager = _manager(monkeypatch, rank=rank, **env)
    draft = _drafter(rank, seed)
    manager._init_mtp(draft.drafter)
    return manager, draft


def _record(mtp, draft, rows=12, key=KEY):
    """The draft pass's recording: every draft layer routes once."""
    held = {}
    with mtp.capture(key):
        for layer_id in DRAFT_IDS:
            held[layer_id] = torch.full((rows, TOP_K), -1, dtype=torch.int32)
            mtp.place(draft.layers[layer_id], held[layer_id])
    return held


def _counts(ids, experts=DRAFT_EXPERTS):
    flat = ids.reshape(-1)
    return torch.bincount(flat[flat >= 0], minlength=experts)


def _cycle(experts=DRAFT_EXPERTS):
    """A placement that trades three experts in a cycle across ranks."""
    p2l = torch.arange(experts)
    p2l[0], p2l[3], p2l[6] = 3, 6, 0
    return p2l.to(torch.int32)


# ---------------------------------------------------------------- ownership
def test_mtp_is_off_by_default(monkeypatch):
    manager = _manager(monkeypatch)
    draft = _drafter()
    manager._init_mtp(draft.drafter)
    runtime = manager.gate_permute
    assert runtime.mtp is None and draft_graph.record_scope is None
    assert manager._mtp_weights == [] and manager._mtp_temp == []
    assert "mtp=0 mtp_layers=0" in runtime.describe()
    # a draft layer's routes are nobody's: nothing kept, nothing counted
    ids = torch.zeros((4, TOP_K), dtype=torch.int32)
    with runtime.decode_capture("k"):
        runtime.place(draft.layers[40], ids)
    runtime.place(draft.layers[40], ids)
    assert runtime._captures["k"] == []
    assert int(manager.monitor._cur_pass_count.sum()) == 0


def test_mtp_needs_gate_permute_and_a_drafter_with_ep_layers(monkeypatch):
    with pytest.raises(RuntimeError, match="needs ATOM_EPLB_GATE_PERMUTE=1"):
        _mtp_manager(monkeypatch, ATOM_EPLB_GATE_PERMUTE=0)
    manager = _manager(monkeypatch, ATOM_EPLB_MTP=1)
    for drafter in (None, SimpleNamespace(), SimpleNamespace(model=_Model([]))):
        with pytest.raises(RuntimeError, match="no EP MoE layer"):
            manager._init_mtp(drafter)
    other = _drafter()
    other.layers[41].ep_rank = 3
    with pytest.raises(RuntimeError, match="draft layer 41"):
        manager._init_mtp(other.drafter)
    hashed = _drafter()
    hashed.gates[42].tid2eid = torch.zeros(1)
    with pytest.raises(RuntimeError, match="hash-routed"):
        manager._init_mtp(hashed.drafter)
    assert manager.gate_permute.mtp is None


def test_draft_layers_get_a_state_of_their_own_from_the_identity(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="atom"):
        manager, draft = _mtp_manager(monkeypatch, rank=2)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    assert mtp.layer_ids == list(DRAFT_IDS) and len(mtp.layers) == 3
    assert (mtp.num_experts, mtp.per_rank, mtp.top_k) == (
        DRAFT_EXPERTS,
        DRAFT_PER,
        TOP_K,
    )
    identity = torch.arange(DRAFT_EXPERTS)
    assert mtp.p2l.shape == (3, DRAFT_EXPERTS) and mtp.p2l.dtype == torch.int32
    assert all(torch.equal(row.long(), identity) for row in mtp.p2l)
    assert all(torch.equal(row, identity) for row in mtp.fold_index)
    assert mtp.dec_load.shape == (3, DRAFT_EXPERTS) and int(mtp.dec_load.sum()) == 0
    assert mtp.dec_raw.shape == (3, DRAFT_EXPERTS + 1)
    # the target layers' state is as wide as before
    meta = manager.live_metadata
    assert meta.physical_to_logical_map.shape == (LAYERS, EXPERTS)
    assert runtime.dec_load.shape == (LAYERS, EXPERTS)
    assert sorted(manager._moe_layers) == [0, 1, 2]
    for layer_id in DRAFT_IDS:
        assert mtp.owns(draft.layers[layer_id]) and not runtime.owns(
            draft.layers[layer_id]
        )
    assert not mtp.owns(manager._moe_layers[0]) and not mtp.owns(_layer(40))
    # this rank's expert-major views of the live weights, and one layer's buffers
    assert len(manager._mtp_weights) == 3
    for n, layer_id in enumerate(DRAFT_IDS):
        for view, name in zip(manager._mtp_weights[n], NAMES):
            assert view.shape[0] == DRAFT_PER
            assert view.data_ptr() == getattr(draft.layers[layer_id], name).data_ptr()
    assert [b.shape for b in manager._mtp_temp] == [
        w.shape for w in manager._mtp_weights[0]
    ]
    assert "mtp=1 mtp_layers=3" in runtime.describe()
    (line,) = _lines(caplog, " mtp: placing")
    assert (
        line.startswith(gp.TAG) and "[40, 41, 42] (8 experts, top-3, 2 a rank)" in line
    )
    assert "budget 4 swaps" in line and "temp buffers own" in line
    assert draft_graph.record_scope is not None


def test_draft_layers_borrow_the_target_temp_buffers_where_an_expert_is_the_same(
    monkeypatch, caplog
):
    manager = _manager(monkeypatch, ATOM_EPLB_MTP=1)
    draft = _drafter()
    # a target layer's temp buffers: 4 experts a rank of the same parts
    manager._reusable_temp_buffers = [
        torch.zeros((4, *part.shape[1:]), dtype=part.dtype) for part in _full_weights(0)
    ]
    with caplog.at_level(logging.INFO, logger="atom"):
        manager._init_mtp(draft.drafter)
    for mine, theirs in zip(manager._mtp_temp, manager._reusable_temp_buffers):
        assert mine.shape[0] == DRAFT_PER and mine.data_ptr() == theirs.data_ptr()
    assert (
        "temp buffers shared with the target layers"
        in _lines(caplog, " mtp: placing")[0]
    )
    # another expert shape: buffers of their own
    manager2 = _manager(monkeypatch, ATOM_EPLB_MTP=1)
    manager2._reusable_temp_buffers = [torch.zeros(4, 9, 9)] * 4
    manager2._init_mtp(_drafter().drafter)
    assert (
        manager2._mtp_temp[0].data_ptr()
        != manager2._reusable_temp_buffers[0].data_ptr()
    )


# ---------------------------------------------------------------- sampling
def test_draft_pass_ids_are_kept_in_its_recording_only(monkeypatch):
    manager, draft = _mtp_manager(monkeypatch)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    ids = torch.zeros((4, TOP_K), dtype=torch.int32)
    recorded = []
    monkeypatch.setattr(manager.monitor, "record", lambda **kw: recorded.append(kw))
    # eager draft passes and prefill passes: nothing kept, nothing launched
    runtime.place(draft.layers[40], ids)
    with runtime.decode_capture("target"):
        runtime.place(draft.layers[40], ids)
    assert mtp._captures == {} and runtime._captures["target"] == [] and recorded == []
    with mtp.capture(KEY):
        runtime.place(draft.layers[41], ids)
        runtime.place(manager._moe_layers[0], ids)  # a target layer: counted, not kept
    assert [(i, t is ids) for i, t in mtp._captures[KEY]] == [(1, True)]
    assert len(recorded) == 1
    assert runtime.release_graphs() == 2 and mtp._captures == {}


def test_record_scope_brackets_the_recording_not_the_warm_run(monkeypatch):
    manager, draft = _mtp_manager(monkeypatch)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    made = []

    def forward(running_bs, **staged):
        ids = torch.full((running_bs * 2, TOP_K), len(made), dtype=torch.int32)
        made.append(ids)
        runtime.place(draft.layers[42], ids)
        return ids

    pass_ = draft_graph.DraftGraph(
        forward=forward, inputs={"anchor_ids": draft_graph.StagedInput()}
    ).bind(SimpleNamespace(max_num_seqs=8), "cpu")
    pass_._graph = SimpleNamespace(record=lambda fn, key: fn())
    pass_.warmup(4)
    assert len(made) == 2  # the warm run, then the recording
    ((index, held),) = mtp._captures[(pass_.name, 4)]
    assert index == 2 and held is made[1]
    # without the scope a recording is made as before
    draft_graph.record_scope = None
    pass_.warmup(2)
    assert len(made) == 4 and (pass_.name, 2) not in mtp._captures


def test_proposer_samples_the_pass_it_just_ran():
    source = (ROOT / "atom/spec_decode/dspark_proposer.py").read_text()
    run = source.index("draft_token_ids, confidence = pass_.run(running_bs, **staged)")
    hook = source.index("eplb.mtp.after_draft_replay(", run)
    assert (
        "(pass_.name, running_bs), scheduled_bs * num_draft"
        in source[hook : hook + 200]
    )
    assert source.index("draft_token_ids = draft_token_ids[:scheduled_bs", run) > hook


def test_sampled_draft_steps_equal_the_logical_routes(monkeypatch):
    manager, draft = _mtp_manager(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=2)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    bucket_rows = 12
    held = _record(mtp, draft, bucket_rows)
    gen = torch.Generator().manual_seed(3)
    want = torch.zeros((3, DRAFT_EXPERTS), dtype=torch.int64)
    live = sampled = 0
    changes = {
        2: (1, _cycle()),
        5: (1, torch.arange(DRAFT_EXPERTS).flip(0)),
        6: (2, _cycle()),
    }
    for step in range(9):
        if step in changes:  # a placement change between two steps
            index, new = changes[step]
            assert mtp.apply_commit(index, mtp.stage_commit(index, new), new)
        rows = 2 + step  # this rank's live draft rows
        is_sampled = (
            step % 2 == 0
        )  # the first replay after a prefill step, then every 2nd
        for n, layer_id in enumerate(DRAFT_IDS):
            logical = torch.randint(0, DRAFT_EXPERTS, (rows, TOP_K), generator=gen)
            ids = held[layer_id]
            ids.fill_(-1)  # the pad rows of the bucket
            ids[:rows] = mtp.fold_index[n][logical].to(torch.int32)  # slots
            ids[rows - 1, 2] = -1  # a pad pick inside a live row
            logical[rows - 1, 2] = -1
            if rows + 1 < bucket_rows:
                ids[rows + 1 :] = 5  # stale rows past the live ones
            if is_sampled:
                want[n] += _counts(logical)
        runtime.after_decode_replay("target", 6)  # the target sampler's step
        assert mtp.due is is_sampled
        mtp.after_draft_replay(KEY, rows)
        assert mtp.due is False
        if is_sampled:
            live += rows
            sampled += 1
    mtp.fold()
    assert mtp.dec_load.tolist() == want.tolist()
    assert (mtp.rows, mtp.steps, mtp.window_rows, mtp.window_steps) == (
        live,
        sampled,
        live,
        sampled,
    )
    # the fold index followed the commits
    for n in range(3):
        assert torch.equal(
            mtp.p2l[n].long()[mtp.fold_index[n]], torch.arange(DRAFT_EXPERTS)
        )
    assert not torch.equal(mtp.p2l[1], mtp.p2l[0]) and not torch.equal(
        mtp.p2l[2], mtp.p2l[0]
    )


def test_a_draft_pass_without_a_recording_is_not_sampled(monkeypatch, caplog):
    manager, draft = _mtp_manager(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    with caplog.at_level(logging.WARNING, logger="atom"):
        for _ in range(3):
            runtime.after_decode_replay("target", 6)
            mtp.after_draft_replay(("eager", 3), 10)
    assert (mtp.steps, mtp.rows) == (0, 0)
    assert len(_lines(caplog, "holds no top-k ids of a draft layer")) == 1
    # a step whose draft pass did not run leaves nothing due for the next one
    held = _record(mtp, draft)
    held[40][:2] = 1
    runtime.after_decode_replay("target", 6)
    assert mtp.due
    monkeypatch.setattr(runtime, "sample_every", 2)
    runtime._replays = 1
    runtime.after_decode_replay("target", 6)  # not a sampled step
    assert not mtp.due
    mtp.after_draft_replay(KEY, 2)
    assert mtp.steps == 0


class _NoTorch:
    def __getattr__(self, name):
        raise AssertionError(f"torch.{name} used")


def test_frozen_nothing_of_the_draft_sampler_runs(monkeypatch):
    manager, draft = _mtp_manager(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    held = _record(mtp, draft)
    held[40][:2] = 1
    manager._rebalancing_stopped = True
    with monkeypatch.context() as patched:
        patched.setattr(gp, "torch", _NoTorch())
        for _ in range(50):
            runtime.after_decode_replay("target", 6)
            mtp.after_draft_replay(KEY, 2)
    assert not mtp.due and (mtp.steps, mtp.rows) == (0, 0)
    assert int(mtp.dec_raw.sum()) == 0


def test_draft_loads_ride_the_main_collective_on_the_cpu_group(monkeypatch):
    manager, _draft = _mtp_manager(monkeypatch)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    mtp.dec_load.copy_(_skewed(4)[:, :DRAFT_EXPERTS])
    mtp.dec_raw[1, 1 + 3] = 7  # held by slot: folded before the reduce
    mtp._raw_dirty = True
    mtp.rows, mtp.steps, mtp.window_rows, mtp.window_steps = 90, 9, 20, 4
    runtime.dec_load.copy_(_skewed(5))
    runtime.pf_total.copy_(_skewed(6))
    runtime.dec_rows, runtime.dec_steps = 70, 7
    runtime.window_rows, runtime.window_steps = 30, 5
    local_mtp = mtp.dec_load.clone()
    local_mtp[1, 3] += 7
    calls = []

    def all_reduce(tensor, op=None, group=None):
        calls.append((tensor.device.type, tensor.numel(), group))
        tensor.mul_(8)  # eight ranks holding the same

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    group = SimpleNamespace(
        world_size=8, device_group=None, cpu_group="gloo", reduce_on_cpu=True
    )
    dec, pf, cum, steps, window, window_steps = runtime.dump_global(group)
    # one collective, on the CPU group, carrying both layer sets
    size = LAYERS * EXPERTS
    assert calls == [("cpu", 2 * size + 3 * DRAFT_EXPERTS + 8, "gloo")]
    assert torch.equal(dec, _skewed(5) * 8) and torch.equal(pf, _skewed(6) * 8)
    assert (cum, steps, window, window_steps) == (80.0, 7, 48.0, 5)
    mtp_dec, mtp_cum, mtp_steps, mtp_window, mtp_window_steps = mtp.reduced
    assert torch.equal(mtp_dec, local_mtp * 8)
    assert (mtp_cum, mtp_steps, mtp_window, mtp_window_steps) == (80.0, 9, 40.0, 4)
    # a planned rebalance starts both windows
    runtime.start_window()
    assert (mtp.window_rows, mtp.window_steps, mtp.rows) == (0, 0, 90)


# ---------------------------------------------------------------- plan
def test_plan_balances_a_skewed_decode_load_within_the_budget(monkeypatch):
    manager, _draft = _mtp_manager(monkeypatch)
    monkeypatch.setattr(eplb, "SWAP_EPS", 0.005)  # the production gain bound
    mtp = manager.gate_permute.mtp
    dec = torch.zeros((3, DRAFT_EXPERTS), dtype=torch.int64)
    dec[0] = torch.tensor([900, 800, 10, 10, 10, 10, 10, 10])  # rank 0 has it all
    dec[1] = 100  # balanced: nothing to move
    dec[2] = torch.tensor([10, 10, 10, 10, 10, 10, 700, 900])
    mtp.reduced = (dec, 40.0, 12, 40.0, 12)
    plan = manager._plan_mtp(mtp)
    assert plan["swaps"][1] == 0 and plan["moved"][1] == 0
    assert 0 < plan["swaps"][0] <= 4 and 0 < plan["swaps"][2] <= 4
    assert plan["moved"] == [2 * n for n in plan["swaps"]]
    identity = torch.arange(DRAFT_EXPERTS, dtype=torch.int32)
    for n in range(3):
        assert torch.equal(plan["p2l"][n].sort().values, identity)
    assert torch.equal(plan["old"], mtp.p2l) and torch.equal(mtp.p2l[0], identity)

    def decode_rows(p2l, n):
        rk = gp.ranks_of(p2l.numpy().astype(np.int64), DRAFT_PER)[n]
        load = dec[n].numpy().astype(float)
        return gp.placement_metrics(load * 0, load, 40.0, rk, RANKS, TOP_K)[3]

    for n in (0, 2):
        assert decode_rows(plan["p2l"], n) < decode_rows(plan["old"], n) - 0.5
    # the same sums give the same plan
    again = manager._plan_mtp(mtp)
    assert torch.equal(again["p2l"], plan["p2l"])
    # no decode load: nothing moves; the budget bounds a layer
    mtp.reduced = (dec * 0, 0.0, 0, 0.0, 0)
    assert manager._plan_mtp(mtp)["moved"] == [0, 0, 0]
    monkeypatch.setattr(eplb, "MTP_SWAP_BUDGET", 1)
    mtp.reduced = (dec, 40.0, 12, 40.0, 12)
    assert manager._plan_mtp(mtp)["swaps"] == [1, 0, 1]


def test_plan_uses_the_draft_passes_own_operating_point(monkeypatch):
    manager, _draft = _mtp_manager(monkeypatch)
    mtp = manager.gate_permute.mtp
    seen = []
    solve = gp.swap_rebalance

    def recording(pf, dec, rows, p2l, **kw):
        seen.append(
            (
                int(pf.sum()),
                rows,
                kw["top_k"],
                kw["budget"],
                kw["weights"],
                kw["num_gpus"],
            )
        )
        return solve(pf, dec, rows, p2l, **kw)

    monkeypatch.setattr(gp, "swap_rebalance", recording)
    dec = _skewed(1)[:, :DRAFT_EXPERTS]
    mtp.reduced = (dec, 50.0, 9, 120.0, 3)  # 120 rows a step in this window
    manager._plan_mtp(mtp)
    mtp.reduced = (dec, 60.0, 9, 0.0, 0)  # a window without a sample
    manager._plan_mtp(mtp)
    # no prefill load and no prefill term; the draft's rows, top-k and budget
    assert seen == [(0, 120.0, TOP_K, 4, (0.0, 1.0, 0.25), RANKS)] * 2


# ---------------------------------------------------------------- migration
def test_draft_layer_moves_through_the_host_transport_with_its_gate(
    monkeypatch, tmp_path
):
    """Four ranks, each a thread with its own manager, weights and gate, meet
    at the transport's barriers: after one layer's switch every rank holds
    the experts its slots now name, byte for byte, and its gate routes each
    row to the slot of the expert it picked before."""
    barrier = threading.Barrier(RANKS)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: barrier.wait(30))
    ranks = []
    for r in range(RANKS):
        manager, draft = _mtp_manager(monkeypatch, rank=r)
        transport = eplb.HostExpertTransport("cpu", str(tmp_path), r)
        manager._migration_group = manager._host_transport = transport
        ranks.append((manager, draft))
    index, layer_id = 1, DRAFT_IDS[1]
    full = ranks[0][1].full[layer_id]
    pristine = _Gate(experts=DRAFT_EXPERTS, seed=100 + index)
    x = torch.randint(
        -8, 9, (40, 8), generator=torch.Generator().manual_seed(1)
    ).float()
    want_ids, want_weights = _route(pristine, x)
    # two rounds: across ranks from the identity, then on from that placement
    first = torch.tensor([5, 1, 2, 0, 4, 3, 7, 6], dtype=torch.int32)
    second = torch.tensor([5, 2, 1, 6, 4, 3, 7, 0], dtype=torch.int32)
    for new in (first, second):
        errors = []

        def run(manager, new=new, errors=errors):
            try:
                manager._switch_mtp_layer(manager.gate_permute.mtp, index, new)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                barrier.abort()

        threads = [threading.Thread(target=run, args=(m,)) for m, _ in ranks]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        l2p = torch.empty(DRAFT_EXPERTS, dtype=torch.int64)
        l2p[new.long()] = torch.arange(DRAFT_EXPERTS)
        for r, (manager, draft) in enumerate(ranks):
            mtp = manager.gate_permute.mtp
            assert torch.equal(mtp.p2l[index], new)
            assert torch.equal(mtp.fold_index[index], l2p)
            for other in (0, 2):  # the other layers stay where they were
                assert torch.equal(mtp.p2l[other].long(), torch.arange(DRAFT_EXPERTS))
            # the bytes: slot s of rank r holds expert new[r * per + s]
            for name, part in zip(NAMES, full):
                live = getattr(draft.layers[layer_id], name)
                for slot in range(DRAFT_PER):
                    logical = int(new[r * DRAFT_PER + slot])
                    assert torch.equal(live[slot], part[logical])
            # the gate: physical ids = slots of the experts picked before
            ids, weights = _route(draft.gates[layer_id], x)
            assert torch.equal(ids, l2p[want_ids]) and torch.equal(
                weights, want_weights
            )
            assert torch.equal(mtp.p2l[index].long()[ids], want_ids)
            # dispatch by id // per reaches this rank for its own slots only
            for other_layer in (DRAFT_IDS[0], DRAFT_IDS[2]):
                assert torch.equal(
                    draft.gates[other_layer].weight.data,
                    _Gate(
                        experts=DRAFT_EXPERTS, seed=100 + DRAFT_IDS.index(other_layer)
                    ).weight.data,
                )
    assert os.listdir(tmp_path) == []  # the writers removed their files
    moved = sum(m._host_transport.take_stats()[2] for m, _ in ranks)
    assert moved > 0 and all(m._host_transport.bytes_moved >= 0 for m, _ in ranks)


@pytest.mark.parametrize("stage", ["prepare", "p2p", "commit"])
def test_failed_draft_layer_switch_is_fatal(monkeypatch, caplog, stage):
    manager, draft = _mtp_manager(monkeypatch)
    mtp = manager.gate_permute.mtp
    before = {i: g.weight.data.clone() for i, g in draft.gates.items()}

    def fail(*args, **kwargs):
        raise OSError(stage)

    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    if stage == "prepare":
        monkeypatch.setattr(mtp, "stage_commit", fail)
    elif stage == "p2p":
        monkeypatch.setattr(eplb, "_migrate_single_layer", fail)
    else:
        monkeypatch.setattr(eplb, "move_from_buffer", fail)
    with (
        caplog.at_level(logging.ERROR, logger="atom"),
        pytest.raises(OSError, match=stage),
    ):
        manager._switch_mtp_layer(mtp, 1, _cycle())
    assert manager.rebalancing_stopped and manager._fatal
    (fatal,) = _lines(caplog, "FATAL")
    assert fatal.startswith(gp.TAG) and f"draft layer 41 failed at '{stage}'" in fatal
    assert ("half switched" in fatal) == (stage == "commit")
    assert all(torch.equal(g.weight.data, before[i]) for i, g in draft.gates.items())
    assert torch.equal(mtp.p2l[1].long(), torch.arange(DRAFT_EXPERTS))


# ---------------------------------------------------------------- rebalance
def _rebalance(manager):
    return sum(1 for _ in manager._rebalance())


def test_rebalance_plans_and_moves_the_draft_layers_after_the_target_layers(
    monkeypatch, caplog, tmp_path
):
    manager, draft = _mtp_manager(
        monkeypatch, ATOM_EPLB_MAX_REBALANCES=2, ATOM_EPLB_DUMP_DIR=tmp_path
    )
    runtime = manager.gate_permute
    # the scripted policy stands in for the target layers' plan
    calls = _scripted_policy(monkeypatch, [True, False, False, False])

    def draft_plans(moves):
        """The draft layers' plan: layer 41 trades two experts when asked."""

        def plan(m):
            old = m.p2l.clone()
            new = old.clone()
            if moves.pop(0):
                new[1, 0], new[1, 7] = old[1, 7], old[1, 0]
            dec = torch.zeros((3, DRAFT_EXPERTS), dtype=torch.int64)
            dec[1, 0] = 30
            return {
                "old": old,
                "p2l": new,
                "moved": (old != new).sum(dim=1).tolist(),
                "swaps": [0, int(not torch.equal(old, new)), 0],
                "dec": dec,
                "rows": 20.0,
                "samples": 6,
                "window_samples": 6,
            }

        return plan

    monkeypatch.setattr(manager, "_plan_mtp", draft_plans([True, True, False, False]))
    switched = []
    real = manager._switch_mtp_layer
    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    monkeypatch.setattr(
        manager,
        "_switch_mtp_layer",
        lambda m, index, new: switched.append(index) or real(m, index, new),
    )
    gate_before = draft.gates[41].weight.data.clone()
    with caplog.at_level(logging.INFO, logger="atom"):
        # target and draft layers both move: the draft chunk is a step of its own
        runtime.pf_total[0, 0] += 1
        assert _rebalance(manager) == 2 and switched == [1]
        assert manager.rebalance_count == 1 and manager._experts_moved_total == 4
        # only a draft layer moves: still a rebalance that counts
        runtime.pf_total[0, 0] += 1
        assert _rebalance(manager) == 1 and switched == [1, 1]
        assert manager.rebalance_count == 2 and manager._experts_moved_total == 6
        assert manager._idle_evaluations == 0 and manager._freeze_due()
    assert torch.equal(draft.gates[41].weight.data, gate_before)  # traded back
    lines = _lines(caplog, " mtp: swaps")
    assert len(lines) == 2 and lines[0].startswith(gp.TAG)
    assert (
        "rebalance #1 mtp: swaps sum=1 max=1 experts_moved=2 | decode picks=30 "
        in lines[0]
    )
    assert "| decode-row max/mean before=" in lines[0] and " after=" in lines[0]
    assert "| active proxy max before=" in lines[0]
    assert len(_lines(caplog, "mtp chunk layers=[41] migrate=")) == 2
    dumps = [torch.load(f) for f in sorted(tmp_path.glob("window_*.pt"))]
    assert len(dumps) == 2 and all(d["executed"] for d in dumps)
    for key in (
        "mtp_decode_counts_window",
        "mtp_decode_counts_cum",
        "mtp_p2l_before",
        "mtp_p2l_after",
    ):
        assert (
            dumps[0][key].shape == (3, DRAFT_EXPERTS)
            and dumps[0][key].dtype == torch.int64
        )
    assert not torch.equal(dumps[0]["mtp_p2l_before"], dumps[0]["mtp_p2l_after"])
    assert torch.equal(dumps[1]["mtp_p2l_before"], dumps[0]["mtp_p2l_after"])
    assert int(dumps[0]["mtp_decode_counts_window"].sum()) == 30
    assert int(dumps[1]["mtp_decode_counts_window"].sum()) == 0
    assert calls == [True, False]


def test_evaluation_that_moves_neither_layer_set_is_idle(monkeypatch):
    manager, _draft = _mtp_manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=3)
    runtime = manager.gate_permute
    # the real plans on loads that are even: the target layers' prefill picks
    # the same for every expert, the draft layers without a sample
    for _ in range(2):
        runtime.pf_total.add_(1)
        assert _rebalance(manager) == 0
    assert manager.rebalance_count == 0 and manager._idle_evaluations == 2
    assert manager._freeze_due()  # converged


def test_rebalance_reads_the_draft_loads_the_reduce_left(monkeypatch, caplog):
    """The real plan on the real sums, end to end on one rank."""
    manager, draft = _mtp_manager(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    runtime, mtp = manager.gate_permute, manager.gate_permute.mtp
    monkeypatch.setattr(eplb, "_migrate_single_layer", lambda **kw: [])
    held = _record(mtp, draft)
    # every live row of layer 40 picks the experts of rank 0: slots 0 and 1
    held[40][:8] = torch.tensor([0, 1, 0], dtype=torch.int32)
    held[41][:8] = torch.tensor([0, 2, 4], dtype=torch.int32)
    held[42][:8] = torch.tensor([1, 3, 5], dtype=torch.int32)
    for _ in range(5):
        runtime.after_decode_replay("target", 6)
        mtp.after_draft_replay(KEY, 8)
    runtime.pf_total.add_(1)  # even: the target layers stay
    with caplog.at_level(logging.INFO, logger="atom"):
        assert _rebalance(manager) == 1  # the draft chunk alone
    (line,) = _lines(caplog, " mtp: swaps")
    assert "decode picks=360 decode samples/rank=5 rows/step=8.0" in line
    assert mtp.reduced[0][0].tolist() == [80, 40, 0, 0, 0, 0, 0, 0]
    # layer 40's two hot experts no longer share a rank; the others were even
    rk = gp.ranks_of(mtp.p2l.numpy().astype(np.int64), DRAFT_PER)
    assert rk[0][0] != rk[0][1]
    assert manager.rebalance_count == 1 and manager._mtp_rows_at_last_solve == 8.0
    assert (mtp.window_rows, mtp.window_steps) == (0, 0) and mtp.steps == 5
