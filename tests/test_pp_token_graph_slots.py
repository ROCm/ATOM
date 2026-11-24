# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefill token graphs under pipeline parallelism (`ATOM_PP_CUDAGRAPH`).

Each forward_vars ring slot records its own token graphs, so a replay reads
the staging buffers of the slot the step was prepared in. No device is used:
a recording is replaced by one that, like a CUDA graph, keeps reading the
tensors it was recorded with (here the recording's forward context, which
holds that slot's `scheduled_tokens_gpu`) and re-reads their contents at
replay. ModelRunner's methods are compiled from its source with stubs for the
module globals they read, as importing it pulls in the GPU stack."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.utils import IntermediateTensors
from atom.utils import forward_context, region_graph
from atom.utils.region_graph import TokenGraph, token_graph

MODEL_RUNNER = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
HIDDEN = 4


class _Recording:
    """A recording that reads what it was recorded reading: the inputs
    `TokenGraph` handed it and the recording step's forward context."""

    replays = 0

    def __init__(self, fn, inputs, kwargs, baked):
        self.fn, self.inputs, self.kwargs, self.baked = fn, inputs, kwargs, baked

    def replay(self):
        _Recording.replays += 1
        live = forward_context._forward_context
        forward_context._forward_context = self.baked
        Stage.replaying = True
        try:
            return self.fn(*self.inputs, **self.kwargs)
        finally:
            Stage.replaying = False
            forward_context._forward_context = live


def _fake_record(fn, *inputs, pool=None, **kwargs):
    fc = forward_context.get_forward_context()
    baked = SimpleNamespace(context=fc.context, tr_graph=None, token_graph_variant=0)
    return _Recording(fn, inputs, kwargs, baked), fn(*inputs, **kwargs)


@pytest.fixture(autouse=True)
def fake_graphs(monkeypatch):
    monkeypatch.setattr(TokenGraph, "_record", staticmethod(_fake_record))
    monkeypatch.setattr(region_graph, "graph_pool", lambda: "pool")
    monkeypatch.setattr(TokenGraph, "_live", type(TokenGraph._live)())
    monkeypatch.setattr(TokenGraph, "_sizes", {})
    monkeypatch.setattr(forward_context, "_forward_context", SimpleNamespace())
    _Recording.replays = 0


@token_graph(per_phase=True)  # as MiniMaxM3Model
class Stage(torch.nn.Module):
    """A PP stage: its rows times the step's `scheduled_tokens_gpu`, which a
    segment reads in place, as MiniMax-M3's MoE sort does."""

    eager_calls = 0
    replaying = False

    def forward(self, input_ids, positions, intermediate_tensors=None):
        Stage.eager_calls += not Stage.replaying
        if intermediate_tensors is None:
            hidden = input_ids.float().unsqueeze(-1).expand(-1, HIDDEN)
        else:
            hidden = intermediate_tensors["hidden_states"]
        scale = forward_context.get_forward_context().context.scheduled_tokens_gpu
        return IntermediateTensors(
            {"hidden_states": hidden * scale, "residual": hidden.clone()}
        )

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                k: torch.zeros(batch_size, HIDDEN, dtype=dtype)
                for k in ("hidden_states", "residual")
            }
        )


# ---------------------------------------------------------------- runner --


class _Buf:
    """`CpuGpuBuffer`: a host array and its device copy."""

    def __init__(self, n):
        self.np = np.zeros(n, dtype=np.int32)
        self.gpu = torch.zeros(n, dtype=torch.int32)

    def clone(self):
        return _Buf(len(self.np))

    def copy_to_gpu(self, n=None):
        self.gpu.copy_(torch.from_numpy(self.np))
        return self.gpu


class _Owner:
    def begin(self):
        pass

    def finish(self):
        pass

    def fail(self):
        pass


class _Event:
    def record(self):
        pass


def _extract(*names):
    tree = ast.parse(MODEL_RUNNER.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == set(names)
    for f in funcs:
        f.decorator_list = []
    return funcs


def _runner_class(monkeypatch, *, pp_size, pp_cudagraph, pp_rank=1, sizes=(8, 4)):
    envs = SimpleNamespace(ATOM_PP_CUDAGRAPH=pp_cudagraph, ATOM_TOKEN_GRAPH_SIZES=sizes)
    pp_group = SimpleNamespace(world_size=pp_size, is_first_rank=pp_rank == 0)

    def set_forward_context(
        attn_metadata, atom_config, context, running_tokens_per_dp_rank, **kwargs
    ):
        forward_context._forward_context = SimpleNamespace(
            context=context,
            tr_graph=None,
            token_graph_variant=kwargs.get("token_graph_variant", 0),
        )

    scope = {
        "envs": envs,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(Event=_Event, synchronize=lambda: None)
        ),
        "CpuGpuBuffer": _Buf,
        "logger": SimpleNamespace(info=lambda *a, **k: None),
        "TokenGraph": TokenGraph,
        "set_forward_context": set_forward_context,
        "get_forward_context": forward_context.get_forward_context,
        "get_pp_group": lambda: pp_group,
    }
    methods = _extract(
        "_init_forward_vars_ring",
        "_advance_forward_vars",
        "_select_forward_vars",
        "_record_tr_graphs",
        "_build_capture",
        "_run_capture_prefill",
    )
    exec(  # noqa: S102 - only this repository's own source
        compile(ast.Module(methods, []), str(MODEL_RUNNER), "exec"), scope
    )
    return type("Runner", (), {f.name: scope[f.name] for f in methods})


class _Builder:
    def __init__(self, runner):
        self.runner = runner
        self.blanked = []

    def blank_cache_write_targets(self):
        self.blanked.append(self.runner._fv_idx)

    def build_for_token_graph_capture(self, tokens):
        context = SimpleNamespace(
            positions=torch.zeros(tokens, dtype=torch.int64),
            scheduled_bs=1,
            scheduled_tokens=tokens,
            scheduled_tokens_gpu=None,
            is_prefill=True,
        )
        return SimpleNamespace(), context


def _runner(monkeypatch, *, pp_size=4, pp_cudagraph=True, pp_rank=1, sizes=(8, 4)):
    cls = _runner_class(
        monkeypatch,
        pp_size=pp_size,
        pp_cudagraph=pp_cudagraph,
        pp_rank=pp_rank,
        sizes=sizes,
    )
    runner = cls()
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp_size, torch_dtype=torch.float32
    )
    runner.enforce_eager = False
    runner.device = "cpu"
    runner.rank = 0
    runner.forward_vars = {
        "input_ids": _Buf(16),
        "decode_src": _Buf(16),
        "batch_info": _Buf(3),
    }
    runner.tokenID_processor = SimpleNamespace()
    return runner


def _ready(runner):
    """The ring, one publication owner per slot, a stage model and its builder."""
    runner._init_forward_vars_ring()
    runner._h2d_owners = [_Owner() for _ in runner._fv_ring]
    runner._h2d_groups = [{} for _ in runner._fv_ring]
    runner.h2d_owner = runner._h2d_owners[runner._fv_idx]
    runner.h2d_groups = runner._h2d_groups[runner._fv_idx]
    runner.model = Stage()
    runner.attn_metadata_builder = _Builder(runner)
    runner.running_tokens_per_dp_rank = np.zeros(1, dtype=np.int32)
    runner.warmup_max_tokens = lambda: 16
    return runner


def _replay_step(runner, slot, tokens, received, *, scale, variant=None):
    """A non-first stage's prefill replay as `run_model` drives it: the slot
    `forward` rotated to, its staged `batch_info`, the received hidden states."""
    runner._select_forward_vars(slot)
    batch_info = runner.forward_vars["batch_info"]
    batch_info.np[:] = (1, scale, 0)
    batch_info.copy_to_gpu()
    context = SimpleNamespace(scheduled_tokens_gpu=batch_info.gpu[1:2], is_prefill=True)
    forward_context._forward_context = SimpleNamespace(
        context=context,
        tr_graph="replay",
        token_graph_variant=runner._fv_idx if variant is None else variant,
    )
    ids = runner.forward_vars["input_ids"].gpu[:tokens]
    # By name, as `run_model` hands it over and `_run_capture_prefill` records.
    out = runner.model(
        ids, ids.long(), intermediate_tensors=IntermediateTensors(received)
    )
    return out["hidden_states"]


def _received(tokens, value):
    return {
        "hidden_states": torch.full((tokens, HIDDEN), float(value)),
        "residual": torch.zeros(tokens, HIDDEN),
    }


# ----------------------------------------------------------------- tests --


def test_pp_without_the_flag_still_requires_eager(monkeypatch):
    runner = _runner(monkeypatch, pp_cudagraph=False)
    with pytest.raises(AssertionError, match="--enforce-eager"):
        runner._init_forward_vars_ring()
    runner.enforce_eager = True
    runner._init_forward_vars_ring()
    assert len(runner._fv_ring) == 4


def test_each_slot_records_its_own_graphs_on_its_own_buffers(monkeypatch):
    runner = _ready(_runner(monkeypatch))
    runner._fv_idx = 2  # wherever warmup left the ring
    runner._record_tr_graphs()
    assert runner._fv_idx == 2
    # Every slot but the one capture_cudagraph already blanked, per bucket.
    assert sorted(set(runner.attn_metadata_builder.blanked)) == [0, 1, 3]

    graphs = [region_graph._variant_graph(runner.model, v) for v in range(4)]
    assert len({id(g) for g in graphs}) == 4
    for slot, graph in enumerate(graphs):
        assert len(graph.keys()) == 2  # buckets 8 and 4
        for segments, _, _ in graph._buckets.values():
            # The batch_info this slot's recording reads in place is its own.
            baked = segments.baked.context.scheduled_tokens_gpu
            own = runner._fv_ring[slot]["batch_info"].gpu
            assert baked.data_ptr() == own[1:2].data_ptr()
    # Staging buffers are distinct per slot; the copy-in inputs are shared,
    # since replays run in stream order.
    ptrs = {s["batch_info"].gpu.data_ptr() for s in runner._fv_ring}
    assert len(ptrs) == 4
    assert all(g._buffers is graphs[0]._buffers for g in graphs)
    assert TokenGraph.captured_sizes() == (4, 8)


def test_a_replay_reads_its_own_slot_and_the_received_states(monkeypatch):
    runner = _ready(_runner(monkeypatch))
    runner._record_tr_graphs()
    Stage.eager_calls = 0
    # Stage every slot first, as the head's back-to-back microbatches do, then
    # replay each: none may see another slot's staging.
    scales = {0: 3, 1: 5, 2: 7, 3: 11}
    for slot, scale in scales.items():
        runner._select_forward_vars(slot)
        info = runner.forward_vars["batch_info"]
        info.np[:] = (1, scale, 0)
        info.copy_to_gpu()
    for slot, scale in scales.items():
        out = _replay_step(runner, slot, 8, _received(8, slot + 1), scale=scale)
        torch.testing.assert_close(out, torch.full((8, HIDDEN), (slot + 1.0) * scale))
    assert Stage.eager_calls == 0
    assert _Recording.replays == 4


def test_replaying_another_slots_graph_reads_that_slots_staging(monkeypatch):
    """What per-slot recording prevents: one graph set serving every slot."""
    runner = _ready(_runner(monkeypatch))
    runner._record_tr_graphs()
    runner._select_forward_vars(0)
    info = runner.forward_vars["batch_info"]
    info.np[:] = (1, 3, 0)
    info.copy_to_gpu()
    out = _replay_step(runner, 1, 4, _received(4, 1), scale=5, variant=0)
    torch.testing.assert_close(out, torch.full((4, HIDDEN), 3.0))


def test_a_bucket_a_slot_did_not_record_runs_eagerly(monkeypatch):
    runner = _ready(_runner(monkeypatch, sizes=(8,)))
    runner._record_tr_graphs()
    # Slot 3 loses its recording of bucket 8: that step runs the module.
    region_graph._variant_graph(runner.model, 3)._buckets.clear()
    Stage.eager_calls = 0
    out = _replay_step(runner, 3, 8, _received(8, 2), scale=5)
    torch.testing.assert_close(out, torch.full((8, HIDDEN), 10.0))
    assert Stage.eager_calls == 1 and _Recording.replays == 0
    # The other slots still replay.
    _replay_step(runner, 2, 8, _received(8, 2), scale=5)
    assert Stage.eager_calls == 1 and _Recording.replays == 1


def test_the_first_stage_records_without_received_states(monkeypatch):
    runner = _ready(_runner(monkeypatch, pp_rank=0, sizes=(4,)))
    runner._record_tr_graphs()
    graph = region_graph._variant_graph(runner.model, 0)
    ((_, inputs, _),) = graph._buckets.values()
    assert len(inputs) == 2  # ids and positions only


def test_without_pp_one_slot_records_as_before(monkeypatch):
    runner = _ready(_runner(monkeypatch, pp_size=1, pp_rank=0, sizes=(8, 4)))
    runner._record_tr_graphs()
    assert len(runner._fv_ring) == 1
    assert "_token_graph_variants" not in runner.model.__dict__
    assert len(runner.model.__dict__["_token_graph"].keys()) == 2
    assert runner.attn_metadata_builder.blanked == []


def test_received_states_are_copied_into_the_recorded_inputs():
    """`IntermediateTensors` arguments: bucketed by their shapes, copied in."""
    graph = TokenGraph()
    seen = []

    def forward(ids, it):
        seen.append(it["hidden_states"].data_ptr())
        return it["hidden_states"] + 1

    forward_context._forward_context = SimpleNamespace(context=None)
    first = IntermediateTensors({"hidden_states": torch.ones(4, HIDDEN)})
    graph.record(forward, (torch.zeros(4), first))
    key = TokenGraph.key((torch.zeros(4), first), {})
    assert key == ((4,), (4, HIDDEN))
    recorded = graph.inputs(key)[1]["hidden_states"]
    fresh = IntermediateTensors({"hidden_states": torch.full((4, HIDDEN), 6.0)})
    out = graph.replay((torch.zeros(4), fresh))
    torch.testing.assert_close(recorded, fresh["hidden_states"])
    torch.testing.assert_close(out, torch.full((4, HIDDEN), 7.0))
    assert seen[-1] == recorded.data_ptr()


def test_a_replay_refuses_tensors_its_recording_did_not_take():
    graph = TokenGraph()
    forward_context._forward_context = SimpleNamespace(context=None)
    it = IntermediateTensors({"hidden_states": torch.ones(4, HIDDEN)})
    graph.record(
        lambda ids, intermediate_tensors: ids,
        (torch.zeros(4),),
        {"intermediate_tensors": it},
    )
    key = TokenGraph.key((torch.zeros(4),), {"intermediate_tensors": it})
    with pytest.raises(TypeError, match="did not take"):
        graph.replay((torch.zeros(4), it), key=key)


def test_weak_output_keeps_intermediate_tensors():
    out = IntermediateTensors({"hidden_states": torch.ones(2, HIDDEN)})
    weak = region_graph._weak_output(out)
    assert isinstance(weak, IntermediateTensors)
    assert weak["hidden_states"].data_ptr() == out["hidden_states"].data_ptr()


def _method(name):
    tree = ast.parse(MODEL_RUNNER.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


@pytest.mark.parametrize("method", ["prepare_inputs", "_run_capture_prefill"])
def test_the_forward_context_names_the_steps_slot(method):
    """A live step and a recording both key their token graphs by the slot
    whose buffers they staged into."""
    calls = [
        node
        for node in ast.walk(_method(method))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "set_forward_context"
    ]
    assert len(calls) == 1
    (variant,) = [k for k in calls[0].keywords if k.arg == "token_graph_variant"]
    assert ast.unparse(variant.value) == "self._fv_idx"
