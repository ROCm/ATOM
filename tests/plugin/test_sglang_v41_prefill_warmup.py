# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the SGLang plugin: the eager prefill warm-up at start-up
(`deepseek_v41_bridge.warm_deepseek_v41_eager_prefill`), on CPU with a
recording builder and model."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import atom.plugin.sglang.runtime.context as runtime_context
from atom.plugin.sglang import deepseek_v41_bridge as bridge
from atom.utils import forward_context


class _Builder:
    block_size = 256
    max_num_batched_tokens = 16512

    def __init__(self):
        self.cache = SimpleNamespace(backing=torch.empty(1, dtype=torch.uint8))
        self.calls = []

    def _prepare(self, batch, running_bs, running_tokens, **kwargs):
        self.calls.append(("prepare", batch, running_bs, running_tokens, kwargs))
        return SimpleNamespace(), torch.arange(running_tokens)

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", int(input_ids.shape[0])))


class _Model:
    def __init__(self, surface):
        self.surface = surface
        self.calls = []

    def __call__(self, **kwargs):
        ctx = forward_context.get_forward_context()
        if self.surface is not None:
            # A forward fills the DSpark taps; the warm-up must drop them.
            assert self.surface._rows == {}
            self.surface._rows["tap"] = 1
        self.calls.append((kwargs, ctx.tr_graph, ctx.attn_metadata))


def _runner(*, dspark=True, chunk=8192, dp=False, slots=(65, 66, 67)):
    surface = SimpleNamespace(_rows={"stale": 1}) if dspark else None
    wrapper = SimpleNamespace(
        atom_config=SimpleNamespace(
            enable_dp_attention=dp, enable_tbo=False, parallel_config=None
        ),
        _filter_model_forward_kwargs=lambda kw: kw,
        _v41_dspark=surface,
    )
    wrapper.model = _Model(surface)
    return SimpleNamespace(
        model=wrapper,
        token_to_kv_pool=SimpleNamespace(capture_slots=slots),
        server_args=SimpleNamespace(chunked_prefill_size=chunk),
    )


@pytest.fixture
def cpu_forward(monkeypatch):
    state = {"set": [], "reset": 0, "ctx": None}

    def set_forward_context(**kwargs):
        state["set"].append(kwargs)
        state["ctx"] = SimpleNamespace(
            attn_metadata=kwargs["attn_metadata"], tr_graph="unset"
        )

    def reset_forward_context():
        state["reset"] += 1
        state["ctx"] = None

    monkeypatch.setattr(forward_context, "set_forward_context", set_forward_context)
    monkeypatch.setattr(forward_context, "reset_forward_context", reset_forward_context)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: state["ctx"])
    monkeypatch.setattr(
        forward_context.DPMetadata,
        "gather_running_tokens",
        staticmethod(lambda parallel, tokens: torch.tensor([tokens])),
    )
    monkeypatch.setattr(
        runtime_context, "plugin_runtime_scope", lambda **kw: nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.delenv(bridge.V41_PREFILL_WARMUP_ENV, raising=False)
    return state


def test_dspark_runs_two_sequences_at_the_chunk_budget(monkeypatch, cpu_forward):
    builder = _Builder()
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    runner = _runner(dspark=True)
    assert bridge.warm_deepseek_v41_eager_prefill(runner) == 8192
    (_, batch, running_bs, running_tokens, kwargs), inputs = builder.calls
    assert (running_bs, running_tokens) == (2, 8192)
    assert kwargs == {"is_prefill": True, "engram_live": False}
    assert batch.num_scheduled_tokens == batch.context_lens == (4096, 4096)
    # Reserved capture STATE slots, every row on padding PAGE 0.
    assert batch.req_ids == batch.state_slots_committed == (65, 66)
    assert batch.block_tables == ((0,) * 16,) * 2
    assert batch.replaying == {} and not batch.is_dummy_run
    assert inputs == ("inputs", 8192)
    ((model_kwargs, tr_graph, metadata),) = runner.model.model.calls
    assert metadata.dummy is True  # no slot reset, Engram lookup or cursor advance
    assert tr_graph is None  # eager: no token graph recorded or replayed
    assert model_kwargs["input_ids"].shape == (8192,)
    assert int(model_kwargs["input_ids"].abs().sum()) == 0
    assert model_kwargs["positions"].shape == (8192,)
    (ctx_kwargs,) = cpu_forward["set"]
    context = ctx_kwargs["context"]
    assert context.is_prefill and not context.is_dummy_run
    assert (context.scheduled_bs, context.scheduled_tokens) == (2, 8192)
    assert (context.running_bs, context.running_tokens) == (2, 8192)
    assert "in_hipgraph" not in ctx_kwargs
    assert cpu_forward["reset"] == 1
    assert runner.model._v41_dspark._rows == {}


def test_without_dspark_one_sequence_and_builder_budget(monkeypatch, cpu_forward):
    builder = _Builder()
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    # No chunk size: the builder's budget.
    runner = _runner(dspark=False, chunk=0)
    assert bridge.warm_deepseek_v41_eager_prefill(runner) == 16512
    batch = builder.calls[0][1]
    assert batch.num_scheduled_tokens == (16512,)
    assert batch.req_ids == (65,)


def test_odd_budget_gives_the_remainder_to_the_last_sequence(monkeypatch, cpu_forward):
    builder = _Builder()
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    assert bridge.warm_deepseek_v41_eager_prefill(_runner(chunk=8191)) == 8191
    assert builder.calls[0][1].num_scheduled_tokens == (4095, 4096)


def test_context_is_reset_when_the_forward_raises(monkeypatch, cpu_forward):
    builder = _Builder()
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    runner = _runner()

    def boom(**kwargs):
        raise RuntimeError("kernel")

    runner.model.model = boom
    with pytest.raises(RuntimeError, match="kernel"):
        bridge.warm_deepseek_v41_eager_prefill(runner)
    assert cpu_forward["reset"] == 1


@pytest.mark.parametrize(
    "case",
    ["kill_switch", "no_builder", "no_cache", "dp_attention", "no_capture_slots"],
)
def test_skips(monkeypatch, cpu_forward, case):
    builder = _Builder()
    runner = _runner()
    if case == "kill_switch":
        monkeypatch.setenv(bridge.V41_PREFILL_WARMUP_ENV, "0")
    elif case == "no_builder":
        builder = None
    elif case == "no_cache":
        builder.cache = None
    elif case == "dp_attention":
        runner = _runner(dp=True)
    elif case == "no_capture_slots":
        runner = _runner(slots=(65,))
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    assert bridge.warm_deepseek_v41_eager_prefill(runner) == 0
    assert cpu_forward["set"] == []
    if builder is not None:
        assert builder.calls == []


def test_kill_switch_values(monkeypatch):
    for raw, on in (
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("", False),
    ):
        monkeypatch.setenv(bridge.V41_PREFILL_WARMUP_ENV, raw)
        assert bridge.v41_prefill_warmup_enabled() is on
    monkeypatch.delenv(bridge.V41_PREFILL_WARMUP_ENV)
    assert bridge.v41_prefill_warmup_enabled()
