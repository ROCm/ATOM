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


_CHUNK = {}


def _runner(*, dspark=True, chunk=8192, slots=(65, 66, 67)):
    surface = SimpleNamespace(_rows={"stale": 1}) if dspark else None
    wrapper = SimpleNamespace(
        atom_config=SimpleNamespace(parallel_config=None),
        _filter_model_forward_kwargs=lambda kw: kw,
        _v41_dspark=surface,
    )
    wrapper.model = _Model(surface)
    # SGLang's resolved schedule bag (`cpu_forward` serves it).
    _CHUNK["size"] = chunk
    return SimpleNamespace(
        model=wrapper, token_to_kv_pool=SimpleNamespace(capture_slots=slots)
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
    from sglang.srt import runtime_context as sglang_config

    monkeypatch.setattr(
        sglang_config,
        "get_schedule",
        lambda: SimpleNamespace(chunked_prefill_size=_CHUNK["size"]),
    )
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
    # A chunk past the builder's budget: the builder's budget.
    runner = _runner(dspark=False, chunk=32768)
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


def test_one_capture_slot_runs_one_sequence(monkeypatch, cpu_forward):
    builder = _Builder()
    monkeypatch.setattr(bridge, "_V41_BUILDER", builder)
    assert bridge.warm_deepseek_v41_eager_prefill(_runner(slots=(65,))) == 8192
    assert builder.calls[0][1].num_scheduled_tokens == (8192,)


def test_switch_parses_as_every_plugin_knob(monkeypatch):
    from atom.plugin.sglang import deepseek_v41_envs

    for raw, on in (
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("", False),
    ):
        monkeypatch.setenv("ATOM_SGLANG_V41_PREFILL_WARMUP", raw)
        assert deepseek_v41_envs.ATOM_SGLANG_V41_PREFILL_WARMUP is on
    monkeypatch.delenv("ATOM_SGLANG_V41_PREFILL_WARMUP")
    assert deepseek_v41_envs.ATOM_SGLANG_V41_PREFILL_WARMUP


# ---------------------------------------------------------------------------
# The same paths through the plugin builder's own `_prepare`
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin_builder(monkeypatch):
    """The plugin's builder class with its own ``_prepare`` and
    ``prepare_model_inputs``: only what they call underneath, native
    ``DeepseekV41MetadataBuilder._prepare`` / ``prepare_model_inputs`` (the
    staging the kernels read) and ``publish_query_metadata``, is recorded. A
    batch missing a field the plugin's ``_prepare`` reads fails here, as at
    start-up."""
    pytest.importorskip("aiter", reason="the V4.1 builder module reaches AITER")
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    cls = bridge._make_builder_cls()
    builder = cls.__new__(cls)
    builder.block_size = 256
    builder.max_bs = 8
    builder.max_num_batched_tokens = 16512
    owner = SimpleNamespace(begin=lambda: None, finish=lambda: None, fail=lambda: None)
    builder.geometry = SimpleNamespace(window_size=128, speculative_tokens=0)
    builder.cache = SimpleNamespace(
        backing=torch.empty(1, dtype=torch.uint8),
        geometry=builder.geometry,
        _sglang_h2d_owner=owner,
    )
    builder.model_runner = SimpleNamespace(h2d_owner=owner, h2d_groups={})
    builder.publish_query_metadata = lambda *args: None
    prepared, inputs = [], []

    def native_prepare(self, batch, running_bs, running_tokens, **kwargs):
        # What native's builder would publish for this batch's decoder.
        prepared.append((batch, self._sglang_decoder_replays, kwargs))
        return SimpleNamespace(cache=self.cache, dummy=False), torch.arange(
            running_tokens
        )

    monkeypatch.setattr(DeepseekV41MetadataBuilder, "_prepare", native_prepare)
    monkeypatch.setattr(
        DeepseekV41MetadataBuilder,
        "prepare_model_inputs",
        lambda self, input_ids, metadata: inputs.append(int(input_ids.shape[0])),
    )
    return SimpleNamespace(builder=builder, prepared=prepared, inputs=inputs)


class TestThroughThePluginPrepare:
    def test_capture_step(self, plugin_builder):
        ids = torch.zeros(300, dtype=torch.int64)
        metadata, context = bridge._prefill_capture_step(
            plugin_builder.builder,
            (128, 128, 44),
            finishing=True,
            slots=(65, 66, 67),
            input_ids=ids,
        )
        ((batch, decoder, kwargs),) = plugin_builder.prepared
        # Native's capture runs the builder's configured decoder.
        assert batch.decoder_replays is True and decoder is True
        assert kwargs == {"is_prefill": True, "engram_live": False}
        assert metadata.dummy and plugin_builder.inputs == [300]
        assert context.running_tokens == 300
        # The per-step verdict is reset after the step.
        assert plugin_builder.builder._sglang_decoder_replays is True

    def test_token_graph_capture(self, plugin_builder):
        builder = plugin_builder.builder
        ids = torch.zeros(512, dtype=torch.int64)
        builder._sglang_capture_inputs = ((65, 66, 67, 68), ids)
        metadata, context = builder.build_for_token_graph_capture(300)
        ((batch, decoder, _),) = plugin_builder.prepared
        assert batch.num_scheduled_tokens == (128, 128, 44) and decoder is True
        assert batch.state_slots_committed == (65, 66, 67)
        assert metadata.dummy and context.running_tokens == 300

    def test_start_up_warm_up(self, plugin_builder, monkeypatch, cpu_forward):
        monkeypatch.setattr(bridge, "_V41_BUILDER", plugin_builder.builder)
        assert bridge.warm_deepseek_v41_eager_prefill(_runner(dspark=True)) == 8192
        ((batch, decoder, _),) = plugin_builder.prepared
        assert batch.num_scheduled_tokens == (4096, 4096) and decoder is True
        assert plugin_builder.inputs == [8192]
