# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU regressions for V4 Mono admission, rollback and native metadata ownership."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "atom/models/deepseek_v4/mono"


def load(monkeypatch, name):
    qualname = f"atom.models.deepseek_v4.mono.{name}"
    spec = importlib.util.spec_from_file_location(
        qualname, MONO / (name.replace(".", "/") + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, qualname, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return NS(
        speculative_config=None,
        tensor_parallel_size=8,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        enable_tbo=False,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        parallel_config=NS(data_parallel_size=1),
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        enable_prefix_caching=False,
        max_num_seqs=512,
        dspark=NS(confidence_schedule=False, ragged=False),
        capture_sizes=[1, 2, 4, 8],
        compilation_config=NS(
            level=3,
            cudagraph_mode=NS(
                name="FULL",
                uses_tr_hybrid=lambda: False,
                decode_mode=lambda: NS(name="FULL"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 4),
        ("enable_expert_parallel", True),
        ("enable_dp_attention", True),
        ("enable_tbo", True),
        ("enable_tbo_decode", True),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("kv_cache_dtype", "bf16"),
        ("index_cache_dtype", "fp8"),
        ("enable_prefix_caching", True),
        ("enable_rapidserve", True),
        ("disagg_is_decode", True),
    ],
)
def test_admission_refuses_unsupported_deployments(monkeypatch, config, field, value):
    profile = load(monkeypatch, "profile")
    model_args = NS(use_index_cache=False, index_topk_freq=1)
    assert (
        profile.deployment_refusal(
            config, profile.ORDINARY_PROFILE, model_args=model_args
        )
        is None
    )
    setattr(config, field, value)
    assert profile.deployment_refusal(
        config, profile.ORDINARY_PROFILE, model_args=model_args
    )


@pytest.mark.parametrize(
    "method,k,block,accepted",
    [
        ("mtp", 3, False, True),
        ("mtp", 1, False, False),
        ("dspark", 7, True, False),
        ("eagle", 3, False, False),
        ("mtp", 3, True, False),
    ],
)
def test_only_ordinary_and_native_mtp3_are_deployment_qualified(
    monkeypatch, config, method, k, block, accepted
):
    p = load(monkeypatch, "profile")
    assert p.select_profile(config, None) is p.ORDINARY_PROFILE
    config.speculative_config = NS(method=method, num_speculative_tokens=k)
    draft = NS(mtp_k=k, is_block_drafter=block)
    selected = p.select_profile(config, draft)
    if method == "dspark":
        assert selected.tokens_per_seq == k + 1
        assert not selected.deployment_qualified
    else:
        assert selected is (p.MTP3_TARGET_PROFILE if accepted else None)
    assert (
        p.deployment_refusal(
            config, selected, model_args=NS(use_index_cache=False, index_topk_freq=1)
        )
        is None
    ) == accepted


@pytest.mark.parametrize(
    "mtp,reuse,freq,pattern,accepted",
    [
        (False, False, 1, None, True),
        (False, True, 4, None, True),
        (True, False, 1, None, True),
        (True, True, 4, None, False),
        (False, True, 4, "custom", False),
        (False, True, 2, None, False),
    ],
)
def test_index_policy_is_not_widened(
    monkeypatch, config, mtp, reuse, freq, pattern, accepted
):
    p = load(monkeypatch, "profile")
    selected = p.MTP3_TARGET_PROFILE if mtp else p.ORDINARY_PROFILE
    args = NS(use_index_cache=reuse, index_topk_freq=freq, index_topk_pattern=pattern)
    assert (p.deployment_refusal(config, selected, model_args=args) is None) == accepted


class NativeBuilder:
    def __init__(self):
        self.events = []
        self.pool_geometry = object()
        self.model_runner = NS(attn_metadata_builder=self)

    def build(
        self, batch, running_bs, running_tokens, max_seqlen_q, *, ubatch_split_bs=None
    ):
        self.events.append(
            ("build", batch, running_bs, running_tokens, max_seqlen_q, ubatch_split_bs)
        )
        return self.events[-1]

    def invalidate_pool_rows(self):
        self.events.append("invalidate")

    def allocate_kv_cache_tensors(self, *, blocks, buf):
        self.events.append("allocate")
        return buf

    def build_for_cudagraph_capture(self, bs, max_q_len=None):
        self.events.append("capture")
        return bs, max_q_len

    def blank_cache_write_targets(self):
        self.events.append("blank")

    def close(self):
        self.events.append("close")


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_native_build_runs_once_with_original_arguments(monkeypatch, keyword, empty):
    native = NativeBuilder()
    wrapper = load(monkeypatch, "metadata").MonoMetadataBuilder(native)
    batch = NS(state_maintenance_ops=NS(empty=empty))
    # Target-only execution must retain the original native build contract.
    wrapper._mono_session = NS(_has_draft_resources=lambda: False)
    result = (
        wrapper.build(
            batch=batch,
            running_bs=4,
            running_tokens=16,
            max_seqlen_q=4,
            ubatch_split_bs=2,
        )
        if keyword
        else wrapper.build(batch, 4, 16, 4, ubatch_split_bs=2)
    )
    assert result == ("build", batch, 4, 16, 4, 2)
    assert native.events == [result]
    assert wrapper._mono_batch is batch
    with pytest.raises(TypeError, match="unexpected keyword"):
        wrapper.build(batch, 4, 16, 4, invalid=True)


def test_metadata_lifecycle_and_state_are_delegated(monkeypatch):
    native = NativeBuilder()
    wrapper = load(monkeypatch, "metadata").MonoMetadataBuilder(native)
    assert native.model_runner.attn_metadata_builder is native
    assert wrapper.pool_geometry is native.pool_geometry
    wrapper.num_blocks = 12
    assert native.num_blocks == 12
    session = NS(adapter=None)

    def bind():
        native.events.append("bind")
        session.adapter = object()

    session.engine = NS(
        cache_rebinding=lambda: native.events.append("retire"),
        cache_bound=bind,
        bind_graphs=lambda: native.events.append("graphs"),
        shutdown=lambda: native.events.append("shutdown"),
    )
    wrapper._mono_session = session
    wrapper.invalidate_pool_rows()
    wrapper.prepare_mono()
    value = object()
    assert wrapper.allocate_kv_cache_tensors(blocks=4, buf=value) is value
    wrapper.blank_cache_write_targets()
    wrapper._mono_batch = object()
    assert wrapper.build_for_cudagraph_capture(2, 4) == (2, 4)
    assert wrapper._mono_batch is None
    wrapper.prepare_mono()
    wrapper.close()
    assert native.events == [
        "retire",
        "invalidate",
        "allocate",
        "graphs",
        "blank",
        "bind",
        "capture",
        "shutdown",
        "close",
    ]


@pytest.fixture
def installation(monkeypatch, config):
    from atom.mono.runtime import deployment

    load(monkeypatch, "profile")
    load(monkeypatch, "metadata")
    dispatch = load(monkeypatch, "dispatch")
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setattr(deployment, "_takes_weight_updates", lambda _: False)
    inner = torch.nn.Module()
    inner.embed = torch.nn.Identity()
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.attn = torch.nn.Module()
        layer.attn.indexer = torch.nn.Module()
        layer.attn.indexer.weights_proj = torch.nn.Identity()
        layers.append(layer)
    inner.layers = torch.nn.ModuleList(layers)
    model = torch.nn.Module()
    model.model = inner
    model.args = NS(use_index_cache=False, index_topk_freq=1)
    native = NativeBuilder()
    state = NS(failure=None, sessions=[], wrappers=0)

    class Session:
        def __init__(self, target, cfg, builder, profile):
            self.group = None
            self.builder = builder
            self.profile = profile
            self.closed = False
            state.sessions.append(self)

        def prepare_storage(self):
            assert native.model_runner.attn_metadata_builder is native
            if state.failure == "prepare_raise":
                raise ValueError("unsupported geometry")
            return state.failure != "prepare_false"

        def close(self):
            self.closed = True

    class Wrapper(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            state.wrappers += 1
            if state.failure == "projection" and state.wrappers == 3:
                raise ValueError("quantized projection")
            self.native = original

    for name, attr, cls in [
        ("session", "V4MonoSession", Session),
        ("embedding", "MonoEmbedding", Wrapper),
        ("projection", "MonoIndexerProjection", Wrapper),
    ]:
        module = ModuleType(f"atom.models.deepseek_v4.mono.{name}")
        setattr(module, attr, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return dispatch, model, native, state


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize(
    "failure", ["projection", "prepare_false", "prepare_raise", "peer_refusal", None]
)
def test_installation_is_atomic(monkeypatch, config, installation, mtp, failure):
    dispatch, model, native, state = installation
    state.failure = failure
    originals = [model.model.embed] + [
        x.attn.indexer.weights_proj for x in model.model.layers
    ]
    draft = None
    block = NS(embed=model.model.embed, head=object())
    if mtp:
        config.speculative_config = NS(method="mtp", num_speculative_tokens=3)
        draft = NS(mtp_k=3, model=NS(model=NS(mtp=[block])))
    if failure == "peer_refusal":
        from atom.mono.runtime.consensus import MonoUnsupported

        def peer_refused(fn, group):
            fn()
            if model.model.embed is not originals[0]:
                raise MonoUnsupported("another TP rank refused mono")

        monkeypatch.setattr(dispatch, "bind_agreed", peer_refused)
    result = dispatch.install_mono_decode(model, config, draft, metadata_builder=native)
    modules = [model.model.embed] + [
        x.attn.indexer.weights_proj for x in model.model.layers
    ]
    if failure is not None:
        assert result is model
        assert all(before is after for before, after in zip(originals, modules))
        assert block.embed is originals[0]
        assert native.model_runner.attn_metadata_builder is native
        assert state.sessions[0].closed
    else:
        assert result.model is model
        assert all(after.native is before for before, after in zip(originals, modules))
        assert native.model_runner.attn_metadata_builder._mono_session is result._mono
        assert not state.sessions[0].closed
        if mtp:
            assert block.embed is model.model.embed


@pytest.mark.parametrize("reason", ["off", "prefix", "rollout"])
def test_early_refusal_does_not_construct_wrappers(
    monkeypatch, config, installation, reason
):
    dispatch, model, native, state = installation
    if reason == "off":
        monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    elif reason == "prefix":
        config.enable_prefix_caching = True
    else:
        from atom.mono.runtime import deployment

        monkeypatch.setattr(deployment, "_takes_weight_updates", lambda _: True)
    assert (
        dispatch.install_mono_decode(model, config, None, metadata_builder=native)
        is model
    )
    assert state.sessions == []
    assert state.wrappers == 0


@pytest.mark.parametrize(
    "field,value", [("eps", 1e-5), ("route_scale", 1.5), ("rope_dim", 32)]
)
def test_kernel_build_receives_keyed_numerical_config(monkeypatch, field, value):
    cfg_module = load(monkeypatch, "config")
    build = load(monkeypatch, "build")
    cfg = cfg_module.V4Config()
    changed = replace(cfg, **{field: value})
    kwargs = build.kernel_kwargs(
        changed,
        tokens_per_seq=1,
        timeline=False,
        moe_mode="a8w4",
        index_topk_refresh=True,
    )
    assert kwargs[field] == value

    def key(c):
        return cfg_module._variant_key(
            c, 1, 8, "a8w4", False, cfg_module.POLL_TIMEOUT_US
        )

    assert key(cfg) != key(changed)
