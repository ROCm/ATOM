# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen's installer owns admission and zero-copy GDN cache adaptation."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.qwen4_exp_mono import config, dispatch
from atom.models.qwen4_exp_mono.cache import MonoCacheBuilder
from atom.models.qwen4_exp_mono.shapes import EXPECTED_LAYER_TYPES


def deployment(tp=1, ep=False):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
        compilation_config=None,
        max_model_len=32768,
        speculative_config=None,
        kv_cache_dtype="bf16",
        torch_dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            num_hidden_layers=48,
            hidden_size=2560,
            hc_count=4,
            hc_lowrank=320,
            num_experts=512,
            num_experts_per_tok=10,
            moe_intermediate_size=640,
            shared_expert_intermediate_size=640,
            layer_types=EXPECTED_LAYER_TYPES,
            ple_layer_ids=[2],
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_attention_heads=24,
            num_key_value_heads=2,
            head_dim=256,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            ple_conv_kernel_size=4,
            ngram_size=3,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            mamba_ssm_dtype="float32",
            output_gate_type="sigmoid",
        ),
    )


@pytest.mark.parametrize(
    "tp,ep,accepted",
    [(1, False, True), (8, True, True), (1, True, False), (8, False, False)],
)
def test_ep_admission_does_not_mutate_engine_config(monkeypatch, tp, ep, accepted):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    candidate = deployment(tp, ep)
    assert (config.config_refusal(candidate) is None) is accepted
    assert candidate.enable_expert_parallel is ep


def test_disabled_installer_keeps_native_model_and_builder(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    model = torch.nn.Linear(2, 2)
    owner = SimpleNamespace()
    native = SimpleNamespace(model_runner=owner)
    owner.attn_metadata_builder = native
    assert dispatch.install_mono_decode(model, deployment(), None, native) is model
    assert owner.attn_metadata_builder is native


def test_installer_adapts_existing_builder_without_importing_runner(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    model = torch.nn.Linear(2, 2)
    owner = SimpleNamespace()
    native = SimpleNamespace(model_runner=owner)
    owner.attn_metadata_builder = native
    wrapped = dispatch.install_mono_decode(model, deployment(), None, native)
    assert isinstance(wrapped, dispatch.MonoDecodeModel)
    assert wrapped._mono.runner is None
    assert wrapped.in_features == model.in_features
    assert isinstance(owner.attn_metadata_builder, MonoCacheBuilder)
    assert owner.attn_metadata_builder._native is native
    assert not hasattr(wrapped, "cudagraph_key")
    assert not hasattr(wrapped, "cudagraph_capture")


@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize(
    "transposed,replayssm,enabled",
    [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_cache_binding_preserves_storage_and_native_layout_rules(
    monkeypatch, tp, transposed, replayssm, enabled
):
    monkeypatch.setenv("ATOM_MONO_ENABLE", str(int(enabled)))
    backing = torch.arange(2 * 128 * 128, dtype=torch.float32).view(1, 2, 128, 128)
    state = backing.transpose(-1, -2) if transposed else backing
    native = SimpleNamespace(
        replayssm=replayssm,
        build_kv_cache_tensor=lambda module: SimpleNamespace(v_cache=state),
        prepare_decode=lambda: "native metadata",
    )
    module = SimpleNamespace(
        base_linear_attention=True,
        impl=SimpleNamespace(
            tp_size=tp, num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128
        ),
    )
    adapter = MonoCacheBuilder(native)
    bound = adapter.build_kv_cache_tensor(module).v_cache
    expected = (
        state.transpose(-1, -2)
        if enabled and not replayssm and not transposed
        else state
    )
    assert bound.data_ptr() == backing.data_ptr()
    assert bound.stride() == expected.stride()
    assert torch.equal(bound, expected)
    assert adapter.prepare_decode() == "native metadata"
    bound[0, 0, 1, 2] = -1
    assert expected[0, 0, 1, 2] == -1


@pytest.mark.parametrize("cache", [None, SimpleNamespace(v_cache=object())])
def test_other_modules_keep_native_binding(cache):
    native = SimpleNamespace(build_kv_cache_tensor=lambda module: cache)
    assert MonoCacheBuilder(native).build_kv_cache_tensor(object()) is cache


@pytest.mark.parametrize("tp,rows", [(1, 1), (1, 2), (1, 4), (8, 1), (8, 2), (8, 4)])
def test_only_ordinary_decode_rows_are_admitted(monkeypatch, tp, rows):
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(
            image_mask=None,
            gdn_metadata=SimpleNamespace(
                num_prefills=0,
                num_decodes=rows,
                num_actual_tokens=rows,
                num_spec_decodes=0,
                replayssm=False,
            ),
        ),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: context)
    ids = torch.zeros(rows, dtype=torch.int32)
    positions = torch.zeros(3, rows, dtype=torch.int64)
    assert dispatch.step_supported(ids, positions, None, None, tp)
    context.context.is_prefill = True
    assert not dispatch.step_supported(ids, positions, None, None, tp)
    context.context.is_prefill = False
    context.attn_metadata.gdn_metadata.num_spec_decodes = rows
    assert not dispatch.step_supported(ids, positions, None, None, tp)


def test_qwen_resource_refusal_is_agreed_and_memoized(monkeypatch):
    pytest.importorskip("aiter")
    pytest.importorskip("flydsl")
    from atom.models.qwen4_exp_mono import runner as runner_module
    from atom.mono.runtime import widths as widths_module

    events = []
    owner = object.__new__(runner_module.QwenMonoRunner)
    owner.tp = 1
    owner.execution = SimpleNamespace(
        build=lambda rows: events.append("build") or object(),
        kernels=lambda build: [],
    )

    def resources(build, rows):
        events.append("resources")
        raise RuntimeError("resource rejection")

    def agree(ok, group):
        events.append(("agree", ok))
        return ok

    owner.execution.check_resources = resources
    monkeypatch.setattr(
        runner_module, "compile_kernels", lambda kernels: events.append("compile")
    )
    monkeypatch.setattr(widths_module, "tp_agree", agree)
    builds = widths_module.WidthBuilds(
        owner._build_width, owner.execution.kernels, None, "Qwen test"
    )
    assert not builds.prepare(1)
    assert events == ["build", "compile", "resources", ("agree", False)]
    assert not builds.prepare(1)
    assert events == ["build", "compile", "resources", ("agree", False)]
