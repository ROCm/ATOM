# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen Mono admission, state views, build setup and GPU state publication."""

import os
from itertools import pairwise
from types import SimpleNamespace as NS

import pytest
import torch

from atom.models.qwen4_exp_mono import dispatch
from atom.models.qwen4_exp_mono.cache import metadata_builder_cls
from atom.models.qwen4_exp_mono.layout import qsa_post_scratch_layout
from atom.models.qwen4_exp_mono.qsa_plan import QsaPlan
from atom.models.qwen4_exp_mono.weights import _bind_rope
from atom.mono.runtime import widths
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.compile import compile_kernels
from atom.mono.runtime.consensus import MonoUnsupported
from tests.conftest import atom_config_double


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tmax", [1, 4, 16])
@pytest.mark.parametrize("is_kda", [False, True])
def test_replay_flush_publication_preserves_uncommitted_tail(tmax, is_kda):
    from atom.models.qwen4_exp_mono.replayssm import _publish_flush_records

    cap, heads, width = 2 * tmax, 2, 128
    k = torch.arange(3 * heads * cap * width, device="cuda", dtype=torch.int32)
    k = (k % 251).to(torch.bfloat16).reshape(3, heads, cap, width)
    gshape = k.shape if is_kda else k.shape[:-1]
    g = torch.arange(gshape.numel(), device="cuda", dtype=torch.float32).reshape(gshape)
    before_k, before_g = k.clone(), g.clone()
    # h=1 gives overlapping source/destination when T>1. Other entries are
    # a prefill sentinel, an empty request, and a graph-padding slot.
    cursor = torch.tensor([1, -1, 1], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
    cu = torch.tensor(
        [0, tmax, 2 * tmax, 2 * tmax, 3 * tmax], device="cuda", dtype=torch.int32
    )
    expected_k, expected_g = before_k.clone(), before_g.clone()
    expected_k[0, :, :tmax] = before_k[0, :, 1 : 1 + tmax]
    expected_g[0, :, :tmax] = before_g[0, :, 1 : 1 + tmax]

    _publish_flush_records(k, g, cursor, slots, cu, tmax)

    # This stage is a lossless copy, including untouched slots and tail rows.
    assert torch.equal(k, expected_k)
    assert torch.equal(g, expected_g)
    assert cursor.tolist() == [1, -1, 1]


def _deployment(tp, quark=False):
    # Checkpoint values, independent of the constants used by the admission gate.
    hf = NS(
        num_hidden_layers=48,
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=320,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
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
    )
    hf.layer_types *= 12
    return atom_config_double(
        hf_config=hf,
        tensor_parallel_size=tp,
        enable_expert_parallel=tp == 8,
        parallel_config=NS(data_parallel_size=1),
        max_model_len=32768,
        runner_qualname=None,  # Ordinary inference; no rollout weight updater.
        torch_dtype=torch.bfloat16,
        kv_cache_dtype="fp8" if quark else "bf16",
        quant_config=(
            NS(
                quant_method="quark",
                online_quant=False,
                hf_quant_config={"global_quant_config": {"weight": {"dtype": "fp4"}}},
            )
            if quark
            else None
        ),
    )


@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize("quark", [False, True])
def test_installer_keyword_and_native_fallback(monkeypatch, tp, quark):
    model, cfg = torch.nn.Identity(), _deployment(tp, quark)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    assert (
        dispatch.install_mono_decode(model, cfg, None, metadata_builder=object())
        is model
    )
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    wrapped = dispatch.install_mono_decode(model, cfg, None, metadata_builder=object())
    assert isinstance(wrapped, dispatch.MonoDecodeModel)
    assert wrapped._mono.runner is None
    assert cfg.enable_expert_parallel is (tp == 8)
    cfg.speculative_config = object()
    assert (
        dispatch.install_mono_decode(model, cfg, None, metadata_builder=object())
        is model
    )


@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_decode_route_requires_ordinary_rows(monkeypatch, tp, rows):
    gdn = NS(
        num_prefills=0,
        num_decodes=rows,
        num_actual_tokens=rows,
        num_spec_decodes=0,
        replayssm=False,
    )
    context = NS(
        context=NS(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=NS(image_mask=None, gdn_metadata=gdn),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: context)
    ids = torch.zeros(rows, dtype=torch.int32)
    positions = torch.zeros(3, rows, dtype=torch.int64)
    assert dispatch.step_supported(ids, positions, None, None, tp)
    assert not dispatch.step_supported(ids.long(), positions, None, None, tp)
    context.context.is_prefill = True
    assert not dispatch.step_supported(ids, positions, None, None, tp)
    context.context.is_prefill = False
    gdn.num_spec_decodes = rows
    assert not dispatch.step_supported(ids, positions, None, None, tp)


@pytest.mark.parametrize("rows,requests", [(4, 1), (8, 2), (16, 4), (16, 3)])
def test_mtp_target_keeps_verify_metadata_and_padding(monkeypatch, rows, requests):
    cfg = _deployment(1, quark=True)
    cfg.quant_config.online_quant = True
    cfg.speculative_config = NS(method="mtp", num_speculative_tokens=3)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    wrapped = dispatch.install_mono_decode(torch.nn.Identity(), cfg, object())
    assert isinstance(wrapped, dispatch.MonoDecodeModel) and wrapped.mtp_target
    gdn = NS(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=requests,
        num_actual_tokens=4 * requests,
        replayssm=True,
        replayssm_max_query_len=4,
    )
    context = NS(
        context=NS(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=NS(image_mask=None, gdn_metadata=gdn),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: context)
    ids = torch.zeros(rows, dtype=torch.int32)
    positions = torch.zeros(3, rows, dtype=torch.int64)
    assert dispatch.step_supported(ids, positions, None, None, mtp_target=True)
    assert not dispatch.step_supported(ids, positions, None, None)
    gdn.replayssm_max_query_len = 1
    assert not dispatch.step_supported(ids, positions, None, None, mtp_target=True)
    gdn.replayssm_max_query_len = 4
    # A truncated/ragged verify and a non-ReplaySSM deployment stay Native.
    gdn.num_actual_tokens -= 1
    assert not dispatch.step_supported(ids, positions, None, None, mtp_target=True)
    gdn.num_actual_tokens += 1
    gdn.replayssm = False
    assert not dispatch.step_supported(ids, positions, None, None, mtp_target=True)
    for spec in (
        NS(method="mtp", num_speculative_tokens=2),
        NS(method="eagle", num_speculative_tokens=3),
    ):
        cfg.speculative_config = spec
        model = torch.nn.Identity()
        assert dispatch.install_mono_decode(model, cfg, object()) is model


@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize(
    "enabled,replay,transposed",
    [
        (True, False, False),
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
)
def test_gdn_cache_view_keeps_native_storage(
    monkeypatch, tp, enabled, replay, transposed
):
    monkeypatch.setenv("ATOM_MONO_ENABLE", str(int(enabled)))
    raw = torch.zeros(1, 48 // tp, 128, 128)
    state = raw.transpose(-1, -2) if transposed else raw

    class Native:
        model_runner = NS(config=_deployment(tp))
        replayssm = replay

        def build_kv_cache_tensor(self, module):
            return NS(v_cache=state)

    module = NS(
        base_linear_attention=True,
        impl=NS(
            tp_size=tp,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
        ),
    )
    bound = metadata_builder_cls(Native)().build_kv_cache_tensor(module).v_cache
    expected = (
        state.transpose(-1, -2) if enabled and not replay and not transposed else state
    )
    assert bound.data_ptr() == raw.data_ptr()
    assert bound.stride() == expected.stride()


@pytest.mark.parametrize("capacity", [1, 8192, 8193, 32768, 262144])
def test_qsa_capacity_fits_disjoint_workspace(capacity):
    plan = QsaPlan(capacity)
    layout = qsa_post_scratch_layout(max_model_len=capacity)
    assert plan.columns * 4 >= capacity
    assert plan.lists[0] * 2048 >= plan.columns and plan.lists[-1] == 1
    assert layout["qsa_scores"][1] >= plan.columns * 4
    assert all(
        off + size <= next_off
        for (off, size), (next_off, _) in pairwise(sorted(layout.values()))
    )


@pytest.mark.parametrize(
    "shape,accepted",
    [
        ((8, 32), True),
        ((8, 1, 1, 32), True),
        ((8, 2, 32), False),
    ],
)
def test_rope_rows_match_fixed_kernel_addressing(shape, accepted):
    cache = torch.empty(shape, dtype=torch.bfloat16)
    rope = NS(cos_cache=cache, sin_cache=cache)
    if accepted:
        cos, sin = _bind_rope(rope, "QSA")
        assert cos is cache and sin is cache
    else:
        with pytest.raises(MonoUnsupported, match="cos_cache"):
            _bind_rope(rope, "QSA")


def test_generator_build_is_compiled_once_and_resource_refusal_is_cached(monkeypatch):
    events = []

    def launcher():
        assert os.environ["COMPILE_ONLY"] == "1"
        events.append("compile")

    def reject(built, width):
        events.append("resources")
        raise MonoUnsupported("resource limit")

    def agree(ok, group):
        events.append(("agree", ok))
        return ok

    abi = KernelAbi(())
    assert compile_kernels(iter([(launcher, abi)])) == [launcher]
    assert events == ["compile"]
    events.clear()
    monkeypatch.setattr(widths, "scratch_bytes", lambda kernel: 0)
    monkeypatch.setattr(widths, "tp_agree", agree)
    builds = widths.WidthBuilds(
        lambda width: width,
        lambda built: iter([(launcher, abi)]),
        None,
        "Qwen test",
        check_resources=reject,
    )
    assert not builds.prepare(1) and not builds.prepare(1)
    assert events == ["compile", "resources", ("agree", False)]
    with pytest.raises(KeyError):
        builds[1]
