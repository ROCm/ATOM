# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and steps Qwen3.8-Flash-Next mono serves, and its host-side contracts (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.qwen38.mono import config as mono_config
from atom.models.qwen38.mono import step_meta
from atom.models.qwen38.mono.kernels.weights import aiter_moe
from atom.models.qwen38.mono.kernels.weights.source import (
    LayoutError,
    WeightSpec,
    check,
)
from atom.mono.runtime import deployment

_HF = {
    "num_hidden_layers": 48,
    "hidden_size": 2560,
    "hc_count": 4,
    "hc_lowrank": 320,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "layer_types": [
        "qwen_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(48)
    ],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ple_layer_ids": [2],
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "mamba_ssm_dtype": "float32",
    "output_gate_type": "sigmoid",
}
_QUANT = {
    "quant_method": "quark",
    "global_quant_config": {
        "weight": {"dtype": "fp4"},
        "input_tensors": {"dtype": "fp4"},
    },
    "layer_quant_config": {},
}


def _atom_config(**overrides):
    """Configuration F as a worker sees it."""
    fields = {
        "tensor_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_dp_attention": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": SimpleNamespace(level=3, cudagraph_mode=None),
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "bf16",
        "kv_cache_block_size": 64,
        "speculative_config": SimpleNamespace(method="mtp", num_speculative_tokens=3),
        "quant_config": SimpleNamespace(quant_method="quark", hf_quant_config=_QUANT),
        "torch_dtype": torch.bfloat16,
        "hf_config": SimpleNamespace(**_HF),
        "max_model_len": 262144,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx950")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", "fp32")
    monkeypatch.delenv("ATOM_QSA_DRAFT_KV_BF16", raising=False)


@pytest.mark.parametrize("replayssm", ["0", "1", None])
def test_configuration_f_is_served_on_either_verify(native, monkeypatch, replayssm):
    """ReplaySSM (ATOM's default under MTP, unset) and the classic verify alike."""
    if replayssm is None:
        monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM")
    else:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", replayssm)
    assert mono_config.config_refusal(_atom_config()) is None
    offload = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    assert mono_config.config_refusal(_atom_config(kv_transfer_config=offload)) is None


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"tensor_parallel_size": 8}, "TP 8"),
        ({"speculative_config": None}, "MTP3"),
        ({"kv_cache_dtype": "bf16"}, "KV / index caches"),
        ({"kv_cache_block_size": 16}, "KV block 16"),
        (
            {"quant_config": SimpleNamespace(quant_method="fp8", hf_quant_config={})},
            "Quark MXFP4",
        ),
        ({"hf_config": SimpleNamespace(**{**_HF, "ple_layer_ids": [3]})}, "PLE layers"),
        # a connector whose kernels can hold the CUs the resident programs wait on
        ({"kv_transfer_config": {"kv_connector": "moriio"}}, "KV connector moriio"),
    ],
)
def test_other_deployments_are_refused(native, overrides, reason):
    config = _atom_config(**overrides)
    assert reason in mono_config.config_refusal(config)
    assert not mono_config.refused_by_switch_only(config)


def test_a_compute_partition_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "_cu_count", lambda: 128)
    assert "128 CUs" in mono_config.config_refusal(_atom_config())


@pytest.mark.parametrize(
    "env,value,reason",
    [
        ("ATOM_MONO_ENABLE", "0", "ATOM_MONO_ENABLE"),
        ("ATOM_REPLAYSSM_CACHE_LEN", "128", "ATOM_REPLAYSSM_CACHE_LEN"),
        ("ATOM_GDN_SSM_DTYPE", "fp16", "ATOM_GDN_SSM_DTYPE"),
        ("ATOM_QSA_DRAFT_KV_BF16", "0", "draft caches"),
    ],
)
def test_engine_switches_are_checked(native, monkeypatch, env, value, reason):
    monkeypatch.setenv(env, value)
    if env == "ATOM_REPLAYSSM_CACHE_LEN":
        monkeypatch.delenv(
            "ATOM_ENABLE_REPLAYSSM"
        )  # (the capacity matters under ReplaySSM only)
    assert reason in mono_config.config_refusal(_atom_config())
    # a served deployment missing only an engine switch is warned about at install
    assert mono_config.refused_by_switch_only(_atom_config()) == (
        env != "ATOM_MONO_ENABLE"
    )


def _fwd(batch=2, spec=4, prefill=False, replayssm=False, decodes=0, idx_rows=None):
    gdn = SimpleNamespace(
        replayssm=replayssm,
        num_prefills=0,
        num_decodes=decodes,
        num_spec_decodes=batch,
        spec_state_indices_tensor=torch.zeros(
            idx_rows or batch, spec, dtype=torch.int32
        ),
    )
    md = SimpleNamespace(gdn_metadata=gdn, image_mask=None)
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill),
        ubatch_slices=None,
        attn_metadata=md,
    )


@pytest.mark.parametrize(
    "fwd,rows,replayssm,served",
    [
        (_fwd(1), 4, False, 1),
        (_fwd(4, idx_rows=8), 16, False, 4),  # a wider slot table
        (_fwd(2, replayssm=True), 8, True, 2),
        (_fwd(5), 20, False, None),  # concurrency > 4: Native
        (_fwd(2), 6, False, None),  # not SPEC rows a request
        (_fwd(2, prefill=True), 8, False, None),
        (_fwd(2, decodes=1), 8, False, None),  # a mixed decode
        (_fwd(2, spec=2), 8, False, None),  # another slot table
        # not the verify the kernels were built for
        (_fwd(2, replayssm=True), 8, False, None),
        (_fwd(2), 8, True, None),
    ],
)
def test_verify_steps_served(fwd, rows, replayssm, served):
    assert (
        step_meta.verify_step(fwd, rows, mono_config.BATCHES, replayssm=replayssm)
        == served
    )


def test_target_puts_fill_every_metadata_arg_but_the_selection():
    batch, width = 2, 8
    m = step_meta.meta_buffers(batch, 4, width, 16, "cpu")
    table = torch.arange(batch * (width + 3), dtype=torch.int32).view(batch, width + 3)[
        :, :width
    ]
    meta = step_meta.StepMeta(
        batch,
        torch.tensor([True, False]),
        table,
        torch.tensor([9, 9], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([5, 6], dtype=torch.int32),
        torch.arange(8, dtype=torch.int32).view(2, 4),
    )
    puts = {id(p.dst): p for p in step_meta.target_puts(m, meta)}
    assert set(puts) == {
        id(v) for k, v in m.items() if k not in ("sel", "h1_mem", "stage")
    }
    assert puts[id(m["seq_lens"])].sub == 4 and puts[id(m["seq_lens"])].fill == 0
    assert not puts[id(m["gdn.conv_off"])].masked
    bt = puts[id(m["block_table"])]
    assert (bt.group, bt.inner, bt.stride) == (width, width, width + 3)
    assert puts[id(m["gdn.ssm_write"])].group == 4


def test_draft_rows_must_be_consecutive_block_table_slots():
    table = [[7, 3], [2, 5]]  # two requests, block 4
    positions = [1, 2, 3, 4, 2, 3, 4, 5]
    slots = [
        7 * 4 + 1,
        7 * 4 + 2,
        7 * 4 + 3,
        3 * 4 + 0,
        2 * 4 + 2,
        2 * 4 + 3,
        5 * 4 + 0,
        5 * 4 + 1,
    ]
    assert step_meta.draft_rows_error(positions, slots, table, 4, 2, 4) is None
    assert (
        step_meta.draft_rows_error(positions, slots[:-1] + [0], table, 4, 2, 4)
        is not None
    )
    assert (
        step_meta.draft_rows_error([1, 2, 4, 5, 2, 3, 4, 5], slots, table, 4, 2, 4)
        is not None
    )


def test_pools_refuse_another_layout_or_dtype():
    from atom.models.qwen38.mono.kernels.geometry import G

    blocks, bs = 3, 64
    kv = torch.zeros(blocks, bs, 2, 256, dtype=torch.bfloat16)
    raw = torch.zeros(blocks, bs, 1, 128, dtype=torch.bfloat16)
    cmp = torch.zeros(blocks, bs // 4, 1, 128, dtype=torch.bfloat16)
    step_meta.QSAPools(G, kv, kv.clone(), raw, cmp)
    fp8 = kv.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="fp32 k_scale"):
        step_meta.QSAPools(G, fp8, fp8.clone(), raw, cmp)
    scale = torch.ones(blocks, bs, 2)
    step_meta.QSAPools(G, fp8, fp8.clone(), raw, cmp, scale, scale.clone())
    with pytest.raises(ValueError, match="index_cmp"):
        step_meta.QSAPools(G, kv, kv.clone(), raw, raw)
    conv = torch.zeros(5, 6, G.conv_dim, dtype=torch.bfloat16)
    pools = step_meta.GDNPools(
        G.gdn, conv.transpose(1, 2), torch.zeros(5, 48, 128, 128)
    )
    assert pools.conv.shape == (5, 6, G.conv_dim)
    pools.check_window(4)
    with pytest.raises(ValueError, match="conv rows"):
        pools.check_window(1)
    with pytest.raises(ValueError, match="state pool is torch.bfloat16"):
        step_meta.GDNPools(
            G.gdn, conv, torch.zeros(5, 48, 128, 128, dtype=torch.bfloat16)
        )
    # ReplaySSM: ATOM's record buffers of the layer and the cursor
    rec = (
        torch.zeros(5, 48, 16, 128, dtype=torch.bfloat16),
        torch.zeros(5, 48, 16, 128, dtype=torch.bfloat16),
        torch.zeros(5, 48, 16),
    )
    cursor = torch.zeros(5, dtype=torch.int32)
    ssm = torch.zeros(5, 48, 128, 128)
    assert step_meta.GDNPools(G.gdn, conv, ssm, rec, cursor).replay == 16
    assert step_meta.GDNPools(G.gdn, conv, ssm).replay == 0
    with pytest.raises(ValueError, match="ReplaySSM records"):
        step_meta.GDNPools(
            G.gdn, conv, ssm, (rec[0], rec[1], rec[2].bfloat16()), cursor
        )
    with pytest.raises(ValueError, match="cursor"):
        step_meta.GDNPools(G.gdn, conv, ssm, rec, None)


@pytest.mark.parametrize("gate_up", [False, True])
def test_expert_layout_checks_reject_dirty_padding(gate_up):
    """Bind-time checks of ATOM's shuffled, padded A8W4 experts reach the padding lanes."""
    experts, stored_n, stored_k, used_n, used_k = 3, 128, 256, 64, 128
    weights = torch.zeros(experts, stored_n, stored_k // 2, dtype=torch.uint8)
    scales = torch.zeros(experts, stored_n, stored_k // 32, dtype=torch.uint8)
    for expert in range(experts):
        ranges = (
            [(0, used_n)]
            if not gate_up
            else [(0, used_n // 2), (stored_n // 2, stored_n // 2 + used_n // 2)]
        )
        for lo, hi in ranges:
            weights[expert, lo:hi, : used_k // 2] = 17 + expert
            scales[expert, lo:hi, : used_k // 32] = (
                torch.arange(1, used_k // 32 + 1) + expert * 11
            )
    weights = aiter_moe.shuffle_weight(weights, gate_up)
    scales = aiter_moe.shuffle_scale(
        scales.reshape(-1, stored_k // 32), experts, gate_up
    )
    params = (("gate_up", gate_up), ("n", used_n), ("k", used_k), ("experts", experts))
    weight_spec = WeightSpec(
        "weights", "aiter.a8w4.moe", tuple(weights.shape), (torch.uint8,), params=params
    )
    scale_spec = WeightSpec(
        "scales", "aiter.e8m0.moe", tuple(scales.shape), (torch.uint8,), params=params
    )
    check(weights, weight_spec)
    check(scales, scale_spec)
    weights[-1].view(-1)[-1] = 1
    with pytest.raises(LayoutError, match="padding"):
        check(weights, weight_spec)
    scales[-1, -1] = 1
    with pytest.raises(LayoutError, match="padding"):
        check(scales, scale_spec)


def test_a_shape_that_does_not_build_stops_startup(monkeypatch):
    """An admitted deployment is served whole: a build failure is raised at startup, and a step never builds."""
    pytest.importorskip("flydsl")
    from atom.models.qwen38.mono import runner
    from atom.models.qwen38.mono.kernels.geometry import G
    from atom.models.qwen38.mono.kernels.program import PlanError

    def no_room(*args, **kwargs):
        raise PlanError("no room")

    monkeypatch.setattr(runner, "compose_layer", no_room)
    kernels = runner.Kernels("Qwen mono", ("delta",), G)
    with pytest.raises(RuntimeError, match="ATOM_MONO_ENABLE=0"):
        kernels.build(1, 4, 4096)
    assert not kernels.serves(1, 4, 4096)
