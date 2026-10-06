# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the DeepSeek-V4.1 mono decode takes."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.deepseek_v41.mono import config as mono_config
from atom.models.deepseek_v41.mono import dispatch
from atom.mono.runtime import deployment


class _Mode:
    def __init__(self, piecewise):
        self.piecewise = piecewise

    def requires_piecewise_compilation(self):
        return self.piecewise


_HF = {
    "sliding_window": 128,
    "index_topk": 512,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "index_n_heads": 32,
    "index_head_dim": 128,
}


def _atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 4,
        "enable_expert_parallel": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": SimpleNamespace(level=0, cudagraph_mode=_Mode(False)),
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "fp8",
        "speculative_config": SimpleNamespace(
            method="dspark", num_speculative_tokens=5
        ),
        "dspark": SimpleNamespace(confidence_schedule=False, ragged=False),
        "hf_config": SimpleNamespace(**_HF),
        "max_model_len": 1 << 20,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_DSV41_SIDE_STREAMS", "0")


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_supported_deployment_has_no_refusal(native, tp):
    assert mono_config.config_refusal(_atom_config(tensor_parallel_size=tp)) is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 1}, "TP 1 not in"),
        ({"tensor_parallel_size": 3}, "query heads 64 is not divisible by TP 3"),
        ({"tensor_parallel_size": 16}, "wo_a groups 8 is not divisible by TP 16"),
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"parallel_config": SimpleNamespace(data_parallel_size=2)}, "DP"),
        ({"pipeline_parallel_size": 2}, "PP"),
        ({"decode_context_parallel_size": 2}, "decode CP"),
        ({"prefill_context_parallel_size": 2}, "prefill CP"),
        ({"enable_tbo": True}, "TBO"),
        ({"enable_tbo_decode": True}, "TBO"),
        (
            {
                "compilation_config": SimpleNamespace(
                    level=0, cudagraph_mode=_Mode(True)
                )
            },
            "piecewise",
        ),
        ({"kv_cache_dtype": "fp4"}, "kv cache"),
        ({"index_cache_dtype": "bf16"}, "index cache"),
        ({"speculative_config": None}, "not DSpark"),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=5
                )
            },
            "not DSpark",
        ),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="dspark", num_speculative_tokens=3
                )
            },
            "speculative tokens",
        ),
        (
            {"dspark": SimpleNamespace(confidence_schedule=True, ragged=False)},
            "confidence",
        ),
        ({"dspark": SimpleNamespace(confidence_schedule=False, ragged=True)}, "ragged"),
        (
            {"hf_config": SimpleNamespace(**{**_HF, "index_topk": 1024})},
            "attention keys",
        ),
        (
            {"hf_config": SimpleNamespace(**{**_HF, "candidate_topk_blocks": 1024})},
            "indexer shape",
        ),
        (
            {"hf_config": SimpleNamespace(**{**_HF, "index_n_heads": 64})},
            "indexer shape",
        ),
    ],
)
def test_unsupported_deployment_is_refused(native, overrides, reason):
    assert reason in mono_config.config_refusal(_atom_config(**overrides))


def test_switches_refuse(native, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    assert mono_config.config_refusal(_atom_config()) == "switched off"
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.setenv("ATOM_DSV41_SIDE_STREAMS", "1")
    assert mono_config.config_refusal(_atom_config()) == "ATOM_DSV41_SIDE_STREAMS"


def test_plugin_mode_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: True)
    assert mono_config.config_refusal(_atom_config()) == "plugin mode"


def _context(monkeypatch, *, prefill=False, ubatch=None, **step_fields):
    step = SimpleNamespace(
        **{
            "decode": True,
            "tentative": True,
            "requests": [object()],
            "width": 6,
            **step_fields,
        }
    )
    fwd = SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill),
        ubatch_slices=ubatch,
        attn_metadata=SimpleNamespace(step=step, image_mask=None),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def test_verify_step_is_routed(monkeypatch):
    _context(monkeypatch)
    assert dispatch.step_supported(torch.zeros(6, dtype=torch.int64), None)


@pytest.mark.parametrize(
    "tokens, kwargs",
    [
        (5, {}),
        (12, {}),
        (6, {"prefill": True}),
        (6, {"ubatch": object()}),
        (6, {"decode": False}),
        (6, {"tentative": False}),
        (6, {"width": 12}),
    ],
)
def test_other_steps_keep_the_original_model(monkeypatch, tokens, kwargs):
    _context(monkeypatch, **kwargs)
    assert not dispatch.step_supported(torch.zeros(tokens, dtype=torch.int64), None)


@pytest.mark.parametrize(
    "tokens, routed", [(6, True), (9, True), (12, True), (13, False), (18, False)]
)
def test_the_target_is_routed_by_its_rows_alone(monkeypatch, tokens, routed):
    # a padded graph width: neither the scheduled request count nor how the rows
    # split into requests is read
    monkeypatch.setattr(dispatch, "MAX_ROWS", 12)
    _context(monkeypatch, width=tokens, requests=[object()])
    ids = torch.zeros(tokens, dtype=torch.int64)
    assert dispatch.step_supported(ids, None) is routed


class _DraftRunner:
    def __init__(self):
        self.rows = []

    def backbone(self, anchor_ids, anchor_positions):
        return "mono"


class _Draft(torch.nn.Module):
    block_size = 5

    def block_backbone(self, input_ids, positions, num_draft):
        return "original"


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("requests, routed", [(1, True), (8, True), (9, False)])
def test_the_draft_is_routed_by_its_request_count(
    monkeypatch, prefill, requests, routed
):
    # the target step before the draft pass (a prefill or a decode) is not read
    _context(monkeypatch, prefill=prefill)
    model = dispatch.MonoDraftModel(_Draft())
    runner = _DraftRunner()
    model._mono = SimpleNamespace(
        runner=runner, ready=lambda rows: runner.rows.append(rows) or True
    )
    ids = torch.zeros(requests, dtype=torch.int64)
    out = model.block_backbone(ids, ids, None)
    assert out == ("mono" if routed else "original")
    assert runner.rows == ([requests * 5] if routed else [])


def test_embeddings_or_images_keep_the_original_model(monkeypatch):
    fwd = _context(monkeypatch)
    ids = torch.zeros(6, dtype=torch.int64)
    assert not dispatch.step_supported(ids, torch.zeros(6, 8))
    fwd.attn_metadata.image_mask = torch.zeros(6, dtype=torch.bool)
    assert not dispatch.step_supported(ids, None)


def test_refused_deployment_is_not_wrapped(native, monkeypatch):
    model = torch.nn.Linear(2, 2)
    assert (
        dispatch.install_mono_decode(
            model, _atom_config(tensor_parallel_size=16), None, None
        )
        is model
    )
    wrapped = dispatch.install_mono_decode(model, _atom_config(), None, None)
    assert isinstance(wrapped, dispatch.MonoDecodeModel)
    # attributes the runner never overrides pass through to the wrapped model
    assert wrapped.in_features == 2


def test_fp4_index_plane_plans_its_walk_with_each_step(native, monkeypatch):
    planners = []
    builder = SimpleNamespace(
        geometry=SimpleNamespace(compress_ratios=((2, 4), (8, 1))),
        add_step_planner=planners.append,
    )
    model = torch.nn.Linear(2, 2)
    dispatch.install_mono_decode(
        model, _atom_config(index_cache_dtype="fp4"), None, builder
    )
    assert [p.ratios for p in planners] == [(2, 8)]
    dispatch.install_mono_decode(model, _atom_config(), None, builder)
    assert len(planners) == 1


def _v41_config():
    return SimpleNamespace(
        hidden_size=5120, hc_mult=4, num_attention_heads=64, o_groups=8,
        q_lora_rank=1280, head_dim=512, o_lora_rank=1024, n_routed_experts=384,
        moe_intermediate_size=2304, index_n_heads=32, index_head_dim=128,
    )  # fmt: skip


class _Block(torch.nn.Module):
    # the deployed routed experts: A8W4 (gate / up interleaved, MXFP8 activations)
    def __init__(self, table, mode, *, interleaved=True, fp8_activations=True):
        super().__init__()
        self.attn = SimpleNamespace(spec=SimpleNamespace(layer_id=2, mode=mode))
        method = SimpleNamespace(
            is_guinterleave=interleaved, fp8_activations=fp8_activations
        )
        self.ffn = SimpleNamespace(experts=SimpleNamespace(quant_method=method))
        self.table = table

    def named_parameters(self):
        for name, (shape, dtype) in self.table.items():
            yield name, torch.empty(shape, dtype=dtype, device="meta")


def test_weight_table_is_the_loaded_tp4_shard():
    from atom.models.deepseek_v41.config import AttentionMode
    from atom.models.deepseek_v41.mono import weights

    table = weights.expected(_v41_config(), AttentionMode.FULL, 4)
    # as a TP4 server loads them (the p1 weight manifest)
    assert table["attn.wqkv_a.weight"] == ((1792, 5120), torch.float8_e4m3fn)
    assert table["attn.wo_a.weight_scale"] == ((64, 128), torch.float8_e8m0fnu)
    assert table["ffn.experts.w13_weight"] == (
        (384, 1280, 2560),
        torch.float4_e2m1fn_x2,
    )
    assert table["ffn.experts.w2_weight"] == ((384, 5120, 320), torch.float4_e2m1fn_x2)
    assert table["ffn.shared_experts.w2.weight_scale"] == (
        (160, 18),
        torch.float8_e8m0fnu,
    )
    assert table["attn.indexer.wq_b.weight"] == ((4096, 1280), torch.float8_e4m3fn)
    window = weights.expected(_v41_config(), AttentionMode.WINDOW, 4)
    assert "attn.indexer.wq_b.weight" not in window


@pytest.mark.parametrize(
    "tp, name, shape",
    [
        (2, "attn.wq_b.weight", (32 * 512, 1280)),
        (2, "attn.wo_b.weight", (5120, 4096)),
        (2, "ffn.experts.w2_weight", (384, 5120, 576)),
        (8, "attn.attn_sink", (8,)),
        (8, "attn.wo_a.weight", (1024, 4096)),
        # 288 padded to the 128-k MXFP4 step
        (8, "ffn.experts.w13_weight", (384, 768, 2560)),
        (8, "ffn.shared_experts.w2.weight_scale", (160, 9)),
    ],
)
def test_weight_table_follows_the_tp_size(tp, name, shape):
    from atom.models.deepseek_v41.config import AttentionMode
    from atom.models.deepseek_v41.mono import weights

    assert weights.expected(_v41_config(), AttentionMode.FULL, tp)[name][0] == shape


def test_a_weight_off_the_table_is_refused():
    from atom.models.deepseek_v41.config import AttentionMode
    from atom.models.deepseek_v41.mono import weights
    from atom.mono.runtime.consensus import MonoUnsupported

    table = weights.expected(_v41_config(), AttentionMode.FULL, 4)
    weights.bind_layer(_Block(table, AttentionMode.FULL), _v41_config(), 4)
    for name, change in (
        ("attn.wq_b.weight", ((8192, 1024), torch.float8_e4m3fn)),
        ("attn.attn_sink", ((16,), torch.bfloat16)),
    ):
        bad = dict(table, **{name: change})
        with pytest.raises(MonoUnsupported, match=name):
            weights.bind_layer(_Block(bad, AttentionMode.FULL), _v41_config(), 4)
    missing = {k: v for k, v in table.items() if k != "ffn_norm.weight"}
    with pytest.raises(MonoUnsupported, match="no ffn_norm.weight"):
        weights.bind_layer(_Block(missing, AttentionMode.FULL), _v41_config(), 4)


@pytest.mark.parametrize(
    "interleaved, fp8_activations", [(False, False), (True, False), (False, True)]
)
def test_routed_experts_off_a8w4_are_refused(interleaved, fp8_activations):
    from atom.models.deepseek_v41.config import AttentionMode
    from atom.models.deepseek_v41.mono import weights
    from atom.mono.runtime.consensus import MonoUnsupported

    table = weights.expected(_v41_config(), AttentionMode.FULL, 4)
    block = _Block(
        table,
        AttentionMode.FULL,
        interleaved=interleaved,
        fp8_activations=fp8_activations,
    )
    with pytest.raises(MonoUnsupported, match="not A8W4"):
        weights.bind_layer(block, _v41_config(), 4)
