# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the DeepSeek-V4 mono decode takes."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v4.mono import config, dispatch, runner
from atom.mono.runtime import deployment


class _Mode:
    def __init__(self, piecewise):
        self.piecewise = piecewise

    def requires_piecewise_compilation(self):
        return self.piecewise


# V4-Pro's config.json (the values the kernel is built for; one MTP layer's ratio last)
_V4_PRO = SimpleNamespace(
    **config.V4_PRO_CONFIG, compress_ratios=list(config.compress_ratios(61, n_mtp=1))
)
# V4-Flash: the same architecture name, other widths and a ratio-0 (sliding-only) start
_V4_FLASH = {
    "hidden_size": 4096,
    "q_lora_rank": 1024,
    "num_attention_heads": 64,
    "o_groups": 8,
    "n_routed_experts": 256,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 1.5,
    "index_topk": 512,
    "num_hidden_layers": 43,
    "compress_ratios": [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 43)] + [0],
}


def _atom_config(**overrides):
    fields = {
        "hf_config": _V4_PRO,
        "tensor_parallel_size": 8,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "fp4",
        "load_dummy": False,
        "kv_cache_block_size": 256,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "speculative_config": None,
        "compilation_config": SimpleNamespace(cudagraph_mode=_Mode(False)),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mtp(k, method="mtp"):
    return SimpleNamespace(method=method, num_speculative_tokens=k)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)


def test_supported_deployment_has_no_refusal(native):
    assert dispatch.config_refusal(_atom_config()) is None


@pytest.mark.parametrize("method", ["mtp", "dspark"])
@pytest.mark.parametrize("k", [1, 2, 3, 6, 7])
def test_a_verify_of_up_to_8_rows_is_served(native, method, k):
    cfg = _atom_config(speculative_config=_mtp(k, method))
    assert dispatch.config_refusal(cfg) is None
    assert dispatch.spec_steps(cfg) == k


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 4}, "32 attention heads a rank at TP 4"),
        ({"tensor_parallel_size": 16}, "TP 16 not in"),
        ({"tensor_parallel_size": 3}, "not divisible by TP 3"),
        ({"parallel_config": SimpleNamespace(data_parallel_size=2)}, "DP"),
        ({"pipeline_parallel_size": 2}, "PP"),
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"prefill_context_parallel_size": 2}, "prefill CP"),
        ({"enable_tbo_decode": True}, "TBO"),
        ({"kv_cache_dtype": "bf16"}, "kv cache"),
        ({"index_cache_dtype": "fp8"}, "index cache"),
        ({"load_dummy": True}, "dummy weights"),
        ({"speculative_config": _mtp(8, "dspark")}, "a verify of 9 rows > 8"),
        ({"speculative_config": _mtp(3, "eagle3")}, "speculative method eagle3"),
        (
            {"compilation_config": SimpleNamespace(cudagraph_mode=_Mode(True))},
            "piecewise",
        ),
    ],
)
def test_each_unsupported_setting_is_refused(native, overrides, reason):
    assert reason in dispatch.config_refusal(_atom_config(**overrides))


def test_heads_a_rank_the_kernel_does_not_hold_are_refused(native):
    hf = SimpleNamespace(**{**vars(_V4_PRO), "num_attention_heads": 256})
    why = dispatch.config_refusal(_atom_config(hf_config=hf))
    assert "32 attention heads a rank at TP 8" in why


def test_v4_flash_is_refused(native):
    hf = SimpleNamespace(**{**vars(_V4_PRO), **_V4_FLASH})
    assert "checkpoint:" in dispatch.config_refusal(_atom_config(hf_config=hf))


@pytest.mark.parametrize(
    "key, value",
    [
        ("routed_scaling_factor", 1.5),
        ("rms_norm_eps", 1e-5),
        ("index_topk", 512),
        ("hidden_size", 4096),
        ("q_lora_rank", 1024),
        ("n_routed_experts", 256),
        ("swiglu_limit", 7.0),
        ("compress_ratios", [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 61)]),
    ],
)
def test_a_checkpoint_value_the_kernel_bakes_in_is_compared(native, key, value):
    hf = SimpleNamespace(**{**vars(_V4_PRO), key: value})
    why = dispatch.config_refusal(_atom_config(hf_config=hf))
    assert why.startswith("checkpoint:") and (key in why or "compress_ratios" in why)


def test_another_kv_block_size_is_refused(native):
    why = dispatch.config_refusal(_atom_config(kv_cache_block_size=64))
    assert "kv block size 64" in why


def test_plugin_mode_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "is_sglang", lambda: True)
    assert dispatch.config_refusal(_atom_config()) == "plugin mode"


# ---------------------------------------------------------------- launch widths


def _runner(spec_steps=0):
    r = object.__new__(runner.Dsv4MonoDecodeRunner)
    r.spec_steps = spec_steps
    return r


@pytest.mark.parametrize(
    "rows, width", [(1, 1), (2, 2), (3, 4), (5, 8), (6, 8), (7, 8), (8, 8)]
)
def test_a_step_pads_to_the_next_launch_width(rows, width):
    assert _runner().launch_width(rows) == width


def test_a_step_past_the_widest_launch_is_not_served():
    assert _runner().launch_width(9) is None


@pytest.mark.parametrize(
    "k, rows, width",
    [(6, 7, 7), (6, 5, 5), (6, 8, None), (3, 4, 4), (3, 3, 3), (3, 5, None), (1, 2, 2)],
)
def test_with_a_drafter_one_requests_verify_is_served(k, rows, width):
    """At most K + 1 rows, one request's verify, launched at its own width; more
    requests keep the original path."""
    assert _runner(k).launch_width(rows) == width


# ------------------------------------------------- a row's place in its request


@pytest.mark.parametrize(
    "seq, want",
    [
        ([0, 1, 2, 3], [0, 0, 0, 0]),  # a token a request
        ([0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]),  # MTP-3, 2 requests
        ([0] * 7, list(range(7))),  # DSpark K = 6, one request
        ([0, 0, 0, 1, 2, 2, 2, 2], [0, 1, 2, 0, 0, 1, 2, 3]),  # ragged
        ([0, 0, 1, 1, 1, -1, -1], [0, 1, 0, 1, 2, 0, 1]),  # a graph's pad rows
    ],
)
def test_run_positions(seq, want):
    got = runner.run_positions(torch.tensor(seq))
    assert got.dtype == torch.int32 and got.tolist() == want


# ------------------------------------------------- the drafter's taps


def _drafter(layer_ids, buffers, captures=True):
    return SimpleNamespace(
        _captures_aux=captures,
        _aux_buffers=buffers,
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(dspark_target_layer_ids=layer_ids)
        ),
    )


def test_no_drafter_taps_nothing():
    assert runner.bind_aux_taps(None, 16).by_layer == {}
    assert runner.bind_aux_taps(_drafter((), (), captures=False), 16).by_layer == {}


def test_dspark_taps_bind_to_their_layers():
    bufs = tuple(torch.zeros(4, 16, dtype=torch.bfloat16) for _ in range(3))
    taps = runner.bind_aux_taps(_drafter((58, 59, 60), bufs), 16)
    assert list(taps.by_layer) == [58, 59, 60] and taps.by_layer[59] is bufs[1]


@pytest.mark.parametrize(
    "layer_ids, buf",
    [
        ((-1,), torch.zeros(4, 16, dtype=torch.bfloat16)),  # an embedding tap
        ((58,), torch.zeros(4, 8, dtype=torch.bfloat16)),  # another width
        ((58,), torch.zeros(4, 16, dtype=torch.float32)),  # another dtype
    ],
)
def test_taps_the_runner_cannot_fill_are_refused(layer_ids, buf):
    with pytest.raises(runner.MonoUnsupported):
        runner.bind_aux_taps(_drafter(layer_ids, (buf,)), 16)


# ------------------------------------------------- the original path's precision


def _experts(**qm):
    fields = {"is_guinterleave": True, "fp8_activations": False, "quant_type": 0}
    fields.update(qm)
    return SimpleNamespace(
        quant_method=SimpleNamespace(**fields),
        w13_weight=SimpleNamespace(dtype=None),
        activation=None,
    )


@pytest.fixture
def resolver(monkeypatch):
    """aiter's activation rule, as a table rows -> dtype (fp8 by default)."""
    import aiter.fused_moe
    from aiter.utility import dtypes

    table = {}
    monkeypatch.setattr(
        aiter.fused_moe,
        "resolve_activation_dtype",
        lambda *a, M, **k: table.get(M, dtypes.fp8),
    )
    return table


def test_the_recipes_a8w4_decode_is_served(resolver):
    assert runner.baseline_refusal(_experts()) is None


def test_mxfp8_activations_are_served_at_any_width(resolver):
    from aiter.utility import dtypes

    resolver.update({m: dtypes.bf16 for m in range(1, 9)})
    assert runner.baseline_refusal(_experts(fp8_activations=True)) is None


def test_a_bf16_decode_is_refused(resolver):
    from aiter.utility import dtypes

    resolver[3] = dtypes.bf16  # AITER_BF16_FP8_MOE_BOUND past 3 rows
    assert "3-row decode" in runner.baseline_refusal(_experts())


def test_a_gate_then_up_bank_is_refused(resolver):
    why = runner.baseline_refusal(_experts(is_guinterleave=False))
    assert "ATOM_MOE_GU_ITLV" in why


def test_the_triton_moe_path_is_refused(resolver):
    why = runner.baseline_refusal(_experts(use_triton_decode=True))
    assert "Triton" in why
