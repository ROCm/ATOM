# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the Qwen3.5-MoE mono decode takes."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.qwen3_5.mono import dispatch
from atom.mono.runtime import deployment


class _Mode:
    def __init__(self, piecewise):
        self.piecewise = piecewise

    def requires_piecewise_compilation(self):
        return self.piecewise


# Qwen3.5-397B-A17B's text config: the shapes the kernel is built for
_HF = {
    "hidden_size": 4096,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "attention_bias": False,
    "mlp_only_layers": [],
    "mtp_num_hidden_layers": 1,
    "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
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
        "compilation_config": SimpleNamespace(level=3, cudagraph_mode=_Mode(False)),
        "kv_cache_dtype": "fp8",
        "kv_cache_block_size": 16,
        "load_dummy": False,
        "speculative_config": None,
        # the multimodal checkpoint nests its text config
        "hf_config": SimpleNamespace(text_config=SimpleNamespace(**_HF)),
        "max_model_len": 1 << 18,
        "enforce_eager": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mtp(k):
    return SimpleNamespace(method="mtp", num_speculative_tokens=k)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    monkeypatch.delenv("ATOM_REPLAYSSM_CACHE_LEN", raising=False)
    for setting, _ in dispatch.baseline_switches():
        monkeypatch.delenv(setting.split("=")[0], raising=False)
    monkeypatch.delenv("ATOM_MONO_CHECK", raising=False)
    monkeypatch.delenv("ATOM_PA_FLYDSL", raising=False)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_supported_deployment_has_no_refusal(native, k):
    config = _atom_config(speculative_config=_mtp(k) if k else None)
    assert dispatch.config_refusal(config) is None


@pytest.mark.parametrize("k", [0, 3, 4, 5])
def test_flydsl_attention_is_followed_up_to_mtp5(native, monkeypatch, k):
    """Under ATOM_PA_FLYDSL the original decode attention is FlyDSL's, which takes a verify of up to 6 tokens
    a request; gluon's takes 4, so past MTP-3 the gluon mode refuses."""
    config = _atom_config(speculative_config=_mtp(k) if k else None)
    if k > 3:
        assert "a verify of" in dispatch.config_refusal(config)
    monkeypatch.setenv("ATOM_PA_FLYDSL", "1")
    assert dispatch.config_refusal(config) is None
    assert "> 6" in dispatch.config_refusal(_atom_config(speculative_config=_mtp(6)))


def test_a_flat_text_config_is_read(native):
    config = _atom_config(hf_config=SimpleNamespace(**_HF))
    assert dispatch.config_refusal(config) is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 2}, "TP 2 not in"),
        ({"tensor_parallel_size": 8}, "TP 8 not in"),
        ({"tensor_parallel_size": 3}, "GDN key heads 16 is not divisible by TP 3"),
        ({"enable_expert_parallel": True}, "expert parallel"),
        ({"parallel_config": SimpleNamespace(data_parallel_size=2)}, "DP"),
        ({"pipeline_parallel_size": 2}, "PP"),
        ({"decode_context_parallel_size": 2}, "decode CP"),
        ({"prefill_context_parallel_size": 2}, "prefill CP"),
        ({"enable_tbo": True}, "TBO"),
        (
            {
                "compilation_config": SimpleNamespace(
                    level=3, cudagraph_mode=_Mode(True)
                )
            },
            "piecewise",
        ),
        ({"kv_cache_dtype": "bf16"}, "kv cache bf16"),
        ({"kv_cache_block_size": 1024}, "KV block size 1024"),
        ({"load_dummy": True}, "dummy weights"),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="eagle3", num_speculative_tokens=3
                )
            },
            "speculative method eagle3",
        ),
        ({"speculative_config": _mtp(4)}, "a verify of 5 tokens a request > 4"),
        (
            {"hf_config": SimpleNamespace(**{**_HF, "num_experts": 256})},
            "num_experts 256: the kernel is built for 512",
        ),
        (
            {"hf_config": SimpleNamespace(**{**_HF, "hidden_size": 2048})},
            "hidden_size 2048",
        ),
    ],
)
def test_unsupported_deployment_is_refused(native, overrides, reason):
    assert reason in dispatch.config_refusal(_atom_config(**overrides))


@pytest.mark.parametrize(
    "field, value",
    [
        ("rms_norm_eps", 1e-5),
        ("hidden_act", "gelu"),
        ("attn_output_gate", False),
        ("attention_bias", True),
        ("qkv_bias", True),
        ("norm_topk_prob", False),
        ("mlp_only_layers", [0]),
        ("decoder_sparse_step", 2),
        ("layer_scale", True),
    ],
)
def test_a_setting_the_kernel_bakes_in_is_compared(native, field, value):
    config = _atom_config(hf_config=SimpleNamespace(**{**_HF, field: value}))
    assert f"{field} {value!r}: the kernel is built for" in dispatch.config_refusal(
        config
    )


@pytest.mark.parametrize("field", ["rms_norm_eps", "hidden_act", "attn_output_gate"])
def test_a_missing_setting_is_refused(native, field):
    hf = {k: v for k, v in _HF.items() if k != field}
    reason = dispatch.config_refusal(_atom_config(hf_config=SimpleNamespace(**hf)))
    assert f"{field} None" in reason


@pytest.mark.parametrize("rope", [{"partial_rotary_factor": 0.5}, {}])
def test_a_rotary_share_other_than_the_kernels_is_refused(native, rope):
    hf = SimpleNamespace(**{**_HF, "rope_parameters": rope})
    reason = dispatch.config_refusal(_atom_config(hf_config=hf))
    assert reason.startswith("partial_rotary_factor")


def test_the_rotary_share_is_read_where_the_original_reads_it(native):
    """A flat ``partial_rotary_factor`` is not what the original attention reads: it rotates the whole
    head without ``rope_parameters``, so that config is refused."""
    hf = {k: v for k, v in _HF.items() if k != "rope_parameters"}
    hf = SimpleNamespace(**hf, partial_rotary_factor=0.25)
    assert dispatch.config_refusal(_atom_config(hf_config=hf)).startswith(
        "partial_rotary_factor 1.0"
    )


def test_the_layer_check_needs_eager_plain_decode(native, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_CHECK", "1")
    why = "ATOM_MONO_CHECK without --enforce-eager or with a drafter"
    assert dispatch.config_refusal(_atom_config()) == why
    assert dispatch.config_refusal(_atom_config(enforce_eager=True)) is None
    mtp = _atom_config(enforce_eager=True, speculative_config=_mtp(3))
    assert dispatch.config_refusal(mtp) == why


def test_a_non_bf16_model_is_refused(native):
    reason = dispatch.config_refusal(_atom_config(torch_dtype=torch.float16))
    assert reason == "model dtype torch.float16"


def test_more_than_one_mtp_layer_is_refused(native):
    hf = SimpleNamespace(**{**_HF, "mtp_num_hidden_layers": 2})
    reason = dispatch.config_refusal(
        _atom_config(hf_config=hf, speculative_config=_mtp(1))
    )
    assert reason == "mtp_num_hidden_layers 2"
    assert dispatch.config_refusal(_atom_config(hf_config=hf)) is None


@pytest.mark.parametrize(
    "setting", [n for n, _ in dispatch.baseline_switches() if "=" in n]
)
def test_a_switch_that_changes_the_original_decode_is_refused(
    native, monkeypatch, setting
):
    name, value = setting.split("=")
    assert dispatch.config_refusal(_atom_config()) is None
    monkeypatch.setenv(name, value)
    assert dispatch.config_refusal(_atom_config()) == setting


@pytest.mark.parametrize(
    "fusion, max_tokens, refused",
    [("1", "8", True), ("1", "16", False), ("0", "8", False)],
)
def test_the_qk_fusion_must_serve_every_launch_width(
    native, monkeypatch, fusion, max_tokens, refused
):
    """Past ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS rows the original attention leaves its fused
    q / k norm + RoPE + cache kernel, which the kernel follows under the QK fusion."""
    monkeypatch.setenv("ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION", fusion)
    monkeypatch.setenv("ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS", max_tokens)
    reason = dispatch.config_refusal(_atom_config())
    if refused:
        assert reason.startswith("ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS <")
    else:
        assert reason is None


def test_replayssm_without_a_drafter_is_refused(native, monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "1")
    reason = dispatch.config_refusal(_atom_config())
    assert reason == "ReplaySSM without speculative decoding"
    assert dispatch.config_refusal(_atom_config(speculative_config=_mtp(3))) is None


def test_plugin_mode_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: True)
    assert dispatch.config_refusal(_atom_config()) == "plugin mode"


def test_switched_off_keeps_the_original_model(native, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    model = torch.nn.Linear(1, 1)
    assert dispatch.install_mono_decode(model, _atom_config(), None, None) is model


def test_a_refused_deployment_keeps_the_original_model(native):
    model = torch.nn.Linear(1, 1)
    config = _atom_config(tensor_parallel_size=8)
    assert dispatch.install_mono_decode(model, config, None, None) is model


@pytest.mark.parametrize(
    "k, env, cache_len, cap",
    [
        (0, None, None, 0),  # no drafter: no records
        (3, None, None, 16),  # gdn_attn's default under speculative decoding
        (3, "0", None, 0),  # the snapshot state
        (0, "1", None, 0),  # forced on, but no verify to keep records for
        (3, None, "4", 8),  # raised to two verify windows
        (7, "1", "32", 32),
    ],
)
def test_replay_cap_follows_gdn_attn(native, monkeypatch, k, env, cache_len, cap):
    if env is not None:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", env)
    if cache_len is not None:
        monkeypatch.setenv("ATOM_REPLAYSSM_CACHE_LEN", cache_len)
    config = _atom_config(speculative_config=_mtp(k) if k else None)
    assert dispatch.replay_cap(config) == cap


def _context(
    monkeypatch, *, prefill=False, dummy=False, draft=False, ubatch=None, **gdn
):
    g = SimpleNamespace(
        **{
            "num_prefills": 0,
            "num_decodes": 0,
            "spec_sequence_masks": None,
            "spec_state_indices_tensor": None,
            "replayssm": False,
            "slot_idx": None,
            **gdn,
        }
    )
    fwd = SimpleNamespace(
        context=SimpleNamespace(is_prefill=prefill, is_dummy_run=dummy, is_draft=draft),
        ubatch_slices=ubatch,
        kv_cache_data=True,
        attn_metadata=SimpleNamespace(gdn_metadata=g),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def _ids(rows):
    return torch.zeros(rows, dtype=torch.int64)


@pytest.mark.parametrize("rows", [1, 5, 8, 16])
def test_plain_decode_step_is_routed(monkeypatch, rows):
    _context(monkeypatch, num_decodes=rows)
    assert dispatch.step_supported(_ids(rows), 1, False)


def test_a_dummy_decode_is_routed_like_any_other(monkeypatch):
    """is_dummy_run is a per-rank flag: a rank that kept the original path on it would leave its
    peers waiting in the kernel's exchange."""
    _context(monkeypatch, dummy=True, num_decodes=4)
    assert dispatch.step_supported(_ids(4), 1, False)


@pytest.mark.parametrize(
    "rows, kwargs",
    [
        (17, {}),
        (4, {"prefill": True}),
        (4, {"draft": True}),
        (4, {"ubatch": object()}),
        (4, {"num_prefills": 1}),
        (4, {"spec_sequence_masks": torch.ones(1)}),  # a verify, but no drafter
        (4, {"replayssm": True}),  # plain decode on ReplaySSM state
    ],
)
def test_other_plain_steps_keep_the_original_model(monkeypatch, rows, kwargs):
    _context(monkeypatch, **kwargs)
    assert not dispatch.step_supported(_ids(rows), 1, False)


def _verify(replay, requests, T=4):
    slots = torch.zeros(requests, 1 if replay else T, dtype=torch.int32)
    return {
        "spec_sequence_masks": torch.ones(requests),
        "spec_state_indices_tensor": slots,
        "replayssm": replay,
        "slot_idx": torch.zeros(requests, dtype=torch.int32) if replay else None,
        "replayssm_max_query_len": T,
    }


def test_a_replay_window_other_than_the_kernels_keeps_the_original_model(monkeypatch):
    _context(monkeypatch, **{**_verify(True, 1), "replayssm_max_query_len": 5})
    assert not dispatch.step_supported(_ids(4), 4, True)


@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("requests", [1, 2, 4])
def test_verify_of_whole_requests_is_routed(monkeypatch, replay, requests):
    _context(monkeypatch, **_verify(replay, requests))
    assert dispatch.step_supported(_ids(4 * requests), 4, replay)


@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("requests, routed", [(1, True), (2, True), (3, False)])
def test_an_mtp5_verify_is_routed_while_it_fits_a_launch(
    monkeypatch, replay, requests, routed
):
    """MTP-5 verifies 6 rows a request: two requests (12 rows) fit a launch, three (18) do not."""
    _context(monkeypatch, **_verify(replay, requests, T=6))
    assert dispatch.step_supported(_ids(6 * requests), 6, replay) == routed


@pytest.mark.parametrize(
    "rows, gdn, replay",
    [
        (6, _verify(True, 2), True),  # not whole requests
        (20, _verify(True, 5), True),  # more rows than a launch holds
        (8, _verify(True, 2), False),  # ReplaySSM state, a snapshot runner
        (8, _verify(False, 2), True),  # snapshot state, a ReplaySSM runner
        (8, {**_verify(True, 2), "num_decodes": 1}, True),  # plain decodes mixed in
        # snapshot slots of another verify width
        (
            8,
            {**_verify(False, 2), "spec_state_indices_tensor": torch.zeros(2, 3)},
            False,
        ),
    ],
)
def test_other_verify_steps_keep_the_original_model(monkeypatch, rows, gdn, replay):
    _context(monkeypatch, **gdn)
    assert not dispatch.step_supported(_ids(rows), 4, replay)


class _DraftRunner:
    def __init__(self, t):
        self.t = t

    def step_t(self, rows):
        return self.t


class _Draft(torch.nn.Module):
    def forward(self, input_ids, positions, hidden_states, *args, **kw):
        return "original"


@pytest.mark.parametrize(
    "kwargs, t",
    [
        ({"prefill": True, "draft": True}, 1),  # the draft's prefill
        ({"draft": False}, 1),
        ({"draft": True}, None),  # a width the propose loop does not use
    ],
)
def test_other_draft_calls_keep_the_original_draft(monkeypatch, kwargs, t):
    _context(monkeypatch, **kwargs)
    model = dispatch.MonoDraftModel(_Draft(), _DraftRunner(t))
    model._mono = SimpleNamespace(ready=lambda rows: True)
    assert model(_ids(4), _ids(4), torch.zeros(4, 8)) == "original"


def test_draft_embeddings_keep_the_original_draft(monkeypatch):
    _context(monkeypatch, draft=True)
    model = dispatch.MonoDraftModel(_Draft(), _DraftRunner(1))
    model._mono = SimpleNamespace(ready=lambda rows: True)
    out = model(_ids(4), _ids(4), torch.zeros(4, 8), inputs_embeds=torch.zeros(4, 8))
    assert out == "original"


@pytest.mark.parametrize(
    "bs, rows, t",
    [
        (1, 4, 4),  # the verify rows (MTP3: four a request)
        (1, 1, 1),  # a later draft step
        (4, 16, 4),
        (4, 4, 1),
        (5, 20, None),  # the target's verify is past a launch: ATOM's ...
        (5, 5, None),  # ... and so is every draft step of the batch
        (8, 8, None),
        (16, 16, None),
        (4, 8, None),  # neither shape
    ],
)
def test_draft_steps_follow_the_verify_width(monkeypatch, bs, rows, t):
    from atom.models.qwen3_5.mono import draft_runner

    fwd = SimpleNamespace(context=SimpleNamespace(scheduled_bs=bs))
    monkeypatch.setattr(draft_runner, "get_forward_context", lambda: fwd)
    runner = object.__new__(draft_runner.Qwen35MonoDraftRunner)
    runner.T = 4
    assert runner.step_t(rows) == t


@pytest.mark.parametrize(
    "bs, rows, t",
    [
        (1, 6, 6),  # MTP-5 verify rows: six a request
        (2, 12, 6),
        (2, 2, 1),  # a later draft step
        (3, 18, None),  # the target's verify is past a launch: ATOM's ...
        (3, 3, None),  # ... and so is every draft step of the batch
    ],
)
def test_mtp5_draft_steps_follow_the_verify_width(monkeypatch, bs, rows, t):
    from atom.models.qwen3_5.mono import draft_runner

    fwd = SimpleNamespace(context=SimpleNamespace(scheduled_bs=bs))
    monkeypatch.setattr(draft_runner, "get_forward_context", lambda: fwd)
    runner = object.__new__(draft_runner.Qwen35MonoDraftRunner)
    runner.T = 6
    assert runner.step_t(rows) == t


@pytest.mark.parametrize(
    "arch", ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]
)
def test_both_architectures_install_mono(arch):
    from atom.model_engine.model_runner import mono_decode_installers

    assert (
        mono_decode_installers[arch]
        == "atom.models.qwen3_5.mono.dispatch.install_mono_decode"
    )


def test_a_step_the_kernel_refuses_runs_the_original_model(monkeypatch):
    """A step whose tensors the kernel does not address (``StepInputError``, raised before any launch)
    is served by the original model, not an engine crash."""
    from atom.models.qwen3_5.mono.op import StepInputError

    class Runner:
        T, replay_cap = 1, 0

        def forward(self, input_ids, positions):
            raise StepInputError("v_cache must be FP8")

    model = torch.nn.Identity()
    model.forward = lambda ids, pos, *a, **k: "original"
    wrapped = dispatch.MonoDecodeModel(model, Runner())
    object.__setattr__(
        wrapped, "_mono", SimpleNamespace(ready=lambda n: True, runner=Runner())
    )
    monkeypatch.setattr(
        dispatch, "get_forward_context", lambda: SimpleNamespace(kv_cache_data=True)
    )
    monkeypatch.setattr(dispatch, "step_supported", lambda *a: True)
    assert wrapped(_ids(4), torch.zeros(4, dtype=torch.int32)) == "original"
