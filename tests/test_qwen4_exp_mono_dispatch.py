# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the Qwen3.8-Flash-Next mono decode takes,
and how its wrapper routes them. No GPU: the runner is never created."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen4_exp.mono import config as mono_config
from atom.models.qwen4_exp.mono import dispatch
from atom.models.qwen4_exp.mono.kernels.config import _FIXED, DEFAULT
from atom.mono.runtime import deployment
from atom.utils import envs


class _Mode:
    def __init__(self, piecewise):
        self.piecewise = piecewise

    def requires_piecewise_compilation(self):
        return self.piecewise


def _hf(mamba_ssm_dtype="float32", **overrides):
    """An HF config as ``to_dict`` gives it: the kernels' geometry, 48 layers
    with every fourth QSA and the PLE on layer 1 (``ple_layer_ids`` count
    from 1)."""
    d = {key: getattr(DEFAULT, name) for name, key in _FIXED.items()}
    d |= {
        "num_experts": 512,
        "num_hidden_layers": 48,
        "vocab_size": 248320,
        "layer_types": list(DEFAULT.layer_types),
        "ple_layer_ids": [2],
        "partial_rotary_factor": 0.25,
    }
    d |= overrides
    return SimpleNamespace(to_dict=lambda: d, mamba_ssm_dtype=mamba_ssm_dtype)


def _atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 1,
        "enable_expert_parallel": False,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "compilation_config": SimpleNamespace(level=0, cudagraph_mode=_Mode(False)),
        "kv_cache_dtype": "bf16",
        "kv_cache_block_size": 16,
        "speculative_config": None,
        "multimodal_config": None,
        "hf_config": _hf(),
        "max_model_len": 1 << 16,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _spec(method="mtp", k=1):
    return SimpleNamespace(method=method, num_speculative_tokens=k)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    monkeypatch.delenv("ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", raising=False)
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)


@pytest.mark.parametrize("kv", ["auto", "bf16", "bfloat16"])
def test_supported_deployment_has_no_refusal(native, kv):
    assert mono_config.config_refusal(_atom_config(kv_cache_dtype=kv)) is None


def test_a_vl_config_reads_its_text_config(native):
    text = _hf().to_dict()
    hf = SimpleNamespace(to_dict=lambda: {"text_config": text})
    assert mono_config.config_refusal(_atom_config(hf_config=hf)) is None


# layer 1, the PLE layer, made QSA
_QSA_AT_1 = [DEFAULT.layer_types[0], "full_attention", *DEFAULT.layer_types[2:]]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 2}, "TP 2 not in"),
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
        ({"kv_cache_dtype": "fp8"}, "kv cache fp8"),
        ({"kv_cache_block_size": 32}, "block size 32"),
        ({"hf_config": _hf(mamba_ssm_dtype="bfloat16")}, "mamba_ssm_dtype"),
        ({"speculative_config": _spec(method="eagle")}, "speculative method eagle"),
        ({"speculative_config": _spec(k=4)}, "4 speculative tokens"),
        # ReplaySSM follows speculative decoding unless switched off
        ({"speculative_config": _spec()}, "ReplaySSM"),
        ({"multimodal_config": object()}, "multimodal"),
        ({"max_model_len": (1 << 18) + 1}, "contexts up to 262145 > 262144"),
        ({"hf_config": _hf(hidden_size=4096)}, "geometry: hidden_size"),
        ({"hf_config": _hf(partial_rotary_factor=0.5)}, "geometry: rotary"),
        ({"hf_config": _hf(ple_layer_ids=[2, 6])}, "PLE layers"),
        ({"hf_config": _hf(ple_layer_ids=[])}, "PLE layers"),
        (
            {"hf_config": _hf(layer_types=_QSA_AT_1)},
            "layer kinds: PLE on a QSA layer",
        ),
    ],
)
def test_unsupported_deployment_is_refused(native, overrides, reason):
    assert reason in mono_config.config_refusal(_atom_config(**overrides))


def test_a_missing_size_is_a_geometry_refusal(native):
    hf = _hf()
    d = hf.to_dict()
    del d["num_experts"]
    assert mono_config.config_refusal(_atom_config(hf_config=hf)).startswith("geometry")


@pytest.mark.parametrize(
    "env, reason",
    [
        ({"ATOM_MONO_ENABLE": "0"}, "switched off"),
        (
            {"ATOM_ENABLE_GDN_DECODE_LOSSY_FAST": "1"},
            "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST",
        ),
        ({"ATOM_ENABLE_REPLAYSSM": "1"}, "ReplaySSM"),
    ],
)
def test_switches_refuse(native, monkeypatch, env, reason):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert mono_config.config_refusal(_atom_config()) == reason


def test_replayssm_switched_off_is_no_refusal(native, monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    assert envs.ATOM_ENABLE_REPLAYSSM is False
    assert mono_config.config_refusal(_atom_config()) is None


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mtp_with_replayssm_switched_off_has_no_refusal(native, monkeypatch, k):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    assert (
        mono_config.config_refusal(_atom_config(speculative_config=_spec(k=k))) is None
    )


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_max_model_len_up_to_the_selection_has_no_refusal(native, monkeypatch, k):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    spec = _spec(k=k) if k else None
    atom_config = _atom_config(speculative_config=spec, max_model_len=(1 << 18) - k)
    assert mono_config.config_refusal(atom_config) is None


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mtp_draft_tokens_count_toward_the_context(native, monkeypatch, k):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    atom_config = _atom_config(speculative_config=_spec(k=k), max_model_len=1 << 18)
    reason = f"max_model_len: contexts up to {(1 << 18) + k} > 262144"
    assert mono_config.config_refusal(atom_config) == reason


@pytest.mark.parametrize(
    "env, spec, on",
    [
        (None, None, False),
        (None, _spec(), True),
        ("0", None, False),
        ("0", _spec(), False),
        ("1", None, True),
        ("1", _spec(), True),
    ],
)
def test_replayssm_follows_spec_unless_switched(native, monkeypatch, env, spec, on):
    if env is not None:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", env)
    on_now = mono_config.replayssm_on(_atom_config(speculative_config=spec))
    assert on_now is on


def test_plugin_mode_is_refused(native, monkeypatch):
    monkeypatch.setattr(deployment, "is_vllm", lambda: True)
    assert mono_config.config_refusal(_atom_config()) == "plugin mode"


# ------------------------------------------------------------------ steps
def _context(
    monkeypatch,
    *,
    prefill=False,
    draft=False,
    context=True,
    ubatch=None,
    qsa=True,
    ple=True,
    gdn=True,
    spec_slots=None,
    slots=True,
    max_seqlen_q=1,
    spec_decodes=0,
    decodes=1,
    accepted=None,
):
    gdn_md = None
    if gdn:
        gdn_md = SimpleNamespace(
            spec_state_indices_tensor=spec_slots,
            non_spec_state_indices_tensor=torch.zeros(4) if slots else None,
            num_spec_decodes=spec_decodes,
            num_decodes=decodes,
            num_accepted_tokens=accepted,
        )
    fwd = SimpleNamespace(
        context=(
            SimpleNamespace(is_prefill=prefill, is_draft=draft) if context else None
        ),
        ubatch_slices=ubatch,
        attn_metadata=SimpleNamespace(
            qsa_metadata=object() if qsa else None,
            ple_metadata=object() if ple else None,
            gdn_metadata=gdn_md,
            max_seqlen_q=max_seqlen_q,
        ),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def _ids(n):
    return torch.zeros(n, dtype=torch.int64)


def test_a_one_row_decode_is_routed(monkeypatch):
    _context(monkeypatch)
    assert dispatch.step_supported(_ids(1), None)


@pytest.mark.parametrize(
    "rows, kwargs",
    [
        (0, {}),
        (5, {}),
        (1, {"prefill": True}),
        (1, {"context": False}),
        (1, {"ubatch": object()}),
        # warmup and profiling forwards: no state pools, no metadata
        (1, {"qsa": False}),
        (1, {"ple": False}),
        (1, {"gdn": False}),
        (1, {"slots": False}),
        # speculative steps
        (1, {"spec_slots": torch.zeros(1)}),
        (1, {"spec_decodes": 1}),
        (1, {"max_seqlen_q": 2}),
    ],
)
def test_other_steps_keep_the_original_model(monkeypatch, rows, kwargs):
    _context(monkeypatch, **kwargs)
    assert not dispatch.step_supported(_ids(rows), None)


def _verify(monkeypatch, k=1, **kwargs):
    """A verify step as ATOM builds it without ReplaySSM: one request's k + 1
    tokens, every decode a spec decode."""
    step = {
        "spec_slots": torch.zeros(1, k + 1),
        "slots": False,
        "max_seqlen_q": k + 1,
        "spec_decodes": 1,
        "decodes": 0,
        "accepted": torch.ones(1),
    }
    _context(monkeypatch, **(step | kwargs))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_one_request_verify_is_routed(monkeypatch, k):
    _verify(monkeypatch, k)
    assert dispatch.step_supported(_ids(k + 1), None, k + 1)


@pytest.mark.parametrize(
    "rows, kwargs",
    [
        # a batched verify (two requests) is ATOM's
        (4, {"max_seqlen_q": 2, "spec_decodes": 2}),
        (2, {"spec_decodes": 2}),
        (2, {"max_seqlen_q": 1}),
        (2, {"decodes": 1}),
        (2, {"spec_slots": None}),
        (2, {"accepted": None}),
        (2, {"prefill": True}),
        (2, {"qsa": False}),
        (2, {"gdn": False}),
        (1, {}),
        (3, {}),
    ],
)
def test_other_steps_under_mtp_keep_the_original_model(monkeypatch, rows, kwargs):
    _verify(monkeypatch, **kwargs)
    assert not dispatch.step_supported(_ids(rows), None, 2)


def test_a_plain_decode_under_mtp_keeps_the_original_model(monkeypatch):
    _context(monkeypatch)
    assert not dispatch.step_supported(_ids(2), None, 2)


def test_embeddings_in_keep_the_original_model(monkeypatch):
    _context(monkeypatch)
    assert not dispatch.step_supported(_ids(1), torch.zeros(1, 8))


@pytest.mark.parametrize(
    "rows, routed", [(1, True), (2, True), (3, True), (4, True), (5, False)]
)
def test_rows_up_to_the_max_are_routed(monkeypatch, rows, routed):
    _context(monkeypatch)
    assert dispatch.step_supported(_ids(rows), None) is routed


# ------------------------------------------------------------------ wrapper
class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Identity()
        self.marker = "inner"

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        return "native"

    def compute_logits(self, hidden_states):
        return ("logits", hidden_states)


def _builder():
    # the class default is False; the installer sets the instance's
    return SimpleNamespace(gdn_state_vk=False)


def test_a_refused_deployment_keeps_the_model(native, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    model, builder = _Model(), _builder()
    assert dispatch.install_mono_decode(model, _atom_config(), None, builder) is model
    assert builder.gdn_state_vk is False


def test_install_wraps_and_asks_for_vk_state(native):
    model, builder = _Model(), _builder()
    mono = dispatch.install_mono_decode(model, _atom_config(), None, builder)
    assert isinstance(mono, dispatch.MonoDecodeModel)
    assert mono.model is model
    assert builder.gdn_state_vk is True
    # created on its first eligible step, after the caches exist
    assert mono._mono.runner is None


class _Lazy:
    def __init__(self, ready):
        self._ready = ready
        self.runner = SimpleNamespace(forward=lambda ids, positions: "mono")
        self.asked = []

    def ready(self, rows):
        self.asked.append(rows)
        return self._ready


@pytest.mark.parametrize(
    "ready, step, intermediate, out",
    [
        (True, {}, None, "mono"),
        # the runner refused (``LazyRunner``): ATOM's decode for good
        (False, {}, None, "native"),
        (True, {"prefill": True}, None, "native"),
        (True, {}, object(), "native"),
    ],
)
def test_forward_routes(native, monkeypatch, ready, step, intermediate, out):
    mono = dispatch.MonoDecodeModel(_Model(), _atom_config(), _builder())
    mono._mono = _Lazy(ready)
    _context(monkeypatch, **step)
    assert mono(_ids(1), _ids(1), intermediate, None) == out


@pytest.mark.parametrize("k", [None, 1, 2, 3])
def test_the_wrapper_reads_its_verify_rows(native, k):
    spec = None if k is None else _spec(k=k)
    config = _atom_config(speculative_config=spec)
    mono = dispatch.MonoDecodeModel(_Model(), config, _builder())
    assert mono.verify_rows == (0 if k is None else k + 1)


def test_the_runner_is_asked_only_for_routable_steps(native, monkeypatch):
    mono = dispatch.MonoDecodeModel(_Model(), _atom_config(), _builder())
    mono._mono = lazy = _Lazy(True)
    _context(monkeypatch, prefill=True)
    mono(_ids(1), _ids(1))
    _context(monkeypatch)
    mono(_ids(1), _ids(1))
    assert lazy.asked == [1]


def test_attributes_pass_through(native):
    inner = _Model()
    mono = dispatch.MonoDecodeModel(inner, _atom_config(), _builder())
    assert mono.marker == "inner"
    assert mono.lm_head is inner.lm_head
    assert mono.compute_logits("h") == ("logits", "h")


# ------------------------------------------------------------------ drafter
def _draft(monkeypatch, k=1, step=0, **kwargs):
    """A draft step after a one-request verify: step 0 over the verify's
    k + 1 rows, a later step (k >= 2) one row; the GDN metadata stays the
    verify's."""
    fields = {
        "draft": True,
        "spec_slots": torch.zeros(1, k + 1),
        "slots": False,
        "max_seqlen_q": k + 1 if step == 0 else 1,
        "spec_decodes": 1,
        "decodes": 0,
        "accepted": torch.ones(1),
    }
    _context(monkeypatch, **(fields | kwargs))


@pytest.mark.parametrize(
    "k, step, rows", [(1, 0, 2), (2, 0, 3), (2, 1, 1), (3, 0, 4), (3, 1, 1), (3, 2, 1)]
)
def test_a_one_request_draft_step_is_routed(monkeypatch, k, step, rows):
    _draft(monkeypatch, k, step)
    assert dispatch.draft_step_supported(_ids(rows), None, k + 1)


@pytest.mark.parametrize(
    "k, step, rows, kwargs",
    [
        # the target's verify, the drafter's prefill step, no context
        (1, 0, 2, {"draft": False}),
        (1, 0, 2, {"prefill": True}),
        (1, 0, 2, {"context": False}),
        (1, 0, 2, {"ubatch": object()}),
        # warmup and profiling forwards
        (1, 0, 2, {"qsa": False}),
        (1, 0, 2, {"gdn": False}),
        # two requests
        (1, 0, 4, {"spec_decodes": 2}),
        (1, 0, 2, {"spec_decodes": 2, "max_seqlen_q": 1}),
        (2, 1, 2, {"spec_decodes": 2}),
        (1, 0, 2, {"decodes": 1}),
        # k = 1 has no later steps; rows and query length disagree
        (1, 0, 1, {"max_seqlen_q": 1}),
        (2, 1, 1, {"max_seqlen_q": 3}),
        (2, 0, 3, {"max_seqlen_q": 1}),
        # k = 3: the verify's 4 rows or one
        (3, 0, 4, {"spec_decodes": 2}),
        (3, 0, 4, {"max_seqlen_q": 1}),
        (3, 1, 2, {}),
        (3, 1, 3, {}),
    ],
)
def test_other_draft_forwards_keep_atoms_layer(monkeypatch, k, step, rows, kwargs):
    _draft(monkeypatch, k, step, **kwargs)
    assert not dispatch.draft_step_supported(_ids(rows), None, k + 1)


def test_draft_embeddings_in_keep_atoms_layer(monkeypatch):
    _draft(monkeypatch)
    assert not dispatch.draft_step_supported(_ids(2), torch.zeros(2, 8), 2)


class _Draft(nn.Module):
    """Stands in for ``Qwen4ExpMTP`` (patched in as it)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Identity()
        self.marker = "draft"

    def forward(
        self, input_ids, positions, hidden_states, inputs_embeds=None, **kwargs
    ):
        return "atom"

    def compute_draft_ids(self, hidden_states, spec_step_idx=0, *, out):
        return ("ids", hidden_states)


@pytest.fixture
def mtp(native, monkeypatch):
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    monkeypatch.setattr(dispatch, "Qwen4ExpMTP", _Draft)


def _mtp_config(k=1):
    return _atom_config(speculative_config=_spec(k=k))


def test_install_wraps_the_drafter(mtp):
    draft = _Draft()
    drafter = SimpleNamespace(model=draft)
    dispatch.install_mono_decode(_Model(), _mtp_config(), drafter, _builder())
    assert isinstance(drafter.model, dispatch.MonoDraftModel)
    assert drafter.model.draft is draft
    assert drafter.model._mono.runner is None


@pytest.mark.parametrize(
    "drafter", [SimpleNamespace(model=nn.Identity()), SimpleNamespace()]
)
def test_other_drafters_are_kept(mtp, drafter):
    before = getattr(drafter, "model", None)
    dispatch.install_mono_decode(_Model(), _mtp_config(), drafter, _builder())
    assert getattr(drafter, "model", None) is before


def test_a_refused_deployment_keeps_the_drafter(mtp, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    draft = _Draft()
    drafter = SimpleNamespace(model=draft)
    dispatch.install_mono_decode(_Model(), _mtp_config(), drafter, _builder())
    assert drafter.model is draft


class _DraftLazy(_Lazy):
    def __init__(self, ready):
        super().__init__(ready)
        self.runner = SimpleNamespace(forward=lambda ids, hidden: "mono")


@pytest.mark.parametrize(
    "ready, step, kwargs, out",
    [
        (True, {}, {}, "mono"),
        # the runner refused (``LazyRunner``): ATOM's draft layer for good
        (False, {}, {}, "atom"),
        (True, {"draft": False}, {}, "atom"),
        (True, {"prefill": True}, {}, "atom"),
        (True, {}, {"extra": 1}, "atom"),
    ],
)
def test_draft_forward_routes(mtp, monkeypatch, ready, step, kwargs, out):
    wrapper = dispatch.MonoDraftModel(_Draft(), _mtp_config(), _builder())
    wrapper._mono = _DraftLazy(ready)
    _draft(monkeypatch, **step)
    assert wrapper(_ids(2), _ids(2), torch.zeros(2, 8), **kwargs) == out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_draft_wrapper_reads_its_verify_rows(mtp, k):
    wrapper = dispatch.MonoDraftModel(_Draft(), _mtp_config(k), _builder())
    assert wrapper.verify_rows == k + 1


def test_the_draft_runner_is_asked_only_for_routable_steps(mtp, monkeypatch):
    wrapper = dispatch.MonoDraftModel(_Draft(), _mtp_config(2), _builder())
    wrapper._mono = lazy = _DraftLazy(True)
    _draft(monkeypatch, 2, prefill=True)
    wrapper(_ids(3), _ids(3), torch.zeros(3, 8))
    _draft(monkeypatch, 2)
    wrapper(_ids(3), _ids(3), torch.zeros(3, 8))
    _draft(monkeypatch, 2, step=1)
    wrapper(_ids(1), _ids(1), torch.zeros(1, 8))
    assert lazy.asked == [3, 1]


def test_draft_attributes_pass_through(mtp):
    draft = _Draft()
    wrapper = dispatch.MonoDraftModel(draft, _mtp_config(), _builder())
    # the drafter reads its model's ``model`` (``set_skip_topk`` and the like)
    assert wrapper.model is draft.model
    assert wrapper.marker == "draft"
    assert wrapper.compute_draft_ids("h", out=None) == ("ids", "h")


def test_the_fp4_head_is_installed_before_the_drafter_is_wrapped(mtp, monkeypatch):
    seen = []
    monkeypatch.setattr(
        dispatch,
        "install_fp4_head",
        lambda model, drafter: seen.append((model, drafter.model)),
    )
    model, draft = _Model(), _Draft()
    drafter = SimpleNamespace(model=draft)
    dispatch.install_mono_decode(model, _mtp_config(), drafter, _builder())
    # the drafter's model is still ATOM's when its head is swapped
    assert seen == [(model, draft)]
    assert isinstance(drafter.model, dispatch.MonoDraftModel)


def test_a_refused_deployment_keeps_the_head(mtp, monkeypatch):
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    seen = []
    monkeypatch.setattr(dispatch, "install_fp4_head", lambda *a: seen.append(a))
    dispatch.install_mono_decode(
        _Model(), _mtp_config(), SimpleNamespace(model=_Draft()), _builder()
    )
    assert not seen
