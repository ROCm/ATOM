# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of the gfx1250 DSpark draft MLA kernels' integration
(``ATOM_KIMI_MONO_MLA_DRAFT``): the switch, ``K3DSparkMLAAttention``'s hook,
the layer binding's refusals, a step's refusals and the decode from a
prepadded q_out."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.mono.runtime.consensus import MonoUnsupported

aiter = pytest.importorskip("aiter", reason="the draft and MLA modules import aiter")
from aiter import QuantType, dtypes


def test_the_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_DRAFT", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_DRAFT is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_DRAFT", "1")
    assert envs.ATOM_KIMI_MONO_MLA_DRAFT is True


def _draft_module():
    try:
        from atom.models import kimi_k3_dspark
    except ImportError as error:
        pytest.skip(f"kimi_k3_dspark needs the GPU stack: {error}")
    return kimi_k3_dspark


def test_switched_off_the_attention_never_asks_the_draft_kernels(monkeypatch):
    dspark = _draft_module()
    from atom.models.kimi_k3.mono import draft_mla

    monkeypatch.setattr(
        draft_mla, "forward", lambda *a: pytest.fail("draft kernels asked with it off")
    )

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    fake = SimpleNamespace(_try_mla_draft=False, fused_qkv_a_proj=stop)
    with pytest.raises(Stop):
        dspark.K3DSparkMLAAttention.forward(fake, "positions", ("x", "scale"))


def test_switched_on_a_q_out_takes_the_attentions_decode(monkeypatch):
    dspark = _draft_module()
    from atom.models.kimi_k3.mono import draft_mla

    monkeypatch.setattr(draft_mla, "forward", lambda attn, pos, hidden: ("out", pos))
    fake = SimpleNamespace(_try_mla_draft=True)
    assert dspark.K3DSparkMLAAttention.forward(fake, "pos", "x") == ("out", "pos")
    monkeypatch.setattr(draft_mla, "forward", lambda *a: None)

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    fake = SimpleNamespace(_try_mla_draft=True, fused_qkv_a_proj=stop)
    with pytest.raises(Stop):  # None: production's projections run
        dspark.K3DSparkMLAAttention.forward(fake, "pos", ("x", "scale"))


def test_a_refused_layer_turns_its_attempts_off(monkeypatch):
    from atom.models.kimi_k3.mono import draft_mla

    def refuse(attn):
        raise MonoUnsupported("no")

    monkeypatch.setattr(draft_mla, "bind", refuse)
    attn = SimpleNamespace(
        mla_draft=None, _try_mla_draft=True, input_quant_prefix="l.a"
    )
    assert draft_mla.forward(attn, torch.zeros(2, dtype=torch.int64), None) is None
    assert attn._try_mla_draft is False


# ------------------------------------------------------------- the binding
def _rope(neox=False, dim=64, dtype=torch.bfloat16):
    cache = torch.zeros(128, 1, 1, 32, dtype=dtype)
    return SimpleNamespace(
        rotary_dim=dim, is_neox_style=neox, cos_cache=cache, sin_cache=cache.clone()
    )


def _impl(heads=8, dcp=8, **change):
    qrep = heads * dcp
    impl = {
        "forward_decode_q_out": lambda q_out, prepadded=False: q_out,
        "is_sparse_mla": False,
        "use_seg_mla": False,
        "kv_shuffled": False,
        "kv_cache_dtype": "fp8",
        "_k_scale": torch.tensor(1.0),
        "_q_scale": torch.tensor(1.0),
        "rotary_emb": _rope(),
        "qrep_enabled": True,
        "qrep_num_heads": qrep,
        "W_K_qrep": torch.zeros(qrep, 512, 128, dtype=torch.float8_e4m3fn),
        "W_K_qrep_scale": torch.ones((), dtype=torch.float32),
        "num_heads": heads,
        "W_K": torch.zeros(heads, 512, 128, dtype=torch.float8_e4m3fn),
        "W_K_scale": torch.ones((), dtype=torch.float32),
        "dcp_world_size": dcp,
        "dcp_kernel_num_heads": qrep,
        "padded_num_heads": 16,
        "_fused_q_head_pad": dcp == 1,
        "layer_num": 98,
    }
    impl.update(change)
    return SimpleNamespace(**impl)


def _linear(rows, cols, e8m0=True, shuffled=True, code=120):
    padded = -(-rows // 128) * 128
    weight = nn.Parameter(
        torch.zeros(padded, cols, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    weight.is_shuffled = shuffled
    scale = torch.full((padded // 128, cols // 128), code, dtype=torch.uint8)
    scale = nn.Parameter(
        scale.view(torch.float8_e8m0fnu) if e8m0 else scale.float(),
        requires_grad=False,
    )
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        bias=None,
        native_a8_group_rows=None,
        quant_type=QuantType.per_1x128,
        blockscale_e8m0_scale=e8m0,
    )


def _norm(**change):
    norm = {
        "use_fused_quant": True,
        "fused_allreduce": False,
        "quant_type": QuantType.per_1x128,
        "params_dtype": dtypes.fp8,
        "_aiter_transpose_scale": True,
        "_aiter_e8m0_scale": True,
    }
    norm.update(change)
    return SimpleNamespace(**norm)


def _attn(impl=None, q_b_heads=64, norm=None, **change):
    vec = lambda n: torch.ones(n, dtype=torch.bfloat16)
    attn = {
        "input_quant_prefix": "layers.0.self_attn.fused_qkv_a_proj",
        "_input_norm": norm or _norm(),
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "fuse_qknorm_quant": True,
        "qknorm_dtype": dtypes.fp8,
        "qknorm_quant_type_value": QuantType.per_1x128.value,
        "fused_qkv_a_proj": _linear(2112, 7168),
        "q_b_proj": _linear(q_b_heads * 192, 1536),
        "q_a_layernorm": SimpleNamespace(weight=vec(1536), eps=1e-5),
        "kv_a_layernorm": SimpleNamespace(weight=vec(512), eps=1e-5),
        "mla_attn": SimpleNamespace(impl=impl or _impl()),
    }
    attn.update(change)
    return SimpleNamespace(**attn)


@pytest.fixture
def preshuffle(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")


def test_qrep_binds_the_dcp_groups_heads_in_place(preshuffle):
    from atom.models.kimi_k3.mono.draft_mla import bind

    attn = _attn()
    layer = bind(attn)
    impl = attn.mla_attn.impl
    assert layer.w_uk is impl.W_K_qrep and layer.heads == 64
    assert (layer.out_heads, layer.compute_all, layer.prepadded) == (64, True, False)
    assert layer.w_a.data_ptr() == attn.fused_qkv_a_proj.weight.data_ptr()
    assert layer.w_qb.data_ptr() == attn.q_b_proj.weight.data_ptr()
    assert layer.max_pos == 128 and layer.prefix == "layers.0.self_attn"


def test_a_padded_dcp_width_is_written_padded(preshuffle):
    from atom.models.kimi_k3.mono.draft_mla import bind

    layer = bind(_attn(_impl(dcp_kernel_num_heads=96)))
    assert (layer.heads, layer.out_heads, layer.prepadded) == (64, 96, True)


def test_without_dcp_the_local_heads_go_padded_and_slot_gated(preshuffle):
    from atom.models.kimi_k3.mono.draft_mla import bind

    layer = bind(_attn(_impl(heads=12, dcp=1, qrep_enabled=False), q_b_heads=12))
    assert (layer.heads, layer.out_heads, layer.compute_all) == (12, 16, False)
    assert layer.prepadded is False


@pytest.mark.parametrize(
    ("impl_change", "message"),
    [
        ({"is_sparse_mla": True}, "sparse"),
        ({"use_seg_mla": True}, "segmented"),
        ({"kv_shuffled": True}, "shuffled"),
        ({"kv_cache_dtype": "auto"}, "not FP8"),
        ({"_k_scale": torch.tensor(0.5)}, "scales"),
        ({"rotary_emb": _rope(neox=True)}, "GPT-J"),
        ({"rotary_emb": _rope(dtype=torch.float32)}, "GPT-J"),
        ({"rotary_emb": _rope(dim=32)}, "GPT-J"),
        ({"qrep_enabled": False}, "QREP"),
        ({"W_K_qrep_scale": None}, "BF16 absorb"),
        ({"W_K_qrep": torch.zeros(64, 512, 128, dtype=torch.bfloat16)}, "W_UK"),
        ({"forward_decode_q_out": None}, "MLAAttention"),
    ],
)
def test_a_layer_the_kernels_would_read_wrong_is_refused(
    preshuffle, impl_change, message
):
    from atom.models.kimi_k3.mono.draft_mla import bind

    with pytest.raises(MonoUnsupported, match=message):
        bind(_attn(_impl(**impl_change)))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"_input_norm": None}, "input norm"),
        ({"_input_norm": _norm(_aiter_e8m0_scale=False)}, "E8M0 FP8 row"),
        ({"_input_norm": _norm(_aiter_transpose_scale=False)}, "E8M0 FP8 row"),
        ({"_input_norm": _norm(use_fused_quant=False)}, "E8M0 FP8 row"),
        ({"qknorm_dtype": torch.bfloat16}, "q norm"),
        ({"fuse_qknorm_quant": False}, "q norm"),
        ({"q_lora_rank": 1024}, "widths"),
        ({"fused_qkv_a_proj": _linear(2112, 7168, e8m0=False)}, "E8M0"),
        ({"q_b_proj": _linear(64 * 192, 1536, shuffled=False)}, "preshuffled"),
        ({"q_b_proj": _linear(64 * 192, 1536, e8m0=False)}, "q_b_proj"),
        ({"q_b_proj": _linear(64 * 192, 1536, code=0)}, "2\\^-127 or NaN"),
        ({"q_b_proj": _linear(48 * 192, 1536)}, "expected"),
    ],
)
def test_a_projection_or_norm_the_kernels_cannot_read_is_refused(
    preshuffle, change, message
):
    from atom.models.kimi_k3.mono.draft_mla import bind

    with pytest.raises(MonoUnsupported, match=message):
        bind(_attn(**change))


# ------------------------------------------------------------ the step
def _step(monkeypatch, rows=7, **change):
    from atom.models.kimi_k3.mono import draft_mla

    context = {"is_dummy_run": False, "is_prefill": False}
    metadata = {
        "slot_mapping": torch.arange(rows, dtype=torch.int64),
        "dtype_q": dtypes.fp8,
    }
    cache = torch.zeros(32, 1, 576, dtype=dtypes.fp8)
    hidden = (
        torch.zeros(rows, 7168, dtype=dtypes.fp8),
        torch.zeros(rows, 56, dtype=dtypes.fp8_e8m0),
    )
    positions = torch.arange(rows, dtype=torch.int64)
    for name, value in change.items():
        if name in context:
            context[name] = value
        elif name == "cache":
            cache = value
        elif name == "hidden":
            hidden = value
        elif name == "positions":
            positions = value
        else:
            metadata[name] = value
    forward = SimpleNamespace(
        context=SimpleNamespace(**context),
        attn_metadata=SimpleNamespace(**metadata),
        kv_cache_data={"layer_98": SimpleNamespace(k_cache=cache)},
    )
    import atom.utils.forward_context as fc

    monkeypatch.setattr(fc, "get_forward_context", lambda: forward)
    layer = SimpleNamespace(attention=SimpleNamespace(layer_num=98))
    return draft_mla.step_refusal(layer, rows, positions, hidden), (cache, forward)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"is_dummy_run": True}, "decode"),
        ({"is_prefill": True}, "decode"),
        ({"hidden": torch.zeros(7, 7168, dtype=torch.bfloat16)}, "pair"),
        ({"hidden": (torch.zeros(7, 7168, dtype=torch.bfloat16), None)}, "input row"),
        ({"positions": torch.arange(7, dtype=torch.int32)}, "positions"),
        ({"slot_mapping": torch.arange(6, dtype=torch.int64)}, "slot mapping"),
        ({"dtype_q": torch.bfloat16}, "q_out"),
        ({"cache": torch.zeros(32, 1, 576, dtype=torch.bfloat16)}, "KV cache"),
        ({"cache": torch.zeros(0, 1, 576, dtype=dtypes.fp8)}, "KV cache"),
    ],
)
def test_a_step_the_kernels_cannot_take_is_refused(monkeypatch, change, message):
    (why, found), _ = _step(monkeypatch, **change)
    assert found is None and message in why


def test_a_step_the_kernels_take_hands_its_tensors(monkeypatch):
    (why, found), (cache, forward) = _step(monkeypatch)
    assert why is None
    x, x_scale, slots, got_cache = found
    assert got_cache is cache and slots is forward.attn_metadata.slot_mapping
    assert x.shape == (7, 7168) and x_scale.numel() == 7 * 56
    (why, _), _ = _step(monkeypatch, rows=65)
    assert "65 rows" in why


# ---------------------------------------------------------- the decode
def _attention_module():
    try:
        from atom.model_ops import attention_mla
    except ImportError as error:
        pytest.skip(f"attention_mla needs the GPU stack: {error}")
    return attention_mla


@pytest.mark.parametrize("prepadded", [False, True])
def test_a_prepadded_q_out_reaches_the_dcp_decode_as_such(monkeypatch, prepadded):
    mla = _attention_module()
    calls = []
    metadata, cache = object(), object()
    context = SimpleNamespace(
        attn_metadata=metadata,
        kv_cache_data={"layer_5": SimpleNamespace(k_cache=cache)},
    )
    monkeypatch.setattr(mla, "get_forward_context", lambda: context)
    fake = SimpleNamespace(
        layer_num=5,
        dcp_world_size=8,
        qrep_enabled=True,
        use_seg_mla=False,
        _dcp_decode=lambda *a, **k: calls.append((a, k)) or "dcp_out",
    )
    assert mla.MLAAttention.forward_decode_q_out(fake, "q_out", prepadded) == "dcp_out"
    assert calls == [(("q_out", cache, metadata, True), {"q_prepadded": prepadded})]


def test_the_dcp_decode_hands_its_padding_on(monkeypatch):
    mla = _attention_module()
    seen = []
    fake = SimpleNamespace(
        _forward_decode=lambda *a, **k: seen.append(k) or ("o", "lse"),
        is_sparse_mla=False,
        _dcp_project_merge_out=lambda o, lse, **k: (o, lse),
        _cp_triton_ctx=None,
    )
    assert mla.MLAAttention._dcp_decode(fake, "q", "c", "m", True, q_prepadded=True) == (
        "o", "lse",
    )  # fmt: skip
    assert seen == [{"return_lse": True, "q_prepadded": True}]
    with pytest.raises(AssertionError):
        mla.MLAAttention._dcp_decode(fake, "q", "c", "m", False, q_prepadded=True)
