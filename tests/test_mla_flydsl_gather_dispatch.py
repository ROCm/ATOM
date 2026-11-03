# SPDX-License-Identifier: MIT
"""Which backend `_kv_b_proj_gather` picks, and when it asks.

Dispatch policy only -- no device. The FlyDSL gather serves one corner (gfx950,
page_size 1, fp8 cache and fp8 weight); a bf16 cache, an unquantized weight or
an MXFP4 one belong to the Triton op, which takes the same arguments. Getting
that wrong is not a slow path, it is a `ValueError` out of aiter mid-forward
that kills the engine, which is how three CI accuracy jobs died.

Both aiter functions are stubbed: what is under test is that ATOM consults the
predicate rather than the import, and consults it once.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")

from atom.model_ops import attention_mla as mla


@pytest.fixture(autouse=True)
def _gfx950_fp8_mha(monkeypatch):
    """Pin the FP8 MHA backend to gfx950 unless a test selects gfx1250."""
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_GFX1250_AVAILABLE", False)
    monkeypatch.setattr(mla, "_ASM_FP8_MHA_AVAILABLE", False)


def _impl(**over):
    """The slice of MLAAttention that `_kv_b_proj_gather` touches."""
    fields = {
        "kv_b_proj": SimpleNamespace(weight=SimpleNamespace(is_shuffled=True)),
        "_k_scale": None,
        "use_flydsl_gather_kv_b_proj": True,
        "_flydsl_gather_ok": None,
        "_flydsl_gather_fp8_ok": None,
    }
    return SimpleNamespace(**{**fields, **over})


@pytest.fixture
def spies(monkeypatch):
    calls = SimpleNamespace(flydsl=0, triton=0, asked=0)

    def supported(*_a, **_k):
        calls.asked += 1
        return calls.answer

    monkeypatch.setattr(mla, "_FLYDSL_GATHER_AVAILABLE", True)
    monkeypatch.setattr(mla, "gather_kv_b_proj_flydsl_supported", supported)
    monkeypatch.setattr(
        mla,
        "gather_kv_b_proj_flydsl",
        lambda *a, **k: setattr(calls, "flydsl", calls.flydsl + 1),
    )
    monkeypatch.setattr(
        mla,
        "gather_kv_b_proj",
        lambda *a, **k: setattr(calls, "triton", calls.triton + 1),
    )
    monkeypatch.setattr(mla, "_maybe_view_mxfp4_weight_for_gather", lambda _proj, w: w)
    return calls


def _gather(impl):
    mla.MLAAttention._kv_b_proj_gather(impl, None, None, None, None, None, None)


@pytest.mark.parametrize("answer, flydsl, triton", [(True, 1, 0), (False, 0, 1)])
def test_backend_follows_the_predicate_not_the_import(spies, answer, flydsl, triton):
    """A shape the kernel declines must reach the Triton op, not raise."""
    spies.answer = answer
    _gather(_impl())
    assert (spies.flydsl, spies.triton) == (flydsl, triton)


def test_the_predicate_is_asked_once(spies):
    """Its terms are fixed by the weights and the cache, so caching is sound;
    asking per gather would put a Python-level shape walk on every layer."""
    spies.answer = True
    impl = _impl()
    for _ in range(5):
        _gather(impl)
    assert spies.asked == 1
    assert spies.flydsl == 5


def test_env_off_never_asks(spies):
    """ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=0 is a decision, not a question."""
    spies.answer = True
    impl = _impl(use_flydsl_gather_kv_b_proj=False)
    _gather(impl)
    assert (spies.flydsl, spies.triton) == (0, 1)
    assert spies.asked == 0


@pytest.mark.parametrize("supported", [False, True])
def test_fp8_gather_checks_output_capability_once(monkeypatch, supported):
    impl = _impl()
    predicate = Mock(return_value=supported)
    launch = Mock()
    monkeypatch.setattr(mla, "_FLYDSL_GATHER_AVAILABLE", True)
    monkeypatch.setattr(mla, "gather_kv_b_proj_flydsl_supported", lambda *a: True)
    monkeypatch.setattr(
        mla, "gather_kv_b_proj_flydsl_fp8_supported", predicate, raising=False
    )
    monkeypatch.setattr(mla, "gather_kv_b_proj_flydsl", launch)
    monkeypatch.setattr(mla, "_maybe_view_mxfp4_weight_for_gather", lambda _proj, w: w)
    k, v = torch.empty(16, 12, 192, dtype=torch.bfloat16), torch.empty(
        16, 12, 128, dtype=torch.bfloat16
    )
    scales = (torch.ones(1), torch.ones(1))
    for _ in range(2):
        result = mla.MLAAttention._kv_b_proj_gather(
            impl, None, None, None, None, k, v, kv_out_scales=scales
        )
        args, kwargs = launch.call_args
        assert args[7].dtype == (torch.float8_e4m3fn if supported else torch.bfloat16)
        assert ("k_out_scale" in kwargs) is supported
        assert (result is not None) is supported
        if supported:
            assert result[0].data_ptr() == k.data_ptr()
        else:
            assert args[7] is k and args[8] is v
    assert predicate.call_count == 1


def _prefill_impl(**over):
    impl = _impl(
        use_asm_fp8_prefill_attn=False,
        use_flydsl_fp8_prefill_attn=True,
        _fp8_mha_ok=None,
        _fp8_mha_backend=None,
        num_heads=12,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_is_zero_pad=False,
        dtype=torch.bfloat16,
        scale=192**-0.5,
        layer_num=0,
        o_proj=lambda x: x,
    )
    for name in (
        "_check_fp8_prefill_mha",
        "_prepare_prefill_k",
        "_drop_rope_pad",
        "_flash_attn_prefill",
    ):
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    impl.__dict__.update(over)
    return impl


@pytest.mark.parametrize(
    "enabled,available,supported",
    [(False, True, True), (True, False, True), (True, True, False), (True, True, True)],
)
def test_fp8_mha_check_cached_and_warning_once(
    monkeypatch, caplog, enabled, available, supported
):
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_AVAILABLE", available)
    monkeypatch.setattr(
        mla.MLAAttention, "_fp8_prefill_fallback_logged", False, raising=False
    )
    predicate = Mock(return_value=supported)
    monkeypatch.setattr(
        mla, "flydsl_flash_attn_fp8_supported", predicate, raising=False
    )
    impl = _prefill_impl(use_flydsl_fp8_prefill_attn=enabled)
    for tokens in (1, 70, 1024):
        impl._check_fp8_prefill_mha(torch.empty(tokens, 12, 192, dtype=torch.bfloat16))
    assert impl._fp8_mha_ok is (enabled and available and supported)
    assert predicate.call_count == int(enabled and available)
    # A second layer must not repeat the warning for the same model/process.
    _prefill_impl(use_flydsl_fp8_prefill_attn=enabled)._check_fp8_prefill_mha(
        torch.empty(1, 12, 192, dtype=torch.bfloat16)
    )
    assert caplog.text.count("Falling back to BF16 attention") == int(
        enabled and not (available and supported)
    )


@pytest.mark.parametrize("supported", [False, True])
def test_gfx1250_fp8_mha_check_asks_the_a8w8_predicate(monkeypatch, supported):
    """gfx1250 asks the a8w8 kernel, once, with Q/K/V shapes and no device memory."""
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_GFX1250_AVAILABLE", True)
    monkeypatch.setattr(mla, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(
        mla.MLAAttention, "_fp8_prefill_fallback_logged", False, raising=False
    )
    monkeypatch.setattr(
        mla,
        "flydsl_flash_attn_fp8_supported",
        Mock(side_effect=AssertionError("gfx950 predicate on gfx1250")),
        raising=False,
    )
    predicate = Mock(return_value=supported)
    monkeypatch.setattr(
        mla, "flash_attn_a8w8_m32x8_supported", predicate, raising=False
    )
    impl = _prefill_impl()
    for tokens in (1, 70, 1024):
        impl._check_fp8_prefill_mha(torch.empty(tokens, 12, 192, dtype=torch.bfloat16))
    assert impl._fp8_mha_ok is supported
    assert impl._fp8_mha_backend == ("flydsl_gfx1250" if supported else None)
    assert predicate.call_count == 1
    mode, q, k, v = predicate.call_args.args
    assert mode == "varlen"
    assert [t.shape for t in (q, k, v)] == [(1, 12, 192), (1, 12, 192), (1, 12, 128)]
    assert all(t.dtype == torch.float8_e4m3fn and t.is_meta for t in (q, k, v))


def test_gfx1250_fp8_prefill_launches_a8w8_with_descales(monkeypatch):
    """The quantized Q/K/V and their descales reach the a8w8 kernel unchanged."""
    q8, k8, v8 = (
        torch.empty(8, 12, d, dtype=torch.float8_e4m3fn) for d in (192, 192, 128)
    )
    qs, ks, vs = torch.full((1,), 0.5), torch.full((1,), 0.25), torch.full((1,), 2.0)
    monkeypatch.setattr(
        mla,
        "fused_qkv_per_tensor_quant",
        lambda *a, **k: (q8, k8, v8, qs, ks, vs, None, None),
    )
    monkeypatch.setattr(
        mla,
        "flydsl_flash_attn_fp8_func",
        Mock(side_effect=AssertionError("gfx950 kernel on gfx1250")),
        raising=False,
    )
    result = (torch.empty(8, 12, 128, dtype=torch.bfloat16), torch.empty(12, 8))
    launch = Mock(return_value=result)
    monkeypatch.setattr(mla, "flash_attn_varlen_a8w8_m32x8", launch, raising=False)
    impl = _prefill_impl(_fp8_mha_ok=True, _fp8_mha_backend="flydsl_gfx1250")
    cu = torch.tensor([0, 8], dtype=torch.int32)
    q = torch.randn(8, 12, 192, dtype=torch.bfloat16)
    out = impl._flash_attn_prefill(
        q,
        torch.randn(8, 12, 192, dtype=torch.bfloat16),
        torch.randn(8, 12, 128, dtype=torch.bfloat16),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=8,
        max_seqlen_k=8,
        min_seqlen_q=8,
        dropout_p=0.0,
        causal=True,
        return_lse=True,
    )
    assert out is result
    launch.assert_called_once_with(
        q8,
        k8,
        v8,
        cu,
        cu,
        8,
        8,
        softmax_scale=impl.scale,
        causal=True,
        return_lse=True,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
    )


def _asm_gfx1250(monkeypatch, device_ok=True):
    """A gfx1250 whose AITER ships the ASM FP8 FMHA; `device_ok` answers its B0 gate."""
    monkeypatch.setattr(mla, "_ASM_FP8_MHA_AVAILABLE", True)
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_GFX1250_AVAILABLE", True)
    monkeypatch.setattr(mla, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(
        mla.MLAAttention, "_fp8_prefill_fallback_logged", False, raising=False
    )
    gate = Mock(return_value=device_ok)
    monkeypatch.setattr(mla, "is_gfx1250_asm_supported", gate, raising=False)
    return gate


@pytest.mark.parametrize("flydsl", [False, True])
def test_asm_fp8_mha_takes_precedence_on_gfx1250(monkeypatch, flydsl):
    """K3's D192x128 heads go to ASM, asked once, without consulting FlyDSL."""
    gate = _asm_gfx1250(monkeypatch)
    for name in ("flash_attn_a8w8_m32x8_supported", "flydsl_flash_attn_fp8_supported"):
        monkeypatch.setattr(
            mla, name, Mock(side_effect=AssertionError(name)), raising=False
        )
    impl = _prefill_impl(
        use_asm_fp8_prefill_attn=True, use_flydsl_fp8_prefill_attn=flydsl
    )
    for tokens in (1, 70, 1024):
        impl._check_fp8_prefill_mha(torch.empty(tokens, 12, 192, dtype=torch.bfloat16))
    assert impl._fp8_mha_ok is True
    assert impl._fp8_mha_backend == "asm"
    assert gate.call_count == 1


@pytest.mark.parametrize("reason", ["device", "head_dim"])
@pytest.mark.parametrize("flydsl", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_asm_fp8_mha_declined_falls_through(
    monkeypatch, caplog, reason, flydsl, explicit
):
    """Where ASM does not apply, the next enabled backend serves the layer; the
    default-on ASM flag alone never warns about the BF16 fallback."""
    _asm_gfx1250(monkeypatch, device_ok=reason != "device")
    monkeypatch.setattr(
        mla, "flash_attn_a8w8_m32x8_supported", Mock(return_value=True), raising=False
    )
    if explicit:
        monkeypatch.setenv("ATOM_USE_ASM_FP8_PREFILL_ATTN", "1")
    else:
        monkeypatch.delenv("ATOM_USE_ASM_FP8_PREFILL_ATTN", raising=False)
    impl = _prefill_impl(
        use_asm_fp8_prefill_attn=True, use_flydsl_fp8_prefill_attn=flydsl
    )
    head_dim = 192 if reason == "device" else 128
    impl._check_fp8_prefill_mha(torch.empty(4, 12, head_dim, dtype=torch.bfloat16))
    assert impl._fp8_mha_backend == ("flydsl_gfx1250" if flydsl else None)
    assert impl._fp8_mha_ok is flydsl
    assert caplog.text.count("Falling back to BF16 attention") == int(
        explicit and not flydsl
    )


@pytest.mark.parametrize("return_lse", [False, True])
def test_asm_fp8_prefill_launch_and_lse_layout(monkeypatch, return_lse):
    """The ASM kernel gets the quantized Q/K/V and descales; its [total_q, heads, 1]
    LSE reaches the merges as [heads, total_q]."""
    q8, k8, v8 = (
        torch.empty(8, 12, d, dtype=torch.float8_e4m3fn) for d in (192, 192, 128)
    )
    qs, ks, vs = torch.full((1,), 0.5), torch.full((1,), 0.25), torch.full((1,), 2.0)
    monkeypatch.setattr(
        mla,
        "fused_qkv_per_tensor_quant",
        lambda *a, **k: (q8, k8, v8, qs, ks, vs, None, None),
    )
    for name in ("flydsl_flash_attn_fp8_func", "flash_attn_varlen_a8w8_m32x8"):
        monkeypatch.setattr(
            mla, name, Mock(side_effect=AssertionError(name)), raising=False
        )
    out = torch.empty(8, 12, 128, dtype=torch.bfloat16)
    lse = torch.arange(8 * 12, dtype=torch.float32).view(8, 12, 1)
    launch = Mock(return_value=(out, lse))
    monkeypatch.setattr(mla, "fmha_fwd_fp8_varlen_asm", launch, raising=False)
    impl = _prefill_impl(_fp8_mha_ok=True, _fp8_mha_backend="asm")
    cu = torch.tensor([0, 8], dtype=torch.int32)
    result = impl._flash_attn_prefill(
        torch.randn(8, 12, 192, dtype=torch.bfloat16),
        torch.randn(8, 12, 192, dtype=torch.bfloat16),
        torch.randn(8, 12, 128, dtype=torch.bfloat16),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=8,
        max_seqlen_k=8,
        min_seqlen_q=8,
        dropout_p=0.0,
        causal=True,
        return_lse=return_lse,
    )
    launch.assert_called_once_with(
        q8,
        k8,
        v8,
        qs,
        ks,
        vs,
        cu,
        cu,
        8,
        8,
        softmax_scale=impl.scale,
        is_causal=True,
        return_lse=return_lse,
    )
    if not return_lse:
        assert result is out
        return
    result_out, result_lse = result
    assert result_out is out
    assert result_lse.shape == (12, 8) and result_lse.is_contiguous()
    torch.testing.assert_close(result_lse, lse.squeeze(-1).t(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "path", ["plain", "cached_single", "cached_chunked", "cached_dcp"]
)
def test_unsupported_fp8_prefill_keeps_bf16_inputs(monkeypatch, path):
    """Decline before quant/gather overwrites data, including the DCP handoff."""
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_AVAILABLE", True)
    monkeypatch.setattr(
        mla, "flydsl_flash_attn_fp8_supported", lambda *a: False, raising=False
    )
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: False)
    for name in (
        "fused_qkv_per_tensor_quant",
        "fused_kv_per_tensor_quant",
        "quant_fp8_per_tensor",
        "flydsl_flash_attn_fp8_func",
        "flash_attn_varlen_a8w8_m32x8",
        "fmha_fwd_fp8_varlen_asm",
    ):
        monkeypatch.setattr(
            mla,
            name,
            Mock(side_effect=AssertionError(f"unexpected {name}")),
            raising=False,
        )
    q = torch.randn(8, 12, 192, dtype=torch.bfloat16)
    kv = torch.randn(8, 12, 256, dtype=torch.bfloat16)
    rope = torch.randn(8, 1, 64, dtype=torch.bfloat16)
    impl = _prefill_impl(kv_b_proj=lambda x: kv)
    attention_calls, gather_calls = [], []

    def attend(**kw):
        assert kw["q"] is q
        assert kw["k"].shape == (8, 12, 192)
        assert kw["v"].shape == (8, 12, 128)
        assert all(kw[n].dtype == torch.bfloat16 for n in ("q", "k", "v"))
        attention_calls.append(kw)
        output = torch.ones(8, 12, 128, dtype=torch.bfloat16)
        return (output, torch.zeros(12, 8)) if kw["return_lse"] else output

    def gather(cache, indptr, indices, cu, k, v, *a, **kw):
        assert kw.get("kv_out_scales") is None
        assert k.dtype == v.dtype == torch.bfloat16
        k.fill_(2)
        v.fill_(3)
        gather_calls.append(kw)

    def dcp(*args, **kw):
        assert kw["q_fp8"] is None and kw["kv_out_scales"] is None
        return None, None

    monkeypatch.setattr(mla, "flash_attn_varlen_func", attend)
    from atom.model_ops.attentions import triton_merge_attn_states as merge

    monkeypatch.setattr(
        merge, "merge_attn_states", lambda **kw: kw["output"].copy_(kw["prefix_output"])
    )
    impl._gather_cached_kv_b_proj = gather
    impl._dcp_compute_prefill_context = dcp
    cu = torch.tensor([0, 8], dtype=torch.int32)
    meta = SimpleNamespace(
        total_kv=8,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        kv_indptr=cu,
        kv_indices=None,
        max_seqlen_q=8,
        max_seqlen_k=8,
        min_seqlen_q=8,
        dropout_p=0.0,
    )
    if path == "plain":
        mla.MLAAttention._forward_prefill_mha(impl, q, None, rope, None, meta)
    elif path == "cached_single":
        mla.MLAAttention._forward_prefill_cached_single_pass(impl, q, None, meta)
    else:
        chunks = SimpleNamespace(
            num_chunks=1,
            is_dcp=path == "cached_dcp",
            k_workspace=torch.empty_like(q),
            v_workspace=torch.empty_like(kv[..., :128]),
            total_tokens=[8],
            kv_indptr=[cu],
            kv_indices=[None],
            cu_seqlens_k=[cu],
            max_seqlen_k=[8],
            shuffle_kv_block_indptr=None,
            shuffle_kv_block_indices=None,
        )
        mla.MLAAttention._forward_prefill_cached_chunked(
            impl, q, None, rope, None, meta, chunks
        )
    assert len(attention_calls) == (2 if path == "cached_chunked" else 1)
    assert len(gather_calls) == int(path in ("cached_single", "cached_chunked"))
    if path != "cached_single":
        torch.testing.assert_close(
            attention_calls[0]["k"],
            torch.cat((kv[..., :128], rope.expand(-1, 12, -1)), dim=-1),
            rtol=0,
            atol=0,
        )
