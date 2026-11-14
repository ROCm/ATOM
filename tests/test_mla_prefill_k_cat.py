"""CPU checks for the MLA prefill K concat kernel (`_cat_prefill_k`):

  AITER_AOT_REPLAY=1 GPU_ARCHS=gfx1250 TRITON_INTERPRET=1 \
      python -m pytest -q tests/test_mla_prefill_k_cat.py

AITER_AOT_REPLAY/GPU_ARCHS only give aiter an fp8 dtype without a GPU (the
existing dispatch tests need it). The Triton kernel runs in the interpreter on
CPU tensors, so the device gate is lifted per test; bit equality is checked on
the int16 view (-0.0, NaN payloads, inf and subnormals included). The
interpreter does not model bf16 arithmetic, which the kernel does not do.
"""

import os
from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
_arch = pytest.importorskip(
    "aiter.ops.triton.utils._triton.arch_info",
    reason="attention_mla imports the AITER runtime",
)
if _arch._CACHED_ARCH is None and not torch.cuda.is_available():
    _arch._CACHED_ARCH = "gfx1250"  # GPU-less: aiter probes the arch at import

from atom.model_ops import attention_mla as mla

interpret_only = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="needs TRITON_INTERPRET=1"
)

SPECIAL = [0x8000, 0x7FC1, 0xFFC3, 0x7F80, 0xFF80, 0x0001, 0x807F, 0x0000]


def _special_bits():
    return torch.tensor(
        [v - 0x10000 if v >= 0x8000 else v for v in SPECIAL], dtype=torch.int16
    )


def _bits(x):
    return x.contiguous().view(torch.int16)


def _inputs(tokens, heads, nope=128, v=128, rope=64, row=2112, seed=0):
    """k_nope / v as the kv_b_proj split views, k_rope as a [T, 1, rope] view of
    the qkv_a row, like the DSR1 prefill hands them over."""
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(tokens, heads * (nope + v), generator=g).to(torch.bfloat16)
    qkv_a = torch.randn(tokens, row, generator=g).to(torch.bfloat16)
    s = _special_bits()
    kv.view(torch.int16)[0, : s.numel()] = s
    last_nope_end = (heads - 1) * (nope + v) + nope
    kv.view(torch.int16)[-1, last_nope_end - s.numel() : last_nope_end] = s
    qkv_a.view(torch.int16)[0, row - rope : row - rope + s.numel()] = s
    qkv_a.view(torch.int16)[-1, row - s.numel() :] = s
    k_nope, v_ = kv.view(tokens, heads, nope + v).split([nope, v], dim=-1)
    k_rope = qkv_a[:, row - rope :].unsqueeze(1)
    return k_nope, v_, k_rope


def _ref(k_nope, k_rope):
    return torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)


def _assert_same(new, ref):
    assert new.shape == ref.shape and new.dtype == ref.dtype
    assert new.stride() == ref.stride() and new.is_contiguous()
    assert torch.equal(_bits(new), _bits(ref))


@pytest.fixture
def kernel_on_cpu(monkeypatch):
    monkeypatch.setattr(mla, "_prefill_k_cat_supported", lambda *a: True)


@interpret_only
@pytest.mark.parametrize(
    "tokens,heads,nope,rope",
    [
        (1, 128, 128, 64),
        (7, 128, 128, 64),
        (33, 16, 128, 64),
        (5, 20, 128, 64),
        (3, 1, 128, 64),
        (4, 64, 256, 64),
    ],
)
def test_kernel_bitwise_equals_cat(kernel_on_cpu, tokens, heads, nope, rope):
    k_nope, _, k_rope = _inputs(tokens, heads, nope=nope, v=128, rope=rope)
    assert not k_nope.is_contiguous()
    _assert_same(mla._cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


@interpret_only
@pytest.mark.parametrize("head_block", [1, 4, 32, 64])
def test_kernel_head_block_independent(kernel_on_cpu, monkeypatch, head_block):
    monkeypatch.setattr(mla, "_PREFILL_K_CAT_HEAD_BLOCK", head_block)
    k_nope, _, k_rope = _inputs(9, 48, seed=head_block)
    _assert_same(mla._cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


@interpret_only
def test_kernel_contiguous_and_expanded_rope(kernel_on_cpu):
    k_nope = torch.randn(6, 8, 128).to(torch.bfloat16)
    k_rope = torch.randn(1, 1, 64).to(torch.bfloat16).expand(6, 1, 64)
    _assert_same(mla._cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


def test_supported_gate(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    k_nope, _, k_rope = _inputs(1024, 16)
    ok = mla._prefill_k_cat_supported
    assert ok(k_nope, k_rope)
    assert not ok(k_nope[:1023], k_rope[:1023])
    assert not ok(k_nope[:0], k_rope[:0])
    assert not ok(k_nope, k_rope.expand(-1, 16, -1))
    assert not ok(k_nope, k_rope.squeeze(1))
    assert not ok(k_nope, k_rope.float())
    assert not ok(k_nope[..., :96], k_rope)
    assert not ok(k_nope, k_rope[..., :48])
    assert not ok(k_nope.transpose(1, 2).contiguous().transpose(1, 2), k_rope)
    assert not ok(k_nope, k_rope.transpose(0, 2).contiguous().transpose(0, 2))


def test_cpu_falls_back_to_cat():
    k_nope, _, k_rope = _inputs(5, 16)
    assert not mla._prefill_k_cat_supported(k_nope, k_rope)
    _assert_same(mla._cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))
    wide = k_rope.expand(-1, 16, -1)
    _assert_same(mla._cat_prefill_k(k_nope, wide), _ref(k_nope, wide))


class _KvB:
    """BF16 kv_b_proj stand-in: the MLPerf checkpoint leaves it unquantized."""

    weight = torch.empty(1, dtype=torch.bfloat16)
    weight_scale = None

    def __init__(self, kv):
        self.kv = kv

    def __call__(self, x):
        return self.kv


def _impl(fp8_ok, heads, kv=None):
    impl = SimpleNamespace(
        _flydsl_fp8_mha_ok=fp8_ok,
        use_flydsl_fp8_prefill_attn=False,
        num_heads=heads,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_is_zero_pad=False,
        scale=192**-0.5,
        kv_b_proj=_KvB(kv),
        o_proj=lambda x: x,
    )
    for name in (
        "_check_flydsl_fp8_mha",
        "_prepare_prefill_k",
        "_drop_rope_pad",
        "_flash_attn_prefill",
    ):
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    return impl


def test_prepare_prefill_k_fp8_split_unchanged():
    k_nope, _, k_rope = _inputs(3, 16)
    k, rope = _impl(True, 16)._prepare_prefill_k(k_nope, k_rope)
    assert k is k_nope and rope is k_rope


@pytest.mark.parametrize("force_kernel", [False, True])
def test_forward_prefill_mha_hands_attention_the_cat(monkeypatch, force_kernel):
    if force_kernel:
        if os.environ.get("TRITON_INTERPRET") != "1":
            pytest.skip("needs TRITON_INTERPRET=1")
        monkeypatch.setattr(mla, "_prefill_k_cat_supported", lambda *a: True)
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: True)
    tokens, heads = 11, 16
    g = torch.Generator().manual_seed(3)
    kv = torch.randn(tokens, heads * 256, generator=g).to(torch.bfloat16)
    qkv_a = torch.randn(tokens, 2112, generator=g).to(torch.bfloat16)
    rope2d = qkv_a[:, 2048:]
    q = torch.randn(tokens, heads, 192, generator=g).to(torch.bfloat16)
    impl = _impl(False, heads, kv)
    seen = {}

    def attend(**kw):
        seen.update(kw)
        return torch.ones(tokens, heads, 128, dtype=torch.bfloat16)

    monkeypatch.setattr(mla, "flash_attn_varlen_func", attend)
    cu = torch.tensor([0, 4, 5, tokens], dtype=torch.int32)
    meta = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=6,
        max_seqlen_k=6,
        min_seqlen_q=1,
        dropout_p=0.0,
    )
    out = mla.MLAAttention._forward_prefill_mha(impl, q, None, rope2d, None, meta)
    assert out.shape == (tokens, heads * 128)
    k_nope, v = kv.view(tokens, heads, 256).split([128, 128], dim=-1)
    _assert_same(seen["k"], _ref(k_nope, rope2d.unsqueeze(1)))
    assert seen["v"].data_ptr() == v.data_ptr() and seen["v"].stride() == v.stride()
    assert seen["q"] is q and seen["causal"] is True
