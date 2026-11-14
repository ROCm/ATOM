# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter import dtypes, fused_qk_rope_concat_and_cache_mla

from atom.model_ops import attention_mla
from atom.model_ops.attention_mla import MLAAttention
from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding
from atom.model_ops.triton_mla_absorb_qpack import mla_absorb_q_pack

# Kimi-K3.1 DSA prefill under QREP: 96 replicated heads, 128 nope, 64 rope,
# 512 latent.
HEADS, NOPE, PE, LORA = 96, 128, 64, 512


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """e4m3 codes as signed step counts from zero, so +0 and -0 coincide."""
    c = codes.view(torch.uint8).to(torch.int32)
    return torch.where(c >= 128, 128 - c, c)


def _pack(q_nope, q_pe, k_nope, k_pe, q_out, kv_cache, slots, rope, q_scale):
    fused_qk_rope_concat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        q_out,
        slots,
        torch.ones(1, device="cuda"),
        q_scale,
        torch.arange(slots.numel(), device="cuda"),
        rope.cos_cache,
        rope.sin_cache,
        is_neox=rope.is_neox_style,
        is_nope_first=True,
        compute_all_q_rope=True,
    )


@pytest.mark.parametrize("q_scale", [1.0, 2.0**-10])
@pytest.mark.parametrize("tokens", [77, 16384])
def test_absorb_into_q_out_matches_two_kernel_path(tokens, q_scale):
    torch.manual_seed(tokens)
    q = torch.randn(tokens, HEADS * (NOPE + PE), device="cuda", dtype=torch.bfloat16)
    w_k = 0.05 * torch.randn(HEADS, LORA, NOPE, device="cuda", dtype=torch.bfloat16)
    # The prefill QREP slice of MLAAttention that _q_proj_and_k_up_proj reads.
    layer = SimpleNamespace(
        qrep_enabled=True,
        qrep_num_heads=HEADS,
        qk_head_dim=NOPE + PE,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=PE,
        q_proj=lambda x, x_scale: x,
        W_K_qrep=w_k,
        W_K_qrep_scale=None,
    )
    ql_nope, q_pe = MLAAttention._q_proj_and_k_up_proj(layer, q, group=True)
    q_nope, _ = MLAAttention._q_proj_and_k_up_proj(
        layer, q, group=True, k_up_proj=False
    )
    # forward_impl sizes q_out from q_nope's leading dims either way.
    assert q_nope.shape == (tokens, HEADS, NOPE)
    assert ql_nope.shape == (tokens, HEADS, LORA)
    kv = torch.randn(tokens, 1, LORA + PE, device="cuda", dtype=torch.bfloat16)
    k_nope, k_pe = kv.split([LORA, PE], dim=-1)
    # Every other token is another DCP rank's: its K is not written, its Q is.
    slots = torch.arange(tokens, device="cuda")
    slots[1::2] = -1
    rope = NoPositionalRotaryEmbedding(PE, PE, tokens, 10000.0, dtype=torch.bfloat16)
    rope = rope.cuda()
    scale = torch.tensor([q_scale], device="cuda")

    def buffers():
        return (
            torch.empty(tokens, HEADS, LORA + PE, device="cuda", dtype=dtypes.fp8),
            torch.zeros(tokens, 1, LORA + PE, device="cuda", dtype=dtypes.fp8),
        )

    ref_q, ref_kv = buffers()
    _pack(ql_nope, q_pe, k_nope, k_pe, ref_q, ref_kv, slots, rope, scale)

    new_q, new_kv = buffers()
    stand_in = q_pe.new_empty(LORA).expand(tokens, HEADS, -1)
    _pack(stand_in, q_pe, k_nope, k_pe, new_q, new_kv, slots, rope, scale)
    mla_absorb_q_pack(q_nope.transpose(0, 1), w_k, new_q, scale)

    torch.testing.assert_close(new_kv.view(torch.uint8), ref_kv.view(torch.uint8))
    torch.testing.assert_close(
        new_q[..., LORA:].view(torch.uint8), ref_q[..., LORA:].view(torch.uint8)
    )
    # The NaN code sits one step past the largest finite one, which the
    # step bound below would accept.
    assert not new_q[..., :LORA].float().isnan().any(), "nope half stores an FP8 NaN"
    steps = (_ordinal(new_q[..., :LORA]) - _ordinal(ref_q[..., :LORA])).abs()
    assert steps.max().item() <= 1, f"nope half off by {steps.max().item()} e4m3 steps"
    # Rounding alone moves about 3% of codes one step; a systematic error
    # under one step moves far more.
    share = (steps == 1).float().mean().item()
    assert share < 0.06, f"{share:.1%} of nope codes one step off"


# GLM-5.x has 192 nope dims, which the absorb kernel does not compile for, and
# 512 overruns gfx950's shared memory. The kernel has not run on gfx942.
@pytest.mark.parametrize(
    "gfx, nope, absorbs",
    [
        ("gfx950", NOPE, True),
        ("gfx950", 192, False),
        ("gfx950", 512, False),
        ("gfx942", NOPE, False),
    ],
)
def test_absorb_gate_takes_two_kernel_path_off_kernel_shapes(
    monkeypatch, gfx, nope, absorbs
):
    monkeypatch.setattr(attention_mla, "get_gfx_runtime", lambda: gfx)
    layer = SimpleNamespace(
        W_K_qrep_scale=None,
        kv_shuffled=False,
        qk_nope_head_dim=nope,
        kv_lora_rank=LORA,
    )
    kv_cache = torch.empty(1, device="cuda", dtype=dtypes.fp8)
    gate = MLAAttention._absorb_into_q_out(layer, True, True, kv_cache, dtypes.fp8)
    assert gate == absorbs


# L=96 would store its second 64-column block over half the rope columns.
@pytest.mark.parametrize("nope, lora", [(NOPE, 96), (192, LORA), (512, LORA)])
def test_absorb_q_pack_rejects_shapes_its_kernel_cannot_take(nope, lora):
    tokens = 2048
    q_nope = torch.randn(HEADS, tokens, nope, device="cuda", dtype=torch.bfloat16)
    w_k = torch.randn(HEADS, lora, nope, device="cuda", dtype=torch.bfloat16)
    q_out = torch.zeros(tokens, HEADS, lora + PE, device="cuda", dtype=dtypes.fp8)
    with pytest.raises(AssertionError, match="no absorb kernel"):
        mla_absorb_q_pack(q_nope, w_k, q_out, torch.ones(1, device="cuda"))
