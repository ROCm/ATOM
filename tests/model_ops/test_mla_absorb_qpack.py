# SPDX-License-Identifier: MIT
import pytest
import torch

if not torch.version.hip or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)

from aiter import dtypes, fused_qk_rope_concat_and_cache_mla
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

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
    q = torch.randn(tokens, HEADS, NOPE + PE, device="cuda", dtype=torch.bfloat16)
    q_nope, q_pe = q.split([NOPE, PE], dim=-1)
    q_nope = q_nope.transpose(0, 1)
    w_k = 0.05 * torch.randn(HEADS, LORA, NOPE, device="cuda", dtype=torch.bfloat16)
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
    ql_nope = batched_gemm_bf16(q_nope, w_k).transpose(0, 1)
    _pack(ql_nope, q_pe, k_nope, k_pe, ref_q, ref_kv, slots, rope, scale)

    new_q, new_kv = buffers()
    stand_in = q_pe.new_empty(LORA).expand(tokens, HEADS, -1)
    _pack(stand_in, q_pe, k_nope, k_pe, new_q, new_kv, slots, rope, scale)
    mla_absorb_q_pack(q_nope, w_k, new_q, scale)

    torch.testing.assert_close(new_kv.view(torch.uint8), ref_kv.view(torch.uint8))
    torch.testing.assert_close(
        new_q[..., LORA:].view(torch.uint8), ref_q[..., LORA:].view(torch.uint8)
    )
    steps = (_ordinal(new_q[..., :LORA]) - _ordinal(ref_q[..., :LORA])).abs()
    assert steps.max().item() <= 1, f"nope half off by {steps.max().item()} e4m3 steps"
