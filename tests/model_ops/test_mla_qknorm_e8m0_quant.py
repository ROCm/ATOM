# SPDX-License-Identifier: MIT
"""MLA q_a/kv_a RMSNorm with E8M0 scales: the norms are within 1 bf16 ulp of a
torch rms_norm, and q and its scale are byte-identical to the consumer's
per_1x128_e8m0_quant of the kernel's own bf16 q norm."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import QuantType, dtypes

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.models.deepseek_v2 import _fuse_rmsnorm_quant


def _norm_quant(q_c, q_w, kv_c, kv_w, q_norm_out=False):
    return _fuse_rmsnorm_quant(
        q_c,
        q_w,
        1e-6,
        kv_c,
        kv_w,
        1e-6,
        dtype_quant=dtypes.fp8,
        group_size=128,
        quant_type=QuantType.per_1x128.value,
        output_unquantized_inp1=q_norm_out,
        transpose_scale=True,
        e8m0_scale=True,
    )


def _inputs(rows):
    g = torch.Generator(device="cuda").manual_seed(20261116 + rows)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    q_c, kv_c = (randn(rows, 1536) * 3).bfloat16(), (randn(rows, 512) * 3).bfloat16()
    return q_c, randn(1536).bfloat16(), kv_c, randn(512).bfloat16()


# Kimi-K3 q_lora 1536 / kv_lora 512; 4100 rows: the kernel's prefill path
@pytest.mark.parametrize("rows", [1, 15, 135, 1009, 4100])
@torch.inference_mode()
def test_mla_qknorm_e8m0_matches_bf16_norm_then_consumer_quant(rows):
    (q, s), q_norm, _, _ = _norm_quant(*_inputs(rows), q_norm_out=True)
    q_ref, s_ref = per_1x128_e8m0_quant(q_norm, dtypes.fp8, True)
    assert s.dtype == dtypes.fp8_e8m0 and s.shape == s_ref.shape
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def _ulps(a, b):
    # bf16 bits as integers ordered like the values (+-0 -> 0)
    a, b = (x.view(torch.int16).int() for x in (a, b))
    a, b = (torch.where(x < 0, -(x & 0x7FFF), x) for x in (a, b))
    return (a - b).abs().max().item()


def _torch_rms_norm(x, w):
    return F.rms_norm(x.float(), x.shape[-1:], w.float(), 1e-6).bfloat16()


# the norms against torch and against the unfused bf16 norm this path replaces
@pytest.mark.parametrize("rows", [1, 15, 135, 1009, 4100])
@torch.inference_mode()
def test_mla_qknorm_e8m0_norms_within_1ulp_of_torch_and_unfused_norm(rows):
    q_c, q_w, kv_c, kv_w = args = _inputs(rows)
    _, q_norm, kv, _ = _norm_quant(*args, q_norm_out=True)
    (q_unfused, _), _, kv_unfused, _ = _fuse_rmsnorm_quant(
        q_c, q_w, 1e-6, kv_c, kv_w, 1e-6, dtype_quant=torch.bfloat16
    )
    assert _ulps(q_norm, _torch_rms_norm(q_c, q_w)) <= 1
    assert _ulps(kv, _torch_rms_norm(kv_c, kv_w)) <= 1
    assert _ulps(q_norm, q_unfused) <= 1
    assert _ulps(kv, kv_unfused) <= 1


# Inductor fails on an auto_functionalized_v2 that returns an E8M0 scale
@torch.inference_mode()
def test_mla_qknorm_e8m0_under_torch_compile():
    def f(*args):
        (q, s), _, kv, _ = _norm_quant(*args)
        return q, s, kv

    args = _inputs(64)
    for out, ref in zip(torch.compile(f, fullgraph=True)(*args), f(*args)):
        assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
