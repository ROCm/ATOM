"""GPU check of the overlay's per-tensor FP8 quant helpers against a torch reference."""
import importlib.util
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a ROCm GPU", allow_module_level=True)
pytest.importorskip("vllm")
spec = importlib.util.spec_from_file_location(
    "overlay", Path(__file__).with_name("aiter_flash_attn.py")
)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
H, NOPE, ROPE, VD = 12, 128, 64, 128


def check(x, x8, descale):
    x = x.float()
    amax = x.abs().max() if x.numel() else torch.tensor(0.0, device=x.device)
    expected = amax / 448 if amax > 0 else torch.tensor(1e-6, device=x.device)
    torch.testing.assert_close(descale.float().reshape(()), expected, rtol=1e-6, atol=0)
    assert x8.shape == x.shape
    ref = (x / expected).clamp(-448, 448)
    # Within one E4M3 step (3 mantissa bits) of the exact scaled value.
    assert ((x8.float() - ref).abs() <= ref.abs() / 8 + 2**-9).all()


@pytest.mark.parametrize("tokens", [0, 1, 7, 4096, 20000, 30000])
def test_kv_quant(tokens):
    torch.manual_seed(tokens)
    kv = torch.randn(tokens, H, NOPE + VD, device="cuda", dtype=torch.bfloat16)
    k_nope, v = kv.split([NOPE, VD], dim=-1)
    k_pe = torch.randn(tokens, 1, ROPE, device="cuda", dtype=torch.bfloat16) * 4
    k = torch.cat((k_nope, k_pe.expand(-1, H, -1)), dim=-1)
    k8, v8, ks, vs = overlay._quant_fp8_kv_per_tensor(k, v)
    check(k, k8, ks)
    check(v, v8, vs)


def test_kv_quant_zero():
    k = torch.zeros(5000, H, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(5000, H, VD, device="cuda", dtype=torch.bfloat16)
    k8, v8, ks, vs = overlay._quant_fp8_kv_per_tensor(k, v)
    assert ks.item() == pytest.approx(1e-6) and vs.item() == pytest.approx(1e-6)
    assert not k8.float().any() and not v8.float().any()


@pytest.mark.parametrize("tokens", [1, 3, 8191])
def test_q_quant_strided(tokens):
    q = torch.randn(tokens, H + 1, NOPE + ROPE, device="cuda", dtype=torch.bfloat16)[:, 1:]
    q8, qs = overlay._quant_fp8_per_tensor(q)
    check(q, q8, qs)
