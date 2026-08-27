"""Check aiter's block-FP8 GEMM against a torch dequant reference on real GLM weights.

If these agree, the reference run can use ATOM's own kernels instead of the
gfx950-broken `finegrained-fp8` hub kernel -- which both makes it fast and proves
the aiter path works on GLM-5.3-Flash shapes.
"""

from __future__ import annotations

import json

import aiter
import torch
import torch.nn.functional as F
from aiter import QuantType, dtypes, get_hip_quant
from safetensors import safe_open

MP = "/models/GLM-5.3-Flash"


def load(key: str, device: str = "cuda"):
    with open(f"{MP}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    with safe_open(f"{MP}/{wm[key]}", framework="pt", device=device) as f:
        return f.get_tensor(key)


def torch_dequant_linear(x, w_fp8, w_scale, block=(128, 128)):
    bn, bk = block
    n, k = w_fp8.shape
    s = w_scale.to(torch.float32)
    s = s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)[:n, :k]
    w = (w_fp8.to(torch.float32) * s).to(torch.bfloat16)
    return F.linear(x, w)


def main() -> None:
    dev = "cuda"
    torch.manual_seed(0)

    # A real block-FP8 weight: layer-0 dense MLP gate_proj, [12288, 4096].
    key = "model.language_model.layers.0.mlp.gate_proj.weight"
    w = load(key, dev)
    ws = load(key + "_scale_inv", dev)
    print(f"weight {tuple(w.shape)} {w.dtype} | scale {tuple(ws.shape)} {ws.dtype}")
    print(f"aiter fp8 dtype: {dtypes.fp8}")

    quant_func = get_hip_quant(QuantType.per_1x128)

    for m in (1, 21, 256):
        x = torch.randn(m, w.shape[1], device=dev, dtype=torch.bfloat16) * 0.05

        ref = torch_dequant_linear(x, w, ws)

        xq, xs = quant_func(x, quant_dtype=dtypes.fp8)
        got = aiter.gemm_a8w8_blockscale(xq, w, xs, ws, dtype=torch.bfloat16)

        diff = (got.float() - ref.float()).abs()
        rel = diff.max() / ref.float().abs().max().clamp(min=1e-6)
        cos = F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
        print(
            f"  M={m:>4}: x_scale={tuple(xs.shape)} out={tuple(got.shape)} "
            f"max_abs={diff.max():.4f} rel={rel:.5f} cos={cos:.6f}"
        )

    print(
        "\nNote: exact equality is not expected -- aiter quantises the activation to "
        "FP8 (which is what the model was trained for), the torch reference keeps it "
        "in BF16. High cosine similarity is the pass condition."
    )


if __name__ == "__main__":
    main()
