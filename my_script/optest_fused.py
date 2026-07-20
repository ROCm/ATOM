import torch
from atom.models.kimi_k3_fused import (
    situ_and_mul, _situ_and_mul_torch,
    rmsnorm_gated, _rmsnorm_gated_torch,
)

torch.manual_seed(0)
dev = "cuda"


def rep(name, a, b):
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    rel = err / (b.abs().max().item() + 1e-9)
    print(f"{name}: max_abs_err={err:.4g} rel={rel:.4g} finite={torch.isfinite(a).all().item()}")


# situ_and_mul: dense/shared-expert act, D up to ~7168, beta=4 linear_beta=25
for M, D in [(512, 3072), (1, 7168), (633, 4096)]:
    x = torch.randn(M, 2 * D, device=dev, dtype=torch.bfloat16)
    rep(f"situ_and_mul M={M} D={D}", situ_and_mul(x, 4.0, 25.0),
        _situ_and_mul_torch(x, 4.0, 25.0))
# no linear_beta branch
x = torch.randn(256, 2 * 3072, device=dev, dtype=torch.bfloat16)
rep("situ_and_mul no-linear", situ_and_mul(x, 1.0, None), _situ_and_mul_torch(x, 1.0, None))

# rmsnorm_gated: KDA o_norm over head_dim
for M, H in [(512 * 32, 128), (69, 128), (1 * 96, 192)]:
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(H, device=dev, dtype=torch.bfloat16)
    g = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    rep(f"rmsnorm_gated M={M} H={H}", rmsnorm_gated(x, w, g, 1e-6),
        _rmsnorm_gated_torch(x, w, g, 1e-6))

# apply_attn_res: [T,H] prefix + [T,B,H] block residual, B=attn_res_block_size=12
from atom.models.kimi_k3_fused import apply_attn_res, _apply_attn_res_torch
for T, B, H in [(512, 12, 7168), (1, 12, 7168), (633, 12, 7168)]:
    ps = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    br = torch.randn(T, B, H, device=dev, dtype=torch.bfloat16)
    pw = torch.randn(H, device=dev, dtype=torch.bfloat16)
    nw = torch.randn(H, device=dev, dtype=torch.bfloat16)
    rep(f"apply_attn_res T={T} B={B} H={H}",
        apply_attn_res(ps, br, pw, nw, 1e-6),
        _apply_attn_res_torch(ps, br, pw, nw, 1e-6))

print("FUSED OP TESTS DONE")
