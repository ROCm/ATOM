import torch

torch.manual_seed(0)
dev = "cuda"


def report(name, a, b):
    a = a.float()
    b = b.float()
    err = (a - b).abs().max().item()
    rel = err / (b.abs().max().item() + 1e-9)
    fin = torch.isfinite(a).all().item()
    print(f"{name}: max_abs_err={err:.4g} rel={rel:.4g} finite={fin}")


# 1) rms_norm (native CK module_rmsnorm_quant) vs torch reference
from aiter.ops.rmsnorm import rms_norm

x = (torch.randn(512, 4096, device=dev, dtype=torch.bfloat16))
w = (torch.randn(4096, device=dev, dtype=torch.bfloat16))
eps = 1e-6
native = rms_norm(x, w, eps)
xf = x.float()
ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16).float() * w.float()
report("rms_norm native-vs-torch", native, ref)

# 2) biased_grouped_topk (native CK module_moe_asm) vs torch reference
from aiter.ops.topk import biased_grouped_topk, biased_grouped_topk_torch

T, E, topk = 512, 896, 16
gating = torch.randn(T, E, device=dev, dtype=torch.bfloat16)
bias = torch.randn(E, device=dev, dtype=torch.bfloat16)
# native CK (in-place output buffers)
n_tw = torch.empty(T, topk, device=dev, dtype=torch.float32)
n_ti = torch.empty(T, topk, device=dev, dtype=torch.int32)
biased_grouped_topk(gating, bias, n_tw, n_ti, 1, 1, True, 1.0)
# torch reference (returns)
r_tw, r_ti = biased_grouped_topk_torch(gating, bias, topk, True, 1, 1)
torch.cuda.synchronize()
# compare expert-id sets per token (order may differ) + weights
id_match = 0
for t in range(T):
    if set(n_ti[t].tolist()) == set(r_ti[t].tolist()):
        id_match += 1
print(f"biased_grouped_topk: expert-id set match {id_match}/{T}; weight max_abs_err="
      f"{(n_tw.sort(-1).values - r_tw.sort(-1).values).abs().max().item():.4g}")

print("OP TESTS DONE")
