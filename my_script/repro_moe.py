import os, torch
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4

dev = "cuda"
E, model_dim, inter_dim, topk = 896, 7168, 3072, 16
M = int(os.environ.get("REPRO_M", "639"))
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def rand_u8(*shape):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev)


# MXFP4 packed weights directly as random uint8 (values are irrelevant for an
# index/shape-dependent OOB). fp4x2 packs 2 fp4 per byte -> last dim K//2.
w1_q = rand_u8(E, inter_dim * 2, model_dim // 2).view(dtypes.fp4x2)
w2_q = rand_u8(E, model_dim, inter_dim // 2).view(dtypes.fp4x2)
# per-1x32 e8m0 block scales are 2D [E*N, K//32], dtype float8_e8m0fnu
# (matches per_1x32_f4_quant output).
w1_scale = rand_u8(E * inter_dim * 2, model_dim // 32).view(torch.float8_e8m0fnu)
w2_scale = rand_u8(E * model_dim, inter_dim // 32).view(torch.float8_e8m0fnu)

w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True).view(torch.uint8)
w2_shuf = shuffle_weight_a16w4(w2_q, 16, False)
w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False).view(torch.uint8)

inp = (torch.randn(M, model_dim, device=dev, dtype=torch.bfloat16)) / 4

# routing mode: spread (each token random experts) or concentrated (all tokens
# -> same experts, like the all-zero-token warmup batch).
if os.environ.get("REPRO_ROUTING", "spread") == "concentrated":
    row = torch.arange(topk, device=dev, dtype=torch.int32)
    topk_ids = row.unsqueeze(0).expand(M, topk).contiguous()
else:
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:topk] for _ in range(M)]
    ).to(torch.int32)
topk_weight = torch.softmax(
    torch.randn(M, topk, device=dev, dtype=torch.float32), dim=-1
).to(torch.bfloat16)

contig = os.environ.get("AITER_GROUPED_DEEPGEMM_CONTIGUOUS")
print("M=%d E=%d topk=%d contiguous=%s" % (M, E, topk, contig), flush=True)
out = fused_moe(
    inp, w1_shuf, w2_shuf, topk_weight, topk_ids,
    activation=ActivationType.Situv2, quant_type=QuantType.per_1x32,
    w1_scale=w1_scale_shuf, w2_scale=w2_scale_shuf,
    beta=4.0, linear_beta=25.0,
)
torch.cuda.synchronize()
print("NO CRASH out.shape=%s finite=%s" % (tuple(out.shape), torch.isfinite(out).all().item()), flush=True)
