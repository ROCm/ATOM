"""Offline numerical harness for the RCCL batched w4a8 (MXFP4) expert MLP.

Question it answers: does ``batched_w4a8_mlp`` correctly consume expert weights
in the SAME shuffled layout the model produces (aiter ``shuffle_weight`` +
``moe_shuffle_scale`` with is_guinterleave=True)? If yes, the garbage bug is not
in the GEMM's layout handling; if no, this pins it.

Recipe (mirrors Fp8/Mxfp4MoEMethod.process_weights_after_loading for per_1x32):
  1. build a plain bf16 gate/up weight, INTERLEAVED as [gate0,up0,gate1,up1,...]
     rows (that's what is_guinterleave encodes), and a plain down weight.
  2. quantize to MXFP4 codes + e8m0 1x32 scales (per_1x32_f4_quant).
  3. apply aiter shuffle_weight(is_guinterleave=True) + moe_shuffle_scale to get
     the exact tensors the model stores.
  4. run batched_w4a8_mlp on the shuffled tensors.
  5. reference: dequant the PLAIN fp4 codes to bf16, do the MLP in bf16.
  6. compare.

Run: python tests/w4a8_real_layout_harness.py
"""

import torch

torch.manual_seed(0)
dev = "cuda"

from atom.model_ops.fused_moe.flydsl_kernels import fp4_utils
from atom.model_ops.fused_moe.rccl_batched_experts import batched_w4a8_mlp
from aiter.ops.shuffle import shuffle_weight, moe_shuffle_scale


def quant_fp4(w_bf):
    """w_bf [N, K] bf16 -> (codes uint8 [N, K/2], scale e8m0 [N, K/32], deq f32 [N,K])."""
    N, K = w_bf.shape
    codes, scale, _ = fp4_utils.per_1x32_f4_quant(w_bf.contiguous())
    # Faithful dequant: unpack fp4 codes -> values (mxfp4_to_f32 unpacks the
    # 2-per-byte packing, returning [N, K]) and multiply by the per-1x32 e8m0
    # block scale. This is exactly what the kernel effectively computes.
    vals = fp4_utils.mxfp4_to_f32(codes)  # [N, K]
    sf = fp4_utils.e8m0_to_f32(scale)  # [N, K/32]
    deq = (vals.reshape(N, K // 32, 32) * sf.reshape(N, K // 32, 1)).reshape(N, K)
    return codes, scale, deq


def main():
    E = 2
    H = 512  # hidden (K for gate_up), multiple of 256
    I = 256  # inter_dim
    C = 128  # tokens per expert
    twoI = 2 * I

    # ---- build plain weights ----
    # gate/up INTERLEAVED rows: row 2i = gate_i, row 2i+1 = up_i  -> [E, 2I, H]
    gate = torch.randn(E, I, H, device=dev, dtype=torch.bfloat16) * 0.05
    up = torch.randn(E, I, H, device=dev, dtype=torch.bfloat16) * 0.05
    w13_bf = torch.empty(E, twoI, H, device=dev, dtype=torch.bfloat16)
    w13_bf[:, 0::2, :] = gate
    w13_bf[:, 1::2, :] = up
    w2_bf = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.05

    x = torch.randn(E, C, H, device=dev, dtype=torch.bfloat16) * 0.1

    # ---- quantize to fp4 + build plain codes/scales/deq per expert ----
    w13_codes, w13_scale, w13_deq = [], [], []
    w2_codes, w2_scale, w2_deq = [], [], []
    for e in range(E):
        c, s, d = quant_fp4(w13_bf[e])
        w13_codes.append(c)
        w13_scale.append(s)
        w13_deq.append(d)
        c, s, d = quant_fp4(w2_bf[e])
        w2_codes.append(c)
        w2_scale.append(s)
        w2_deq.append(d)
    w13_codes = torch.stack(w13_codes)
    w13_scale = torch.stack(w13_scale)
    w13_deq = torch.stack(w13_deq)
    w2_codes = torch.stack(w2_codes)
    w2_scale = torch.stack(w2_scale)
    w2_deq = torch.stack(w2_deq)

    # ---- reference: bf16 MLP from dequantized PLAIN codes ----
    gate_up_ref = torch.einsum(
        "ech,enh->ecn", x.float(), w13_deq
    )  # [E,C,2I] interleaved
    g = gate_up_ref[..., 0::2]
    u = gate_up_ref[..., 1::2]
    act_ref = torch.nn.functional.silu(g) * u  # [E,C,I]
    out_ref = torch.einsum("eci,ehi->ech", act_ref, w2_deq)  # [E,C,H]

    # ---- model layout: aiter shuffle_weight + moe_shuffle_scale (is_guinterleave=True) ----
    import torch.nn as nn

    w13_p = nn.Parameter(
        w13_codes.view(torch.float4_e2m1fn_x2)
        if hasattr(torch, "float4_e2m1fn_x2")
        else w13_codes
    )
    w2_p = nn.Parameter(
        w2_codes.view(torch.float4_e2m1fn_x2)
        if hasattr(torch, "float4_e2m1fn_x2")
        else w2_codes
    )
    # BATCHED layout: both w13 and w2 shuffled with gate_up=False (the sweep
    # below found this is what the FlyDSL batched kernel wants).
    w13_shuf = shuffle_weight(w13_p.data, is_guinterleave=True, gate_up=False)
    w2_shuf = shuffle_weight(w2_p.data, is_guinterleave=True, gate_up=False)
    # scales: flatten to 2D [E*N, K/32] then moe_shuffle_scale (gate_up=False)
    w13_s2d = w13_scale.reshape(-1, w13_scale.shape[-1])
    w2_s2d = w2_scale.reshape(-1, w2_scale.shape[-1])
    w13_scale_shuf = moe_shuffle_scale(w13_s2d, E, is_guinterleave=True, gate_up=False)
    w2_scale_shuf = moe_shuffle_scale(w2_s2d, E, is_guinterleave=True, gate_up=False)

    # ---- run the production batched MLP on the model-layout tensors ----
    out = batched_w4a8_mlp(
        x, w13_shuf, w13_scale_shuf, w2_shuf, w2_scale_shuf, activation=None
    )
    rel = (out.float() - out_ref).norm() / out_ref.norm()
    print(
        f"[full MLP] batched_w4a8_mlp (aiter-shuffled layout) vs ref: rel={rel:.4f}  "
        f"({'PASS' if rel < 0.15 else 'FAIL'})"
    )

    # ---- FULL batched_expert_compute (resort + GEMM + unsort) on dispatched rows ----
    # This is the exact production function. Build R dispatched rows (topk==1),
    # each tagged with a GLOBAL expert id, some experts under-full (padding),
    # then compare against a per-row dequant-MLP reference.
    from atom.model_ops.fused_moe.rccl_batched_experts import batched_expert_compute

    ep_rank = 0
    R = 40
    # random rows, each routed to one of this rank's E local experts (global ids
    # for ep_rank=0 are 0..E-1)
    disp_ids = torch.randint(0, E, (R, 1), device=dev, dtype=torch.int32)
    disp_a1 = torch.randn(R, H, device=dev, dtype=torch.bfloat16) * 0.1

    fused = batched_expert_compute(
        dispatch_a1=disp_a1,
        dispatch_ids=disp_ids,
        w13=w13_shuf,
        w13_scale=w13_scale_shuf,
        w2=w2_shuf,
        w2_scale=w2_scale_shuf,
        activation=None,
        local_num_experts=E,
        capacity=None,
        ep_rank=ep_rank,
    )  # [R, H]

    # reference per row: dequant-MLP of that row's expert
    ref_rows = torch.empty(R, H, device=dev, dtype=torch.float32)
    for r in range(R):
        e = int(disp_ids[r, 0])
        gu = disp_a1[r].float() @ w13_deq[e].t()  # [2I]
        g = gu[0::2]
        u = gu[1::2]
        act = torch.nn.functional.silu(g) * u  # [I]
        ref_rows[r] = act @ w2_deq[e].t()  # [H]
    rel2 = (fused.float() - ref_rows).norm() / ref_rows.norm()
    print(
        f"[full compute] batched_expert_compute vs per-row ref: rel={rel2:.4f}  "
        f"({'PASS' if rel2 < 0.15 else 'FAIL'})"
    )

    # ---- ISOLATED single gate_up GEMM: sweep weight/scale shuffle variants ----
    # Reference for gate_up alone: x @ w13_deq^T  (interleaved 2I output)
    from atom.model_ops.fused_moe.flydsl_batched_gemm import batched_mxfp8_mxfp4_gemm
    from atom.model_ops.fused_moe.rccl_batched_experts import _mxfp8_quant_batched
    from atom.model_ops.fused_moe.flydsl_kernels import fp4_utils as fu

    gu_ref = torch.einsum("ech,enh->ecn", x.float(), w13_deq)  # [E,C,2I]
    aq, as_ = _mxfp8_quant_batched(x)  # A-side MXFP8 (already shuffled inside)

    def try_variant(name, w, ws):
        try:
            o = batched_mxfp8_mxfp4_gemm(aq, w, as_, ws)
            r = (o.float() - gu_ref).norm() / gu_ref.norm()
            print(
                f"  [gate_up] {name:42s} rel={r:.4f} {'<-- PASS' if r < 0.15 else ''}"
            )
        except Exception as e:
            print(f"  [gate_up] {name:42s} ERR {type(e).__name__}: {str(e)[:60]}")

    N = 2 * I
    Kdiv32 = H // 32
    # weight variants
    w_aiter = w13_shuf
    w_raw = w13_p.data
    w_fly = fu.shuffle_weight_w4(w13_p.data.view(torch.uint8), 16, True, True).view(
        torch.float4_e2m1fn_x2
    )
    # scale variants (all as [E, N, K/32])
    s_aiter = w13_scale_shuf.reshape(E, N, Kdiv32)
    s_raw = w13_scale  # [E, N, K/32] unshuffled
    s_fly = fu.shuffle_scale_w4(w13_scale.reshape(-1, Kdiv32), E, True).reshape(
        E, N, Kdiv32
    )
    print("  --- weight/scale shuffle sweep (gate_up only) ---")
    try_variant("w=aiter  s=aiter(model)", w_aiter, s_aiter)
    try_variant("w=aiter  s=raw", w_aiter, s_raw)
    try_variant("w=aiter  s=fly", w_aiter, s_fly)
    try_variant("w=fly    s=fly", w_fly, s_fly)
    try_variant("w=fly    s=raw", w_fly, s_raw)
    try_variant("w=raw    s=raw", w_raw, s_raw)


if __name__ == "__main__":
    main()
