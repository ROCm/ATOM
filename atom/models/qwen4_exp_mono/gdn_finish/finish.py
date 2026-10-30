"""Same-stream publication and gated norm/PTPC, with independent CTA owners."""

import torch
import triton
import triton.language as tl

from atom.model_ops.qwen4_exp.ops.gated import _sigmoid_rmsnorm_quant_kernel

from .device import publish_device


@triton.jit
def finish_kernel(
    X,
    Z,
    W,
    OUT,
    SCALE,
    BK,
    BG,
    CURSOR,
    SLOTS,
    CU,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    ZS0: tl.constexpr,
    ZS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    TOKENS: tl.constexpr,
    HV: tl.constexpr,
    CAP: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < TOKENS:
        _sigmoid_rmsnorm_quant_kernel(
            X,
            Z,
            W,
            OUT,
            SCALE,
            XS0,
            XS1,
            ZS0,
            ZS1,
            HV,
            128,
            EPS,
            FP8_MAX,
            triton.next_power_of_2(HV),
            128,
        )
    else:
        at = pid - TOKENS
        publish_device(
            BK,
            BG,
            CURSOR,
            SLOTS,
            CU,
            4,
            CAP,
            128,
            128,
            4,
            KS0,
            KS2,
            KS1,
            GS0,
            GS2,
            GS1,
            False,
            at // HV,
            at % HV,
        )


def finish(x, z, weight, eps, buf_k, buf_g, cursor, slots, cu):
    tokens, heads, dim = x.shape
    assert dim == 128 and heads == 48 and tokens == 4 * (cu.numel() - 1)
    assert x.dtype == z.dtype == torch.bfloat16 and x.shape == z.shape
    assert buf_k.shape[-1] == 128 and buf_k.shape[1] == heads
    output = torch.empty(
        (tokens, heads * dim), device=x.device, dtype=torch.float8_e4m3fn
    )
    scale = torch.empty((tokens, 1), device=x.device, dtype=torch.float32)
    binary = finish_kernel[(tokens + heads * (cu.numel() - 1),)](
        x,
        z,
        weight,
        output,
        scale,
        buf_k,
        buf_g,
        cursor,
        slots,
        cu,
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        buf_k.stride(0),
        buf_k.stride(1),
        buf_k.stride(2),
        buf_g.stride(0),
        buf_g.stride(1),
        buf_g.stride(2),
        tokens,
        heads,
        buf_k.shape[2],
        eps,
        torch.finfo(output.dtype).max,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, scale, binary
