from aiter.ops.triton.moe.moe_op_gemm_a4w4 import swizzle_scales  # same for a4 and a16


def check_and_swizzle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = swizzle_scales(scale)
        return scale, "CDNA4_SCALE"
    else:
        return scale, None
