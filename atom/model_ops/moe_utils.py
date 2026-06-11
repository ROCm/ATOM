from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import swizzle_scales  # same for a4 and a16
import torch

import logging

logger = logging.getLogger("atom")  # debug


def check_and_swizzle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = swizzle_scales(scale)
        return scale, "CDNA4_SCALE"
    else:
        return scale, None
