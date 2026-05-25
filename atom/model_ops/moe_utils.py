from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import swizzle_scales # same for a4 and a16
from aiter.ops.triton.moe.moe_routing.routing import ExptData
import torch

import logging
logger = logging.getLogger("atom") # debug

def check_and_swizzle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = swizzle_scales(scale)
        return scale, "CDNA4_SCALE"
    else:
        return scale, None


def quantize(x, dtype):
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "fp8":
        scale = x.abs().max().item() / 448.0
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x = x.to(fp8e4_dtype)
        return x, scale
    elif dtype == "mx8":
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x, scale = downcast_to_mxfp(x, fp8e4_dtype, axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def compute_expt_data(hist, n_expts_tot, n_gates, block_m):
    # Similar in form to aiter compute_expt_data_torch, but isolated for ATOM
    # offset for each experts
    device = hist.device
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device), token_offs_raw))
    token_offs_raw = token_offs_raw.int()
    # maximum number of tiles for all values of `block_m` considered
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // block_m)
    # fill up tile offset/infos for each block
    n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
    token_offs_pad = torch.cumsum(n_tiles, dim=0)
    token_offs_pad = torch.cat((torch.zeros(1, device=device), token_offs_pad))
    token_offs_pad = token_offs_pad.int()

    # # compute data required to drive ragged batch matmul
    # #block_pid_map = -torch.ones(max_n_tiles, device=device) # this is the issue
    # for e in range(n_expts_tot):
    #     offset = token_offs_pad[e]
    #     for b in range(n_tiles[e]):
    #         block_pid_map[offset + b] = (b << 16) + e
    # block_pid_map = block_pid_map.int()
    positions = torch.arange(max_n_tiles, dtype=torch.int32, device=device)
    expert_idx = torch.searchsorted(token_offs_pad[1:], positions, right=True)
    block_id = positions - token_offs_pad[expert_idx.long()]
    val = (block_id << 16) | expert_idx.to(torch.int32)
    mask = positions < token_offs_pad[-1]
    block_pid_map = torch.where(mask, val, torch.full_like(val, -1)).int()
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)