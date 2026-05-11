# # edit as required but these are all the triton_kernels components used 
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import swizzle_scales
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

# #moe
# class FlexCtx:
#     lhs_data: InFlexData = InFlexData()
#     rhs_data: InFlexData = InFlexData()
#     out_data: OutFlexData = OutFlexData()
#     acc_data: InFlexData = InFlexData()

# class PrecisionConfig:
#     max_num_imprecise_acc: int | None = None
#     allow_tf32: bool = True
#     flex_ctx: FlexCtx = FlexCtx()
#     acc_scale: float = 1.0
#     flexpoint_saturate_inf: bool = False
#     report_quantization_err_fn: Callable | None = None
#     a_mx_scale: torch.Tensor | Tensor | None = None
#     a_microblock_size: int | None = None
#     b_mx_scale: torch.Tensor | Tensor | None = None
#     b_microblock_size: int | None = None
#     c_mx_scale: torch.Tensor | Tensor | None = None
#     c_microblock_size: int | None = None
#     c_value_pack_factor: int = 1
#     out_dtype: torch.dtype | None = None
#     enforce_bitwise_invariance: bool = False

# #both
# @dataclass(frozen=True)
# class BaseFlexData:
#     dtype: torch.dtype | None = None

#     def view(self, x: torch.Tensor):
#         if self.dtype is None:
#             return x
#         return x.view(self.dtype)

#     def reinterpret(self, x):
#         if self.dtype is None or x.dtype.itemsize > 1:
#             return x
#         return x.view(self.dtype)


# @dataclass(frozen=True)
# class InFlexData(BaseFlexData):
#     scale: torch.Tensor | None = None

#     @property
#     def is_per_batch(self):
#         return False if self.scale is None else len(self.scale) > 1


# @dataclass(frozen=True)
# class OutFlexData(BaseFlexData):
#     expected_scale: torch.Tensor | None = None
#     actual_scale: torch.Tensor | None = None
#     checksum_scale: torch.Tensor | None = None

#     @property
#     def is_per_batch(self):
#         return False if self.expected_scale is None else len(self.expected_scale) > 1

#     def __iter__(self):
#         yield self.expected_scale
#         yield self.actual_scale
#         yield self.checksum_scale

# #fused moe

# def wrap_torch_tensor(torch_tensor, dtype=None, shape=None, shape_max=None, layout=None):
#     if dtype is None:
#         dtype = torch_tensor.dtype
#     dtype = torch_dtype_to_dtype(dtype)
#     if shape is None:
#         shape = list(torch_tensor.shape)
#         if dtype == FP4:
#             shape[torch_tensor.stride().index(1)] *= (8 * torch_tensor.dtype.itemsize) // dtype.bitwidth
#     if shape_max is None:
#         shape_max = list(shape)
#     if layout is None:
#         # For a strided (dense) tensor we only track which dimension has unit stride.
#         # This is consistent with how we expand `shape` for packed sub-byte dtypes.
#         major_dim = torch_tensor.stride().index(1) if 1 in torch_tensor.stride() else -1
#         layout = StridedLayout(major_dim=major_dim - torch_tensor.ndim)
#     return Tensor(Storage(torch_tensor, layout), dtype=dtype, shape=shape, shape_max=shape_max)


# def convert_layout(tensor: Tensor, layout: Layout, **layout_transformation_kwargs):
#     shape = list(tensor.shape)
#     # convert `tensor` into canonical form
#     transformation = tensor.storage.layout.make_transformation(shape, tensor.dtype == FP4)
#     canonical_data = transformation.unswizzle_data(tensor.storage.data)
#     # convert canonical form to `layout`
#     transformation = layout.make_transformation(shape, tensor.dtype == FP4, **layout_transformation_kwargs)
#     # print("convert layout ", torch.cuda.memory_summary(0, abbreviated=True))
#     new_data = transformation.swizzle_data(canonical_data)
#     return Tensor(Storage(new_data, layout), shape=list(tensor.shape), dtype=tensor.dtype)

# # FP4 but thats not in the tensor file like it's meant to be

# @dataclass(frozen=True)
# class StridedLayout(Layout):

#     # NOTE: We only encode the (logical) major dimension; the full dimension order is
#     # derived from the tensor rank. This keeps the API minimal while still allowing
#     # "which dim is contiguous/packed" to be expressed.
#     #
#     # For a tensor of rank `R`, the derived order is:
#     #   base = list(reversed(range(R)))
#     #   swap base[0] with base[index(major_dim)]
#     #   order = base
#     #
#     # This matches the previous default `order=list(reversed(range(R)))` when
#     # `major_dim == R - 1`.
#     major_dim: int = -1

#     def __post_init__(self):
#         if not isinstance(self.major_dim, int):
#             raise TypeError(f"StridedLayout(major_dim=...) must be an int, got {type(self.major_dim)}")

#     def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
#         return StridedLayoutTransformation(shape, is_fp4, self.order(len(shape)))

#     @property
#     def name(self):
#         return "STRIDED"

#     def swizzle_block_shape(self, block_shape):
#         return block_shape

#     def order(self, rank: int) -> list[int]:
#         """
#         Returns the minor->major dimension order for a given tensor rank.

#         `self.major_dim` supports negative indexing (like Python).
#         """
#         if rank <= 0:
#             return []
#         if not (-rank <= self.major_dim < rank):
#             raise ValueError(f"Invalid StridedLayout.major_dim={self.major_dim} for rank={rank}")
#         major_dim = self.major_dim if self.major_dim >= 0 else self.major_dim + rank
#         base = list(reversed(range(rank)))
#         # Preserve the previous behavior: derive from canonical reversed order, then
#         # swap the requested major dimension into position 0.
#         idx = base.index(major_dim)
#         base[0], base[idx] = base[idx], base[0]
#         return base

# #entire gfx1250 file, should be 950 but i cannot find that for the life of me
# import math
# import torch
# from dataclasses import dataclass
# import triton
# import triton.language as tl
# from .base import Layout, LayoutTransformation

# NON_K_PRESHUFFLE_BLOCK_SIZE = 128


# @dataclass(frozen=True)
# class GFX1250MXScaleLayout(Layout):

#     @property
#     def name(self):
#         return "GFX1250_SCALE"

#     def make_transformation(self, shape: list[int], is_fp4: bool) -> LayoutTransformation:
#         return GFX1250MXScaleLayoutTransformation(shape, is_fp4)

#     def swizzle_block_shape(self, block_shape):
#         SCALE_K = block_shape[-2]
#         N = block_shape[-1]
#         return block_shape[:-2] + [N // NON_K_PRESHUFFLE_BLOCK_SIZE, SCALE_K * NON_K_PRESHUFFLE_BLOCK_SIZE]


# @dataclass(frozen=True)
# class GFX1250MXScaleLayoutTransformation(LayoutTransformation):

#     def __post_init__(self) -> None:
#         *leading_shape, K_SCALE, N = self.shape
#         B = math.prod(leading_shape)
#         ALIGN_K_SCALE = 4 if K_SCALE > 4 else K_SCALE
#         ALIGN_N = NON_K_PRESHUFFLE_BLOCK_SIZE
#         K_SCALE_pad = math.ceil(K_SCALE / ALIGN_K_SCALE) * ALIGN_K_SCALE
#         N_pad = math.ceil(N / ALIGN_N) * ALIGN_N
#         object.__setattr__(self, "leading_shape", leading_shape)
#         object.__setattr__(self, "B", B)
#         object.__setattr__(self, "ALIGN_K_SCALE", ALIGN_K_SCALE)
#         object.__setattr__(self, "ALIGN_N", ALIGN_N)
#         object.__setattr__(self, "K_SCALE_pad", K_SCALE_pad)
#         object.__setattr__(self, "N_pad", N_pad)
#         object.__setattr__(self, "K_SCALE", K_SCALE)
#         object.__setattr__(self, "N", N)

#     def swizzle_data(self, data):
#         data = torch.nn.functional.pad(data, (0, self.N_pad - self.N, 0, self.K_SCALE_pad - self.K_SCALE))
#         data = data.transpose(-1, -2)
#         data = data.view(-1, self.N_pad // self.ALIGN_N, 4, self.ALIGN_N // 4, self.K_SCALE_pad // self.ALIGN_K_SCALE,
#                          self.ALIGN_K_SCALE)
#         data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
#         data = data.reshape(self.B, self.N_pad // self.ALIGN_N, self.K_SCALE_pad * self.ALIGN_N)
#         return data.transpose(-1, -2)

#     def unswizzle_data(self, data):
#         data = data.transpose(-1, -2)
#         data = data.view(-1, self.N_pad // self.ALIGN_N, self.K_SCALE_pad // self.ALIGN_K_SCALE, self.ALIGN_N // 4, 4,
#                          self.ALIGN_K_SCALE)
#         data = data.permute(0, 1, 4, 3, 2, 5)
#         data = data.reshape(*self.leading_shape, self.N_pad, self.K_SCALE_pad)
#         return data.transpose(-1, -2)[..., :self.K_SCALE, :self.N].contiguous()

#     def swizzle_block_shape(self, block_shape):
#         SCALE_K = block_shape[-2]
#         N = block_shape[-1]
#         return block_shape[:-2] + [N // self.ALIGN_N, SCALE_K * self.ALIGN_N]


# @triton.jit
# def unswizzle_mx_scale_gfx1250(x, BLOCK_N: tl.constexpr, MX_SCALE_BLOCK_K: tl.constexpr,
#                                N_PRESHUFFLE_FACTOR: tl.constexpr = NON_K_PRESHUFFLE_BLOCK_SIZE):
#     SCALE_KWIDTH: tl.constexpr = 4 if MX_SCALE_BLOCK_K >= 4 else MX_SCALE_BLOCK_K
#     x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // SCALE_KWIDTH, N_PRESHUFFLE_FACTOR // 4, 4,
#                   SCALE_KWIDTH)
#     x = x.permute(0, 3, 2, 1, 4)
#     x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
#     return x

# # routing file bc it got deleted from main repo
# import torch
# import triton
# from dataclasses import dataclass, field
# from .routing_details._routing_compute import _combined_routing_compute
# from .routing_details._routing_compute import _combined_routing_memset
# from .routing_details._routing_compute import _routing_clear_bitmatrix
# from .routing_details._expt_data import _expt_data_memset
# from .routing_details._expt_data import _expt_data_compute
# from .target_info import is_hip


# @dataclass
# class GatherIndx:
#     """
#     Indices for an operation that performs:
#     Y = X[src_idx, :]
#     """
#     # array such that `dst_idx[src_idx] = arange(0, N)`
#     src_indx: torch.Tensor
#     dst_indx: torch.Tensor


# @dataclass
# class ScatterIndx:
#     """
#     Indices for an operation that performs:
#     Y[dst_idx, :] = X
#     """
#     # array such that `dst_idx[src_idx] = arange(0, N)`
#     src_indx: torch.Tensor
#     dst_indx: torch.Tensor


# @dataclass
# class ExptData:
#     # hist[i] is the number of tokens routed to expert i
#     hist: torch.Tensor
#     # token_offs_raw[i] is the offset of the first token routed
#     # to expert i in an expert-sorted array
#     token_offs_raw: torch.Tensor
#     # token_offs_pad[block][i] is the offset of the first token routed
#     # to expert i in an expert-sorted array, assuming histogram
#     # rounded to the next multiple of `block`
#     token_offs_pad: dict[int, torch.Tensor]
#     # block_id_map[block] contain one value for each `pid`` launched by
#     # the matrix multiplication kernel launched with BLOCK_M=block:
#     # - the value is -1 if the `pid` has no work to do
#     # - otherwise, the value is two int16 (packed as an int32) that
#     #   correspond respectively to (1) the expert assigned to
#     #   the tokens processed by this pid; (2) the block assigned to the
#     #   tokens processed by this pid (think `pid_m` in a regular matmul)
#     # see `test_routing.py` for a reference implementation and more details
#     block_pid_map: dict[int, torch.Tensor]

#     def __post_init__(self):
#         if self.hist is not None:
#             assert self.hist.dtype == torch.int32
#         if self.token_offs_raw is not None:
#             assert self.token_offs_raw.dtype == torch.int32
#         if self.token_offs_pad is not None:
#             for v in self.token_offs_pad.values():
#                 assert v.dtype == torch.int32
#         if self.block_pid_map is not None:
#             for v in self.block_pid_map.values():
#                 assert v.dtype == torch.int32


# @dataclass
# class RoutingData:
#     gate_scal: torch.Tensor = field()
#     expt_hist: torch.Tensor = field()
#     n_expts_tot: int = field()
#     n_expts_act: int = field()
#     expt_data: ExptData = None

#     # Used to make perf annotation cleaner: when we use expert sharding, we can
#     # use this to tell the "expected" number of local tokens per expert, because
#     # the actual number can vary per each input.
#     expected_tokens_per_expt: int = field(default=None)

#     def n_blocks(self, n_rows, block_m):
#         if n_rows <= self.n_expts_tot:
#             return n_rows
#         else:
#             return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1


# # --------------------------
# # sort tokens by expert
# # --------------------------


# class SortTokens(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, expt_scal, expt_indx, n_expts_tot, bitmatrix):
#         HIST_BLOCK_M = 32
#         INDX_OFFS_BLOCK_M = 512
#         MEMSET_BLOCK = 1024
#         cdiv = triton.cdiv

#         device = expt_scal.device
#         dtype = expt_scal.dtype
#         n_tokens_raw, _ = bitmatrix.shape
#         n_tokens_pad, n_expts_act = expt_scal.shape
#         n_gates_pad = n_tokens_pad * n_expts_act

#         hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
#         hist = hist[:n_expts_tot]
#         assert hist.dtype == torch.int32
#         # scratchpad
#         expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
#         combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
#         # output
#         topk_indx = combined_indx[:n_gates_pad]
#         gate_indx = combined_indx[n_gates_pad:]
#         gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)

#         token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1a, blocks2a, MEMSET_BLOCK_A, HIST2_BLOCK_M, block_m_log2_start, block_m_num = _compute_expt_data_internal(
#             hist, n_expts_tot, n_gates_pad)

#         blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
#         blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)

#         _combined_routing_memset[(blocks1a + blocks1b, )](
#             combined_indx, n_gates_pad * 2, -1, MEMSET_BLOCK, hist,  #
#             expt_offs, hist.shape[0], n_expts_tot, partial_hist,  # inputs
#             partial_hist.shape[0], partial_hist.stride(0), partial_hist.stride(1),  # outputs
#             token_offs_combined, token_offs_combined.stride(0),  #
#             blocks1a, block_pid_map,  #
#             block_m_log2_start, SIZES=block_m_num, BLOCK_A=MEMSET_BLOCK_A,  # optimization parameters
#             BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
#         )

#         indx_offs = partial_hist

#         _combined_routing_compute[(blocks2a + blocks2b, )](
#             topk_indx, gate_indx, gate_scal,  # outputs
#             expt_scal, expt_indx, indx_offs, indx_offs.stride(0), indx_offs.stride(1),  # inputs
#             expt_offs, n_tokens_raw,  # input shape
#             HIST_BLOCK_M, n_expts_act,  # constants
#             hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
#             block_m_log2_start, block_m_num, HIST2_BLOCK_M, blocks2a,  # etc.
#         )

#         ctx.n_tokens_raw = n_tokens_raw
#         ctx.n_tokens_pad = n_tokens_pad
#         ctx.n_expts_act = n_expts_act
#         ctx.save_for_backward(gate_indx)
#         return hist, topk_indx, gate_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map

#     @staticmethod
#     def backward(ctx, _0, _1, _2, dgate_scal, _3, _4, _5):
#         (gate_indx, ) = ctx.saved_tensors
#         dgate_scal = dgate_scal[gate_indx]
#         dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
#         return dgate_scal, None, None, None


# def sort_tokens(expt_scal, expt_indx, n_expts_tot, bitmatrix):
#     return SortTokens.apply(expt_scal, expt_indx, n_expts_tot, bitmatrix)


# # --------------------------
# # prune routing
# # --------------------------


# class PruneRouting(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep):
#         from .compaction import compaction
#         n_tokens_pad = expt_scal.shape[0]
#         assert n_expts_tot % simulated_ep == 0
#         _routing_clear_bitmatrix[(n_tokens_pad, )](
#             bitmatrix.storage.data,
#             bitmatrix.storage.data.stride(0),
#             bitmatrix.storage.data.stride(1),
#             bitmatrix.storage.data.shape[1],
#             n_expts_tot // simulated_ep,
#             BLOCK_N=512,
#         )
#         # perform compaction to update expt_scal / expt_indx
#         expt_scal, expt_indx = compaction(expt_scal, expt_indx, bitmatrix)
#         n_expts_tot = n_expts_tot // simulated_ep
#         bitmatrix.shape[-1] = n_expts_tot
#         return expt_scal, expt_indx, bitmatrix


# def prune_routing(expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep):
#     return PruneRouting.apply(expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep)


# # --------------------------
# # expt_data
# # --------------------------


# def log2_power_of_two(x):
#     assert x > 0 and (x & (x - 1)) == 0, "x must be a power of two"
#     return x.bit_length() - 1


# block_m_log2_start = 4


# def _compute_expt_data_internal(expt_hist, n_expts_tot, n_gates):

#     MEMSET_BLOCK = 512
#     HIST2_BLOCK_M = 512
#     device = expt_hist.device
#     n_expts_tot = n_expts_tot
#     cdiv = triton.cdiv
#     # block_ms are all powers-of-two between 16 and 128 (inclusive)
#     block_m_log2_end = 9 if is_hip() else 8
#     block_m_num = block_m_log2_end - block_m_log2_start
#     if n_gates <= n_expts_tot:
#         max_n_tiles = n_gates
#     else:
#         max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // 2**block_m_log2_start)
#     # allocate memory
#     pad = lambda x: cdiv(x, MEMSET_BLOCK) * MEMSET_BLOCK
#     dtype = torch.int32

#     token_offs_combined = torch.empty((block_m_num + 1, pad(n_expts_tot + 1)), dtype=dtype, device=device)

#     token_offs_raw = token_offs_combined[0][:n_expts_tot + 1]
#     token_offs_pad = token_offs_combined[1:]

#     block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=dtype, device=device)
#     memset_grid = torch.numel(block_pid_map) // MEMSET_BLOCK  # exact division
#     # compute outputs
#     token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
#     block_pid_map = block_pid_map[:, :max_n_tiles]

#     blocks1 = memset_grid + block_m_num + 1
#     blocks2 = n_expts_tot * block_m_num
#     return token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1, blocks2, MEMSET_BLOCK, HIST2_BLOCK_M, block_m_log2_start, block_m_num


# def _unpack_into_dict(x):

#     block_m_log2_end = block_m_log2_start + x.shape[0]
#     x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
#     return x


# def compute_expt_data(expt_hist, n_expts_tot, n_gates):

#     if expt_hist is None:
#         return ExptData(None, None, None, None)

#     # this just computes the kernel arguments:
#     token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1, blocks2, MEMSET_BLOCK, HIST2_BLOCK_M, block_m_log2_start, block_m_num = _compute_expt_data_internal(
#         expt_hist, n_expts_tot, n_gates)

#     _expt_data_memset[(blocks1, )](
#         expt_hist, n_expts_tot,  #
#         token_offs_combined, token_offs_combined.stride(0),  #
#         block_pid_map,  #
#         block_m_log2_start, SIZES=block_m_num, BLOCK=MEMSET_BLOCK,  # optimization parameters
#         num_warps=4)
#     _expt_data_compute[(blocks2, )](
#         expt_hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
#         block_m_log2_start, SIZES=block_m_num, BLOCK=HIST2_BLOCK_M,  # optimization parameters
#         num_warps=4)

#     token_offs_pad = _unpack_into_dict(token_offs_pad)
#     block_pid_map = _unpack_into_dict(block_pid_map)
#     return ExptData(expt_hist, token_offs_raw, token_offs_pad, block_pid_map)


# # --------------------------
# # routing
# # --------------------------


# def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
#     hist, topk_indx, gate_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map = sort_tokens(
#         expt_scal, expt_indx, n_expts_tot, bitmatrix)
#     token_offs_pad = _unpack_into_dict(token_offs_pad)
#     block_pid_map = _unpack_into_dict(block_pid_map)
#     expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

#     # pack the matmul data structure
#     gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
#     scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
#     return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx


# def routing(logits, n_expts_act, sm_first=False, expt_indx=None, simulated_ep=1, n_rows=None):
#     from .topk import topk
#     if sm_first:
#         logits = torch.softmax(logits, dim=-1)
#     expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act,  #
#                                            apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
#     n_expts_tot = logits.shape[-1] // simulated_ep
#     # mutate bitmatrix
#     if simulated_ep > 1:
#         expt_scal, expt_indx, bitmatrix = prune_routing(expt_scal, expt_indx, bitmatrix, logits.shape[-1], simulated_ep)

#     return routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act)


# # --------------------------
# # torch reference
# # --------------------------


# def compute_expt_data_torch(hist, n_expts_tot, n_gates):
#     # offset for each experts
#     device = hist.device
#     token_offs_raw = torch.cumsum(hist, dim=0)
#     token_offs_raw = torch.cat((torch.zeros(1, device=device), token_offs_raw))
#     token_offs_raw = token_offs_raw.int()
#     # maximum number of tiles for all values of `block_m` considered
#     block_ms = [16, 32, 64, 128]
#     if is_hip():
#         block_ms.append(256)
#     if n_gates <= n_expts_tot:
#         max_n_tiles = n_gates
#     else:
#         # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
#         # ceil_div(x, y): -(-x // y)
#         max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))
#     # fill up tile offset/infos for each block
#     token_offs_pad = dict()
#     block_pid_map = dict()
#     for block_m in block_ms:
#         n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
#         token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
#         token_offs_pad[block_m] = torch.cat((torch.zeros(1, device=device), token_offs_pad[block_m]))
#         token_offs_pad[block_m] = token_offs_pad[block_m].int()
#         # compute data required to drive ragged batch matmul
#         block_pid_map[block_m] = -torch.ones(max_n_tiles, device=device)
#         for e in range(n_expts_tot):
#             offset = token_offs_pad[block_m][e]
#             for b in range(n_tiles[e]):
#                 block_pid_map[block_m][offset + b] = (b << 16) + e
#         block_pid_map[block_m] = block_pid_map[block_m].int()
#     return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


# def routing_torch(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
#     has_user_provided_indx = expt_indx is not None
#     n_gates_pad = logits.shape[0] * n_expts_act

#     if n_rows is not None:
#         logits = logits[:n_rows, :]

#     def topk(vals, k, expt_indx):
#         # topk of experts
#         if has_user_provided_indx:
#             tk_indx = expt_indx
#         else:
#             tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
#         tk_indx = tk_indx.long()
#         tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
#         tk_indx = tk_indx.int()
#         return tk_val, tk_indx

#     _, n_expts_tot = logits.shape
#     if sm_first:
#         logits = torch.softmax(logits, dim=-1)
#     expt_scal, expt_indx = topk(logits, n_expts_act, expt_indx)
#     if not sm_first:
#         expt_scal = torch.softmax(expt_scal, dim=-1)
#     # sort each token's selections by expert
#     if not has_user_provided_indx:
#         expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
#         expt_scal = torch.gather(expt_scal, 1, sort_indices)
#     # flatten topk data
#     expt_scal = expt_scal.reshape(-1)
#     expt_indx = expt_indx.reshape(-1).to(torch.int32)
#     # sort by expert_id so experts are contiguous for the matmul
#     topk_indx = torch.argsort(expt_indx, stable=True)
#     gate_indx = torch.argsort(topk_indx, stable=True)
#     gate_scal = expt_scal[topk_indx]
#     hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1).int()  # histogram of tokens over experts
#     # pack the matmul data structure
#     gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
#     scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
#     # compute expt_data
#     expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
#     return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx