# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from functools import partial as functools_partial
from typing import Callable, Optional

import torch
from aiter import (
    QuantType,
    dtypes,
    gemm_a4w4,
    gemm_a8w8,
    gemm_a8w8_blockscale_bpreshuffle,
    gemm_a8w8_bpreshuffle,
    get_hip_quant,
)
from torch import nn

from atom.config import QuantizationConfig, get_current_atom_config
from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz, requantize_with_max_scale
from atom.models.utils import get_quant_config_for_layer

# import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.tuned_gemm import tgemm
from aiter.utility import fp4_utils
from atom.utils.decorators import mark_trace

from atom.model_ops.utils import shuffle_weights
from atom.utils import envs

logger = logging.getLogger("atom")


def use_triton_gemm() -> bool:
    return envs.ATOM_USE_TRITON_GEMM


if use_triton_gemm():
    try:
        from aiter.ops.triton.gemm_afp4wfp4 import (
            gemm_afp4wfp4_preshuffle,
        )  # noqa: E402

        # For Triton FP8 Blockscale GEMM is mostly slower then AITER GEMM, we turn off Triton FP8 GEMM
        # from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale_preshuffle as gemm_a8w8_blockscale_bpreshuffle_triton
    except ImportError as e:
        logger.warning(f"Triton FP4 GEMM not available: {e}")
        gemm_afp4wfp4_preshuffle = None
    gemm_a8w8_blockscale_bpreshuffle_triton = None
else:
    gemm_afp4wfp4_preshuffle = None
    gemm_a8w8_blockscale_bpreshuffle_triton = None
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE


def divide(numerator, denominator):
    assert (
        numerator % denominator == 0
    ), f"numerator {numerator} denominator {denominator}"
    return numerator // denominator


def gemm_a4w4_quant_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=otype, device=x.device)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=gemm_a4w4_quant_fake, mutates_args=[])
def gemm_a4w4_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    otype: torch.dtype,
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype,
    input_scale: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    if gemm_afp4wfp4_preshuffle is None:
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                scale=input_scale,
                shuffle=True,
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        m = x.view(-1, x.size(-1)).shape[0]
        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        y = gemm_a4w4(
            x,
            weight,
            x_scale,
            weight_scale,
            y,
        )
    else:
        m, k = x.view(-1, x.size(-1)).shape
        n = weight.shape[0]

        y = torch.empty(
            (
                (m + MXFP4_QUANT_BLOCK_SIZE - 1)
                // MXFP4_QUANT_BLOCK_SIZE
                * MXFP4_QUANT_BLOCK_SIZE,
                output_size,
            ),
            dtype=otype,
            device=x.device,
        )
        if x_scale is None:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                x,
                quant_dtype=params_dtype,
                shuffle=(m >= MXFP4_QUANT_BLOCK_SIZE),
            )
        else:
            x_scale = x_scale.view(torch.float8_e8m0fnu)
            x = x.view(torch.float4_e2m1fn_x2)

        if m >= MXFP4_QUANT_BLOCK_SIZE:
            x_scale = x_scale.view(torch.uint8).view(
                x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            )
        else:
            x_scale = x_scale[:m, ...].view(torch.uint8)

        y = gemm_afp4wfp4_preshuffle(
            x.view(torch.uint8),
            weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
            x_scale,
            weight_scale.view(torch.uint8).view(
                weight_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            y=y,
        )

    return y[:m, ...]


def gemm_a8w8_blockscale_preshuffle_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_preshuffle_fake, mutates_args=[])
def gemm_a8w8_blockscale_preshuffle_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if gemm_a8w8_blockscale_bpreshuffle_triton is not None:
        weight_shuffled = weight.reshape(weight.shape[0] // 16, weight.shape[1] * 16)
        y = gemm_a8w8_blockscale_bpreshuffle_triton(
            x, weight_shuffled, x_scale, w_scale, dtype
        )
    else:
        y = gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, w_scale, dtype)
    return y


@mark_trace
class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | list[int],
        tp_dim: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = False,
        source_quant_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        if quant_config is None:
            quant_config = QuantizationConfig()
        self.source_quant_dtype = source_quant_dtype
        quant_type = quant_config["quant_type"]
        params_dtype = quant_config["quant_dtype"]
        super().__init__()
        self.prefix = prefix
        self.reduce_results = reduce_results
        self.input_size = input_size
        self.output_size = (
            output_size if isinstance(output_size, int) else sum(output_size)
        )
        self.tp_dim = tp_dim
        self.tp_rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_group().world_size
        self.output_partition_sizes = (
            output_size if isinstance(output_size, list) else [output_size]
        )
        if tp_dim == 1:
            self.input_size = divide(input_size, self.tp_size)
        elif tp_dim == 0:
            self.output_size = divide(self.output_size, self.tp_size)
            self.output_partition_sizes = [
                divide(s, self.tp_size) for s in self.output_partition_sizes
            ]

        if self.source_quant_dtype is not None:
            weight_size = (self.output_size, self.input_size)
            self.weight = nn.Parameter(
                torch.empty(weight_size, dtype=self.source_quant_dtype),
                requires_grad=False,
            )
        else:
            weight_size = (
                (self.output_size, self.input_size)
                if params_dtype not in [dtypes.fp4x2, dtypes.i4x2]
                else (self.output_size, self.input_size // 2)
            )
            self.weight = nn.Parameter(
                torch.empty(weight_size, dtype=params_dtype),
                requires_grad=False,
            )
        if bias:
            output_type = get_current_atom_config().torch_dtype
            self.bias = nn.Parameter(
                torch.empty(self.output_size, dtype=output_type), requires_grad=False
            )
            self.bias.weight_loader_process = self.weight_loader_process
        else:
            self.register_parameter("bias", None)
        self.quant_type = quant_type
        self.params_dtype = params_dtype

        if quant_type != QuantType.No and self.source_quant_dtype is None:
            if quant_type == QuantType.per_Tensor:
                self.weight_scale = nn.Parameter(
                    torch.empty(len(self.output_partition_sizes), 1, dtype=dtypes.fp32),
                    requires_grad=False,
                )
                if not quant_config["is_dynamic"]:
                    self.input_scale = nn.Parameter(
                        torch.empty(
                            len(self.output_partition_sizes), 1, dtype=dtypes.fp32
                        ),
                        requires_grad=False,
                    )
                    self.input_scale.weight_loader_process = self.weight_loader_process
                    self.input_scale.weight_loader = self.weight_loader
            elif quant_type == QuantType.per_Token:
                self.weight_scale = nn.Parameter(
                    torch.empty(self.output_size, 1, dtype=dtypes.fp32),
                    requires_grad=False,
                )
            elif quant_type == QuantType.per_1x128:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        (self.output_size + 127) // 128,
                        (self.input_size + 127) // 128,
                        dtype=dtypes.fp32,
                    ),
                    requires_grad=False,
                )
            elif quant_type == QuantType.per_1x32:
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        self.output_size,
                        (self.input_size + 31) // 32,
                        dtype=dtypes.fp8_e8m0,
                    ),
                    requires_grad=False,
                )
            self.weight.weight_loader_process = self.weight_loader_process
            self.weight_scale.weight_loader_process = self.weight_loader_process
        else:
            self.weight.weight_loader_process = self.weight_loader_process
            self.register_parameter("weight_scale", None)
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self.weight_loader
        if self.weight_scale is not None:
            self.weight_scale.weight_loader = self.weight_loader
        self.need_normalize_e4m3fn_to_e4m3fnuz = params_dtype == torch.float8_e4m3fnuz

    @staticmethod
    def weight_loader_process(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        post_process_func: Callable = lambda a: a,
    ):
        if (
            param.data.dtype != loaded_weight.dtype
            and param.data.element_size() == loaded_weight.element_size()
        ):
            param.data = param.data.view(loaded_weight.dtype)
        param.data.copy_(post_process_func(loaded_weight))

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)

    def process_weights_after_loading(self):
        if (
            self.quant_type == QuantType.per_Tensor
            and len(self.output_partition_sizes) > 1
        ):
            weight_scale, weight = requantize_with_max_scale(
                weight=self.weight.data,
                weight_scale=self.weight_scale.data,
                logical_widths=self.output_partition_sizes,
                normalize_e4m3fn_to_e4m3fnuz=self.need_normalize_e4m3fn_to_e4m3fnuz,
            )
            self.weight.data = weight
            self.weight_scale.data = weight_scale.view(-1)
            if hasattr(self, "input_scale"):
                self.input_scale.data = (
                    self.input_scale.data.max() * 2.0
                    if self.need_normalize_e4m3fn_to_e4m3fnuz
                    else self.input_scale.data.max()
                )
        elif self.need_normalize_e4m3fn_to_e4m3fnuz:
            self.weight.data, self.weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                self.weight.data, self.weight_scale.data
            )
        if (
            self.source_quant_dtype == torch.bfloat16
            and self.quant_type == QuantType.per_1x32
            and self.params_dtype == torch.float4_e2m1fn_x2
        ):
            quant_func = get_hip_quant(QuantType.per_1x32)
            w_q, w_s = quant_func(
                self.weight.data,
                quant_dtype=self.params_dtype,
                shuffle=False,
            )
            self.weight.data = w_q
            self.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
            shuffle_weights(self.weight)
            # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        else:
            if (
                self.quant_type == QuantType.per_Token
                and self.params_dtype == dtypes.fp8
            ) or (self.quant_type in [QuantType.per_1x32, QuantType.per_1x128]):
                shuffle_weights(self.weight)
                # self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)
        # shuffle weight scale once so no reshuffling for every gemm
        if self.quant_type == QuantType.per_1x32:
            self.weight_scale.data = fp4_utils.e8m0_shuffle(self.weight_scale.data)

    def forward(
        self, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None, otype=dtypes.bf16
    ) -> torch.Tensor:
        if self.quant_type.value == QuantType.No.value:
            y = tgemm.mm(
                x,
                self.weight,
                self.bias,
                otype=otype,
            )
        else:
            if x_scale is None:
                quant_func = get_hip_quant(self.quant_type)
                if self.quant_type.value == QuantType.per_1x128.value:
                    quant_func = functools_partial(quant_func, transpose_scale=True)
                if self.quant_type.value != QuantType.per_1x32.value:
                    x, x_scale = quant_func(
                        x,
                        quant_dtype=self.params_dtype,
                        scale=getattr(self, "input_scale", None),
                    )
            if self.quant_type.value == QuantType.per_Tensor.value:
                y = tgemm.mm(
                    x,
                    self.weight,
                    self.bias,
                    otype=otype,
                    scale_a=x_scale,
                    scale_b=self.weight_scale,
                )
            elif self.quant_type.value == QuantType.per_Token.value:
                if self.params_dtype == dtypes.i8:
                    y = gemm_a8w8(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        self.bias,
                        dtype=otype,
                    )
                else:
                    y = gemm_a8w8_bpreshuffle(
                        x,
                        self.weight,
                        x_scale,
                        self.weight_scale,
                        dtype=otype,
                    )
                    if self.bias is not None:
                        y += self.bias
            elif self.quant_type.value == QuantType.per_1x128.value:
                y = gemm_a8w8_blockscale_preshuffle_impl(
                    x, self.weight, x_scale, self.weight_scale, dtype=otype
                )
                if self.bias is not None:
                    y += self.bias
            elif self.quant_type.value == QuantType.per_1x32.value:
                y = gemm_a4w4_quant(
                    x,
                    x_scale,
                    self.weight,
                    otype,
                    self.weight_scale.data,
                    self.params_dtype,
                    getattr(self, "input_scale", None),
                    self.output_size,
                )
                if self.bias is not None:
                    y += self.bias
        if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
            y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
        return y


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            tp_dim=None,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=kwargs.get("prefix", ""),
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param.weight_loader_process(param_data, loaded_weight)


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        **kwargs,
    ):
        self.tp_dim = 0
        super().__init__(
            input_size,
            output_size,
            self.tp_dim,
            bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=kwargs.get("prefix", ""),
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class MergedColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.output_sizes = output_sizes
        if quant_config is not None and prefix:
            quant_config = get_quant_config_for_layer(quant_config, prefix)
        super().__init__(
            input_size,
            output_sizes,
            tp_dim=0,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset //= 128
                shard_size //= 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = loaded_shard_id
                shard_size = 1

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.weight_loader_process(param_data, loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        **kwargs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = get_tp_group().world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        input_size = hidden_size
        output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
        ]

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=kwargs.get("prefix", ""),
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            shard_rank = self.tp_rank
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset //= 128
                shard_size //= 128
            elif self.quant_type == QuantType.per_Tensor:
                loaded_weight = loaded_weight.view(1, 1).repeat(self.tp_size, 1)
                shard_offset = ["q", "k", "v"].index(loaded_shard_id)
                shard_size = 1

        start_idx = shard_rank * shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.weight_loader_process(param_data, loaded_weight)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        source_quant_dtype: torch.dtype = None,
        prefix: str = "",
        **kwargs,
    ):
        self.tp_rank = get_tp_group().rank_in_group
        if quant_config is not None and prefix:
            quant_config = get_quant_config_for_layer(quant_config, prefix)
        super().__init__(
            input_size,
            output_size,
            tp_dim=1,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            source_quant_dtype=source_quant_dtype,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param is not getattr(self, "bias", None):
            shard_size = param_data.size(self.tp_dim)
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.view(1, 1)
            if loaded_weight.size(self.tp_dim) == 1 and self.tp_size > 1:
                loaded_weight = loaded_weight.repeat(1, self.tp_size)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        else:
            if self.tp_size > 0 and self.tp_rank != 0:
                loaded_weight.zero_()
        param.weight_loader_process(param_data, loaded_weight)


class MergedReplicatedLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        source_quant_dtype: torch.dtype = None,
        **kwargs,
    ):
        self.output_sizes = output_size
        super().__init__(
            input_size,
            sum(output_size),  # ？
            bias=bias,
            quant_config=quant_config,
            source_quant_dtype=source_quant_dtype,
            prefix=kwargs.get("prefix", ""),
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):  # ？
        param_data = param.data
        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        if param is getattr(self, "weight_scale", None) or param is getattr(
            self, "input_scale", None
        ):
            if self.quant_type == QuantType.per_1x128:
                shard_offset = (
                    sum(self.output_sizes[:loaded_shard_id]) + 128 - 1
                ) // 128
                shard_size = (self.output_sizes[loaded_shard_id] + 128 - 1) // 128
            elif self.quant_type == QuantType.per_Tensor:
                shard_offset = loaded_shard_id
                shard_size = 1
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        param.weight_loader_process(param_data, loaded_weight)
