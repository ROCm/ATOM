# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 AtomRL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code adapted from the verl project
# (https://github.com/verl-project/verl).
# Original: verl/utils/ulysses.py

"""Utilities for DeepSpeed Ulysses Sequence Parallelism.

DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

_ULYSSES_SEQUENCE_PARALLEL_GROUP: ProcessGroup | None = None


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::set_ulysses_sequence_parallel_group
def set_ulysses_sequence_parallel_group(group: ProcessGroup) -> None:
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::get_ulysses_sequence_parallel_group
def get_ulysses_sequence_parallel_group() -> ProcessGroup | None:
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::get_ulysses_sequence_parallel_world_size
def get_ulysses_sequence_parallel_world_size(
    group: ProcessGroup | None = None,
) -> int:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::get_ulysses_sequence_parallel_rank
def get_ulysses_sequence_parallel_rank(group: ProcessGroup | None = None) -> int:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::slice_input_tensor
def slice_input_tensor(
    x: Tensor,
    dim: int,
    padding: bool = True,
    group: ProcessGroup | None = None,
) -> Tensor:
    """Slice a tensor along *dim* for the local SP rank."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank(group)
    dim_size = x.size(dim)
    if padding and dim_size % sp_world:
        pad_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, dim, pad_size)
    parts = x.size(dim) // sp_world
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[tuple(slc)].contiguous()


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::all_to_all_tensor
def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: ProcessGroup | None = None,
    async_op: bool = False,
):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::SeqAllToAll
class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple:
        input_t = (
            torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
            if ctx.async_op
            else grad_output[0]
        )
        return (
            None,
            all_to_all_tensor(
                input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False
            ),
            None,
            None,
            None,
            None,
        )


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::gather_seq_scatter_heads
def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-all: gather sequence dim, scatter head dim."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::gather_heads_scatter_seq
def gather_heads_scatter_seq(
    x: Tensor,
    head_dim: int,
    seq_dim: int,
    group: ProcessGroup | None = None,
) -> Tensor:
    """All-to-all: gather head dim, scatter sequence dim."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::all_gather_tensor
def all_gather_tensor(
    local_tensor: Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Tensor:
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::Gather
class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::gather_outputs_and_unpad
def gather_outputs_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int | None = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: ProcessGroup | None = None,
) -> Tensor:
    """Gather a tensor across SP ranks and optionally remove padding."""
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int)
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::ulysses_pad
def ulysses_pad(
    input_ids_rmpad: Tensor,
    position_ids_rmpad: Tensor | None = None,
    sp_size: int = 1,
    pad_value: int = 0,
) -> tuple[Tensor, Tensor | None, int]:
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(
            input_ids_rmpad, (0, pad_size), value=pad_value
        )
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(
                pad_size, device=position_ids_rmpad.device
            ).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(
                    position_ids_rmpad.size(0), 1, 1
                )
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


# Adapted from verl (https://github.com/verl-project/verl)
# Original: verl/utils/ulysses.py::ulysses_pad_and_slice_inputs
def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: Tensor,
    position_ids_rmpad: Tensor | None = None,
    sp_size: int = 1,
    skip_position_ids_rmpad: bool = False,
    pad_value: int = 0,
) -> tuple[Tensor, Tensor | None, int]:
    """Pad and slice input_ids to be divisible by sp_size.

    Pre-forward utility for Ulysses sequence parallelism.

    Args:
        input_ids_rmpad: shape of ``[bsz, seqlen]``
        position_ids_rmpad: shape of ``[bsz, seqlen]``, where bsz must be 1
        sp_size: ulysses sequence parallelism size
        skip_position_ids_rmpad: whether to skip slicing position_ids
        pad_value: padding value for input_ids

    Returns:
        Padded and sliced ``input_ids``, ``position_ids``, and ``pad_size``.
    """
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
        input_ids_rmpad,
        position_ids_rmpad,
        sp_size,
        pad_value=pad_value,
    )
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None and not skip_position_ids_rmpad:
        position_ids_rmpad = slice_input_tensor(
            position_ids_rmpad, dim=1, padding=False
        )
    return input_ids_rmpad, position_ids_rmpad, pad_size


def validate_ulysses_config(num_heads: int, ulysses_sequence_size: int) -> None:
    if ulysses_sequence_size > 1:
        assert num_heads % ulysses_sequence_size == 0, (
            f"num_heads ({num_heads}) must be divisible by "
            f"ulysses sequence size ({ulysses_sequence_size})"
        )
