# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn


from atom.utils import mark_spliting_op
from .attention_mla import MLAModules
from atom.config import get_current_atom_config
from atom.utils.selector import get_attn_backend


def fake_(
    q: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    output_shape = list(q.shape)
    if use_mla:
        output_shape[-1] = 7168
    # If we fusion rmsnorm and quant, the input dtype is fp8, but actually we use bf16 for output.
    atom_config = get_current_atom_config()
    output_dtype = atom_config.torch_dtype
    output = torch.zeros(output_shape, dtype=output_dtype, device=q.device)

    return output


# Dynamo will not try to inspect any of the internal operations for prefill or decode
# This way, although attention operation is complicated,
# we can still capture the model's computation graph as a full-graph
@mark_spliting_op(is_custom=True, gen_fake=fake_, mutates_args=[])
def unified_attention_with_output_base(
    q: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
    qkv: torch.Tensor,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    if use_mla:
        return self.impl.forward(q, k, v, positions, q_scale, qkv)
    else:
        return self.impl.forward(layer=self,
                                 query=q,
                                 key=k,
                                 value=v,
                                 position=positions,
                                 q_scale=q_scale,
                                 qkv=qkv)

class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for attention

    This class defines the interface that all attention implementations must follow
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the forward() method"
        )

