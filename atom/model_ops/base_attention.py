# from flash_attn import flash_attn_with_kvcache
from dataclasses import dataclass

import aiter
import torch
import triton
import triton.language as tl
from torch import nn
from typing import Optional


from atom.utils import mark_spliting_op
from .attention_mla import MLAModules
from atom.config import get_current_atom_config
from atom.utils.selector import get_attn_backend


def fake_(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
) -> torch.Tensor:
    output_shape = list(q.shape)
    if use_mla:
        output_shape[-1] = 7168
    output = torch.zeros(output_shape, dtype=q.dtype, device=q.device)

    return output


# Dynamo will not try to inspect any of the internal operations for prefill or decode
# This way, although attention operation is complicated,
# we can still capture the model's computation graph as a full-graph
@mark_spliting_op(is_custom=True, gen_fake=fake_, mutates_args=[])
def unified_attention_with_output_base(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    use_mla: bool,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.impl.forward(q, k, v, positions)


class Attention(nn.Module):

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
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num
        self.mla_modules = mla_modules
        self.use_mla = use_mla
        self.base_attention = None
        self.kv_cache = torch.tensor([])
        self.indexer = mla_modules.indexer if mla_modules is not None else None
        self.sinks = sinks

        atom_config = get_current_atom_config()
        block_size = atom_config.kv_cache_block_size
        self.attn_backend = get_attn_backend(
            block_size,
            use_mla=self.use_mla,
        )
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            layer_num,
            mla_modules,
            sinks=sinks,
            sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
        )

        compilation_config = atom_config.compilation_config
        self.layer_name = f"MLA_{layer_num}" if self.use_mla else f"MHA_{layer_num}"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor = None,
    ):
        output = torch.ops.aiter.unified_attention_with_output_base(
            q, k, v, positions, self.layer_name, self.use_mla
        )
        return output
