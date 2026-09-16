# SPDX-License-Identifier: MIT
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry


def geometry(config, block=4):
    return V41PoolGeometry(
        config.num_hidden_layers,
        tuple(
            (owner, config.compress_ratios[owner])
            for owner in config.kv_source_layer_ids
        ),
        block,
        config.sliding_window,
        config.head_dim,
        config.index_head_dim,
    )


def metadata_buffers(batch_size, tokens, blocks, device="cpu"):
    import torch

    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.utils import CpuGpuBuffer

    buffers = {
        name: CpuGpuBuffer(
            *shape,
            dtype=torch.int64 if name == "positions" else torch.int32,
            device=device,
            pin_memory=torch.device(device).type != "cpu",
        )
        for name, shape in {
            "positions": (tokens,),
            "cu_seqlens_q": (batch_size + 1,),
            "batch_id_per_q_token": (tokens,),
            "block_tables": (batch_size, blocks),
            "input_ids": (tokens,),
        }.items()
    }
    buffers.update(
        DeepseekV4AttentionMetadataBuilder._state_slot_buffers(
            batch_size,
            device,
            read_side=False,
        )
    )
    return buffers
