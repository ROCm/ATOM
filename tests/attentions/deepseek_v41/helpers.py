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
        layer_ratios=tuple(
            sorted(set(config.compress_ratios[: config.num_hidden_layers]))
        ),
        index_topk=config.index_topk,
    )


def metadata_buffers(batch_size, tokens, blocks, device="cpu", geometry=None):
    """The `forward_vars` a hand-assembled builder needs, declared once.

    `geometry` adds the compressor plan buffers for the ratios that geometry
    owns. Pass the same object the builder gets: the names are keyed by ratio,
    so a second geometry would declare buffers no forward looks up.
    """
    import torch
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

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
    if geometry is not None:
        buffers.update(
            DeepseekV41MetadataBuilder._compress_plan_buffers(
                geometry,
                tokens,
                batch_size,
                device,
            )
        )
    return buffers
