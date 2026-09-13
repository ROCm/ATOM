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
