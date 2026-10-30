"""Draft-only stateless compaction after the complete Native QSA call."""

import torch


def tail(layer, streams, block, raw, selected_rows=None):
    # QSA has already published all queries' state. Only these rows are used
    # by the head and following draft invocation. No state is compacted here.
    if selected_rows is not None:
        streams = torch.index_select(streams, 0, selected_rows)
        block = torch.index_select(block, 0, selected_rows)
        raw = torch.index_select(raw, 0, selected_rows)
    hc = layer.mlp_hyper_connection
    mixed, (streams, raw) = hc.mix_fused(streams, (block, raw))
    return hc.apply_pending(streams, (layer.mlp(mixed), raw))


def forward(layer, positions, hidden, selected_rows=None):
    mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden)
    block = layer.self_attn(positions, mixed)
    return tail(layer, streams, block, raw, selected_rows)


def validate_layer(layer):
    assert layer.tp_size == 1 and layer.ple is None and not layer.fused_hc
    assert layer.layer_type in ("full_attention", "qwen_sparse_attention")
    assert layer._attn_input_quant_dtype() is None
    assert not layer.mlp.fuse_shared and layer.mlp.shared_expert is not None
    assert layer.mlp.n_routed_experts == 512
    assert layer.mlp.experts.w13_weight.dtype == torch.bfloat16
    assert layer.mlp.experts.w2_weight.dtype == torch.bfloat16
    for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
        assert hc.hc_count == 4 and hc.hidden_size == 2560 and hc.fused_rank == 320
        assert hc.hc_norm.group_size == 2560
        assert hc.hc_norm.variance_epsilon == 1e-6
        assert hc.fused_w_cat.shape == (324, 10240)
        assert hc.fused_w_cat.dtype == torch.bfloat16
