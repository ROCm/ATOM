"""Explicit complete draft state + prepared head input, with Native BF16 GEMMs."""

import json
import os

import torch

from atom.model_ops.qwen4_exp.ops.gated import sigmoid_mul
from atom.models.qwen4_exp_mono.draft.compact import validate_layer
from atom.models.qwen4_exp_mono.draft.experts import NativeExperts


def forward(model, positions, hidden, selected_rows):
    layer = model.layers[0]
    mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(hidden)
    if os.environ.get("QWEN_DRAFT_QSA_FUSED") == "1":
        if not getattr(model, "_draft_qsa_fused_logged", False):
            assert not torch.cuda.is_current_stream_capturing()
            model._draft_qsa_fused_logged = True
            print(
                "QWEN_DRAFT_QSA_FUSED_BOUND "
                + json.dumps(
                    {
                        "kv_format": str(layer.self_attn.kv_format),
                        "index_format": str(layer.self_attn.index_format),
                        "query_lengths": [1, 4],
                    }
                ),
                flush=True,
            )
        from atom.models.qwen4_exp_mono.draft.attention import attention

        block = attention(layer.self_attn, positions, mixed)
    else:
        block = layer.self_attn(positions, mixed)
    if selected_rows is not None:
        streams, block, raw = [
            torch.index_select(x, 0, selected_rows) for x in (streams, block, raw)
        ]
    hc = layer.mlp_hyper_connection
    mixed, (streams, raw) = hc.mix_fused(streams, (block, raw))
    if mixed.shape[0] == 1:
        if not hasattr(model, "_q1_experts"):
            assert not torch.cuda.is_current_stream_capturing()
            validate_layer(layer)
            ex = layer.mlp.experts
            assert ex.top_k == 10 and ex.renormalize and ex.scoring_func == "softmax"
            assert ex.expert_map is None and not ex.apply_router_weight_on_input
            model._q1_experts = NativeExperts(ex.w13_weight, ex.w2_weight, 10)
            print(
                "QWEN_DRAFT_TAIL_BOUND "
                + json.dumps(
                    {
                        "rows": 1,
                        "topk": 10,
                        "expert_dtype": str(ex.w13_weight.dtype),
                        "block_m": model._q1_experts.bm,
                        "full_hc_state": True,
                        "prepared_head_input": True,
                    }
                ),
                flush=True,
            )
        mlp = layer.mlp
        logits = mlp.gate(mixed)
        weights, ids = mlp.experts.select_experts(
            hidden_states=mixed,
            router_logits=logits[:, :512],
            top_k=10,
            use_grouped_topk=False,
            renormalize=True,
            scoring_func="softmax",
        )
        routed = model._q1_experts(mixed, ids, weights)
        shared = mlp.shared_expert(mixed)
        out = sigmoid_mul(shared, logits[:, 512:], routed)
    else:
        out = layer.mlp(mixed)
    head_input, (state, _) = model.hyper_connection_mixer.mix_fused(streams, (out, raw))
    return state.view(-1, model.hc_count, model.hidden_size), head_input
