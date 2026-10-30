# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared pointer bindings for Qwen kernel ABIs; never allocate or copy tensors."""


def pointer(value):
    return 0 if value is None else value.data_ptr()


def hc_args(bound):
    return {
        "hc_norm": pointer(bound.norm),
        "hc_down_inject": pointer(bound.down_inject),
        "hc_up": pointer(bound.up),
    }


def moe_args(bound):
    return {
        "router": pointer(bound.router),
        "shared_w13": pointer(bound.shared_w13),
        "shared_w2": pointer(bound.shared_w2),
        "routed_w13": pointer(bound.w13),
        "routed_w13_scale": pointer(bound.w13_scale),
        "routed_w2": pointer(bound.w2),
        "routed_w2_scale": pointer(bound.w2_scale),
    }


def bind_static_weights(weights, tp):
    """Resolve weight pointers once; tensor ownership remains with LayerWeights."""
    finish = {**hc_args(weights.mlp_hc), **moe_args(weights.moe)}
    if tp == 8:
        finish["expert_map"] = pointer(weights.moe.expert_map)
    gdn = (
        {
            name: pointer(getattr(weights.gdn, attr))
            for name, attr in _GDN_BINDINGS.items()
        }
        if weights.gdn is not None
        else {}
    )
    return {
        "finish": finish,
        "batch_moe": {**finish, "expert_map": pointer(weights.moe.expert_map)},
        "attn_hc": hc_args(weights.attn_hc),
        "qsa": qsa_projection_args(weights.qsa) if weights.qsa is not None else {},
        "batch_qsa": (
            {"output_weight": pointer(weights.qsa.o_proj)}
            if weights.qsa is not None
            else {}
        ),
        "batch_gdn": gdn,
        "gdn": {
            (_GDN_TP1_NAMES[name] if tp == 1 else name): value
            for name, value in gdn.items()
        },
    }


def gdn_state_args(step, tp):
    return {
        (_GDN_TP1_NAMES[name] if tp == 1 else name): pointer(getattr(step, attr))
        for name, attr in _GDN_STATE_BINDINGS.items()
    }


def output_args(streams_out, view):
    return {
        "streams_out": pointer(streams_out),
        "pending_out": pointer(view.pending_out),
        "pending_raw": pointer(view.pending_raw),
        "moe_out": pointer(view.moe_out),
    }


def qsa_projection_args(bound):
    return {
        "qkv_weight": pointer(bound.qkv),
        "q_norm": pointer(bound.q_norm),
        "k_norm": pointer(bound.k_norm),
        "index_weight": pointer(bound.index_qk),
        "index_q_norm": pointer(bound.index_q_norm),
        "index_k_norm": pointer(bound.index_k_norm),
        "cos": pointer(bound.cos),
        "sin": pointer(bound.sin),
        "index_cos": pointer(bound.index_cos),
        "index_sin": pointer(bound.index_sin),
    }


_QSA_METADATA = ("block_tables", "token_to_req", "logical_positions")
_QSA_CACHE = ("k_cache", "v_cache", "compressed_key_cache")


def _qsa_args(step, names):
    values = {name: pointer(getattr(step, name)) for name in names}
    values.update(
        table_stride=step.block_tables.stride(0), page_size=step.k_cache.shape[1]
    )
    return values


def qsa_cache_args(step):
    return _qsa_args(
        step,
        (
            *_QSA_METADATA,
            *_QSA_CACHE,
            "slot_mapping",
            "compressed_slot_mapping",
            "raw_key_cache",
            "rope_position_cache",
        ),
    )


def qsa_decode_args(step):
    values = _qsa_args(step, (*_QSA_METADATA, *_QSA_CACHE, "seq_lens"))
    values.update(
        table_width=step.block_tables.shape[1], cache_pages=step.k_cache.shape[0]
    )
    return values


_GDN_BINDINGS = {
    "projection_weight": "qkvzba",
    "conv_weight": "conv",
    "a_log": "a_log",
    "dt_bias": "dt_bias",
    "norm_weight": "norm",
    "output_weight": "out",
}
_GDN_STATE_BINDINGS = {
    "conv_state": "conv_state",
    "state_read": "state_read",
    "state_write": "state_write",
    "state": "recurrent_state",
}
_GDN_TP1_NAMES = {
    "projection_weight": "gdn_qkvzba",
    "conv_weight": "gdn_conv",
    "a_log": "gdn_a_log",
    "dt_bias": "gdn_dt_bias",
    "norm_weight": "gdn_norm",
    "output_weight": "gdn_out",
    "conv_state": "gdn_conv_state",
    "state_read": "gdn_state_read",
    "state_write": "gdn_state_write",
    "state": "gdn_recurrent_state",
}


def apply_ple(layer, streams, pending, input_ids):
    """Apply the model's PLE after the previous layer's deferred output is ready."""
    if layer.ple is not None:
        if pending is not None:
            streams = layer.attn_hyper_connection.apply_pending(streams, pending)
            pending = None
        streams = layer._apply_ple(streams, input_ids)
    return streams, pending
