# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Eager-only comparison of the Qwen mono finish with the original modules."""

import logging

from atom.mono.runtime.compare import compare

logger = logging.getLogger("atom")


def _reference(layer, streams, attn_out, attn_raw):
    """Run the original MLP HC seam and MoE on the mono finish's inputs."""
    mixed, (streams_out, raw) = layer.mlp_hyper_connection.mix_fused(
        streams, (attn_out, attn_raw)
    )
    return streams_out, raw, layer.mlp(mixed)


def check_finish(
    rank,
    layer_id,
    layer,
    streams,
    attn_out,
    attn_raw,
    mono_streams,
    mono_raw,
    mono_moe,
):
    """Log each finish hand-off and the reference path's own repeatability."""
    ref_streams, ref_raw, ref_moe = _reference(layer, streams, attn_out, attn_raw)
    again_streams, again_raw, again_moe = _reference(layer, streams, attn_out, attn_raw)
    if rank == 0:
        logger.info(
            "Qwen mono check layer %d: ref streams %s; ref raw %s; "
            "ref moe %s; streams %s; raw %s; moe %s",
            layer_id,
            compare(again_streams, ref_streams),
            compare(again_raw, ref_raw),
            compare(again_moe, ref_moe),
            compare(mono_streams, ref_streams),
            compare(mono_raw, ref_raw),
            compare(mono_moe, ref_moe),
        )


def check_gdn(
    rank,
    layer_id,
    mono_out,
    ref_out,
    mono_conv_state,
    ref_conv_state,
    mono_recurrent_state,
    ref_recurrent_state,
):
    """Log the complete GDN output and both persistent state hand-offs."""
    if rank == 0:
        logger.info(
            "Qwen mono GDN check layer %d: out %s; conv state %s; recurrent state %s",
            layer_id,
            compare(mono_out, ref_out),
            compare(mono_conv_state, ref_conv_state),
            compare(mono_recurrent_state, ref_recurrent_state),
        )



def check_gdn_reduce(rank, layer_id, mono_out, ref_out):
    """Log the TP8 attention endpoint after the fixed-order peer pull."""
    if rank == 0:
        logger.info(
            "Qwen mono GDN reduce check layer %d: out %s",
            layer_id,
            compare(mono_out, ref_out),
        )


def check_qsa_pre(
    rank,
    layer_id,
    mono_streams,
    ref_streams,
    mono_raw,
    ref_raw,
    mono_prepared,
    ref_prepared,
    mono_cache,
    ref_cache,
):
    """Log every public QSA K1 seam and the cache rows K1 owns."""

    if rank != 0:
        return
    cache = "; ".join(
        f"{name} {compare(mono_cache[name], ref_cache[name])}" for name in mono_cache
    )
    logger.info(
        "Qwen mono QSA K1 check layer %d: streams %s; raw %s; query %s; "
        "gate %s; index query %s; cache %s",
        layer_id,
        compare(mono_streams, ref_streams),
        compare(mono_raw, ref_raw),
        compare(mono_prepared.query, ref_prepared.query),
        compare(mono_prepared.gate, ref_prepared.gate),
        compare(mono_prepared.index_query, ref_prepared.index_query),
        cache,
    )


def check_qsa_post(rank, layer_id, mono_attn_out, ref_attn_out):
    """Log the K2 o_proj endpoint before the common finish consumes it."""

    if rank == 0:
        logger.info(
            "Qwen mono QSA K2 check layer %d: o_proj %s",
            layer_id,
            compare(mono_attn_out, ref_attn_out),
        )
