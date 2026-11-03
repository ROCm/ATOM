# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Adapt M3's per-forward top-k memo to the runner's PP transport hooks."""


def validate_pp_index_reuse(config, *, pipeline_parallel_size, enable_tbo):
    if pipeline_parallel_size <= 1 or not enable_tbo:
        return
    from atom.models.minimax_m3.model import (
        _should_skip_minimax_m3_index_topk,
        _sparse_attention_layer_ids,
    )

    if any(
        _should_skip_minimax_m3_index_topk(config, layer)[0]
        for layer in _sparse_attention_layer_ids(config)
    ):
        raise ValueError(
            "MiniMax-M3 index reuse with pipeline parallelism does not support "
            "TBO: microbatch metadata does not carry PP top-k state. "
            "Disable --enable-tbo and --enable-tbo-decode, or use PP=1."
        )


class MiniMaxM3PPTopK:
    # Each TP rank selects its own blocks; these tensors must not be sliced or
    # broadcast by the shared activation transport's TP optimization.
    rank_local_keys = ("sparse_kv_indices",)

    @classmethod
    def create(cls, config, start_layer, end_layer, pp):
        from atom.models.minimax_m3.model import (
            _should_skip_minimax_m3_index_topk,
            _sparse_attention_layer_ids,
        )

        sparse_layers = sorted(_sparse_attention_layer_ids(config))
        if (
            pp.world_size <= 1
            or not getattr(config, "use_index_cache", False)
            or not sparse_layers
        ):
            return None

        def needs_topk(boundary):
            # Dense-only stages relay until the next sparse layer.
            return next(
                (
                    _should_skip_minimax_m3_index_topk(config, layer)[0]
                    for layer in sparse_layers
                    if layer >= boundary
                ),
                False,
            )

        adapter = cls()
        adapter.recv_needs_sparse = not pp.is_first_rank and needs_topk(start_layer)
        adapter.send_needs_sparse = not pp.is_last_rank and needs_topk(end_layer)
        return adapter

    def load(self, intermediate_tensors, forward_context):
        # Only logical selections cross PP; attention rebuilds local page metadata.
        md = forward_context.attn_metadata.sparse_attention_metadata
        md._index_topk_cache_state = {}
        if self.recv_needs_sparse:
            md._index_topk_cache_state["pp_topk"] = intermediate_tensors.tensors.pop(
                "sparse_kv_indices"
            )

    def store(self, model_output, forward_context):
        if forward_context.context.is_dummy_run:
            # Dummy forwards skip attention and produce no top-k memo.
            indices = None
        else:
            md = forward_context.attn_metadata.sparse_attention_metadata
            state = md._index_topk_cache_state
            indices = state["topk"]["value"][0] if "topk" in state else state["pp_topk"]
        model_output.tensors["sparse_kv_indices"] = indices
