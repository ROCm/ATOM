"""Stateless DeepSeek-V4 operation wrappers and helpers.

Exports are resolved lazily so importing this package does not load optional
GPU kernels or the TileKernels-backed QAT implementation.
"""

from importlib import import_module

_EXPORTS = {
    "all_gather_cp": "cp_utils",
    "apply_rotary_emb": "rope",
    "fp8_simulate": "qat",
    "fp8_simulate_qat": "qat",
    "get_batched_indexer_bwd": "indexer",
    "get_batched_indexer_fwd": "indexer",
    "get_compress_topk_idxs_cp": "cp_utils",
    "get_dsa_topk_fn": "dsa_topk",
    "get_freqs_cis_for_cp": "cp_utils",
    "get_indexer_impl": "indexer",
    "get_q_positions_for_cp": "cp_utils",
    "get_sparse_attn_fn": "sparse_mla",
    "get_sparse_mla_backend": "sparse_mla",
    "get_window_topk_idxs_cp": "cp_utils",
    "infer_compress_ratio": "indexer_utils",
    "make_causal_cu_seqlens": "indexer_utils",
    "mask_logits_with_cu_seqlens": "indexer_utils",
    "rotate_activation": "utils",
    "wrapped_precompute_freqs_cis": "rope",
}

__all__ = [
    "all_gather_cp",
    "apply_rotary_emb",
    "fp8_simulate",
    "fp8_simulate_qat",
    "get_batched_indexer_bwd",
    "get_batched_indexer_fwd",
    "get_compress_topk_idxs_cp",
    "get_dsa_topk_fn",
    "get_freqs_cis_for_cp",
    "get_indexer_impl",
    "get_q_positions_for_cp",
    "get_sparse_attn_fn",
    "get_sparse_mla_backend",
    "get_window_topk_idxs_cp",
    "infer_compress_ratio",
    "make_causal_cu_seqlens",
    "mask_logits_with_cu_seqlens",
    "rotate_activation",
    "wrapped_precompute_freqs_cis",
]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
