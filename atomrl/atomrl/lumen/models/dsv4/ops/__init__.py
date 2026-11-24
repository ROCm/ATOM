# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lazy compatibility facade for DSV4 model modules and canonical ops."""

from importlib import import_module

_LOCAL_EXPORTS = {
    "DeepSeekV4Compressor": "compressor",
    "DeepSeekV4HyperConnectionUtil": "hyper_connection",
    "HCHeadParams": "hyper_connection",
}

_CANONICAL_EXPORTS = (
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
)

__all__ = [
    "DeepSeekV4Compressor",
    "DeepSeekV4HyperConnectionUtil",
    "HCHeadParams",
    "all_gather_cp",
    "apply_rotary_emb",
    "configure_mhc_backend",
    "fp8_simulate",
    "fp8_simulate_qat",
    "get_batched_indexer_bwd",
    "get_batched_indexer_fwd",
    "get_compress_topk_idxs_cp",
    "get_dsa_topk_fn",
    "get_freqs_cis_for_cp",
    "get_indexer_impl",
    "get_mhc_backend",
    "get_q_positions_for_cp",
    "get_sparse_attn_fn",
    "get_sparse_mla_backend",
    "get_window_topk_idxs_cp",
    "infer_compress_ratio",
    "log_mhc_backend",
    "make_causal_cu_seqlens",
    "mask_logits_with_cu_seqlens",
    "rotate_activation",
    "wrapped_precompute_freqs_cis",
]


def get_mhc_backend() -> str:
    """Return the fixed MHC implementation family for legacy callers."""
    return "triton"


def configure_mhc_backend(backend: str | None = None) -> str:
    """Validate legacy backend requests without restoring runtime dispatch."""
    if backend is not None and backend.lower() != "triton":
        raise ValueError(
            "DSV4 MHC now uses the AIter Triton implementation exclusively"
        )
    return get_mhc_backend()


def log_mhc_backend() -> str:
    """Return the fixed backend label for legacy bootstrap logging."""
    return get_mhc_backend()


def __getattr__(name: str):
    local_module = _LOCAL_EXPORTS.get(name)
    if local_module is not None:
        value = getattr(import_module(f".{local_module}", __name__), name)
    elif name in _CANONICAL_EXPORTS:
        value = getattr(import_module("atomrl.lumen.ops.dsv4"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
