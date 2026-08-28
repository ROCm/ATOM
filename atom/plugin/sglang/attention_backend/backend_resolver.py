from typing import Any

import torch


def resolve_attn_backend(forward_batch: Any) -> Any:
    try:
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )

        if has_forward_context():
            backend = get_attn_backend()
            if backend is not None:
                return backend
    except Exception:  # noqa: BLE001, S110 - forward context is optional
        pass

    return getattr(forward_batch, "attn_backend", None)


def resolve_mamba_req_pool(forward_batch: Any, linear_backend: Any) -> Any:
    token_pool = getattr(forward_batch, "token_to_kv_pool", None)
    candidates = (
        getattr(token_pool, "_atom_kimi_k3_req_pool", None),
        getattr(linear_backend, "req_to_token_pool", None),
        getattr(forward_batch, "req_to_token_pool", None),
    )
    for pool in candidates:
        if pool is not None and hasattr(pool, "get_mamba_indices"):
            return pool

    try:
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            has_forward_context,
        )

        if has_forward_context():
            pool = get_req_to_token_pool()
            if pool is not None and hasattr(pool, "get_mamba_indices"):
                return pool
    except Exception:  # noqa: BLE001, S110 - forward context is optional
        pass
    return None


def reconstruct_linear_metadata(
    forward_batch: Any, linear_backend: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pool = resolve_mamba_req_pool(forward_batch, linear_backend)
    if pool is None:
        return None

    indices = pool.get_mamba_indices(forward_batch.req_pool_indices)
    translate = getattr(pool, "translate_mamba_indices", None)
    if translate is not None:
        indices = translate(indices)

    mode = forward_batch.forward_mode
    batch_size = forward_batch.batch_size
    # SGLang records the request count before DP/MLP-sync appends dummy rows.
    # Without padding this field is None and every batch row is real.
    real_batch_size = getattr(forward_batch, "_original_batch_size", None)
    real_batch_size = batch_size if real_batch_size is None else int(real_batch_size)
    real_batch_size = min(real_batch_size, batch_size)
    device = indices.device
    if real_batch_size < indices.shape[0]:
        # Mark DP/MLP-sync padding rows so they cannot read or write state.
        indices = indices.clone()
        indices[real_batch_size:] = -1

    if mode.is_decode_or_idle():
        # Give each real decode request one token and every padded row zero tokens.
        query_start_loc = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
        query_start_loc[: real_batch_size + 1] = torch.arange(
            real_batch_size + 1, dtype=torch.int32, device=device
        )
        query_start_loc[real_batch_size + 1 :] = real_batch_size
    elif mode.is_extend():
        # Build variable-length query offsets using only real extend requests.
        query_start_loc = torch.empty(
            (batch_size + 1,), dtype=torch.int32, device=device
        )
        if real_batch_size:
            # End at the final real request instead of a synthetic padded row.
            query_start_loc[:real_batch_size] = forward_batch.extend_start_loc[
                :real_batch_size
            ]
            end = (
                forward_batch.extend_start_loc[real_batch_size - 1]
                + forward_batch.extend_seq_lens[real_batch_size - 1]
            )
        else:
            # An empty real batch makes every synthetic row a zero-length query.
            end = 0
        query_start_loc[real_batch_size:] = end
    else:
        return None

    return query_start_loc, indices.to(dtype=torch.int32, device=device)
