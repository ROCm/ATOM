"""Assemble four native head shards into matching views of one output."""


def attend_original_head_shards(
    query,
    pool,
    prefix_ids,
    prefix_indptr,
    window,
    window_ids,
    window_indptr,
    sink,
    softmax_scale,
    shard_heads,
):
    """Keep the native head count, KV split policy and reduction arithmetic.

    OPUS BF16 shares Q's row/head strides with its output. The dense full Q
    and full output therefore use identical shard views. A non-dense Q keeps
    the original copy-and-concatenate path. Q remains available to observers.
    """
    import torch

    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

    if query.ndim != 3 or query.shape[1] != 4 * shard_heads:
        raise ValueError("Owner attention requires four original head shards")
    args = (pool, prefix_ids, prefix_indptr, window, window_ids, window_indptr)
    if not query.is_contiguous():
        return torch.cat(
            [
                sparse_attn_v4_paged_2src(
                    query[:, start : start + shard_heads].contiguous(),
                    *args,
                    sink[start : start + shard_heads],
                    softmax_scale,
                )
                for start in range(0, query.shape[1], shard_heads)
            ],
            dim=1,
        )
    output = torch.empty_like(query)
    for start in range(0, query.shape[1], shard_heads):
        q_part = query[:, start : start + shard_heads]
        out_part = output[:, start : start + shard_heads]
        # This is a correctness condition for the native shared-stride ABI.
        if q_part.stride() != out_part.stride():
            raise ValueError("Native Q and output shard strides must match")
        result = sparse_attn_v4_paged_2src(
            q_part,
            *args,
            sink[start : start + shard_heads],
            softmax_scale,
            out=out_part,
        )
        # The gfx1250 Triton fallback may return its own allocated output.
        if result is not out_part:
            out_part.copy_(result)
    return output


def attend_fp8_head_shards(
    query,
    pool,
    prefix_ids,
    prefix_indptr,
    window,
    window_ids,
    window_indptr,
    sink,
    softmax_scale,
    shard_heads,
    *,
    mode=None,
):
    """V4.1 MXFP8 owner attention, with an opt-in copy-free path.

    ``native`` keeps four dense H16 calls. ``a0`` keeps their split policy but
    passes strided Q/scales and writes into one output. ``h64`` uses one native
    H64 call only at the two rows tested with complete real-weight windows;
    other rows use A0. Small M can choose different H16/H64 split counts, so
    changing head count there is not an exact arithmetic substitution.
    """
    import torch

    from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
        sparse_attn_v41_paged_2src,
    )
    from atom.utils import envs

    mode = envs.ATOM_DSV41_OWNER_FP8_ATTENTION if mode is None else mode
    if mode not in ("native", "a0", "h64"):
        raise ValueError(f"Unknown owner FP8 attention mode: {mode}")

    values, scales = query
    if (
        values.ndim != 3
        or values.shape[1] != 4 * shard_heads
        or scales.shape != (*values.shape[:-1], values.shape[-1] // 32)
    ):
        raise ValueError("FP8 owner attention requires four original head shards")
    args = (pool, prefix_ids, prefix_indptr, window, window_ids, window_indptr)
    if mode != "native":
        if values.shape[1:] != (64, 512) or shard_heads != 16:
            raise ValueError("Optimized FP8 owner attention requires four H16 shards")
        if mode == "h64" and values.shape[0] in (221, 384):
            return sparse_attn_v41_paged_2src(query, *args, sink, softmax_scale)
        output = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
        for start in range(0, 64, 16):
            sparse_attn_v41_paged_2src(
                (values[:, start : start + 16], scales[:, start : start + 16]),
                *args,
                sink[start : start + 16],
                softmax_scale,
                out=output[:, start : start + 16],
            )
        return output
    return torch.cat(
        [
            sparse_attn_v41_paged_2src(
                (
                    values[:, start : start + shard_heads].contiguous(),
                    scales[:, start : start + shard_heads].contiguous(),
                ),
                *args,
                sink[start : start + shard_heads],
                softmax_scale,
            )
            for start in range(0, values.shape[1], shard_heads)
        ],
        dim=1,
    )
