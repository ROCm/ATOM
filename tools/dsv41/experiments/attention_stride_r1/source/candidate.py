"""A0: four original FP8 H16 calls with strided inputs and caller-owned output.

Private candidate for a complete owner attention window. Arithmetic, CSR,
sink, native split selection and projection boundaries remain native.
"""


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
):
    import torch
    from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
        sparse_attn_v41_paged_2src,
    )

    values, scales = query
    if (
        values.ndim != 3
        or values.shape[1:] != (64, 512)
        or shard_heads != 16
        or scales.shape != (*values.shape[:-1], 16)
    ):
        raise ValueError("A0 requires native four-H16 V4.1 FP8 owner geometry")
    output = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    for start in range(0, 64, 16):
        # The native V4.1 ABI has independent Q, scale and output strides.
        # Its alignment/last-dimension guards still run in the native wrapper.
        sparse_attn_v41_paged_2src(
            (values[:, start : start + 16], scales[:, start : start + 16]),
            pool,
            prefix_ids,
            prefix_indptr,
            window,
            window_ids,
            window_indptr,
            sink[start : start + 16],
            softmax_scale,
            out=output[:, start : start + 16],
        )
    return output
