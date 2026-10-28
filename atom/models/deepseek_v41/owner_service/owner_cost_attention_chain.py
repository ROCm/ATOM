"""Actual native owner front/cache/attention/head assembly for cost diagnostics.

No serving model is installed. Cache, Step, inputs and weights must be bound
by the caller's source-aligned cost or complete private target diagnostic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FullOwnerHeads:
    q_weight: object
    q_scale: object
    a_weight: object
    a_scale: object
    b_weight: object
    b_scale: object
    sink: object

    @classmethod
    def gather_native(cls, attention, group):
        """Preserve original preshuffled N shards and four separate K shards."""
        import torch
        import torch.distributed as dist

        if group.world_size != 4:
            raise ValueError("Owner diagnostic requires the actual TP4 group")
        for module in (attention.wq_b, attention.wo_a, attention.wo_b):
            if not getattr(module.weight, "is_shuffled", False):
                raise ValueError(
                    "Bind original native preshuffled head weights explicitly"
                )

        def gather(value, stack=False):
            dtype = value.dtype
            transport = value.view(torch.uint8) if value.element_size() == 1 else value
            transport = transport.contiguous()
            parts = [torch.empty_like(transport) for _ in range(4)]
            dist.all_gather(parts, transport, group=group.device_group)
            combined = torch.stack(parts) if stack else torch.cat(parts, dim=0)
            return combined.view(dtype)

        return cls(
            gather(attention.wq_b.weight),
            gather(attention.wq_b.weight_scale),
            gather(attention.wo_a.weight),
            gather(attention.wo_a.weight_scale),
            gather(attention.wo_b.weight, True),
            gather(attention.wo_b.weight_scale, True),
            gather(attention.attn_sink),
        )


def forward_owner(
    attention,
    hidden,
    cache,
    step,
    rope,
    heads,
    tp_rank,
    sum_original_tp,
    *,
    observe=None,
):
    """Include real stream joins, all cache writers/scorer and four native heads.

    The diagnostic must separately compare independently built CSR against all
    four source TP CSR. A single owner's computed CSR is not evidence that those
    source CSR agree. Every recurring copy below is part of this graph's cost.
    """
    import torch
    import triton
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
    from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant

    from atom.model_ops.blockscale import native_quant_linear, quantize_fp8
    from atom.utils.region_graph import join_stream

    from .owner_grouped_attention import attend_grouped_head_shards

    # Current main's Mega-MHC returns BF16 for cache/index readers together
    # with the QKV projection's already quantized input and scale.
    hidden_fp8 = hidden_scale = None
    if isinstance(hidden, tuple):
        hidden, hidden_fp8, hidden_scale = hidden

    if cache.current_step is not step or cache.workspace is None:
        raise ValueError("Actual private cache/Step/scorer must be bound")
    if not step.decode or step.replay is not None or cache.packed:
        raise ValueError("This diagnostic assembles native BF16 tentative decode")
    if attention.heads * 4 != heads.sink.numel() or not 0 <= tp_rank < 4:
        raise ValueError("Full owner heads must preserve all four original TP shards")
    rows, dim = step.width, attention.head_dim
    # begin_forward belongs to the caller's complete 40-layer pass, once per
    # recorded invocation; clearing here would destroy REUSE/candidate readers.
    compressed = attention._fork_compress(hidden, cache, step, rope)
    q_lora, kv_pre = attention.project_qkv(
        hidden if hidden_fp8 is None else hidden_fp8, hidden_scale
    )
    if observe is not None:
        observe("q_lora", q_lora)
        observe("kv_pre", kv_pre)
    qr, qr_scale, kv_normed = attention.qk_norm(q_lora, kv_pre)
    if observe is not None:
        observe("qr", qr)
        observe("qr_scale", qr_scale)
        observe("kv_normed", kv_normed)
    attention._fork_select(
        hidden,
        qr,
        qr_scale,
        cache,
        step,
        rope,
        compressed=compressed,
    )
    query = native_quant_linear(
        qr,
        heads.q_weight,
        heads.q_scale,
        x_scale=qr_scale,
        weight_preshuffled=True,
        dtype=torch.bfloat16,
    ).view(rows, attention.heads * 4, dim)
    if observe is not None:
        observe("query_pre_rope", query)
    _, window_kv = cache.rope_quant_window(
        attention.spec.layer_id,
        query,
        kv_normed,
        rope,
        step,
    )
    join_stream(attention.index_stream, attention.compress_stream)
    csr = cache.attention_indices(attention.spec, step)
    cache.write_window(attention.spec.layer_id, window_kv, step)
    if observe is not None:
        observe("query", query)
        for name, value in zip(
            ("prefix_ids", "prefix_indptr", "window_ids", "window_indptr"),
            csr,
            strict=True,
        ):
            observe(name, value)
        from .owner_attention_observation import (
            observe_prefix_readers,
            observe_window_readers,
        )

        compressor_project = getattr(
            getattr(attention, "compressor", None), "project", None
        )
        if compressor_project is not None and hasattr(
            compressor_project, "last_values"
        ):
            observe("compressor_input", compressor_project.last_input)
            observe("compressor_values", compressor_project.last_values)
            if compressor_project.last_scores is not None:
                observe("compressor_scores", compressor_project.last_scores)
        observe_prefix_readers(cache.pool, csr[0], csr[1], observe)

        observe_window_readers(
            cache.ring_view(attention.spec.layer_id), csr[2], csr[3], observe
        )
    # Keep native per-TP head geometry. No rank-0 selection is broadcast and no
    # arithmetic is substituted by a newly compiled attention implementation.
    output = attend_grouped_head_shards(
        query,
        cache.pool,
        csr[0],
        csr[1],
        cache.ring_view(attention.spec.layer_id),
        csr[2],
        csr[3],
        heads.sink,
        attention.softmax_scale,
        attention.heads,
    )
    if observe is not None:
        observe("attention", output)
    groups = attention.groups * 4
    encoded, scale = inverse_rope_group_quant(
        output,
        step.positions,
        rope.cos_cache,
        rope.sin_cache,
        num_groups=groups,
        quant_group_size=32,
    )
    if observe is not None:
        observe("encoded", encoded)
        observe("encoded_scale", scale)
    mid = batched_gemm_a8w8_mxscale_bpreshuffle(
        encoded,
        heads.a_weight.view(groups, attention.o_rank, -1),
        scale,
        heads.a_scale.view(groups, attention.o_rank // 32, -1),
        dtype=output.dtype,
    )
    if observe is not None:
        observe("mid", mid)
    original_k = heads.b_weight.shape[-1]
    x, xs = quantize_fp8(mid.view(rows, 4, original_k))
    # Batched GEMM retains four original K2048 boundaries and BF16 partials.
    partial = batched_gemm_a8w8_mxscale_bpreshuffle(
        x,
        heads.b_weight,
        xs,
        heads.b_scale,
        dtype=output.dtype,
    )
    if observe is not None:
        observe("four_original_K_BF16_partials", partial)
    settled = torch.empty(
        (rows, heads.b_weight.shape[-2]), dtype=output.dtype, device=hidden.device
    )
    sum_original_tp[(triton.cdiv(settled.numel(), 1024),)](
        partial,
        settled,
        OWNER=tp_rank,
        H=settled.shape[-1],
        SIZE=settled.numel(),
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    return settled.view(1, rows, -1), {
        "qr": qr,
        "qr_scale": qr_scale,
        "kv": kv_normed,
        "query": query,
        "CSR": csr,
        "attention": output,
        "encoded": encoded,
        "encoded_scale": scale,
        "mid": mid,
        "four_original_K_BF16_partials": partial,
    }
