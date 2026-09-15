# SPDX-License-Identifier: MIT
"""Diagnostic only: give M=1 the same mHC/wo_a arithmetic as M=2..64."""

import torch.nn.functional as F


def fixed_attention_reduction():
    """Diagnostic only: reuse V4's kernel with the small-batch D-chunk layout."""
    from atom.model_ops.v4_kernels import paged_decode

    original = paged_decode._paged_decode_reduce_kernel

    class FixedReduction:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kwargs["D_CHUNK"] = 32
                return original[(grid[0], grid[1], kwargs["BLOCK_D"] // 32)](
                    *args, **kwargs
                )

            return launch

    paged_decode._paged_decode_reduce_kernel = FixedReduction()


def pad_single_row_projections():
    from atom.model_ops.deepseek_v41 import mhc
    from atom.models.deepseek_v41 import attention

    grouped, hc = attention.grouped_output_projection, mhc.hc_projection

    def grouped_probe(hidden, weight):
        if hidden.is_cuda and hidden.shape[0] * hidden.shape[1] == 1:
            padded = F.pad(hidden.flatten(0, 1), (0, 0, 0, 0, 0, 127))
            return grouped(padded[None], weight)[:, :1].view(
                *hidden.shape[:-1], weight.shape[1]
            )
        return grouped(hidden, weight)

    def hc_probe(hidden, weight):
        if hidden.is_cuda and hidden.numel() // hidden.shape[-1] == 1:
            padded = F.pad(hidden.reshape(1, -1), (0, 0, 0, 127))
            return hc(padded, weight)[:1].view(*hidden.shape[:-1], weight.shape[0])
        return hc(hidden, weight)

    attention.grouped_output_projection = grouped_probe
    mhc.hc_projection = hc_probe


def ordered_reductions():
    """Diagnostic only: sum rank partials in the same order for every row."""
    import torch
    from aiter.dist.parallel_state import get_tp_group

    group = get_tp_group()
    original = group.all_reduce

    def reduce_probe(partial, *args, **kwargs):
        if partial.dtype != torch.float32 or partial.shape[-1] != 5120:
            return original(partial, *args, **kwargs)
        flat = partial.reshape(-1, partial.shape[-1]).contiguous()
        gathered = group.all_gather(flat, dim=0).view(group.world_size, *flat.shape)
        result = gathered[0].clone()
        for rank in range(1, group.world_size):
            result.add_(gathered[rank])
        return result.view_as(partial)

    group.all_reduce = reduce_probe


def serial_verify_operations(names):
    """Causal probe: change only named operations during target verification."""
    import torch

    from atom.model_ops.engram_layer import EngramOp
    from atom.models.deepseek_v41 import attention
    from atom.models.deepseek_v41.execution import DenseGraphExecutor
    from atom.utils.forward_context import get_forward_context

    stats = {name: {"calls": 0, "rows": 0, "max_rows": 0} for name in names}
    stats["engram_graph_stage_bypasses"] = 0

    def observed(name, rows):
        item = stats[name]
        item["calls"] += 1
        item["rows"] += rows
        item["max_rows"] = max(item["max_rows"], rows)

    def verifying():
        forward = get_forward_context()
        return forward.attn_metadata.step.tentative and not forward.context.is_draft

    if "engram" in names:
        original = EngramOp.forward

        def engram_probe(self, hidden, embeddings, token_mask=None):
            if not verifying():
                return original(self, hidden, embeddings, token_mask)
            flat = hidden.reshape(-1, *hidden.shape[-2:])
            observed("engram", flat.shape[0])
            embeds = embeddings.reshape(-1, embeddings.shape[-1])
            mask = None if token_mask is None else token_mask.reshape(-1)
            output = [
                original(
                    self,
                    flat[i : i + 1],
                    embeds[i : i + 1],
                    None if mask is None else mask[i : i + 1],
                )
                for i in range(flat.shape[0])
            ]
            return torch.cat(output).view_as(hidden)

        EngramOp.forward = engram_probe
        graph_run = DenseGraphExecutor.run

        def graph_probe(self, function, *args, bucket, capture=False):
            layer = getattr(function, "__self__", None)
            if (
                not capture
                and verifying()
                and function.__name__ == "prepare_attention"
                and getattr(layer, "engram", None) is not None
            ):
                key = (function, bucket, tuple(x is None for x in args))
                entry = self.entries.get(key)
                if entry is not None:
                    # The original graph captured with tentative=False, so its
                    # replay cannot execute the Python intervention above.
                    # Preserve its exact input bucket/padding and bypass only
                    # this stage, leaving prefill and all other graphs intact.
                    self._copy(entry.inputs, args)
                    stats["engram_graph_stage_bypasses"] += 1
                    output = function(*entry.inputs)
                    length = args[0].shape[1]
                    return tuple(value[:, :length] for value in output)
            return graph_run(self, function, *args, bucket=bucket, capture=capture)

        DenseGraphExecutor.run = graph_probe

    if "attention" in names:
        decode = attention.sparse_attn_v4_paged_decode

        def attention_probe(query, pool, indices, indptr, sink, scale, *args, **kwargs):
            if not verifying():
                return decode(
                    query, pool, indices, indptr, sink, scale, *args, **kwargs
                )
            observed("attention", query.shape[0])
            # Keep CSR offsets into the shared index array; the kernel accepts
            # a nonzero first offset and must see exactly the same visible KV.
            return torch.cat(
                [
                    decode(
                        query[i : i + 1],
                        pool,
                        indices,
                        indptr[i : i + 2],
                        sink,
                        scale,
                        *args,
                        **kwargs,
                    )
                    for i in range(query.shape[0])
                ]
            )

        attention.sparse_attn_v4_paged_decode = attention_probe

    return stats
