# SPDX-License-Identifier: MIT
"""Large row-sharded prefill: transport one quantization to both FFN readers."""

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.deepseek_v41.row_sharded_mhc import (
    eligible,
    v41_mhc_reduce_scatter,
)
from atom.model_ops.module_dispatch_ops import _dual_stream_is_active

from .prefill_router import owned_router_logits


def supported(moe, local, rows, metadata):
    shared = moe.shared_experts
    return (
        eligible(rows, get_tp_group().world_size)
        and local.ndim == 2
        and local.dtype == torch.bfloat16
        and local.shape == ((rows + 3) // 4, 5120)
        and not moe.is_hash_layer
        and getattr(metadata, "image_mask", None) is None
        and moe.experts.supports_prequantized_input()
        and shared is not None
        and getattr(shared.gate_up_proj, "native_a8_group_rows", None) == 32
        and not shared.w2.reduce_results
    )


def _gather(value, rows, group):
    # The BF16 transport copies bytes; it performs no numeric conversion.
    # Keep the communicator's original size/dtype eligibility guard.
    dtype = value.dtype
    encoded = value.contiguous().view(torch.bfloat16)
    comm = getattr(group.device_communicator, "ca_comm", None)
    if comm is not None and comm.should_custom_ag(encoded):
        full = comm.custom_all_gather(encoded, dim=0)
    else:
        full = encoded.new_empty((encoded.shape[0] * 4, encoded.shape[1]))
        dist.all_gather_into_tensor(full, encoded, group=group.device_group)
    return full.view(dtype)[:rows]


def _gather_routes(weights, ids, rows, group):
    # Transport the native top-k fields as 32-bit words; no conversion of ids
    # or probabilities. Keep the original communicator eligibility checks.
    packet = torch.cat((weights.view(torch.int32), ids), dim=1)
    full = _gather(packet, rows, group)
    return full[:, :6].contiguous().view(torch.float32), full[:, 6:].contiguous()


def prequantized_owned_ffn(moe, local, rows):
    """Preserve native shared/routed/RS math and full-M router fallback.

    All tensors are graph-owned allocations, not a shared cache keyed by a
    reduced shape. Eager calls also own their buffers independently.
    """
    group = get_tp_group()
    logits = owned_router_logits(moe, local, rows, group)
    selected = None
    if getattr(moe, "prefill_route_packet", False):
        selected = moe._topk(local, logits, 6, True)
    value, scale = quantize_fp8(local)
    value = _gather(value, rows, group)
    scale = _gather(scale, rows, group)
    if selected is None:
        logits = _gather(logits, rows, group)
    else:
        selected = _gather_routes(*selected, rows, group)
        logits = None

    def shared():
        return moe.shared_experts(value, x_scale=scale)

    def routed():
        # This op already owns the compile boundary. Reuse the module's exact
        # routing/quantization method with an explicit input-scale argument.
        return moe.experts.forward_impl(
            value, logits, x_scale=scale, preselected_routes=selected
        )

    if _dual_stream_is_active(moe, rows):
        main = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(main)
        with torch.cuda.stream(moe.alt_stream):
            shared_output = shared()
        routed_output = routed()
        main.wait_stream(moe.alt_stream)
    else:
        shared_output = shared()
        routed_output = routed()
    return v41_mhc_reduce_scatter(routed_output + shared_output)
