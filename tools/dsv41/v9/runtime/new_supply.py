# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
# SPDX-License-Identifier: MIT
"""Opt-in owner FP8 supply with the full-row native router schedule."""

import torch
from aiter.jit.utils.chip_info import get_gfx
from aiter.tuned_gemm import flydsl_gemm, get_GEMM_A16W16_config

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.module_dispatch_ops import _dual_stream_is_active
from atom.models.deepseek_v41.prefill_ffn import _gather

from atom.models.deepseek_v41.owner_service.owner_ffn_dispatch import finish_output

KERNEL = (
    "flydsl_hgemm_abf16_wbf16_bf16_t32x64x64x6_ks1_w1x2x1_bias0_ktail0_gm4_pft_gfx950"
)


def supply_config(moe, capacity, metadata, group):
    """Unknown schedules and BF16 communication backends keep native supply.

    The whitelist refers to the full packet's lookup, never local-M lookup.
    All packet rows, including padding holes, retain their rank-major order.
    """
    shared, experts, weight = moe.shared_experts, moe.experts, moe.gate.weight
    if not (
        capacity == 384
        and group.world_size == 4
        and (experts.tp_size, experts.ep_size, experts.dp_size) == (4, 1, 1)
        and not moe.is_hash_layer
        and moe.n_activated_experts == 6
        and getattr(metadata, "image_mask", None) is None
        and experts._comm_fused_moe is None
        and experts.supports_prequantized_input()
        and shared is not None
        and getattr(shared.gate_up_proj, "native_a8_group_rows", None) == 32
        and not shared.w2.reduce_results
        and weight.shape == (384, 5120)
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and not getattr(weight, "is_shuffled", False)
        and moe.gate.bias is None
        and get_gfx() == "gfx950"
    ):
        return None
    config = get_GEMM_A16W16_config(
        M=4 * capacity,
        N=384,
        K=5120,
        bias=False,
        dtype=str(weight.dtype),
        otype=str(weight.dtype),
        scaleAB=False,
        bpreshuffle=False,
    )
    if (
        config.get("libtype") == "flydsl"
        and config.get("kernelName") == KERNEL
        and config.get("splitK") == 1
    ):
        return config
    return None


def supplied_forward(moe, normed, config, group, reduce_scatter):
    """One owner quantization, three byte-preserving AGs, native FFN and RS.

    A fused mHC caller supplies (BF16, FP8, E8M0). BF16 still feeds the gate.
    Shared and routed readers consume the same full-M values and scales.
    """
    quantized = isinstance(normed, tuple)
    local = normed[0] if quantized else normed
    local = local.reshape(-1, 5120)
    capacity = local.shape[0]
    if local.dtype != torch.bfloat16 or not local.is_contiguous():
        raise ValueError("Owner supply requires contiguous BF16 router input")
    logits = flydsl_gemm(
        local, moe.gate.weight, config["solidx"], otype=local.dtype, config=config
    )
    value, scale = normed[1:] if quantized else quantize_fp8(local)
    assert capacity == 384 and group.world_size == 4
    local_fields = [
        value.reshape(capacity, 5120),
        scale.reshape(capacity, 160),
        logits.reshape(capacity, 384),
    ]
    comm = group.device_communicator.ca_comm
    assert comm is not None and not comm.disabled and comm.fully_connected
    assert comm._ops_all_gather is None, "Unsupported communicator family"
    gathered = comm.custom_all_gatherv(local_fields, [384] * 4)
    assert gathered is not None, "Owner AGV refused exact eligible contract"
    assert [tuple(t.shape) for t in gathered] == [
        (1536, 5120),
        (1536, 160),
        (1536, 384),
    ]
    assert all(
        out.dtype == inp.dtype and out.is_contiguous()
        for inp, out in zip(local_fields, gathered)
    )
    hits["agv"] = hits.get("agv", 0) + 1
    if VALIDATE:
        assert not torch.cuda.is_current_stream_capturing()
        references = [_gather(t, 1536, group) for t in local_fields]
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(gathered, references)):
            equal = torch.equal(
                out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
            )
            checks.append(
                {
                    "field": i,
                    "equal": equal,
                    "shape": list(out.shape),
                    "dtype": str(out.dtype),
                }
            )
            assert equal, "AGV differs from three native gathers"
    value, scale, logits = gathered

    def shared():
        return moe.shared_experts(value, x_scale=scale)

    def routed():
        return moe.experts.forward_impl(value, logits, x_scale=scale)

    if _dual_stream_is_active(moe, 4 * capacity):
        main = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(main)
        with torch.cuda.stream(moe.alt_stream):
            shared_output = shared()
        routed_output = routed()
        main.wait_stream(moe.alt_stream)
    else:
        shared_output = shared()
        routed_output = routed()
    return finish_output(
        routed_output,
        shared_output,
        False,
        group.rank_in_group,
        capacity,
        reduce_scatter,
    )


VALIDATE = False
hits = {}
checks = []
