"""Bind the V7 HC launcher and its larger tagged-router allocation atomically.

The 20-argument ABI is unchanged, but the old V5/V6 296-byte flags allocation
is invalid for V7. A new 16448-byte owner and every HcStatic.flags pointer are
installed together before any HC execution. G1/G2 bindings stay unchanged.
"""

import torch

import atom.models.qwen4_exp_mono.chain.moe_front_hc_v5_device as v5
import atom.models.qwen4_exp_mono.chain.moe_front_hc_v7_device as v7
from atom.models.qwen4_exp_mono.chain.moe_front_hc_v7_contract import (
    FLAGS_BYTES,
    ROUTER_VALUES,
    SELECT_READY_OFFSET,
)
from atom.mono.runtime.compile import compile_kernels, launcher_resources


def replace_hc_kernel_and_flags(prepared):
    assert prepared.metadata["hc_source"] == v5.SOURCE
    kernel = v7.build(False)
    compile_kernels([(kernel, v7.ABI)])
    resources = launcher_resources(kernel)
    assert resources and all(
        not row["scratch"]
        and not row.get("vgpr_spill", 0)
        and not row.get("sgpr_spill", 0)
        and row["vgpr"] <= 128
        and row["lds"] <= 65536
        for row in resources
    ), resources
    flags = torch.zeros(
        FLAGS_BYTES // 4, device=prepared.flags.device, dtype=torch.int32
    )
    assert FLAGS_BYTES == 16448 and flags.data_ptr() % 8 == 0
    bound = tuple(item._replace(flags=flags.data_ptr()) for item in prepared.hc_layers)
    assert len(bound) == 48 and all(
        item.pack(0, 0, 0)[18] == flags.data_ptr() for item in bound
    )
    previous_bytes = prepared.flags.numel() * prepared.flags.element_size()
    # All potentially failing work completed above; publish the whole binding
    # set before returning the prepared object to the single-threaded runner.
    prepared.flags, prepared.hc_layers, prepared.hc_kernel = flags, bound, kernel
    prepared.metadata.update(
        hc_source=v7.SOURCE,
        hc_resources=resources,
        hc_replaced_source=v5.SOURCE,
        hc_coherence_variant=7,
        hc_flags_bytes=FLAGS_BYTES,
        hc_previous_flags_bytes=previous_bytes,
        hc_router_value_pairs=ROUTER_VALUES,
        hc_selection_ready_offset=SELECT_READY_OFFSET,
        hc_publication="BF16-rounded FP32 value/tag; per-thread pair poll; selection drained publish",
    )
    return prepared
