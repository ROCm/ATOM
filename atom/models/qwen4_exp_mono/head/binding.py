# SPDX-License-Identifier: MIT
"""Prepare the global certificate head while retaining original BF16 refinement."""

import json

import torch

from .global_certificate import HeadScreenGlobalCertificate
from .packed import HeadScreenW4


def prepare(current):
    """Bind once after draft weights load, before warmup or graph capture."""
    weight = current.weight
    assert weight.shape == (248320, 2560) and weight.dtype == torch.bfloat16
    assert weight.is_contiguous() and not torch.cuda.is_current_stream_capturing()
    scalar = HeadScreenW4(weight)
    selected = HeadScreenGlobalCertificate(scalar)
    assert selected.weight.data_ptr() == weight.data_ptr()
    assert selected.packed.data_ptr() == scalar.packed.data_ptr()
    assert bool(
        (selected.we_global <= 2.0**40).all()
        & (selected.wq_global <= 2.0**40).all()
        & (selected.wnorm <= 2.0**40).all()
    )
    example = (
        torch.linspace(-1.0, 1.0, 2560, device=weight.device)
        .to(torch.bfloat16)
        .view(1, -1)
    )
    out = torch.empty((1,), dtype=torch.int32, device=weight.device)
    for owner in (current, selected):
        result = owner.run(example, out=out)
        assert result.data_ptr() == out.data_ptr()
    torch.cuda.synchronize()
    resources = {}
    for name, binary in [
        ("input_certificate", selected.last_stats),
        ("screen", selected.last_screen),
        ("anchor", selected.last_anchor),
        ("refine", selected.last_refine),
    ]:
        assert binary.n_spills == 0 and binary.metadata.shared <= 65536
        resources[name] = {
            "vgpr": binary.n_regs,
            "spills": binary.n_spills,
            "lds": binary.metadata.shared,
        }
    # Retain the same owners and warmup allocations as the measured version.
    selected.previous = current
    print(
        "QWEN_GLOBAL_HEAD_BOUND "
        + json.dumps(
            {
                "enabled": True,
                "selected": type(selected).__name__,
                "rows": 1,
                "group": 64,
                "original_bf16_weight_retained": True,
                "packed_weight_shared": True,
                "resources": resources,
            }
        ),
        flush=True,
    )
    return selected
