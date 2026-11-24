# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2024-2026 ATOM Project Contributors

"""The DSpark context projection, `main_norm(main_proj(concat(taps)))`.

`write_context_kv` normalizes the projected context with `main_norm` to bf16
and lets the stages' fused `wkv` GEMM quantize that, the original model's
rounding. The taps are quantized where they lie rather than joined first, so
the claim here is that the taps and the same rows joined into one block (a PP
relay) project to the same bytes.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises AITER GPU kernels; needs a real GPU", allow_module_level=True
    )

pytest.importorskip("aiter", reason="RMSNorm is an AITER kernel")

from atom.model_ops.layernorm import RMSNorm
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark

DEV = "cuda"
HIDDEN = 5120  # config.hidden_size
EPS = 1e-20  # config.rms_norm_eps
TAPS = 3  # dspark_target_layer_ids


class _Draft:
    """The part of `DeepseekV41DSpark` the context projection reads."""

    project_context = DeepseekV41DSpark.project_context
    _project_taps = DeepseekV41DSpark._project_taps

    def __init__(self, norm):
        self.mtp = [SimpleNamespace(main_proj=_main_proj, main_norm=norm)]


def _main_proj(x, x_scale):
    """Stands in for `main_proj`: takes the fp8 input and its scales, as the
    real one is handed them, and sums the taps' dequantized slices."""
    scale = x_scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=-1)
    values = (x.float() * scale).unflatten(-1, (TAPS, HIDDEN))
    return values.sum(-2).to(torch.bfloat16)


_main_proj.runs_bf16 = False  # an FP8 `main_proj`


@pytest.mark.parametrize("width", [1, 37, 384])
def test_context_projection_takes_taps_as_the_joined_block(single_rank, width):
    gen = torch.Generator(device=DEV).manual_seed(200 + width)
    norm = RMSNorm(HIDDEN, EPS).to(device=DEV, dtype=torch.bfloat16)
    norm.weight.data.copy_(
        0.5 + torch.rand(HIDDEN, generator=gen, device=DEV, dtype=torch.float32)
    )
    draft = _Draft(norm)
    # A power of two per 32-column group, so groups pick different scales.
    group_gain = 2.0 ** torch.randint(
        -4, 5, (HIDDEN // 32,), generator=gen, device=DEV
    ).repeat_interleave(32)
    taps = [
        (torch.randn(width, HIDDEN, generator=gen, device=DEV) * group_gain).to(
            torch.bfloat16
        )
        for _ in range(TAPS)
    ]
    with torch.inference_mode():
        hidden = draft.project_context(taps)
        joined_hidden = draft.project_context([torch.cat(taps, dim=-1)])
    assert hidden.shape == (1, width, HIDDEN)
    assert torch.equal(hidden, joined_hidden)
