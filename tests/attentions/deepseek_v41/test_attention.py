# SPDX-License-Identifier: MIT
"""Accepted model math over paged/relocated state versus private P04 caches."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.models.deepseek_v41.attention import Attention
from atom.models.deepseek_v41.config import build_attention_topology
from tests.attentions.deepseek_v41.helpers import geometry

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tie", ["small_position", "large_position"])
def test_attention_math_and_odd_tail_survive_exact_checkpoint(
    small_config, single_rank, tie, packed
):
    torch.manual_seed(711)
    config = small_config
    config.index_topk_tie_break = tie
    topology = build_attention_topology(config)
    with torch.device("cuda"):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layers = [Attention(config, spec) for spec in topology]
        finally:
            torch.set_default_dtype(previous)
        rope = RotaryEmbedding(32, 32, base=10000)
        for layer in layers:
            for name, parameter in layer.named_parameters():
                if parameter.dtype == torch.float8_e8m0fnu:
                    value = torch.full(parameter.shape, 2**-5).to(parameter.dtype)
                elif parameter.dtype == torch.float8_e4m3fn:
                    value = (torch.randn(parameter.shape) * 8).to(parameter.dtype)
                elif name.endswith("norm.weight"):
                    value = torch.ones(parameter.shape, dtype=parameter.dtype)
                else:
                    value = (torch.randn(parameter.shape) * 0.1).to(parameter.dtype)
                parameter.data.copy_(value)
            # Parent first, as the model's own traversal does: the layer's hook
            # dequantizes wo_a and cancels the FP8 post-load steps that would
            # otherwise shuffle a matrix `torch.einsum` then reads.
            for module in layer.modules():
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
    geo = replace(geometry(config), packed=packed)
    paged = PagedAttentionCache(geo, 40, 3, "cuda")
    private = EagerAttentionCache(config, topology, 1, 32, "cuda")
    spec = PagedStateCheckpointSpec(
        geo.page_bytes, geo.state_bytes, geo.layout_id, geo.state_bytes
    )
    copies = StateCopies(paged, spec, 3)
    blocks = (30, 21, 25, 24, 26, 22, 27, 20)
    slot = 2
    for n, length in enumerate((3, 1, 17, 1)):
        position = private.position
        span = RequestSpan(51, position, 0, length, slot, blocks)
        step = paged.begin_step([span])
        history = paged.prepare_state(step)
        eager_step = private.begin_step(position, length, 1)
        for layer in layers:
            x = torch.randn(1, length, 64, dtype=torch.bfloat16, device="cuda")
            with torch.inference_mode():
                expected = layer(x, private, eager_step, rope)
                actual = layer(x, paged, step, rope)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        paged.finish_step(step, history)
        private.finish_step(eager_step)
        if n == 0:
            units = tuple(range(spec.units_per_checkpoint))
            copies.execute(
                [CheckpointStoreOp(slot, units, spec.image_bytes, spec.layout_id)],
                [CheckpointRestoreOp(0, units, spec.image_bytes, spec.layout_id)],
            )
            slot = 0
    np.testing.assert_array_equal(paged.cursor[slot].cpu().numpy(), [22, -1, -1, -1])
