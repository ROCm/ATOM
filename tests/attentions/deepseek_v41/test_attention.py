# SPDX-License-Identifier: MIT
"""Accepted model math over paged/relocated state versus private P04 caches."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
    INDEX_FP8_SCALE_FMT,
    MAIN_FP4,
)
from atom.model_ops.blockscale import quantize_fp4
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.models.deepseek_v41.attention import Attention
from atom.models.deepseek_v41.config import build_attention_topology
from tests.attentions.deepseek_v41.helpers import geometry


def _where(location, text):
    return f"{location}\n{text}"


def round_to_fp8_row(value):
    """`indexer_k_quant_and_cache`'s arithmetic, in torch.

    One scale per row, rounded to a power of two, and the value cast to E4M3
    against it. Written out rather than called through `quantize_fp8` because
    what has to match is this kernel's rule, not ATOM's own -- the row is the
    quantization block, the floor on the amax is the kernel's, and the power-
    of-two step is `INDEX_FP8_SCALE_FMT`.
    """
    amax = value.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    if INDEX_FP8_SCALE_FMT == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    stored = (value.float() * scale.reciprocal()).to(torch.float8_e4m3fn)
    return (stored.float() * scale).to(torch.bfloat16)


class QuantizedOracle(EagerAttentionCache):
    """A private cache that stores on the grid the format under test does.

    The oracle has to store what the format under test stores. This cache is
    BF16 and would keep the unrounded value, so a quantized paged pool would
    differ from it by the grid rather than by anything the paged bookkeeping
    did. Rounding here keeps the comparison bit-exact and leaves the
    quantization with the storage format instead of putting it in the model.
    """

    def __init__(self, *args, main_dtype, index_dtype, **kwargs):
        super().__init__(*args, **kwargs)
        # `index_dtype` is the grid its rows are rounded to, which is also what
        # the model's own QAT follows. `packed` and `scores_paged` stay as the
        # base class has them: this cache still runs the eager kernels and
        # still has no pages to score over.
        self.main_dtype, self.index_dtype = main_dtype, index_dtype

    def compress(self, owner, compressor, values, scores, step, rope):
        latent, rows = super().compress(owner, compressor, values, scores, step, rope)
        if latent is not None and self.main_dtype == "fp4":
            self.main[owner][:, rows] = quantize_fp4(
                self.main[owner][:, rows], dequantize=True, **MAIN_FP4
            )
        return latent, rows

    def write_index(self, owner, step, rows, index, ratio):
        rounded = (
            round_to_fp8_row(index)
            if self.index_dtype == "fp8"
            else quantize_fp4(index, dequantize=True)
        )
        super().write_index(owner, step, rows, rounded, ratio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize(
    "main_dtype,index_dtype", [("bf16", "bf16"), ("bf16", "fp8"), ("fp4", "fp4")]
)
@pytest.mark.parametrize("tie", ["small_position", "large_position"])
def test_attention_math_and_odd_tail_survive_exact_checkpoint(
    small_config, single_rank, tie, main_dtype, index_dtype
):
    torch.manual_seed(711)
    config = small_config
    config.index_topk_tie_break = tie
    # The schedule below has to outrun the coarsest PAGE under test, and
    # the fixture's own cap is shorter than that.
    config.max_position_embeddings = 128
    # The paged scorer's gluon kernel refuses to compile under
    # `index_n_heads * index_head_dim = 4096`, which is exactly what the
    # published config has. Every arm takes it so the three stay comparable.
    config.index_head_dim = 128
    topology = build_attention_topology(config)
    with torch.device("cuda"):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layers = [Attention(config, spec) for spec in topology]
        finally:
            torch.set_default_dtype(previous)
        rope = RotaryEmbedding(32, 128, base=10000)
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
    # An FP8 index plane addresses whole tiles, so its PAGE has to hold a
    # whole number of them for the ratio-2 owner as well.
    block = 32 if index_dtype == "fp8" else 4
    geo = replace(
        geometry(config, block=block),
        packed=main_dtype == "fp4",
        index_dtype=index_dtype,
    )
    paged = PagedAttentionCache(geo, 40, 3, "cuda")
    private = (
        EagerAttentionCache(config, topology, 1, 128, "cuda")
        if (main_dtype, index_dtype) == ("bf16", "bf16")
        else QuantizedOracle(
            config,
            topology,
            1,
            128,
            "cuda",
            main_dtype=main_dtype,
            index_dtype=index_dtype,
        )
    )
    spec = PagedStateCheckpointSpec(
        geo.paged_bytes, geo.state_bytes, geo.layout_id, geo.state_bytes
    )
    copies = StateCopies(paged, spec, 3)
    # Scattered and non-monotonic, and enough of them that the schedule below
    # spans several at the coarsest PAGE under test -- a request that fits in
    # one page cannot tell a page-order mistake from a correct addressing.
    blocks = (30, 21, 25, 24, 26, 22, 27, 20, 31, 19, 29, 18, 28, 17, 23, 16, 15, 14)
    slot = 2
    for n, length in enumerate((3, 1, 17, 1, 42, 5)):
        position = private.position
        span = RequestSpan(51, position, 0, length, slot, blocks)
        step = paged.begin_step([span])
        history = paged.prepare_state(step)
        eager_step = private.begin_step(position, length, 1)
        for depth, layer in enumerate(layers):
            x = torch.randn(1, length, 64, dtype=torch.bfloat16, device="cuda")
            with torch.inference_mode():
                expected = layer(x, private, eager_step, rope)
                actual = layer(x, paged, step, rope)
            # Name the step and the layer: a bare tensor mismatch here says
            # only that the paged path drifted somewhere in four steps and
            # five layers, which is most of the search.
            where = (
                f"step {n} (position {position}, length {length}), layer {depth} "
                f"{topology[depth].mode.value} ratio={topology[depth].ratio}"
            )
            torch.testing.assert_close(
                actual, expected, rtol=0, atol=0, msg=partial(_where, where)
            )
        paged.advance_cursor(step, history)
        private.finish_step(eager_step)
        if n == 0:
            units = tuple(range(spec.units_per_checkpoint))
            copies.execute(
                [CheckpointStoreOp(slot, units, spec.image_bytes, spec.layout_id)],
                [CheckpointRestoreOp(0, units, spec.image_bytes, spec.layout_id)],
            )
            slot = 0
    np.testing.assert_array_equal(paged.cursor[slot].cpu().numpy(), [69, -1, -1, -1])
