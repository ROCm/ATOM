# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Dense FP8 with 2-D [32, 32] ue8m0 block scales (DeepSeek-V4.1-Flash).

The checkpoint ships one power-of-two ``float8_e8m0fnu`` scale per 32x32 weight
tile, so a weight ``[out, in]`` carries a scale ``[out/32, in/32]``. ATOM keeps
``QuantType.per_1x128`` for it (the enum lives in AITER C++) and carries the
real grid on ``LayerQuantConfig.block_n / block_k / scale_fmt``.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.config import QuantizationConfig
from atom.quant_spec import LayerQuantConfig

QuantType = pytest.importorskip("aiter").QuantType

# (out, in) of every dense FP8 GEMM weight in DeepSeek-V4.1-Flash.
REAL_SHAPES = [
    (1280, 5120),  # attn.wq_a
    (32768, 1280),  # attn.wq_b
    (512, 5120),  # attn.wkv
    (5120, 8192),  # attn.wo_b
    (2304, 5120),  # ffn.shared_experts.w1 / w3
    (5120, 2304),  # ffn.shared_experts.w2
    (25600, 6144),  # engram.wkv
]

V41_FLASH_QUANT_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _quant_config(quantization_config):
    return QuantizationConfig(
        SimpleNamespace(
            torch_dtype=torch.bfloat16, quantization_config=quantization_config
        )
    )


# ---------------------------------------------------------------------------
# 1. Spec parsing
# ---------------------------------------------------------------------------


class TestSpecParsing:
    def test_v41_flash_config(self):
        spec = _quant_config(V41_FLASH_QUANT_CONFIG).global_quant_config
        # No new enum member: the grid rides on the spec, not on quant_type.
        assert spec.quant_type == QuantType.per_1x128
        assert spec.quant_dtype == torch.float8_e4m3fn
        assert spec.is_dynamic is True
        assert (spec.block_n, spec.block_k) == (32, 32)
        assert spec.scale_fmt == "ue8m0"
        assert spec.block_scale_grid == (32, 32)
        assert spec.has_fine_block_scale
        assert spec.is_ue8m0_block_scale

    def test_default_spec_is_legacy_128_grid(self):
        spec = LayerQuantConfig()
        assert (spec.block_n, spec.block_k) == (None, None)
        assert spec.scale_fmt is None
        assert spec.block_scale_grid == (128, 128)
        assert not spec.has_fine_block_scale
        assert not spec.is_ue8m0_block_scale

    @pytest.mark.parametrize(
        "wbs, expected_type",
        [
            ([128, 128], QuantType.per_1x128),
            ([1, 128], QuantType.per_1x32),  # pre-existing [1, K] -> MXFP8 rule
            ([1, 32], QuantType.per_1x32),
        ],
    )
    def test_existing_block_sizes_unchanged(self, wbs, expected_type):
        spec = _quant_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": wbs,
            }
        ).global_quant_config
        assert spec.quant_type == expected_type
        assert spec.block_scale_grid == (128, 128)
        assert not spec.has_fine_block_scale
        assert not spec.is_ue8m0_block_scale

    def test_ue8m0_on_a_128_grid_stays_on_the_legacy_path(self):
        # scale_fmt alone must not switch the scale dtype: the 128x128 grid
        # keeps its env-driven fp32/e8m0 choice.
        spec = _quant_config(
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
            }
        ).global_quant_config
        assert spec.scale_fmt == "ue8m0"
        assert not spec.has_fine_block_scale
        assert not spec.is_ue8m0_block_scale

    def test_fbgemm_and_bare_fp8_unchanged(self):
        fbgemm = _quant_config(
            {"quant_method": "fbgemm_fp8", "activation_scheme": "static"}
        ).global_quant_config
        assert fbgemm.quant_type == QuantType.per_Tensor
        assert fbgemm.block_scale_grid == (128, 128)
        assert not fbgemm.has_fine_block_scale

        bare = _quant_config(
            {"quant_method": "fp8", "activation_scheme": "static"}
        ).global_quant_config
        assert bare.quant_type == QuantType.per_Tensor
        assert not bare.has_fine_block_scale

    def test_compressed_tensors_unchanged(self):
        spec = _quant_config(
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "g0": {
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "strategy": "channel",
                        }
                    }
                },
                "ignore": ["lm_head"],
            }
        ).global_quant_config
        assert spec.quant_type == QuantType.per_Token
        assert not spec.has_fine_block_scale

    def test_mxfp4_unchanged(self):
        spec = _quant_config(
            {"quant_method": "mxfp4", "weight_dtype": "fp4_e2m1"}
        ).global_quant_config
        assert spec.quant_type == QuantType.per_1x32
        assert not spec.has_fine_block_scale


# ---------------------------------------------------------------------------
# 2. Scale allocation and sharding
# ---------------------------------------------------------------------------


class _FakeTPGroup:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank_in_group = rank


@pytest.fixture
def linear_mod(monkeypatch):
    """`atom.model_ops.linear` with a fake TP group installed per test."""
    from atom.model_ops import linear

    def use_tp(world_size, rank):
        monkeypatch.setattr(
            linear, "get_tp_group", lambda: _FakeTPGroup(world_size, rank)
        )

    linear.use_tp = use_tp
    try:
        yield linear
    finally:
        del linear.use_tp


TP_SIZES = [1, 2, 4, 8]


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("out_size, in_size", REAL_SHAPES)
def test_column_parallel_scale_shape(linear_mod, tp_size, out_size, in_size):
    linear_mod.use_tp(tp_size, tp_size - 1)
    qc = _quant_config(V41_FLASH_QUANT_CONFIG)
    layer = linear_mod.ColumnParallelLinear(
        in_size, out_size, quant_config=qc, prefix="attn.wq_a"
    )
    # Column parallel splits dim 0, so the scale splits dim 0 by out/32/tp.
    assert tuple(layer.weight.shape) == (out_size // tp_size, in_size)
    assert tuple(layer.weight_scale.shape) == (
        out_size // tp_size // 32,
        in_size // 32,
    )
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu
    assert (layer.block_scale_n, layer.block_scale_k) == (32, 32)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("out_size, in_size", REAL_SHAPES)
def test_row_parallel_scale_shape(linear_mod, tp_size, out_size, in_size):
    linear_mod.use_tp(tp_size, tp_size - 1)
    qc = _quant_config(V41_FLASH_QUANT_CONFIG)
    layer = linear_mod.RowParallelLinear(
        in_size, out_size, quant_config=qc, prefix="attn.wo_b"
    )
    # Row parallel splits dim 1, so the scale splits dim 1 by in/32/tp.
    assert tuple(layer.weight.shape) == (out_size, in_size // tp_size)
    assert tuple(layer.weight_scale.shape) == (
        out_size // 32,
        in_size // tp_size // 32,
    )
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_column_parallel_shard_follows_the_weight(linear_mod, tp_size):
    """Every rank must load exactly its own slab of scale rows."""
    out_size, in_size = 1280, 5120
    full_scale = (
        torch.arange(out_size // 32 * (in_size // 32), dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .view(out_size // 32, in_size // 32)
        .view(torch.float8_e8m0fnu)
    )
    seen = []
    for rank in range(tp_size):
        linear_mod.use_tp(tp_size, rank)
        layer = linear_mod.ColumnParallelLinear(
            in_size,
            out_size,
            quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
            prefix="attn.wq_a",
        )
        layer.weight_scale.weight_loader(layer.weight_scale, full_scale)
        seen.append(layer.weight_scale.data.view(torch.uint8).clone())
    rebuilt = torch.cat(seen, dim=0)
    assert torch.equal(rebuilt, full_scale.view(torch.uint8))


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_row_parallel_shard_follows_the_weight(linear_mod, tp_size):
    out_size, in_size = 5120, 8192
    full_scale = (
        torch.arange(out_size // 32 * (in_size // 32), dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .view(out_size // 32, in_size // 32)
        .view(torch.float8_e8m0fnu)
    )
    seen = []
    for rank in range(tp_size):
        linear_mod.use_tp(tp_size, rank)
        layer = linear_mod.RowParallelLinear(
            in_size,
            out_size,
            quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
            prefix="attn.wo_b",
        )
        layer.weight_scale.weight_loader(layer.weight_scale, full_scale)
        seen.append(layer.weight_scale.data.view(torch.uint8).clone())
    rebuilt = torch.cat(seen, dim=1)
    assert torch.equal(rebuilt, full_scale.view(torch.uint8))


def test_merged_column_parallel_scale_offsets(linear_mod):
    """A fused gate/up scale must be split on 32-row, not 128-row, boundaries."""
    linear_mod.use_tp(1, 0)
    layer = linear_mod.MergedColumnParallelLinear(
        5120,
        [2304, 2304],
        quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
        prefix="ffn.shared_experts.gate_up",
    )
    assert tuple(layer.weight_scale.shape) == (2 * 2304 // 32, 5120 // 32)
    for shard_id in (0, 1):
        shard = torch.full(
            (2304 // 32, 5120 // 32), 127 + shard_id, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        layer.weight_loader(layer.weight_scale, shard, shard_id)
    data = layer.weight_scale.data.view(torch.uint8)
    assert torch.all(data[: 2304 // 32] == 127)
    assert torch.all(data[2304 // 32 :] == 128)


def test_non_divisible_shard_is_rejected(linear_mod):
    """out/32 must divide by TP, otherwise a scale row would straddle ranks."""
    linear_mod.use_tp(8, 0)
    with pytest.raises(AssertionError, match="block scale needs the per-rank"):
        linear_mod.ColumnParallelLinear(
            5120,
            32 * 8 + 32,
            quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
            prefix="odd",
        )


def test_legacy_128_grid_allocation_is_unchanged(linear_mod):
    linear_mod.use_tp(1, 0)
    qc = _quant_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    layer = linear_mod.ColumnParallelLinear(5120, 1280, quant_config=qc, prefix="w")
    assert tuple(layer.weight_scale.shape) == (1280 // 128, 5120 // 128)
    assert (layer.block_scale_n, layer.block_scale_k) == (128, 128)
    assert not layer.has_fine_block_scale


# ---------------------------------------------------------------------------
# 3. Numerics on GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "out_size, in_size, tokens", [(512, 5120, 64), (1280, 5120, 17)]
)
def test_gpu_matches_dequant_matmul(linear_mod, out_size, in_size, tokens):
    from aiter.ops.triton.quant.quant_fp8_blockwise import quant_fp8_blockwise

    linear_mod.use_tp(1, 0)
    torch.manual_seed(0)
    layer = linear_mod.ColumnParallelLinear(
        in_size,
        out_size,
        quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
        prefix="attn.wkv",
    ).to("cuda")
    w_q = (torch.randn(out_size, in_size, device="cuda") * 40).to(torch.float8_e4m3fn)
    # e8m0 exponents around 2^0 (bias 127) so the dequantized weight stays in range.
    w_s = torch.randint(
        120, 134, (out_size // 32, in_size // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    layer.weight.data = w_q
    layer.weight_scale.data = w_s

    x = torch.randn(tokens, in_size, device="cuda", dtype=torch.bfloat16)
    y = layer(x)

    x_q, x_s = quant_fp8_blockwise(
        x.contiguous(),
        block_size=32,
        fp8_max=448.0,
        quant_dtype=torch.float8_e4m3fn,
        scale_fmt="ue8m0",
    )
    x_deq = x_q.float() * x_s.float().repeat_interleave(32, dim=1)
    w_deq = w_q.float() * w_s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    ref = x_deq @ w_deq.T

    assert y.shape == (tokens, out_size)
    assert y.dtype == torch.bfloat16
    rel = (y.float() - ref).abs().max() / ref.abs().max()
    assert rel < 1e-2, f"relative error {rel:.3g} too large"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_row_parallel_keeps_the_weight_unshuffled(linear_mod):
    """The fine GEMM consumes a plain (N, K) weight, and 3-D input round-trips."""
    linear_mod.use_tp(1, 0)
    torch.manual_seed(0)
    out_size, in_size = 5120, 2304
    layer = linear_mod.RowParallelLinear(
        in_size,
        out_size,
        quant_config=_quant_config(V41_FLASH_QUANT_CONFIG),
        prefix="ffn.shared_experts.w2",
    ).to("cuda")
    layer.weight.data = (torch.randn(out_size, in_size, device="cuda") * 40).to(
        torch.float8_e4m3fn
    )
    layer.weight_scale.data = torch.randint(
        124, 131, (out_size // 32, in_size // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    layer.process_weights_after_loading()

    assert not getattr(layer.weight, "is_shuffled", False)
    assert tuple(layer.weight.shape) == (out_size, in_size)
    assert not layer.supports_out()

    for shape in [(37, in_size), (3, 11, in_size)]:
        y = layer(torch.randn(*shape, device="cuda", dtype=torch.bfloat16))
        assert tuple(y.shape) == (*shape[:-1], out_size)
        assert torch.isfinite(y).all()
