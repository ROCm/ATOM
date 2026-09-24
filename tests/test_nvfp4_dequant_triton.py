"""The Triton NVFP4 decoder must reproduce the torch reference bit for bit.

`dequantize_nvfp4` runs once per NVFP4 weight at load time and feeds the MXFP4
re-quantizer, so a wrong nibble order, a misindexed block scale or a dropped
global scale shows up only as degraded accuracy much later. The torch path it
replaced (`_dequantize_nvfp4_torch`) is kept as the oracle here.
"""

import pytest
import torch

# Order matters: importing triton without an active GPU driver raises rather
# than failing the import, so the device check has to come first.
if not torch.cuda.is_available():
    pytest.skip(
        "drives a Triton kernel on real tensors; needs a real GPU",
        allow_module_level=True,
    )

pytest.importorskip("triton")

from atom.quantization.quark.utils import (
    _dequantize_nvfp4_torch,
    dequantize_nvfp4,
)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers so signed zeros are compared, not just values."""
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _assert_matches_reference(weight, scale, global_scale, out_dtype, high_first):
    got = dequantize_nvfp4(
        weight, scale, global_scale, out_dtype=out_dtype, high_nibble_first=high_first
    )
    ref = _dequantize_nvfp4_torch(weight, scale, global_scale, out_dtype, high_first)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    # NaN block scales (E4M3 0x7F/0xFF) propagate through both paths but the
    # hardware convert does not preserve the payload bits, so compare those
    # positions as "NaN on both sides" instead of bit for bit.
    nan = got.isnan()
    assert torch.equal(nan, ref.isnan())
    assert torch.equal(_bits(got)[~nan], _bits(ref)[~nan])
    return got


def _random_nvfp4(rows: int, k: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randint(
        0, 256, (rows, k // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    # *20 pushes part of the range past E4M3 max so saturation is covered too.
    scale = (torch.randn(rows, k // 16, device="cuda", generator=gen) * 20).to(
        torch.float8_e4m3fn
    )
    return weight, scale


# Every NVFP4 weight carries a global scale, so the shape sweeps below use a
# plain per-row one and the dedicated tests cover the other layouts.
def _global_scale(rows: int) -> torch.Tensor:
    return torch.linspace(0.25, 2.0, rows, device="cuda").view(-1, 1)


@pytest.mark.parametrize("rows", [1, 3, 8, 17, 129])
@pytest.mark.parametrize("k", [16, 48, 256, 272, 4096])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_matches_torch_reference_across_shapes(rows, k, out_dtype):
    weight, scale = _random_nvfp4(rows, k)
    _assert_matches_reference(weight, scale, _global_scale(rows), out_dtype, False)


@pytest.mark.parametrize("high_nibble_first", [False, True])
def test_nibble_order_is_honoured(high_nibble_first):
    weight, scale = _random_nvfp4(8, 128)
    g = _global_scale(8)
    _assert_matches_reference(weight, scale, g, torch.float32, high_nibble_first)
    # The two orders must actually differ, or this proves nothing.
    swapped = dequantize_nvfp4(weight, scale, g, high_nibble_first=True)
    straight = dequantize_nvfp4(weight, scale, g, high_nibble_first=False)
    assert not torch.equal(swapped, straight)


@pytest.mark.parametrize("global_kind", ["scalar", "one_element_2d", "per_row"])
def test_global_scale_layouts(global_kind):
    rows, k = 64, 256
    weight, scale = _random_nvfp4(rows, k)
    if global_kind == "scalar":
        # What the MoE path passes: w13_weight_scale_2[expert, shard], 0-dim.
        global_scale = torch.tensor(0.375, device="cuda")
    elif global_kind == "one_element_2d":
        global_scale = torch.tensor([[0.375]], device="cuda")
    else:
        # What the Linear path passes: one value per output row, shape (N, 1).
        global_scale = torch.linspace(-2.0, 2.0, rows, device="cuda").view(-1, 1)
    _assert_matches_reference(weight, scale, global_scale, torch.float32, False)


def test_global_scale_is_applied_on_top_of_the_block_scale():
    # Low nibble is the even logical value: byte 0x57 decodes to [6, 3].
    # 2.0 * 3.0 = 6.0 is not 1.0, so dropping either scale changes the result.
    packed = torch.full((1, 8), 0x57, dtype=torch.uint8, device="cuda")
    block_scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn, device="cuda")

    out = dequantize_nvfp4(packed, block_scale, torch.tensor(3.0, device="cuda"))

    assert torch.equal(
        out, torch.tensor([[36.0, 18.0] * 8], dtype=torch.float32, device="cuda")
    )


def test_every_byte_and_block_scale_pattern():
    """Exhaustive: all 256 packed byte values x all 256 E4M3 scale encodings."""
    rows = packed_cols = 256
    weight = (
        torch.arange(packed_cols, dtype=torch.uint8, device="cuda")
        .expand(rows, packed_cols)
        .contiguous()
    )
    scale = (
        torch.arange(rows, dtype=torch.uint8, device="cuda")
        .view(torch.float8_e4m3fn)[:, None]
        .expand(rows, packed_cols // 8)
        .contiguous()
    )
    global_scale = torch.linspace(-3.0, 3.0, rows, device="cuda").view(-1, 1)
    for out_dtype in (torch.float32, torch.bfloat16):
        for high_first in (False, True):
            _assert_matches_reference(
                weight, scale, global_scale, out_dtype, high_first
            )


def test_non_contiguous_and_row_sliced_inputs():
    weight = torch.randint(0, 256, (16, 128), dtype=torch.uint8, device="cuda")
    scale = torch.randn(16, 16, device="cuda").to(torch.float8_e4m3fn)
    # Column slice: rows keep their original (wider) stride.
    _assert_matches_reference(
        weight[:, :64], scale[:, :8], _global_scale(16), torch.float32, False
    )
    # Row stride of 2, as a strided view would hand the kernel.
    _assert_matches_reference(
        weight[::2], scale[::2], _global_scale(8), torch.float32, False
    )


def test_empty_weight_returns_empty_without_launching():
    weight = torch.empty(0, 8, dtype=torch.uint8, device="cuda")
    scale = torch.empty(0, 1, dtype=torch.float8_e4m3fn, device="cuda")
    out = dequantize_nvfp4(weight, scale, torch.tensor(0.5, device="cuda"))
    assert out.shape == (0, 16) and out.dtype == torch.float32


@pytest.mark.parametrize("default_dtype", [torch.bfloat16, torch.float16])
def test_out_dtype_defaults_to_the_model_dtype(default_dtype):
    """ModelRunner installs the model dtype before loading; decode follows it.

    Decoding to FP32 would be thrown away -- `quant_mxfp4_online_even` casts
    anything wider down to BF16 -- so the default has to track the model dtype
    rather than being pinned. Emitting it directly must round exactly once,
    i.e. match FP32-then-cast bit for bit.
    """
    weight, scale = _random_nvfp4(64, 256)
    global_scale = _global_scale(64)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        got = dequantize_nvfp4(weight, scale, global_scale)
    finally:
        torch.set_default_dtype(previous)

    assert got.dtype == default_dtype
    wide = dequantize_nvfp4(weight, scale, global_scale, out_dtype=torch.float32)
    assert torch.equal(got.view(torch.int16), wide.to(default_dtype).view(torch.int16))


def test_cuda_and_cpu_paths_agree():
    weight, scale = _random_nvfp4(32, 128)
    global_scale = torch.rand(32, 1, device="cuda") + 0.5
    gpu = dequantize_nvfp4(weight, scale, global_scale)
    cpu = dequantize_nvfp4(weight.cpu(), scale.cpu(), global_scale.cpu())
    assert torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rejects_the_same_bad_inputs_on_both_devices(device):
    """Validation lives in the shared helpers, so it cannot drift per device."""
    weight = torch.zeros(4, 8, dtype=torch.uint8, device=device)
    scale = torch.zeros(4, 1, dtype=torch.float8_e4m3fn, device=device)

    good = torch.ones(4, 1, device=device)

    with pytest.raises(TypeError, match="uint8 storage"):
        dequantize_nvfp4(weight.view(torch.int8), scale, good)
    # Raw scale bytes and bit-reinterpreted FNUZ scales both decode to finite,
    # plausible values that agree across the two paths.
    for bad_scale in (scale.view(torch.uint8), scale.view(torch.float8_e4m3fnuz)):
        with pytest.raises(TypeError, match="float8_e4m3fn"):
            dequantize_nvfp4(weight, bad_scale, good)
    with pytest.raises(ValueError, match="divisible by"):
        dequantize_nvfp4(weight[:, :3], scale, good)
    with pytest.raises(ValueError, match="does not match"):
        dequantize_nvfp4(weight, scale[:, :1].expand(4, 2).contiguous(), good)
    with pytest.raises(ValueError, match="scalar or one value per weight row"):
        dequantize_nvfp4(weight, scale, torch.zeros(3, 1, device=device))
    # NVFP4 without a global scale is not NVFP4; neither path may guess one.
    with pytest.raises(ValueError, match="two-level format"):
        dequantize_nvfp4(weight, scale, None)
