# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Blockwise2DFP8Param / Blockwise2DFP8Gathered: the FSDP2 param-storage wrappers
for a frozen blockwise2d base weight.

The param holds a BF16 master (FSDP2 shards it like any BF16 param); the
all-gather extension quantizes the local shard to FP8 + 2D scale and the gathered
form (Blockwise2DFP8Gathered) carries FP8 + scale for the forward GEMM."""

import pytest
import torch

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fp8():
    return (
        torch.float8_e4m3fnuz
        if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
        else torch.float8_e4m3fn
    )


@_CUDA
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_blockwise2d_fp8_param_allgather_roundtrip(world_size):
    """Simulate FSDP2 dim-0 sharding: shard the BF16 master, run pre_all_gather
    per shard (quantizes the shard), concat (= all-gather), run post_all_gather.
    The gathered FP8 must equal a single full-weight blockwise2d quant (per-shard
    quant is exact because dim-0 shards never split a 128-row tile)."""
    from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from atomrl.lumen.quantize.comm_tensor import (
        Blockwise2DFP8Gathered,
        Blockwise2DFP8Param,
    )

    torch.manual_seed(0)
    block = 128
    N, K = block * world_size * 2, block * 3  # N % (block*world_size) == 0
    w = (torch.randn(N, K) * 0.05).cuda().to(torch.bfloat16)
    fp8_ref, scale_ref = _quant_blockwise2d_weight(w, _fp8(), block)

    rows = N // world_size
    gathered_fp8, gathered_scale, meta, last = [], [], None, None
    for r in range(world_size):
        shard = Blockwise2DFP8Param(
            w[r * rows : (r + 1) * rows].contiguous(), _fp8(), block
        )
        (fp8_s, scale_s), meta = shard.fsdp_pre_all_gather(mesh=None)
        gathered_fp8.append(fp8_s)
        gathered_scale.append(scale_s)
        last = shard

    full_fp8 = torch.cat(gathered_fp8, dim=0)
    full_scale = torch.cat(gathered_scale, dim=0)
    result = last.fsdp_post_all_gather((full_fp8, full_scale), meta, torch.bfloat16)[0]

    assert isinstance(result, Blockwise2DFP8Gathered)
    assert result._fp8.dtype == _fp8(), "gathered weight must stay FP8 (no dequant)"
    assert result.shape == (N, K)
    # Per-shard quant == full-weight quant (tiles don't cross dim-0 shard boundary).
    assert torch.equal(result._fp8.view(torch.uint8), fp8_ref.view(torch.uint8))
    assert torch.equal(result._scale, scale_ref)


@_CUDA
def test_blockwise2d_gathered_dequant_matches_weight():
    """Dequantizing the gathered FP8 + 2D scale must approximate the BF16 weight."""
    from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight
    from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Gathered
    from conftest import compute_snr  # type: ignore

    torch.manual_seed(0)
    block, N, K = 128, 256, 384
    w = (torch.randn(N, K) * 0.05).cuda().to(torch.bfloat16)
    fp8, scale = _quant_blockwise2d_weight(w, _fp8(), block)
    g = Blockwise2DFP8Gathered(fp8, scale, torch.bfloat16, block)

    deq = _dequant_fp8_weight(g._fp8, g._scale, block)
    assert compute_snr(w.float(), deq.float()) > 30


@_CUDA
def test_blockwise2d_gathered_forward_matches_inplace():
    """A Linear whose weight is a Blockwise2DFP8Gathered (the FSDP2 all-gathered
    form) must match the in-place per-step blockwise2d quant path in fwd and DGrad."""
    from atomrl.lumen.ops.quantize.linear import _quant_blockwise2d_weight
    from atomrl.lumen.quantize import disable, enable
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Gathered
    from atomrl.lumen.quantize.config import QuantConfig
    from conftest import compute_snr  # type: ignore
    from torch import nn

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    x0 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    cfg = lambda: QuantConfig.from_str(scaling="blockwise2d", block_size=128)

    outs, gins = {}, {}
    for wrapped in (False, True):
        m = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
        m.weight.requires_grad_(False)
        m.weight.data.copy_(w)
        enable(m, config=cfg())
        if wrapped:
            fp8, scale = _quant_blockwise2d_weight(w, _fp8(), 128)
            m.weight = nn.Parameter(
                Blockwise2DFP8Gathered(fp8, scale, torch.bfloat16, 128),
                requires_grad=False,
            )
        x = x0.clone().requires_grad_(True)
        y = m(x)
        y.sum().backward()
        outs[wrapped] = y.detach().float()
        gins[wrapped] = x.grad.float()
        disable(m)

    assert (
        compute_snr(outs[False], outs[True]) > 40
    ), "wrapped fwd != in-place quant fwd"
    assert compute_snr(gins[False], gins[True]) > 20, "wrapped DGrad != in-place DGrad"


@_CUDA
def test_wrap_frozen_base_only_wraps_frozen_quant_linears():
    """_wrap_frozen_base_as_blockwise2d_fp8 wraps frozen patched Linears (holding the
    BF16 master), leaves trainable and non-patched ones untouched."""
    from atomrl.lumen.models.fsdp import _wrap_frozen_base_as_blockwise2d_fp8
    from atomrl.lumen.quantize import enable
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from atomrl.lumen.quantize.config import QuantConfig
    from torch import nn

    frozen = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    frozen.weight.requires_grad_(False)
    enable(frozen, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))

    trainable = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    enable(
        trainable, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128)
    )

    model = nn.ModuleList([frozen, trainable])
    n = _wrap_frozen_base_as_blockwise2d_fp8(model, _fp8(), 128)

    assert n == 1
    assert isinstance(frozen.weight, Blockwise2DFP8Param)
    assert frozen.weight._tensor.dtype == torch.bfloat16  # holds BF16 master
    assert not isinstance(trainable.weight, Blockwise2DFP8Param)


@_CUDA
def test_wrap_frozen_base_skips_shard_misaligned():
    """A weight whose dim0 is block-aligned but NOT block*world_size-aligned (e.g.
    lm_head N=vocab) must be skipped — its per-rank shard wouldn't be 128-aligned."""
    from atomrl.lumen.models.fsdp import _wrap_frozen_base_as_blockwise2d_fp8
    from atomrl.lumen.quantize import enable
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from atomrl.lumen.quantize.config import QuantConfig
    from torch import nn

    # 384 % 128 == 0 but 384 % (128*4) == 128 != 0 → not shardable over 4 ranks.
    lm = nn.Linear(256, 384, bias=False).cuda().to(torch.bfloat16)
    lm.weight.requires_grad_(False)
    enable(lm, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))

    n = _wrap_frozen_base_as_blockwise2d_fp8(lm, _fp8(), 128, world_size=4)
    assert n == 0
    assert not isinstance(lm.weight, Blockwise2DFP8Param)


@_CUDA
def test_wrap_frozen_base_cpu_weight():
    """Big models load on CPU before fully_shard; the helper wraps the CPU BF16
    master with no kernel call (quant is deferred to the all-gather extension)."""
    from atomrl.lumen.models.fsdp import _wrap_frozen_base_as_blockwise2d_fp8
    from atomrl.lumen.quantize import enable
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param
    from atomrl.lumen.quantize.config import QuantConfig
    from torch import nn

    m = nn.Linear(256, 256, bias=False).to(torch.bfloat16)  # stays on CPU
    m.weight.requires_grad_(False)
    enable(m, config=QuantConfig.from_str(scaling="blockwise2d", block_size=128))
    assert m.weight.device.type == "cpu"

    n = _wrap_frozen_base_as_blockwise2d_fp8(m, _fp8(), 128)
    assert n == 1
    assert isinstance(m.weight, Blockwise2DFP8Param)
    assert (
        m.weight._tensor.device.type == "cpu"
        and m.weight._tensor.dtype == torch.bfloat16
    )


@_CUDA
def test_blockwise2d_fp8_param_flatten_roundtrip():
    """__tensor_flatten__/__tensor_unflatten__ must preserve the BF16 master + meta."""
    from atomrl.lumen.quantize.comm_tensor import Blockwise2DFP8Param

    torch.manual_seed(0)
    w = (torch.randn(256, 256) * 0.05).cuda().to(torch.bfloat16)
    p = Blockwise2DFP8Param(w, _fp8(), 128)
    names, meta = p.__tensor_flatten__()
    inner = {n: getattr(p, n) for n in names}
    p2 = Blockwise2DFP8Param.__tensor_unflatten__(inner, meta, p.shape, p.stride())
    assert torch.equal(p2._tensor, w)
    assert p2._fp8_dtype == _fp8() and p2._block_size == 128
