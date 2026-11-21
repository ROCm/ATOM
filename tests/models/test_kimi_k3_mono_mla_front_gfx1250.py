# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA front's builds (CPU) and, on a gfx1250 GPU, its outputs:
the GEMV against an f64 reference of the dequantized weights, the norms and
q_b_proj's quant bit for bit against AITER's kernels run on the front's own
unnormalized rows, and the hand-off words restored."""

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the MLA front builds with FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import (
    mla_front_gfx1250 as mf,
)
from atom.models.kimi_k3.mono.kernels.stages import (
    mla_front_layout as lay,
)
from atom.models.kimi_k3.mono.kernels.stages import tdm_timeline

GATE_ROWS = 1536


@pytest.mark.parametrize("rows", (1, 8, 24, 57, 58, 64))
def test_the_front_is_one_launch_filling_the_gpu(rows):
    kernel = mf.build_mla_front(rows, GATE_ROWS)
    assert kernel.grid == (256,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is mf.MLA_FRONT_ABI
    staging = "_r" if rows <= 57 else "_st3"
    assert kernel.name == (
        f"k3_mla_front_gfx1250_s{rows}_g{GATE_ROWS}{staging}_n28_d{lay.GATE_DELAY}"
    )


def test_norm_placement_and_gate_delay_build_apart():
    gate = mf.build_mla_front(8, GATE_ROWS, norm_ctas=0, gate_delay=0)
    assert gate.grid == (228,) and gate.name.endswith("_s8_g1536_r")
    assert mf.build_mla_front(8, GATE_ROWS, norm_ctas=8).grid == (236,)


def test_the_abi_takes_the_runner_buffers_and_the_checks_tap_last():
    names = mf.MLA_FRONT_ABI.names
    assert names[:7] == ("x", "w_a", "s_a", "w_g", "s_g", "q_norm", "kv_norm")
    assert names.index("q_scale") == names.index("q_fp8") + 1
    assert names[-2:] == ("partials", "qkv_seen")


def test_the_timeline_is_off_by_default():
    assert tdm_timeline.key() is None
    assert "_tl" not in mf.build_mla_front(8, GATE_ROWS).name


def _gfx1250():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


gpu = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")


def _fp8_weight(rows, gen, dev):
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.quantization.quark.utils import quant_weight_online

    w = (torch.randn(rows, 7168, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    q, s = quant_weight_online(
        w, QuantType.per_1x128, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    scale = s.float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)
    deq = q.float().double() * scale.double()
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous(), deq


@gpu
@pytest.mark.parametrize("rows", (8, 24, 64))
def test_outputs_match_the_reference_and_aiters_norm_and_quant_bits(rows):
    from aiter import QuantType, dtypes
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.linear import per_1x128_e8m0_quant

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows)
    w_a, s_a, d_a = _fp8_weight(lay.QKV_A_ROWS, gen, dev)
    w_g, s_g, d_g = _fp8_weight(GATE_ROWS, gen, dev)

    def bf16(*shape, scale=1.0, shift=0.0):
        values = torch.randn(shape, generator=gen, device=dev) * scale + shift
        return values.to(torch.bfloat16)

    q_norm, kv_norm = bf16(1536, scale=0.1, shift=1.0), bf16(512, scale=0.1, shift=1.0)
    qkv_c = torch.full((lay.qkv_words(),), -1, dtype=torch.int32, device=dev)
    partials = torch.empty(
        lay.partial_bytes(rows, GATE_ROWS), dtype=torch.uint8, device=dev
    )

    def e(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=dev)

    out = {
        "q": e(rows, 1536),
        "q_fp8": e(rows, 1536, dtype=dtypes.fp8),
        "q_scale": e(rows, 12, dtype=dtypes.fp8_e8m0),
        "kv": e(rows, 512),
        "k_rope": e(rows, 64),
        "gate": e(rows, GATE_ROWS),
        "qkv_seen": e(rows, 2048),
    }
    kernel = mf.build_mla_front(rows, GATE_ROWS)
    for _ in range(2):  # the second launch reads the words the first restored
        x = bf16(rows, 7168)
        p = torch.Tensor.data_ptr
        args = {
            "x": p(x), "w_a": p(w_a), "s_a": p(s_a), "w_g": p(w_g),
            "s_g": p(s_g), "q_norm": p(q_norm), "kv_norm": p(kv_norm),
            "qkv_c": p(qkv_c), "partials": p(partials),
        }  # fmt: skip
        args.update({name: p(t) for name, t in out.items()})
        kernel.launcher(
            *mf.MLA_FRONT_ABI.pack(args), stream=torch.cuda.current_stream()
        )
        torch.cuda.synchronize()
        assert bool((qkv_c == -1).all())

        ref_a = x.double() @ d_a.T
        ref = {
            "q_c": ref_a[:, :1536],
            "kv_c": ref_a[:, 1536:2048],
            "k_rope": ref_a[:, 2048:],
            "gate": x.double() @ d_g.T,
        }
        got = {
            "q_c": out["qkv_seen"][:, :1536],
            "kv_c": out["qkv_seen"][:, 1536:],
            "k_rope": out["k_rope"],
            "gate": out["gate"],
        }
        for name, value in got.items():
            # BF16 rounding of f32 sums taken in another order
            err = (value.double() - ref[name]).abs().max().item()
            assert err <= 5e-3 * ref[name].abs().max().item(), name
            exact = (value == ref[name].to(torch.bfloat16)).double().mean().item()
            assert exact > 0.999, name

        q, kv = torch.empty_like(got["q_c"]), torch.empty_like(got["kv_c"])
        fused_qk_rmsnorm(
            q_out_quantized=q,
            q_out_scale=None,
            q=got["q_c"],
            q_weight=q_norm,
            q_epsilon=1e-6,
            k_out=kv,
            k=got["kv_c"],
            k_weight=kv_norm,
            k_epsilon=1e-6,
            quant_type=QuantType.No,
        )
        q8, qs = per_1x128_e8m0_quant(out["q"], dtypes.fp8, True)

        def bits(t):
            return t.contiguous().view(torch.uint8)

        assert torch.equal(bits(out["kv"]), bits(kv))
        assert torch.equal(bits(out["q"]), bits(q))
        assert torch.equal(bits(out["q_fp8"]), bits(q8))
        assert torch.equal(bits(out["q_scale"]), bits(qs))
