# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA back half's builds (CPU) and, on a gfx1250 GPU, one rank's
launch without a merge (DCP 1, the all-reduce among one rank) bit for bit
against the kernels it replaces there: the gluon ``batched_gemm_a16w8`` W_UV,
AITER's ``fused_sigmoid_mul`` and ``k2_oproj_ar_gfx1250``. The merge needs
peers: ``tools/kimi_k3_mono_mla_back_tp_check.py`` checks it against aiter's
``dcp_merge_ll_gfx1250`` under torchrun."""

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the MLA back half builds with FlyDSL")

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import mla_back_gfx1250 as mb
from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay
from atom.mono.plan.architecture import GFX1250


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=1,
    )


@pytest.mark.parametrize("rows", (1, 17, 64))
def test_the_back_half_is_one_resident_launch(rows):
    kernel = mb.build_mla_back(_plan(rows), 8, 8, 12)
    assert kernel.grid == (256,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is mb.MLA_BACK_ABI
    assert kernel.name.startswith(f"k3_mla_back_gfx1250_tp8_d8_h12_s{rows}")
    assert kernel.plan == lay.BackPlan(rows, 8, 8, 12)
    tray = mb.build_mla_back(_plan(rows), 4, 4, 24, bound=1 << 20)
    assert tray.name.startswith(f"k3_mla_back_gfx1250_tp4_d4_h24_s{rows}")
    assert tray.name.endswith(f"_b{1 << 20}") or f"_b{1 << 20}_" in tray.name


def test_the_abi_takes_the_partials_first_and_the_ranks_last():
    names = mb.MLA_BACK_ABI.names
    assert names[:2] == ("o", "lse") and names[-2:] == ("rank", "dcp_rank")
    assert names.index("merge_peers") == names.index("peers") + 1


def test_the_libm_constants_are_the_merge_kernels():
    import struct

    def as_bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    assert as_bits(mb.LOG2E_HI) == 0x3FB8AA3B and as_bits(mb.LOG2E_LO) == 0x32A5705F
    assert as_bits(mb.LN2_HI) == 0x3F317217 and as_bits(mb.LN2_LO) == 0x3377D1CF
    assert as_bits(mb.EXP_UNDERFLOW) == 0xC2CE8ED0
    assert as_bits(mb.EXP_OVERFLOW) == 0x42B17218
    assert as_bits(mb.GATE_LOG2E) == as_bits(1.44269504089)
    # the infinities cross the DSL as the i32 holding their bits
    assert struct.pack("<i", mb.POS_INF_BITS) == struct.pack("<f", float("inf"))
    assert struct.pack("<i", mb.NEG_INF_BITS) == struct.pack("<f", float("-inf"))


def _gfx1250():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


gpu = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")
KEEP = []


def _bits(x):
    return x.contiguous().view(torch.int16)


@gpu
@pytest.mark.parametrize("rows", (1, 17))
def test_one_ranks_output_is_the_replaced_kernels_bits(rows):
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as bmm,
    )
    from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.utils import dynamic_per_batched_tensor_quant
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.quantization.quark.utils import quant_weight_online

    heads, proj = 12, 12 * lay.V_DIM
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows)

    def bf16(*shape, scale=1.0):
        return (torch.randn(shape, generator=gen, device=dev) * scale).to(
            torch.bfloat16
        )

    w_uv, s_uv = dynamic_per_batched_tensor_quant(
        bf16(heads, lay.V_DIM, lay.KV_LORA, scale=0.05), dtype=dtypes.fp8
    )
    s_uv = s_uv.reshape(1).contiguous()
    q, s_o = quant_weight_online(
        bf16(HIDDEN, proj, scale=0.02),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    w_o, s_o = shuffle_weight(q, layout=(16, 16)).contiguous(), s_o.contiguous()
    o = bf16(rows, heads, lay.KV_LORA)
    lse = torch.randn(rows, heads, generator=gen, device=dev).float().contiguous()
    gate = bf16(rows, proj, scale=2.0)

    def bank(nbytes):
        buffer = torch.zeros(nbytes + 1024, dtype=torch.uint8, device=dev)
        peers = torch.tensor([buffer.data_ptr()], dtype=torch.int64, device=dev)
        epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
        return buffer, peers, epochs

    kernel = mb.build_mla_back(_plan(rows), 1, 1, heads, bound=1 << 22)
    _, peers, epochs = held = bank(kernel.plan.peer_bytes)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    core = torch.empty(rows, proj, dtype=torch.bfloat16, device=dev)
    flags = torch.zeros(kernel.plan.flag_words, dtype=torch.int32, device=dev)
    KEEP.extend([kernel, held, out, core, flags, w_uv, s_uv, w_o, s_o, o, lse, gate])
    p = torch.Tensor.data_ptr
    args = mb.MLA_BACK_ABI.pack({
        "o": p(o), "lse": p(lse), "w_uv": p(w_uv), "s_uv": p(s_uv), "gate": p(gate),
        "w_o": p(w_o), "w_o_scale": p(s_o), "out": p(out), "core": p(core),
        "flags": p(flags), "peers": p(peers), "merge_peers": p(peers),
        "epochs": p(epochs), "rank": 0, "dcp_rank": 0,
    })  # fmt: skip
    kernel.launcher(*args, stream=torch.cuda.current_stream())

    y = torch.empty(rows, heads, lay.V_DIM, dtype=torch.bfloat16, device=dev)
    bmm(o.transpose(0, 1), w_uv, s_uv, group_size=128, YQ=y, transpose_bm=True)
    want_core = torch.empty(rows, proj, dtype=torch.bfloat16, device=dev)
    fused_sigmoid_mul(y.reshape(rows, proj), gate, out=want_core)
    oproj = oar.build_oproj_allreduce(_plan(rows), 1, proj)
    _, opeers, oepochs = oheld = bank(oar.oproj_peer_bytes(1, rows))
    want = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    KEEP.extend([oproj, oheld, want, want_core, y])
    oproj.launcher(
        p(want_core), p(w_o), p(s_o), p(want), p(opeers), p(oepochs), 0,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(_bits(core), _bits(want_core))
    assert torch.equal(_bits(out), _bits(want))
    assert int(epochs[0]) == 1 and bool((epochs == 1).all())


def test_poll_together_switch_defaults_off_and_takes_more_than_one_rank(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER is False
    plain = mb.build_mla_back(_plan(8), 8, 8, 12)
    assert "_pt" not in plain.name
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER", "1")
    polled = mb.build_mla_back(_plan(8), 8, 8, 12)
    assert "_pt" in polled.name and polled.name.replace("_pt", "") == plain.name
    assert polled.abi is plain.abi and polled.grid == plain.grid
    # one rank in both groups keeps the volatile loads, asked or not
    assert "_pt" not in mb.build_mla_back(_plan(8), 1, 1, 12).name
    assert "_pt" not in mb.build_mla_back(_plan(8), 1, 1, 12, poll_together=True).name
    assert mb.build_mla_back(_plan(8), 8, 8, 12, poll_together=False).name == plain.name
    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER", raising=False)
    # tool builds are named apart; marks and late pushes take a bound
    marked = mb.build_mla_back(
        _plan(8), 8, 8, 12, bound=9, poll_together=True, poll_marks=True,
        test_swap_late=2,
    )  # fmt: skip
    assert "_pt_b9_gm_sw2" in marked.name
    with pytest.raises(ValueError, match="bound"):
        mb.build_mla_back(_plan(8), 8, 8, 12, poll_marks=True)
    with pytest.raises(ValueError, match="bound"):
        mb.build_mla_back(_plan(8), 8, 8, 12, test_swap_late=2)
