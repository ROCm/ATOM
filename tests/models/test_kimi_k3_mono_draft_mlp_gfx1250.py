# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft MLP / o_proj kernels' builds (CPU) and, on a
gfx1250 GPU, their outputs bit for bit against the kernels the draft layer
runs: ``draft_proj_ar_gfx1250`` against ``per_1x128_e8m0_quant``, the
dispatch's FlyDSL mxfp8_128 GEMM and the all-reduce's rank-order fp32 sum
(one rank; and as one rank of eight, the other ranks' lines placed by the
host), ``draft_gate_up_gfx1250`` against the GEMM at the width's split-K,
its reduce and ``silu_and_mul``."""

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the draft kernels build with FlyDSL")

from atom.models.kimi_k3.mono.kernels.stages import draft_gate_up_gfx1250 as gk
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
from atom.mono.runtime.consensus import MonoUnsupported

HIDDEN = 7168
BOUND = 1 << 22


@pytest.mark.parametrize(
    ("rows", "tp", "k"), [(1, 1, 1024), (7, 8, 1792), (56, 4, 2048)]
)
def test_the_projection_all_reduce_is_one_launch(rows, tp, k):
    kernel = dp.build_draft_proj_ar(rows, tp, k)
    assert kernel.grid == (lay.CTAS,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is dp.DRAFT_PROJ_AR_ABI
    assert kernel.name == f"k3_draft_proj_ar_gfx1250_tp{tp}_s{rows}_k{k}"
    assert kernel.abi.names[-3:] == ("peers", "epochs", "rank")


def test_a_bounded_or_profile_build_takes_its_trailing_arguments():
    bounded = dp.build_draft_proj_ar(7, 1, 1024, poll_bound=BOUND)
    assert bounded.name.endswith(f"_b{BOUND}")
    assert bounded.abi.names[-1] == "flags"
    profile = dp.build_draft_proj_ar(7, 1, 1024, poll_bound=BOUND, profile=True)
    assert profile.name.endswith("_profile") and profile.abi.names[-1] == "stamps"


def test_a_shape_the_projection_kernel_does_not_take_is_refused():
    for rows, tp, k in ((0, 8, 1792), (65, 8, 1792), (33, 4, 3584), (7, 8, 1152)):
        with pytest.raises(MonoUnsupported):
            dp.build_draft_proj_ar(rows, tp, k)


@pytest.mark.parametrize(("rows", "split"), [(1, 4), (7, 4), (21, 8), (48, 8), (64, 4)])
def test_gate_up_is_one_launch(rows, split):
    kernel = gk.build_draft_gate_up(rows, 1792, split)
    assert kernel.grid == (224,) and kernel.block == (256,)
    assert kernel.launch_count == 1 and kernel.abi is gk.DRAFT_GATE_UP_ABI
    assert kernel.name == f"k3_draft_gate_up_gfx1250_s{rows}_i1792_sk{split}"
    assert "partials" in kernel.abi.names and kernel.abi.names[-1] == "gu"
    for bad in ((49, 1792, 1), (7, 1792, 2), (7, 1800, 4)):
        with pytest.raises(MonoUnsupported):
            gk.build_draft_gate_up(*bad)


def _gfx1250():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


gpu = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")


def _bf16(gen, dev, *shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def _fp8_linear(gen, dev, rows, cols):
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.quantization.quark.utils import quant_weight_online

    q, s = quant_weight_online(
        _bf16(gen, dev, rows, cols, scale=0.02),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _partial(x, shard):
    from aiter import dtypes, gemm_a8w8_blockscale_bpreshuffle

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.linear import per_1x128_e8m0_quant

    x8, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    return gemm_a8w8_blockscale_bpreshuffle(x8, shard[0], xs, shard[1], torch.bfloat16)


def _activation(gen, dev, rows, k):
    x = _bf16(gen, dev, rows, k, scale=0.5)
    x[0, :64] *= 40.0
    x[rows - 1, 128:256] = 0.0
    x.view(torch.int16)[rows // 2, 320:324] = torch.tensor(
        [3, -32768 + 5, 127, -32768], dtype=torch.int16, device=dev
    )
    return x


def _launch(kernel, x, shard, out, peers, epochs, rank, flags):
    p = torch.Tensor.data_ptr
    kernel.launcher(
        *kernel.abi.pack({
            "x": p(x), "w": p(shard[0]), "w_scale": p(shard[1]), "out": p(out),
            "peers": p(peers), "epochs": p(epochs), "rank": rank, "flags": p(flags),
        }),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip


@gpu
@pytest.mark.parametrize(("rows", "k"), [(7, 1024), (21, 1792), (56, 1792)])
def test_one_ranks_projection_is_productions_bits(rows, k):
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows + k)
    shard = _fp8_linear(gen, dev, HIDDEN, k)
    kernel = dp.build_draft_proj_ar(rows, 1, k, poll_bound=BOUND)
    bank = torch.zeros(lay.peer_bytes(1, rows), dtype=torch.uint8, device=dev)
    peers = torch.tensor([bank.data_ptr()], dtype=torch.int64, device=dev)
    epochs = torch.zeros(lay.EPOCH_COUNTERS, dtype=torch.int32, device=dev)
    flags = torch.zeros(4, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    for launch in range(3):  # both banks, and the first again
        x = _activation(gen, dev, rows, k)
        _launch(kernel, x, shard, out, peers, epochs, 0, flags)
        want = _partial(x, shard)
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(want))
        assert bool((epochs == launch + 1).all()) and int(flags[0]) == 0


def _lines(partial, rows, epoch):
    count = lay.triples(rows)
    pairs = torch.zeros(
        count * 3, HIDDEN // 2, dtype=torch.int32, device=partial.device
    )
    pairs[:rows] = partial.contiguous().view(torch.int32).view(rows, HIDDEN // 2)
    lines = torch.empty(count, HIDDEN // 2, 4, dtype=torch.int32, device=partial.device)
    lines[:, :, :3] = pairs.view(count, 3, HIDDEN // 2).permute(0, 2, 1)
    lines[:, :, 3] = epoch
    return lines.view(-1, 4)


@gpu
@pytest.mark.parametrize(
    ("tp", "rank", "rows", "k"), [(8, 3, 7, 1024), (4, 0, 21, 2048)]
)
def test_one_rank_of_a_tp_group_sums_its_peers_lines_in_rank_order(tp, rank, rows, k):
    """The kernel built for a TP group, launched as one rank on one GPU, the
    peers' banks local buffers between guard bands and the other ranks'
    production partials placed by the host as their lines: the output is the
    rank-order fp32 sum, the rank's own lines land in its slice of every
    bank, and nothing else is written."""
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(tp * 100 + rows)
    shards = [_fp8_linear(gen, dev, HIDDEN, k) for _ in range(tp)]
    kernel = dp.build_draft_proj_ar(rows, tp, k, poll_bound=BOUND)
    guard, nbytes = 4096, lay.peer_bytes(tp, rows)
    banks = [
        torch.full((guard + nbytes + guard,), 0xA5, dtype=torch.uint8, device=dev)
        for _ in range(tp)
    ]
    for bank in banks:
        bank[guard : guard + nbytes] = 0
    peers = torch.tensor(
        [b.data_ptr() + guard for b in banks], dtype=torch.int64, device=dev
    )
    epochs = torch.zeros(lay.EPOCH_COUNTERS, dtype=torch.int32, device=dev)
    flags = torch.zeros(4, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    lines = lay.triples(rows) * (HIDDEN // 2)

    def slots(bank):
        return bank[guard : guard + nbytes].view(torch.int32).view(2, tp, lines, 4)

    for epoch in (1, 2):
        xs = [_activation(gen, dev, rows, k) for _ in range(tp)]
        partials = [_partial(xs[p], shards[p]) for p in range(tp)]
        for p in range(tp):
            if p != rank:
                slots(banks[rank])[epoch & 1, p] = _lines(partials[p], rows, epoch)
        torch.cuda.synchronize()
        expected = [b.clone() for b in banks]
        for b in expected:
            slots(b)[epoch & 1, rank] = _lines(partials[rank], rows, epoch)
        _launch(kernel, xs[rank], shards[rank], out, peers, epochs, rank, flags)
        torch.cuda.synchronize()
        acc = partials[0].cpu().float()
        for partial in partials[1:]:
            acc = acc + partial.cpu().float()
        assert torch.equal(_bits(out.cpu()), _bits(acc.to(torch.bfloat16)))
        assert all(torch.equal(b, e) for b, e in zip(banks, expected))
        assert int(flags[0]) == 0 and bool((epochs == epoch).all())


@gpu
@pytest.mark.parametrize("rows", (7, 21))
def test_gate_up_and_the_activation_are_productions_bits(rows):
    from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle, silu_and_mul

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.model_ops.layernorm import _aiter_rms_quant
    from atom.models.kimi_k3.mono import draft_dispatch

    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(rows)
    inter = 1792
    shard = _fp8_linear(gen, dev, 2 * inter, HIDDEN)
    split = draft_dispatch.mxfp8_split_k(rows, 2 * inter, HIDDEN, "gate_up_proj")
    kernel = gk.build_draft_gate_up(rows, inter, split)
    act = torch.empty(rows, inter, dtype=torch.bfloat16, device=dev)
    gu = torch.empty(rows, 2 * inter, dtype=torch.bfloat16, device=dev)
    partials = torch.empty(
        lay.gate_up_partial_bytes(inter), dtype=torch.uint8, device=dev
    )
    p = torch.Tensor.data_ptr
    # the last scale drives the gate past the SiLU's exp range fixups, where
    # the HIP kernel's denormal flush shows
    for scale in (1.0, 60.0):
        norm = _bf16(gen, dev, HIDDEN, scale=0.1, shift=1.0)
        x8, xs, _ = _aiter_rms_quant(
            _bf16(gen, dev, rows, HIDDEN, scale=0.5),
            (norm.float() * scale).to(torch.bfloat16), 1e-6,
            QuantType.per_1x128.value, True, _bf16(gen, dev, rows, HIDDEN),
            dtypes.fp8, True,
        )  # fmt: skip
        x8 = x8.contiguous()
        kernel.launcher(
            *kernel.abi.pack({
                "x": p(x8), "x_scale": p(xs), "w": p(shard[0]), "w_scale": p(shard[1]),
                "act": p(act), "partials": p(partials), "gu": p(gu),
            }),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        want_gu = gemm_a8w8_blockscale_bpreshuffle(
            x8, shard[0], xs, shard[1], torch.bfloat16
        )
        want = torch.empty_like(act)
        silu_and_mul(want, want_gu)
        torch.cuda.synchronize()
        assert torch.equal(_bits(gu), _bits(want_gu))
        assert torch.equal(_bits(act), _bits(want))


def _bound_linear(shard, prefix, **change):
    """A linear as the wiring binds it, around a real FP8 shard."""
    from types import SimpleNamespace

    from aiter import QuantType, dtypes
    from torch import nn

    weight = nn.Parameter(shard[0], requires_grad=False)
    weight.is_shuffled = True
    linear = {
        "weight": weight,
        "weight_scale": nn.Parameter(shard[1], requires_grad=False),
        "bias": None,
        "native_a8_group_rows": None,
        "quant_type": QuantType.per_1x128,
        "blockscale_e8m0_scale": True,
        "params_dtype": dtypes.fp8,
        "input_scale": None,
        "is_output_padded": False,
        "_k_padded": False,
        "tp_dim": 1,
        "reduce_results": True,
        "prefix": prefix,
        "mono_proj": None,
        "_try_mono_proj": True,
    }
    linear.update(change)
    return SimpleNamespace(**linear)


@gpu
def test_the_wiring_runs_a_decode_steps_o_proj_and_mlp_on_the_kernels(monkeypatch):
    """``draft_mlp`` end to end at one rank on real shards: the binding, the
    banks, a width's builds through ``WidthBuilds`` and the resident-hazard
    check, the launches -- production's bits; a width past the switches is
    production's."""
    from types import SimpleNamespace

    from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle, silu_and_mul

    import atom.utils.forward_context as fc

    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.model_ops.activation import SiluAndMul

    pytest.importorskip("triton", reason="the kernel under test needs triton")
    from atom.model_ops.layernorm import _aiter_rms_quant
    from atom.models.kimi_k3.mono import draft_mlp

    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_GATE_UP_ROWS", "35")
    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_AR_ROWS", "48")
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(41)
    runner = draft_mlp.DraftMlpRunner(dev, 1, 0, None)
    monkeypatch.setattr(draft_mlp, "runner_for", lambda device: runner)
    monkeypatch.setattr(
        fc,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, is_prefill=False)
        ),
    )
    rows, inter, heads = 7, 1792, 1024
    o_shard = _fp8_linear(gen, dev, HIDDEN, heads)
    o_proj = _bound_linear(o_shard, "layers.0.self_attn.o_proj")
    for _ in range(2):  # the second step replays the bound layer and width
        x = _activation(gen, dev, rows, heads)
        out = draft_mlp.forward_oproj(o_proj, x)
        want = _partial(x, o_shard)
        torch.cuda.synchronize()
        assert out is not None and torch.equal(_bits(out), _bits(want))
    assert o_proj.mono_proj.k == heads and runner.k_oproj == heads
    assert runner.kernels[("o_proj", rows)].name == (
        f"k3_draft_proj_ar_gfx1250_tp1_s{rows}_k{heads}"
    )

    gu_shard = _fp8_linear(gen, dev, 2 * inter, HIDDEN)
    down_shard = _fp8_linear(gen, dev, HIDDEN, inter)
    norm = SimpleNamespace(
        use_fused_quant=True, fused_allreduce=False, quant_type=QuantType.per_1x128,
        params_dtype=dtypes.fp8, _aiter_transpose_scale=True, _aiter_e8m0_scale=True,
    )  # fmt: skip
    mlp = SimpleNamespace(
        prefix="layers.0.mlp",
        _input_norm=norm,
        gate_up_proj=_bound_linear(
            gu_shard, "layers.0.mlp.gate_up_proj", tp_dim=0, reduce_results=False
        ),
        down_proj=_bound_linear(down_shard, "layers.0.mlp.down_proj"),
        act_fn=SiluAndMul(),
        mono_mlp=None,
        _try_mono_mlp=True,
    )
    weight = _bf16(gen, dev, HIDDEN, scale=0.1, shift=1.0)

    def normed(count):
        x8, xs, _ = _aiter_rms_quant(
            _bf16(gen, dev, count, HIDDEN, scale=0.5), weight, 1e-6,
            QuantType.per_1x128.value, True, _bf16(gen, dev, count, HIDDEN),
            dtypes.fp8, True,
        )  # fmt: skip
        return x8.contiguous(), xs

    x8, xs = normed(rows)
    out = draft_mlp.forward_mlp(mlp, (x8, xs))
    gu = gemm_a8w8_blockscale_bpreshuffle(
        x8, gu_shard[0], xs, gu_shard[1], torch.bfloat16
    )
    act = torch.empty(rows, inter, dtype=torch.bfloat16, device=dev)
    silu_and_mul(act, gu)
    want = _partial(act, down_shard)
    torch.cuda.synchronize()
    assert out is not None and torch.equal(_bits(out), _bits(want))
    assert mlp.mono_mlp.inter == inter and runner.inter == inter
    assert runner.kernels[("gate_up", rows)].name.startswith("k3_draft_gate_up_gfx1250")
    assert runner.kernels[("down", rows)].k == inter
    assert not runner.refused
    # 56 rows: past both switches, the whole MLP is production's
    assert draft_mlp.forward_mlp(mlp, normed(56)) is None
    assert draft_mlp.forward_oproj(o_proj, _activation(gen, dev, 56, heads)) is None


def test_poll_together_switch_defaults_off_and_takes_more_than_one_rank(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER", raising=False)
    assert envs.ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER is False
    plain = dp.build_draft_proj_ar(7, 8, 1792)
    assert "_pt" not in plain.name
    monkeypatch.setenv("ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER", "1")
    polled = dp.build_draft_proj_ar(7, 8, 1792)
    assert polled.name == plain.name + "_pt" and polled.abi is plain.abi
    assert polled.grid == plain.grid
    # one rank keeps the volatile loads
    assert "_pt" not in dp.build_draft_proj_ar(7, 1, 1024).name
    assert dp.build_draft_proj_ar(7, 8, 1792, poll_together=False).name == plain.name
    monkeypatch.delenv("ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER", raising=False)
    # tool builds are named apart; a late push takes a bound
    late = dp.build_draft_proj_ar(
        7, 8, 1792, poll_bound=9, poll_together=True, test_swap_late=3
    )
    assert late.name == plain.name + "_pt_b9_sw3"
    assert late.abi is dp.DRAFT_PROJ_AR_BOUNDED_ABI
    with pytest.raises(ValueError, match="bound"):
        dp.build_draft_proj_ar(7, 8, 1792, test_swap_late=3)
