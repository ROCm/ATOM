# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 hi storage boundaries against ATOM's quantizer, including noncanonical scales.

The bridge cases also run with TRITON_INTERPRET=1 on CPU. The KVPrep, draft
writer, and load conversion case runs the actual hi blocks on gfx950.
"""

import os

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402 - guarded by importorskip

from atom.model_ops.blockscale_kernels.quantization import (  # noqa: E402
    AMAX_FLOOR,
    FP8_DTYPE,
    quantize_fp8_kernel,
)
from atom.models.deepseek_v41.mono import hi_bridge as bridge  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def test_fp8_floor_matches_atom():
    pytest.importorskip("flydsl")
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import kv

    assert kv.AMAX_FLOOR == AMAX_FLOOR.value


def _device():
    if INTERPRET:
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip("a GPU, or TRITON_INTERPRET=1")
    return "cuda"


def _quantize(x):
    q = torch.empty_like(x, dtype=FP8_DTYPE)
    s = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 32), dtype=torch.uint8, device=x.device
    )
    quantize_fp8_kernel[(triton.cdiv(x.numel(), 1024),)](x, q, s, x.numel(), False)
    return torch.cat((q.view(torch.uint8), s), dim=-1)


def _dequantize(packed):
    q = packed[..., :512].contiguous().view(FP8_DTYPE).float()
    scale = (packed[..., 512:].int() << 23).view(torch.float32)
    return (q.view(*q.shape[:-1], 16, 32) * scale[..., None]).flatten(-2).bfloat16()


def _values(shape, device):
    torch.manual_seed(871)
    x = torch.randn(shape, device=device).bfloat16()
    flat = x.view(-1, 32)
    # Zero, tiny/subnormal, a scale boundary, signs and a large exponent range.
    flat[::7] = 0
    flat[1::7] *= 1e-12
    flat[2::7] *= 2**-120
    flat[3::7] *= 2**60
    flat[4::7] = 0
    flat[4::7, 0] = 225
    flat[4::7, 1] = 100
    return x


@triton.jit
def _import(
    pool,
    positions,
    bids,
    slots,
    rings,
    layers,
    pos_out,
    bounds,
    START: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING: tl.constexpr,
    NB: tl.constexpr,
    RQ: tl.constexpr,
    RB: tl.constexpr,
):
    bridge._ring_in(
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        RQ,
        pool,
        positions,
        bids,
        slots,
        rings,
        layers,
        pos_out,
        bounds,
        START,
        SLOT_ROWS,
        RING,
        NB,
        1 << 20,
        128,
        512,
        RB,
        NB,
        True,
    )


@triton.jit
def _export(
    pool,
    positions,
    bids,
    slots,
    rings,
    packed_rings,
    layers,
    START: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING: tl.constexpr,
    NB: tl.constexpr,
    RQ: tl.constexpr,
):
    bridge._ring_out(
        tl.program_id(0),
        tl.program_id(1),
        RQ,
        pool,
        positions,
        bids,
        slots,
        rings,
        layers,
        START,
        SLOT_ROWS,
        RING,
        NB,
        1 << 20,
        128,
        512,
        NB,
        True,
        packed_rings,
    )


@pytest.mark.parametrize(
    "nb,ctx,padded",
    [(1, 0, False), (1, 127, False), (1, 133, False), (2, 1001, False), (2, 133, True)],
)
def test_bridge_preserves_original_fp8_bytes_and_padding(nb, ctx, padded):
    device = _device()
    rq, start, slot_rows, ring_slots = 6, 7, 1024, 133
    layer_ids = [0, 2, 3]
    layers = torch.tensor(layer_ids, dtype=torch.int32, device=device)
    x = _values((start + nb * slot_rows, 512), device)
    pool = _quantize(x)
    before = pool.clone()
    decoded = _dequantize(pool)
    positions = torch.cat(
        [
            torch.arange(ctx + b * 17, ctx + b * 17 + rq, device=device)
            for b in range(nb)
        ]
    )
    bids = torch.arange(nb, device=device, dtype=torch.int32).repeat_interleave(rq)
    if padded:
        bids[-rq:] = -1
    slots = torch.arange(nb - 1, -1, -1, device=device, dtype=torch.int32)
    rings = torch.full(
        (len(layer_ids), nb, 128, 512),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    pos_out = torch.zeros(nb, dtype=torch.int32, device=device)
    rb = triton.next_power_of_2(nb * rq)
    bounds = torch.zeros(3, rb, dtype=torch.int32, device=device)
    _import[(len(layer_ids), nb, 128)](
        pool,
        positions,
        bids,
        slots,
        rings,
        layers,
        pos_out,
        bounds,
        start,
        slot_rows,
        ring_slots,
        nb,
        rq,
        rb,
    )
    assert torch.equal(pool, before)
    for m, layer in enumerate(layer_ids):
        for b in range(nb):
            pos = ctx + b * 17
            for s in range(128):
                q = pos - 128 + (s - pos % 128 + 128) % 128
                expected = torch.zeros(512, dtype=torch.bfloat16, device=device)
                if q >= 0 and not (padded and b == nb - 1):
                    expected = decoded[
                        start
                        + layer * ring_slots
                        + int(slots[b]) * slot_rows
                        + q % ring_slots
                    ]
                assert torch.equal(rings[m, b, s], expected)
    new = _values((len(layer_ids), nb, rq, 512), device)
    encoded = _quantize(new)
    packed_rings = torch.zeros(
        (len(layer_ids), nb, 128, 528), dtype=torch.uint8, device=device
    )
    expected_pool = before.clone()
    for m, layer in enumerate(layer_ids):
        for b in range(nb):
            for j in range(rq):
                pos = ctx + b * 17 + j
                packed_rings[m, b, pos % 128] = encoded[m, b, j]
                rings[m, b, pos % 128] = _dequantize(encoded[m, b, j : j + 1])[0]
                if not (padded and b == nb - 1):
                    expected_pool[
                        start
                        + layer * ring_slots
                        + int(slots[b]) * slot_rows
                        + pos % ring_slots
                    ] = encoded[m, b, j]
    _export[(len(layer_ids), nb * rq)](
        pool,
        positions,
        bids,
        slots,
        rings,
        packed_rings,
        layers,
        start,
        slot_rows,
        ring_slots,
        nb,
        rq,
    )
    assert torch.equal(pool, expected_pool)
    if not INTERPRET:
        # Fixed buffers and padding under replay, including all untouched pool bytes.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.copy_(before)
            _export[(len(layer_ids), nb * rq)](
                pool,
                positions,
                bids,
                slots,
                rings,
                packed_rings,
                layers,
                start,
                slot_rows,
                ring_slots,
                nb,
                rq,
            )
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, expected_pool)


@pytest.mark.parametrize("nb", [1, 2])
def test_kvprep_draft_writer_and_load_match_atom_quantizer(nb):
    if INTERPRET or not torch.cuda.is_available():
        pytest.skip("the actual hi blocks need gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950")
    pytest.importorskip("flydsl")
    import flydsl.expr as fx
    from atom.models.deepseek_v41.mono.hi.blocks.base import Channel, Fmt
    from atom.models.deepseek_v41.mono.hi.blocks.gfx950.attention.mla_window import (
        KVPrep,
    )
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import buffer_ops as bo
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import kv, mailbox as mb
    from atom.models.deepseek_v41.mono.hi.device.gfx950.ops import ld_i32, rsrc, traced
    from atom.models.deepseek_v41.mono.hi.plan.program import Program, Scenario, Stage
    from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel
    from atom.models.deepseek_v41.mono.hi.v41.draft import RingStore

    @traced
    def feed(k, row):
        if k.tid < 256:
            word = ld_i32(k.args["input"], row * 256 + k.tid)
            mb.put(k.mb["kv"], row * 256 + k.tid, word, k.tag)

    @traced
    def read_qat(k, row):
        if k.tid < 256:
            word = mb.poll([(k.mb["kvn"], row * 256 + k.tid, 1)], k.tag)[0][0]
            bo.buffer_store(word, rsrc(k.args["qat"]), row * 256 + k.tid)

    @traced
    def decode(k, row):
        if k.tid < 64:
            words = kv.load8(
                fx.Int64(k.args["encoded"]) + fx.Int64(row) * 528, k.tid * 8
            )
            bo.buffer_store(words, rsrc(k.args["decoded"]), row * 256 + k.tid * 4)

    rq, rows = 6, 6 * nb
    scenario = Scenario(nb, rq)
    ch, out = Channel("kv", 512, Fmt.BF16), Channel("kvn", 512, Fmt.BF16)
    stages = [
        Stage(
            "feed",
            rows,
            feed,
            writes=ch.regions(rows, ("feed",)),
            args=("input",),
            key=("feed",),
        )
    ]
    stages += KVPrep("prep", ch, out, "pos", "rope", kv8=True).stages(scenario)
    stages += RingStore("store", out, "ring", "geo", 0, kv8=True).stages(scenario)
    stages += [Stage("read", rows, read_qat, args=("qat",), key=("read_qat",))]
    kernel = MonoKernel(Program(f"test_fp8_kvprep_b{nb}", scenario, stages))
    reader = MonoKernel(
        Program(
            f"test_fp8_load_b{nb}",
            scenario,
            [
                Stage(
                    "decode", rows, decode, args=("encoded", "decoded"), key=("load8",)
                )
            ],
        )
    )
    x = _values((rows, 512), "cuda")
    expected = _quantize(x)
    qat = torch.empty_like(x)
    ring = torch.full((nb * 256, 528), 173, dtype=torch.uint8, device="cuda")
    pos = torch.tensor(
        [127 + 17 * b for b in range(nb)], device="cuda", dtype=torch.int32
    )
    bids = torch.arange(nb, device="cuda", dtype=torch.int32).repeat_interleave(rq)
    slots = torch.arange(nb, device="cuda", dtype=torch.int32)
    geo = torch.tensor(
        [[nb, 256, 133, 1, 1, 0, 0, 0]] * 3, device="cuda", dtype=torch.int32
    )
    rope = torch.ones(256, 32, dtype=torch.bfloat16, device="cuda")
    sine = torch.zeros_like(rope)
    table = kernel.table(
        {
            "input": x,
            "qat": qat,
            "ring": ring,
            "pos": pos,
            "rope": rope,
            "rope.sin": sine,
            "geo": geo,
            "cx.bids": bids,
            "cx.slots": slots,
        }
    )
    buffers = kernel.buffers(uncached=True)
    kernel.launch(table, buffers)
    torch.cuda.synchronize()
    expected_ring = torch.full_like(ring, 173)
    for b in range(nb):
        for j in range(rq):
            expected_ring[b * 256 + (127 + 17 * b + j) % 133] = expected[b * rq + j]
    assert torch.equal(ring, expected_ring)
    assert torch.equal(qat, _dequantize(expected))
    decoded = torch.empty_like(x)
    read_table = reader.table({"encoded": expected, "decoded": decoded})
    read_buffers = reader.buffers(uncached=True)
    reader.launch(read_table, read_buffers)
    torch.cuda.synchronize()
    assert torch.equal(decoded, _dequantize(expected))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ring.fill_(173)
        buffers.bump()
        kernel.launch(table, buffers)
        read_buffers.bump()
        reader.launch(read_table, read_buffers)
    for _ in range(2000):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(ring, expected_ring)
    assert torch.equal(qat, _dequantize(expected))
    assert torch.equal(decoded, _dequantize(expected))


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_pool_quantizes_its_bf16_entry_once(ratio):
    if INTERPRET or not torch.cuda.is_available():
        pytest.skip("the actual hi pool stage needs gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950")
    pytest.importorskip("flydsl")
    from atom.models.deepseek_v41.mono.hi.blocks.base import Channel, Fmt
    from atom.models.deepseek_v41.mono.hi.blocks.gfx950.compressor import Compressor
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import buffer_ops as bo
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import mailbox as mb, paged
    from atom.models.deepseek_v41.mono.hi.device.gfx950.ops import ld_i32, rsrc, traced
    from atom.models.deepseek_v41.mono.hi.plan.program import Program, Scenario, Stage
    from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel

    nb, rq, rows = 2, 6, 12
    sc = Scenario(nb, rq)
    xch, output = Channel("x", 5120, Fmt.BF16), Channel("entry", 512, Fmt.BF16)
    torch.manual_seed(59)
    partial = torch.randn(rows, 512, dtype=torch.float32, device="cuda")
    scores = torch.randn_like(partial)
    kv_state = torch.randn(nb, 7, 512, dtype=torch.float32, device="cuda")
    score_state = torch.randn_like(kv_state)
    pos = torch.tensor([11, 22], device="cuda", dtype=torch.int32)
    req = torch.zeros(paged.REQ_WORDS, device="cuda", dtype=torch.int32)
    req[paged.GEO : paged.GEO + 3] = torch.tensor(
        [1, 128, 7 * 512], device="cuda", dtype=torch.int32
    )
    req[paged.BID : paged.BID + nb] = torch.arange(nb, device="cuda")
    req[paged.SLOT : paged.SLOT + nb] = torch.arange(nb, device="cuda")
    req[paged.LIVE : paged.LIVE + nb] = 1
    req[paged.ROW_LIVE : paged.ROW_LIVE + rows] = 1
    # A dead row must not mutate its destination, at either compression ratio.
    req[paged.ROW_LIVE + rows - 1] = 0
    table_pages = torch.arange(nb, device="cuda", dtype=torch.int32).view(nb, 1)
    rope = torch.ones(256, 32, device="cuda", dtype=torch.bfloat16)
    sine = torch.zeros_like(rope)

    def arm(kv8):
        comp = Compressor(
            "comp",
            xch,
            ratio,
            output,
            "pos",
            "rope",
            "cache",
            128,
            kv_state="kv_state" if ratio == 2 else "",
            score_state="score_state" if ratio == 2 else "",
            slots=7,
            kslices=1,
            qat=False,
            kv8=kv8,
        )
        plan = comp.plan(sc)
        parts = [(plan.kv_part, "partial")]
        if ratio == 2:
            parts.append((plan.score_part, "scores"))

        def feed(k, row):
            for name, arg in parts:
                word = ld_i32(k.args[arg], row * 512 + k.tid)
                mb.put(k.mb[name], row * 512 + k.tid, word, k.tag)

        @traced
        def drain(k, row):
            if k.tid < 256:
                word = mb.poll([(k.mb["entry"], row * 256 + k.tid, 1)], k.tag)[0][0]
                bo.buffer_store(word, rsrc(k.args["out"]), row * 256 + k.tid)

        writes = tuple(
            region
            for name, _ in parts
            for region in Channel(name, 512, Fmt.F32).regions(rows, ("feed",))
        )
        stages = [
            Stage(
                "feed",
                rows,
                feed,
                writes=writes,
                args=tuple(arg for _, arg in parts),
                key=(tuple(parts),),
            )
        ]
        stages += [
            comp.stages(sc)[-1],
            Stage("drain", rows, drain, args=("out",), key=("entry",)),
        ]
        kernel = MonoKernel(
            Program(
                f"test_compressor_r{ratio}_kv8_{int(kv8)}",
                sc,
                stages,
                sgpr_spill_ok=True,
            )
        )
        cache = (
            torch.full((nb * 128, 528), 173, device="cuda", dtype=torch.uint8)
            if kv8
            else torch.zeros(nb * 128, 512, device="cuda", dtype=torch.bfloat16)
        )
        out = torch.empty(rows, 512, device="cuda", dtype=torch.bfloat16)
        args = dict(
            partial=partial,
            scores=scores,
            pos=pos,
            rope=rope,
            cache=cache,
            req=req,
            ptab=table_pages,
            out=out,
            kv_state=kv_state.clone(),
            score_state=score_state.clone(),
        )
        args["rope.sin"] = sine
        args[plan.aux] = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        tab = kernel.table(args)
        buf = kernel.buffers(uncached=True)
        kernel.launch(tab, buf)
        torch.cuda.synchronize()
        return cache, out, args

    bf16, plain, astate = arm(False)
    packed, decoded, bstate = arm(True)
    quantized = _quantize(bf16)
    expected = torch.full_like(packed, 173)
    for b in range(nb):
        for j in range(rq):
            p = int(pos[b]) + j
            if (p + 1) % ratio == 0 and (b * rq + j) != rows - 1:
                at = b * 128 + p // ratio
                expected[at] = quantized[at]
    assert torch.equal(packed, expected)
    assert torch.equal(decoded, _dequantize(_quantize(plain)))
    assert torch.equal(astate["kv_state"], bstate["kv_state"])
    assert torch.equal(astate["score_state"], bstate["score_state"])
