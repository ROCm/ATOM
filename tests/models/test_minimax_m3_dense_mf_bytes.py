"""DMF (aiter ``flydsl_dmf``, the dense layers' post-attention half in one launch) is bit-exact (GPU, gfx1250 only).

One GPU with TP4 peers emulated in four local k5 inbox buffers: the peers' packs are placed, the launch runs rank R's
part. Against the production launches on the same inputs and state:

* out / residual out, the four inboxes and the k5 counters vs Stage O (aiter ``flydsl_oproj_ar``: production's o_proj
  + k5 all-reduce + residual + Gemma RMSNorm, bit for bit);
* h / its scales and the down output vs production's dense MLP on Stage O's output: ``quantize_fp8`` (the k5 MXFP8
  epilogue's bytes) -> gate_up -> ``swiglu_oai_split_mxfp8`` -> down (``native_quant_linear``, preshuffled 1x32).

Random weights in the served layouts (the arithmetic order, not the values, is under test); eager over the three
inbox rounds, then graph replays."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("DMF is a gfx1250 GPU kernel", allow_module_level=True)
gfx = pytest.importorskip("aiter.jit.utils.chip_info", exc_type=ImportError)
if gfx.get_gfx() != "gfx1250":
    pytest.skip("DMF is gfx1250 only", allow_module_level=True)
dmf = pytest.importorskip("atom.models.minimax_m3.mono.gfx1250.dmf", exc_type=ImportError)

from aiter.ops.flydsl.oproj_ar import flydsl_oproj_ar  # noqa: E402
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight  # noqa: E402

from atom.model_ops.blockscale import native_quant_linear, quantize_fp8  # noqa: E402
from atom.model_ops.swiglu_oai import swiglu_oai_split_mxfp8  # noqa: E402

DEV = torch.device("cuda", 0)
H, I, K, T = 6144, 3072, 2048, 8  # hidden, dense intermediate a rank (12288 / TP4), o_proj K, rows
RANKS, ROW_PACKS, WAVES, BUFS, MAX_ROWS = 4, 768, 8, 3, 256  # the k5 inbox layout
SLOT = MAX_ROWS * ROW_PACKS
INBOX = BUFS * RANKS * SLOT * 16
GUARD = 1 << 16
CTR_WORDS = 2 * MAX_ROWS * WAVES + 16
ALPHA, BETA, LIMIT, EPS = 1.702, 1.0, 7.0, 1e-6


def _fix(y):
    """bf16 rows as k5 packs: the 0xffff sentinel half-word stored as 0x7fff."""
    h = y.contiguous().view(torch.int16).clone()
    h[h == -1] = 0x7FFF
    return h.view(torch.uint8).reshape(-1)


def _e8m0(g, shape, lo, hi):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int32).to(torch.uint8)


def _fp8(g, shape, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("rank", [0, 3])
def test_dmf_matches_production_launches(rank):
    g = torch.Generator().manual_seed(17 + rank)
    # the dense MLP's weights in the served layouts: gate / up rows merged, (16, 16) preshuffle, n32k4 scales
    gu = shuffle_weight(_fp8(g, (2 * I, H)).to(DEV), layout=(16, 16))
    gus = shuffle_mxfp8_bpreshuffle_scale(_e8m0(g, (2 * I, H // 32), 116, 122).view(torch.float8_e8m0fnu).to(DEV))
    dn = shuffle_weight(_fp8(g, (H, I)).to(DEV), layout=(16, 16))
    dns = shuffle_mxfp8_bpreshuffle_scale(_e8m0(g, (H, I // 32), 116, 122).view(torch.float8_e8m0fnu).to(DEV))
    # o_proj's MXFP8 input and weight (the attention's output side), residual, norm, the peers' partial rows
    x = _fp8(g, (T, K), 2.0).to(DEV)
    xs = _e8m0(g, (T, K // 32), 118, 124).to(DEV)
    w = _fp8(g, (H, K), 2.0).to(DEV)
    ws = _e8m0(g, (H, K // 32), 116, 122).to(DEV)
    res = torch.randn(T, H, generator=g).to(torch.bfloat16).to(DEV)
    nw = (torch.randn(H, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
    peers = [_fix((torch.randn(T, H, generator=g) * 0.05).to(torch.bfloat16)).to(DEV) for _ in range(RANKS)]
    bufs = [torch.empty(INBOX + 2 * GUARD, dtype=torch.uint8, device=DEV) for _ in range(RANKS)]
    ctr = torch.zeros(CTR_WORDS, dtype=torch.int32, device=DEV)
    inbox = [b.data_ptr() + GUARD for b in bufs]

    def place(rnd):
        for b in bufs:
            b.fill_(0xFF)
            b[:GUARD].fill_(0xA5)
            b[GUARD + INBOX:].fill_(0xA5)
        inb = bufs[rank][GUARD:GUARD + INBOX]
        for s in range(RANKS):
            if s != rank:
                o = ((rnd * RANKS + s) * SLOT) * 16
                inb[o:o + T * ROW_PACKS * 16] = peers[s]
                j = ((((rnd + 2) % BUFS) * RANKS + s) * SLOT) * 16
                inb[j:j + T * ROW_PACKS * 16] = 0x11  # junk the re-arm must clear
        ctr.zero_()
        ctr[: T * WAVES] = rnd
        ctr[MAX_ROWS * WAVES: MAX_ROWS * WAVES + T * WAVES] = rnd

    refs = []
    for rnd in range(BUFS):
        place(rnd)
        out, res_out = flydsl_oproj_ar(x, xs, w, ws, res, nw, EPS, inbox, ctr.data_ptr(), SLOT, rank, maxit=200000)
        xq, xqs = quantize_fp8(out)
        h, hs = swiglu_oai_split_mxfp8(
            native_quant_linear(xq, gu, gus, x_scale=xqs, weight_group_rows=1, weight_preshuffled=True),
            alpha=ALPHA, beta=BETA, limit=LIMIT,
        )
        down = native_quant_linear(h, dn, dns, x_scale=hs, weight_group_rows=1, weight_preshuffled=True)
        torch.cuda.synchronize()
        refs.append([t.clone() for t in (out, res_out, h.view(torch.uint8), hs.view(torch.uint8), down, ctr)]
                    + [b.clone() for b in bufs])

    scratch = dmf.dmf_scratch(DEV)
    outs = dmf.DmfOutputs(DEV)
    tab = dmf.dmf_table(gu.view(torch.uint8), gus.view(torch.uint8), dn.view(torch.uint8), dns.view(torch.uint8),
                        outs, scratch)

    def launch():
        return dmf.flydsl_dmf(x, xs, w, ws, res, nw, EPS, inbox, ctr.data_ptr(), SLOT, rank, tab, alpha=ALPHA,
                              beta=BETA, limit=LIMIT, maxit=200000)

    def check(got, rnd):
        for i, (a, b) in enumerate(zip(got, refs[rnd])):
            assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (rnd, i)
        assert int(scratch["err"][0]) == 0, "poll timeouts"

    def got_of(o, r):
        return [o, r, outs.h, outs.hs, outs.down, ctr] + bufs

    for i in range(2 * BUFS):  # eager: the launch count walks the three hand-off copies twice
        place(i % BUFS)
        outs.down.fill_(-7.0)
        o, r = launch()
        torch.cuda.synchronize()
        check(got_of(o, r), i % BUFS)
    place(0)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        go, gr = launch()
    torch.cuda.current_stream().wait_stream(stream)
    for i in range(2 * BUFS):
        place(i % BUFS)
        outs.down.fill_(-7.0)
        graph.replay()
        torch.cuda.synchronize()
        check(got_of(go, gr), i % BUFS)
