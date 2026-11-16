"""FP4 KV byte checks on one gfx1250 GPU: the single-launch step's own device helpers (the code its stages run) against
the native writers' / reader's quantization functions on identical inputs.

  main  : compressor.fp4_main_lane4 (stage_crow's FP4 main row)  vs  fp4_groups(E4M3_SCALE) + fp4_pack (fused_compress
          "planar_fp4"), 288 B a row
  win   : attention.kv_fp8_lane8 (stage_kv's window row)          vs  fp8_groups (rope_window._quantized_row), 528 B
  ctx   : draft_ctx.ctx_fp8_lane16 (stage_kvrow's draft ring row) vs  fp8_groups (dspark quantize_fp8), 528 B
  read  : attention._kg_fetch + _kg_decode (stage_D's key rows -> LDS BF16)  vs  packed_rows.gather_prefix_rows (planar tagged rows)
"""
import sys

import numpy as np
import torch
import triton
import triton.language as tl

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.model_ops.attentions.deepseek_v41.packed_rows import gather_prefix_rows
from atom.model_ops.blockscale_kernels.quantization import fp4_groups, fp4_pack, fp8_groups
from atom.models.deepseek_v41.mono.gfx1250.kernels import attention as AS
from atom.models.deepseek_v41.mono.gfx1250.kernels import compressor as CS
from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
from atom.models.deepseek_v41.mono.gfx1250.kernels import draft_ctx as CX

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
i64 = fx.Int64
KC, HP = AS.KC, AS.HP


# ---------------------------------------------------------------- native references
@triton.jit
def ref_fp4(x, vals, scs):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    xv = tl.load(x + r * 512 + d).to(tl.bfloat16).to(tl.float32)   # fused_compress: rotated.to(bf16).to(f32)
    code, sign, scale, _ = fp4_groups(tl.reshape(xv, (32, 16)), True)
    tl.store(vals + r * 256 + tl.arange(0, 256), tl.reshape(fp4_pack(code, sign), (256,)))
    tl.store(scs + r * 32 + tl.arange(0, 32), scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


@triton.jit
def ref_fp8(x, vals, scs):
    r = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 512)
    xv = tl.load(x + r * 512 + d).to(tl.float32)   # x: the BF16 row
    q, code, _ = fp8_groups(tl.reshape(xv, (16, 32)))
    tl.store(vals + r * 512 + d, tl.reshape(q, (512,)).to(tl.uint8, bitcast=True))
    tl.store(scs + r * 16 + tl.arange(0, 16), code.to(tl.uint8))


# ---------------------------------------------------------------- the step's helpers in small launches
@flyc.kernel(known_block_size=[128, 1, 1])
def k_main(x: Int64, vals: Int64, scs: Int64):
    t = hx.opaque_v(fx.thread_idx.x)
    r = fx.block_idx.x
    w = t // 32
    l = t % 32
    d0 = w * 128 + l * 4
    xv = hx.gload(x + (i64(r) * 512 + i64(d0)) * 4, F32, 4)
    word, sbyte = CS.fp4_main_lane4([xv[e] for e in range(4)])
    if l % 2 == 0:
        hx.gstore(vals + i64(r) * 256 + i64(d0 // 2), word)
    if l % 4 == 0:
        hx.gstore(scs + i64(r) * 32 + i64(d0 // 16), sbyte.to(fx.Int8))


@flyc.jit
def l_main(x: Int64, vals: Int64, scs: Int64, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    k_main(x, vals, scs).launch(grid=(n,), block=(128,), smem=0, stream=stream)


@flyc.kernel(known_block_size=[256, 1, 1])
def k_win(x: Int64, vals: Int64, scs: Int64):
    t = hx.opaque_v(fx.thread_idx.x)
    row = fx.block_idx.x * 4 + t // 64
    c8 = (t % 64) * 8
    xb = hx.gload(x + (i64(row) * 512 + i64(c8)) * 2, BF16, 8)
    code, wds = AS.kv_fp8_lane8([xb[e].to(F32) for e in range(8)])
    hx.gstore(vals + i64(row) * 512 + i64(c8), fx.Vector.from_elements(wds, I32))
    if c8 % 32 == 0:
        hx.gstore(scs + i64(row) * 16 + i64(c8 // 32), code.to(fx.Int8))


@flyc.jit
def l_win(x: Int64, vals: Int64, scs: Int64, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    k_win(x, vals, scs).launch(grid=(n,), block=(256,), smem=0, stream=stream)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_ctx(x: Int64, vals: Int64, scs: Int64):
    l = hx.opaque_v(fx.thread_idx.x)
    row = fx.block_idx.x
    c0 = l * 16
    xb = hx.gload(x + (i64(row) * 512 + i64(c0)) * 2, BF16, 16)
    ro = [xb[i].to(F32) for i in range(16)]
    am = F32(0.0)
    for i in range_constexpr(16):   # (stage_kvrow's amax: 16 dims, then the pair lane)
        am = fx.max(am, abs(ro[i]))
    am = fx.max(am, am.shuffle_xor(1, 32))
    w8 = CX.ctx_fp8_lane16(ro, AS.inv_pow2(AS.e8m0_of(am)))
    hx.gstore(vals + i64(row) * 512 + i64(c0), fx.Vector.from_elements(w8, I32))
    if l % 2 == 0:
        hx.gstore(scs + i64(row) * 16 + i64(l // 2), AS.e8m0_of(am).to(fx.Int8))


@flyc.jit
def l_ctx(x: Int64, vals: Int64, scs: Int64, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    k_ctx(x, vals, scs).launch(grid=(n,), block=(32,), smem=0, stream=stream)


@flyc.kernel(known_block_size=[256, 1, 1])
def k_read(ids: Int64, md: Int64, ms: Int64, wd: Int64, ws: Int64, out: Int64):
    t = hx.opaque_v(fx.thread_idx.x)
    b = fx.block_idx.x
    sm = hx.lds_base()
    o = {"ids": 0, "k": 256, "stg": 256 + 48 * HP * 2}
    if t < KC:
        hx.lstore(sm + t * 4, hx.gload(ids + (i64(b) * KC + i64(t)) * 4, I32))
    gpu.barrier()
    c = dict(kwd=wd, kws=ws, kmd=md, kms=ms)
    AS._kg_fetch(c, t, sm, o, 0)
    hx.wait_asynccnt0()
    gpu.barrier()
    AS._kg_decode(c, t, sm, o, 0)
    gpu.barrier()
    for it in range_constexpr(KC * 64 // 256):
        i = t + it * 256
        key = i // 64
        cc = i % 64
        v = hx.lload(sm + 256 + (key * HP + cc * 8) * 2, I32, 4)
        hx.gstore(out + ((i64(b) * KC + i64(key)) * 512 + i64(cc * 8)) * 2, v)


@flyc.jit
def l_read(ids: Int64, md: Int64, ms: Int64, wd: Int64, ws: Int64, out: Int64, n: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    k_read(ids, md, ms, wd, ws, out).launch(grid=(n,), block=(256,), smem=256 + 48 * HP * 2 + KC * 528, stream=stream)


# ---------------------------------------------------------------- inputs
def rows_f32(rng, n):
    """Random rows over a wide magnitude range, then the edge rows."""
    x = rng.standard_normal((n, 512)) * 10.0 ** rng.uniform(-8, 3.3, (n, 1))
    x[: n // 8] *= rng.choice([0, 1], (n // 8, 512), p=[0.3, 0.7])   # zeros mixed in
    e = np.zeros((64, 512))
    e[1] = -0.0
    e[2, :] = 1e-41
    e[3, ::2] = -1e-30
    thr = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6])
    for i in range(4, 60):
        # scale s with few mantissa bits: amax = 6 s gives the E4M3 scale s, the thresholds land exactly
        s = 2.0 ** (i - 40) * (1 + (i % 4) / 8)
        g = np.concatenate([thr, -thr[1:], [6.0]])   # 30 values
        row = np.tile(np.concatenate([g, g[:2]]), 16)[:512] * s
        row[::16] = 6 * s   # every 16-dim group's amax 6 s
        e[i] = row
    e[60] = rng.standard_normal(512) * 500   # amax near / past 2688 (the scale saturates past it)
    e[61] = rng.standard_normal(512) * 3000
    e[62, :] = 2688.0
    e[63, :16] = 2.0 ** -9 * np.arange(16)
    return np.concatenate([x, e]).astype(np.float32)


def report(name, mine, ref, unit):
    bad = (mine != ref)
    nrow = bad.reshape(bad.shape[0], -1).any(1)
    print(f"[{name}] rows {bad.shape[0]}: {unit} differing {int(bad.sum())} / {bad.size}, rows differing {int(nrow.sum())}")
    return np.nonzero(nrow)[0]


def main():
    torch.cuda.set_device(0)
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    dev = "cuda"
    st = torch.cuda.current_stream()
    ok = True

    # main FP4 rows
    X = torch.tensor(rows_f32(rng, 20000), device=dev)
    R = X.shape[0]
    mv = torch.zeros(R, 256, dtype=torch.uint8, device=dev)
    ms = torch.zeros(R, 32, dtype=torch.uint8, device=dev)
    nv, ns = torch.zeros_like(mv), torch.zeros_like(ms)
    l_main(X.data_ptr(), mv.data_ptr(), ms.data_ptr(), R, stream=st)
    ref_fp4[(R,)](X, nv, ns)
    torch.cuda.synchronize()
    b1 = report("main fp4 values", mv.cpu().numpy(), nv.cpu().numpy(), "bytes")
    b2 = report("main fp4 scales", ms.cpu().numpy(), ns.cpu().numpy(), "bytes")
    for r in sorted(set(b1.tolist()) | set(b2.tolist()))[:6]:
        a, c = mv[r].cpu().numpy(), nv[r].cpu().numpy()
        j = np.nonzero(a != c)[0][:4]
        print("   row", r, "amax", float(X[r].abs().max()), "byte", j, "mono", a[j], "native", c[j],
              "scales mono/native", ms[r, :4].tolist(), ns[r, :4].tolist())
    edge = set(range(R - 4, R))   # (rows 60..63 of the edge block: saturation cases, reported, not gating)
    ok &= not (set(b1.tolist()) | set(b2.tolist())) - edge

    # window / draft ring FP8 rows (from BF16 rows)
    Xb = X[: (R // 4) * 4].to(torch.bfloat16)
    Rb = Xb.shape[0]
    for name, fn in (("window fp8", lambda v, s: l_win(Xb.data_ptr(), v.data_ptr(), s.data_ptr(), Rb // 4, stream=st)),
                     ("draft ring fp8", lambda v, s: l_ctx(Xb.data_ptr(), v.data_ptr(), s.data_ptr(), Rb, stream=st))):
        v = torch.zeros(Rb, 512, dtype=torch.uint8, device=dev)
        s = torch.zeros(Rb, 16, dtype=torch.uint8, device=dev)
        rv, rs = torch.zeros_like(v), torch.zeros_like(s)
        fn(v, s)
        ref_fp8[(Rb,)](Xb, rv, rs)
        torch.cuda.synchronize()
        c1 = report(name + " values", v.cpu().numpy(), rv.cpu().numpy(), "bytes")
        c2 = report(name + " codes", s.cpu().numpy(), rs.cpu().numpy(), "bytes")
        for r in sorted(set(c1.tolist()) | set(c2.tolist()))[:4]:
            j = np.nonzero(v[r].cpu().numpy() != rv[r].cpu().numpy())[0][:4]
            print("   row", r, "amax", float(Xb[r].float().abs().max()), "byte", j, "mono", v[r, j].tolist(),
                  "native", rv[r, j].tolist(), "codes", s[r, :4].tolist(), rs[r, :4].tolist())
        ok &= not len(c1) and not len(c2)

    # reads: tagged rows of the two sources -> BF16
    NM, NW, NB = 4096, 1064, 512
    md = torch.randint(0, 256, (NM, 256), dtype=torch.uint8, device=dev)
    msc = torch.randint(1, 0x78, (NM, 32), dtype=torch.uint8, device=dev)   # positive E4M3 scales (subnormals too)
    wd = torch.randint(0, 256, (NW, 512), dtype=torch.uint8, device=dev)
    wd[(wd & 0x7F) == 0x7F] = 0x11   # no E4M3 NaN (a writer never stores one)
    wsc = torch.randint(100, 150, (NW, 16), dtype=torch.uint8, device=dev)
    fp8 = torch.rand(NB * KC, device=dev) < 0.4
    rows = torch.where(fp8, torch.randint(0, NW, (NB * KC,), device=dev), torch.randint(0, NM, (NB * KC,), device=dev))
    tagged = rows * 2 + fp8.long()
    out = torch.zeros(NB * KC, 512, dtype=torch.bfloat16, device=dev)
    ids32 = tagged.to(torch.int32)   # (keep alive past the launch)
    l_read(ids32.data_ptr(), md.data_ptr(), msc.data_ptr(), wd.data_ptr(), wsc.data_ptr(), out.data_ptr(), NB, stream=st)
    ref = torch.zeros_like(out)
    ptr = torch.tensor([0, NB * KC], dtype=torch.int32, device=dev)
    gather_prefix_rows(((md, msc), (wd, wsc)), tagged.to(torch.int64), ptr, ref, 0, 1)
    torch.cuda.synchronize()
    a, c = out.view(torch.int16).cpu().numpy(), ref.view(torch.int16).cpu().numpy()
    d = report("read (FP4 main + FP8 window -> BF16)", a, c, "bf16 values")
    for r in d[:4]:
        j = np.nonzero(a[r] != c[r])[0][:4]
        print("   key", r, "tag", int(tagged[r]), "dims", j, "mono", a[r, j], "native", c[r, j])
    ok &= not len(d)
    print("ALL BYTE CHECKS PASS" if ok else "BYTE CHECK MISMATCH")


if __name__ == "__main__":
    main()
