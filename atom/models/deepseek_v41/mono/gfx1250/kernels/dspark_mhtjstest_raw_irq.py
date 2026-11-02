from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_mhtjstest_raw_handoff import *
@traced
def raw_irq_group(c, t, tile, dc, out, rope):
    """Token t, columns 32 dc .. of a head tile's heads (PV out [head][column]
    at ``out``, the dims' ``rope`` (cos, sin): ``irq_rope_loads``) -> inverse
    RoPE, group-32 FP8 quant -> XO / XOS."""
    tid, dims = c["tid"], c["d"]
    mine = tid < HEAD_TILE * COLS // 2
    h = fx.min(tid, HEAD_TILE * COLS // 2 - 1) // (COLS // 2)
    head, live = tile_head(dims, tile, h)
    mine = only_live(mine, live)
    tt = tid % (COLS // 2)
    d = dc * COLS + tt * 2
    out = out + h * COLS
    e = fx.ptr_load(out + tt * 2)
    o = fx.ptr_load(out + (tt * 2 + 1))
    cs, sn = rope
    # inverse rotation, kept fp32 (bf16 products are exact: any contraction agrees)
    e1 = e * cs + o * sn
    o1 = o * cs - e * sn
    amax = butterfly(fx.max(abs(e1), abs(o1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = pow2(254 - code)
    # this thread's two bytes and its neighbour's make a word (4 dims)
    q0, q1 = clamp_fp8(e1 * inv), clamp_fp8(o1 * inv)
    n0, n1 = xshfl(q0, 1), xshfl(q1, 1)
    w = fp8_pack4(q0, q1, n0, n1)
    if mine & (tt % 2 == 0):
        put_xo_word(c,t * dims.xo_words + (head * HEAD_DIM + d) // 4,w)
    if mine & (tt % 16 == 0):
        put_xo_scale(c,t * (dims.heads * HEAD_DIM // 32) + (head * HEAD_DIM + d) // 32,code)
