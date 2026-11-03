# SPDX-License-Identifier: MIT
"""Native n32k4 B/SB jobs: construct once, carry across K, private wave LDS.

gfx950 contracts: complete K accumulators, eight computing waves, bounded
weight prefetch before input waits. gfx1250: native operand/scale layout,
TDM jobs with advancing offsets, three LDS stages and two register operands.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_ug import *
from flydsl.expr.rocdl import tdm_ops
from flydsl._mlir.dialects import llvm

BUFFERS = 6  # UGB6: deeper per-wave UG weight ring
B_WORDS = 512  # two preshuffled N16 tiles, one K128 each
SB_WORDS = 32  # one native n32k4 scale superrow
SLOT_WORDS = B_WORDS + SB_WORDS
RING_WORDS = WAVES * BUFFERS * SLOT_WORDS


@traced
def ready(outstanding):
    """Wave-private READY, with compiler-visible asynchronous LDS mutation.

    The TDM intrinsic alone allowed reused SB scalar loads to be eliminated,
    and tail B loads to precede their wait. Native uses a workgroup barrier
    for shared tiles. Our slots are private to one wave: retain its tensor
    wait, memory clobber and scheduler boundary without a CTA rendezvous.
    """
    llvm.inline_asm(None, [], f"s_wait_tensorcnt {outstanding}", "~{memory}",
                    has_side_effects=True)
    fx.rocdl.sched_barrier(0)


@traced
def retire_reads():
    """Retire old LDS reads before TDM can overwrite a recycled slot."""
    fx.rocdl.sched_barrier(0)
    llvm.inline_asm(None, [], "s_wait_dscnt 0", "~{memory}",
                    has_side_effects=True)


def tensor(pointer, shape, stride):
    return fx.Tensor(fx.make_view(pointer, fx.make_layout(shape, stride)))


@traced
def jobs(c, weight, scale, e, tile, inner, outer, scale_cols):
    """One descriptor per B/SB job, outside every K loop and input wait."""
    e, tile = fresh(e), fresh(tile)
    address = weight + (fx.Int64(e) * outer * inner // 8 + fx.Int64(tile) * inner * 2) * 4
    pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), address)
    bw = tensor(pointer, (2, 256), (inner * 2, 1))
    ba = fx.rocdl.make_tdm_atom(bw, [2, 256], strides=[inner * 2, None],
                              num_warps=1, cache_modifier=CM_NT)
    address = scale + ((fx.Int64(e) * outer // 32 + fx.Int64(tile // 2)) * (scale_cols * 8)) * 4
    pointer = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), address)
    sw = tensor(pointer, (1, 32), (32, 1))
    sa = fx.rocdl.make_tdm_atom(sw, [1, 32], strides=[32, None],
                              num_warps=1, cache_modifier=CM_NT)
    return ba, bw, sa, sw


@traced
def issue(c, job, slot, step):
    ba, bw, sa, sw = job
    base = c['native_ring'] + c['wave'] * (BUFFERS * SLOT_WORDS) + slot * SLOT_WORDS
    # imm_offset is bytes even when the tensor's element type is int32.
    # Native add_tdm_loads likewise uses k_adv=SC_INNER*4 for scale jobs.
    fx.copy(ba, bw, tensor(base, (2, 256), (256, 1)), imm_offset=fx.Int64(step) * 1024)
    fx.copy(sa, sw, tensor(base + B_WORDS, (1, 32), (32, 1)), imm_offset=fx.Int64(step) * 128)


def operands():
    return dict(x=fx.make_rmem_tensor(16, fx.Int32),
                w=[fx.make_rmem_tensor(8, fx.Int32) for _ in range(2)],
                sx=fx.make_rmem_tensor(1, fx.Int32),
                sw=fx.make_rmem_tensor(2, fx.Int32))


@traced
def prime(c, job, ks=None):
    for i in range_constexpr(BUFFERS):
        step=ug_step(ks,i) if ks is not None else fx.Int32(i)
        issue(c, job, fx.Int32(i), step)


@traced
def load(c, slot, step, row, inner, code_stride, out):
    lane = c['lane']
    base = c['native_ring'] + c['wave'] * (BUFFERS * SLOT_WORDS) + slot * SLOT_WORDS
    at = lane // 16 * 64 + lane % 16 * 4
    for tile in range_constexpr(2):
        halves = [fx.Vector(fx.ptr_load(base + tile * 256 + at + p * 128,
                    result_type=fx.Vector.make_type(4, fx.Int32))) for p in range(2)]
        out['w'][tile].store(halves[0].shuffle(halves[1], list(range(8))))
    out['sw'].store(fx.Vector.from_elements([
        fx.ptr_load(base + B_WORDS + tile * 16 + lane % 16) for tile in range(2)], fx.Int32))
    xl = c['native_x'] + row * lds_row(inner // 4) + step * 32 + lane // 16 * 4
    chunks = [fx.Vector(fx.ptr_load(xl + j * 8,
                  result_type=fx.Vector.make_type(4, fx.Int32))) for j in range(4)]
    out['x'].store(fx.Vector.from_elements([chunks[j][q] for j in range(4) for q in range(4)], fx.Int32))
    out['sx'].store(fx.Vector.from_elements([
        fx.ptr_load(c['native_s'] + row * code_stride + step)], fx.Int32))


def ug_step(ks, i):
    """Keep gfx950's exact 8+8+4 per-slice K order, twenty K128 steps."""
    q = fx.Int32(i)
    r = q // 8
    return (r < 2).select(r * 16 + ks * 8 + q % 8,
                           fx.Int32(32) + ks * 4 + q % 8)


@traced
def pipeline(c, job, steps, row, inner, ks=None):
    """Small-M native post-compute issue and operand carry, complete K.

    Only this wave consumes its B/SB slots. READY and REUSE fences cover
    asynchronous LDS writes and old reads, including compiler scheduling.
    A/SA was filled once by the CTA and stays live for the complete K.
    """
    def k_at(i):
        return ug_step(ks, i) if ks is not None else fx.Int32(i)
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(16, 16, 128,
        fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32))
    acc = [fx.make_rmem_tensor(8, fx.Float32) for _ in range(2)]
    for a in acc:a.store(fx.full(8, 0., fx.Float32))
    carry = [operands(), operands()]
    ready(2 * (BUFFERS - 1))
    load(c, fx.Int32(0), k_at(0), row, inner, inner // 128, carry[0])
    # Two rolled phases keep code and descriptor lifetimes bounded. Parity
    # dispatch names actual rmem slots, rather than selecting register pointers.
    for i in range(steps - BUFFERS):
        current = carry[0]
        nxt = carry[1]
        sx, sw = current['sx'].load()[0], current['sw'].load()
        fx.gemm(atom, acc[0], current['w'][0], current['x'], acc[0], scale_a=sw[0], scale_b=sx)
        ready(2 * (BUFFERS - 2))
        load(c, (i + 1) % BUFFERS, k_at(i + 1), row, inner, inner // 128, nxt)
        fx.gemm(atom, acc[1], current['w'][1], current['x'], acc[1], scale_a=sw[1], scale_b=sx)
        retire_reads()
        issue(c, job, i % BUFFERS, k_at(i + BUFFERS))
        # Carry by copy: vector values remain in registers, never spill to LDS.
        current['x'].store(nxt['x'].load())
        current['sx'].store(nxt['sx'].load())
        current['sw'].store(nxt['sw'].load())
        for t in range_constexpr(2):current['w'][t].store(nxt['w'][t].load())
    for i in range_constexpr(steps - BUFFERS, steps):
        current = carry[0]
        sx, sw = current['sx'].load()[0], current['sw'].load()
        fx.gemm(atom, acc[0], current['w'][0], current['x'], acc[0], scale_a=sw[0], scale_b=sx)
        if const_expr(i + 1 < steps):
            ready(2 * (steps - i - 2))
            load(c, fx.Int32((i + 1) % BUFFERS), k_at(i + 1), row, inner, inner // 128, carry[1])
        fx.gemm(atom, acc[1], current['w'][1], current['x'], acc[1], scale_a=sw[1], scale_b=sx)
        if const_expr(i + 1 < steps):
            for name in ('x', 'sx', 'sw'):current[name].store(carry[1][name].load())
            for t in range_constexpr(2):current['w'][t].store(carry[1]['w'][t].load())
    ready(0)
    return [a.load() for a in acc]
