"""Time-bounded spin for the fused MoE kernels: poll until ``pending`` clears; trap (llvm.trap -> the process faults,
nothing keeps spinning on the GPU) once DEADLINE_TICKS of the 100 MHz steady counter have passed. The clock is read
every 64 polls only (a clock read on every poll slows the co-resident working waves)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono import prims

DEADLINE_TICKS = 200_000_000  # 2 s


def clk():
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))


@ASTRewriter.transform
def spin_t(load_all, pending):
    v = load_all()
    t0 = clk()
    n = fx.Int32(0)
    late = fx.Int32(0)
    while pending(v) & (late == 0):
        prims.compiler_fence()
        v = load_all()
        n = n + 1
        if (n & 63) == 0:
            late = ((clk() - t0) > DEADLINE_TICKS).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        prims._trap()
    return v
