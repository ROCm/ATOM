"""Explicit full-K2 state; disjoint attention/MoE epoch domains.

The host clears this allocation at the existing step fence, after all prior
consumers leave. No state is reset by a live producer.
"""
from atom.models.deepseek_v41.mono.kernels.attn_pre import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_index import late_pointer
from atom.models.deepseek_v41.mono.gfx1250.kernels.dspark_index import scalar_word
from flydsl._mlir.dialects import llvm

ROWS = 48
ROW_BLOCKS = 3
SPLITS = 160
PAIRED_PRODUCERS = SPLITS // 2
TAGS = 256
COL_BLOCKS = 20
SQ_BYTES = ROWS * SPLITS * 8
RSTD_OFFSET = SQ_BYTES
MHC_COUNTS_OFFSET = RSTD_OFFSET + ROWS * 8
DOWN_COUNTS_OFFSET = MHC_COUNTS_OFFSET + TAGS * ROW_BLOCKS * 4
DOWN_READY_OFFSET = DOWN_COUNTS_OFFSET + TAGS * COL_BLOCKS * 4
RAW_XO_OFFSET = DOWN_READY_OFFSET + COL_BLOCKS * 8
RAW_XOS_OFFSET = RAW_XO_OFFSET + ROWS * (16*512//4) * 4
RAW_XB_OFFSET = RAW_XOS_OFFSET + ROWS * (16*512//32) * 4
RAW_XBS_OFFSET = RAW_XB_OFFSET + ROWS * (2048//4) * 4
RAW_XO_READY_OFFSET = RAW_XBS_OFFSET + ROWS * (2048//32) * 4
RAW_XB_READY_OFFSET = RAW_XO_READY_OFFSET + ROWS * 16 * 8
STATE_BYTES = RAW_XB_READY_OFFSET + ROWS * 64 * 8
STATE_WORDS = STATE_BYTES // 4
__all__ = (
    "ROW_BLOCKS", "COL_BLOCKS", "RSTD_OFFSET", "MHC_COUNTS_OFFSET",
    "DOWN_COUNTS_OFFSET", "DOWN_READY_OFFSET", "STATE_WORDS", "STATE_BYTES",
    "RAW_XO_OFFSET", "RAW_XOS_OFFSET", "RAW_XB_OFFSET", "RAW_XBS_OFFSET",
    "RAW_XO_READY_OFFSET", "RAW_XB_READY_OFFSET", "drain", "arrive",
)


@traced
def drain():
    llvm.inline_asm(None, [], "s_wait_storecnt 0", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)
    gpu.barrier()


@traced
def arrive(root, offset, stride, epoch, index):
    address = late_pointer(root, 220) + fx.Int64(offset) + (
        fx.Int64(epoch) * stride + fx.Int64(index)) * 4
    ptr = fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,
                                       fx.AddressSpace.Global, 4), address)
    return fx.Int32(fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent"))
