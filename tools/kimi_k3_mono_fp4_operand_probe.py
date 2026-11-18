"""The split-K MXFP4 slab path in isolation, one CTA: wave 0's TDM copy of a
64-row x 128-byte slab (the latent's ``issue_weight`` fp4 branch, same
descriptor) or a plain buffer-load fill of the same LDS image, then every
wave's ``_fp4_operand`` of the four 16-row tasks at unit scale. Dumps the LDS
image and the BF16 operands and compares both with the host's expectation.

  mode tdm32: TDM copy, rows 32 dwords apart (no pad)
  mode tdm33: TDM copy, rows 33 dwords apart (pad dword every 32) - the default
  mode fill33: buffer loads + LDS stores, rows 33 dwords apart

SCALE_WORD (default 127): the E8M0 word handed to ``_fp4_operand``; DUMP=1
saves the raw tensors. COMPILE_ONLY=1: build the three kernels only (no GPU
needed). Run on one gfx1250 GPU: python tools/kimi_k3_mono_fp4_operand_probe.py"""

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gt
from atom.mono.device.ops import rsrc

K = 7168
ROWS = 64
ROW0 = 64
SLAB = 5
WORDS = 32
THREADS = 256
COMPILE_ONLY = os.environ.get("COMPILE_ONLY") == "1"
SCALE_WORD = int(os.environ.get("SCALE_WORD", "127"), 0)


def build(mode):
    pitch = 32 if mode == "tdm32" else 33
    tdm = mode.startswith("tdm")

    @fx.struct
    class Lds:
        buf: fx.Array[fx.Int32, ROWS * 33, 16]

    @flyc.kernel(name=f"fp4_copy_probe_{mode}", known_block_size=[THREADS, 1, 1])
    def kernel(w: Int64, lds_out: Int64, op_out: Int64):
        tid = fx.thread_idx.x
        lane = tid % 32
        wave = tid // 32
        buf = fx.SharedAllocator().allocate(Lds).peek().buf.ptr
        for i in range_constexpr(ROWS * 33 // THREADS + 1):
            item = tid + THREADS * i
            if item < ROWS * 33:
                fx.ptr_store(fx.Int32(-1), buf + item)
        gpu.barrier()
        if tdm:
            if wave == 0:
                gt.load_rows(
                    w + fx.Int64(ROW0 * (K // 2) + SLAB * 128),
                    K // 4,
                    buf,
                    ROWS,
                    ROWS,
                    padded=False,
                    pad_every=WORDS if pitch == 33 else 0,
                    cache_policy=1,
                    whole=True,
                    uniform=True,
                    width=WORDS * 2,
                )
                rocdl.s_wait_tensorcnt(0)
        else:
            for i in range_constexpr(ROWS * WORDS // THREADS):
                item = tid + THREADS * i
                word = bo.buffer_load(
                    rsrc(w),
                    (ROW0 + item // WORDS) * (K // 8) + SLAB * WORDS + item % WORDS,
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.ptr_store(fx.Int32(word), buf + item // WORDS * pitch + item % WORDS)
        gpu.barrier()
        for i in range_constexpr(ROWS * 33 // THREADS + 1):
            item = tid + THREADS * i
            if item < ROWS * pitch:
                bo.buffer_store(fx.Int32(fx.ptr_load(buf + item)), rsrc(lds_out), item)
        for t in range_constexpr(4):
            words = fx.Vector(
                gt._fp4_operand(buf + t * 16 * pitch, wave, lane, SCALE_WORD, pitch)
            ).bitcast(fx.Int32)
            for half in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [words[half * 4 + i] for i in range_constexpr(4)], fx.Int32
                    ),
                    rsrc(op_out),
                    ((t * 8 + wave) * 32 + lane) * 8 + half * 4,
                )

    @flyc.jit
    def launch(
        w: Int64,
        lds_out: Int64,
        op_out: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(w, lds_out, op_out).launch(grid=(1,), block=(THREADS,), stream=stream)

    return launch, pitch


def expected(packed):
    from atom.model_ops.quant_v4 import dequant_fp4_e2m1

    rows = packed[ROW0 : ROW0 + ROWS].cpu()
    image = rows[:, SLAB * 128 : SLAB * 128 + 128].contiguous().view(torch.int32)
    deq = dequant_fp4_e2m1(
        rows.view(torch.int8),
        torch.full((ROWS, K // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        32,
        torch.bfloat16,
    )[:, SLAB * 256 : SLAB * 256 + 256]
    ops = torch.empty(4, 8, 32, 16, dtype=torch.bfloat16)
    for t in range(4):
        for wave in range(8):
            for lane in range(32):
                k0 = wave * 32 + (lane // 16) * 8
                ks = list(range(k0, k0 + 8)) + list(range(k0 + 16, k0 + 24))
                ops[t, wave, lane] = deq[t * 16 + lane % 16, ks]
    return image, ops


modes = os.environ.get("MODES", "tdm33,tdm32,fill33").split(",")
if COMPILE_ONLY:
    for mode in modes:
        launch, _ = build(mode)
        launch(0x1000, 0x2000, 0x3000)
        print("compiled", mode)
    raise SystemExit
dev = torch.device("cuda", 0)
g = torch.Generator(device=dev).manual_seed(5)
packed = torch.randint(
    0, 256, (256, K // 2), generator=g, device=dev, dtype=torch.uint8
)
image, ops = expected(packed)
for mode in modes:
    launch, pitch = build(mode)
    lds = torch.zeros(ROWS * pitch, dtype=torch.int32, device=dev)
    op = torch.zeros(4 * 8 * 32 * 16, dtype=torch.bfloat16, device=dev)
    launch(
        packed.data_ptr(),
        lds.data_ptr(),
        op.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    got = lds.view(ROWS, pitch).cpu()
    image_bad = (got[:, :WORDS] != image).nonzero()
    # float compare: E2M1 code 8 (-0) converts to -0.0, the host table's +0.0
    op_bad = (op.view(4, 8, 32, 16).cpu().float() != ops.float()).nonzero()
    if os.environ.get("DUMP"):
        torch.save(
            {
                "packed": packed.cpu(),
                "got": op.view(4, 8, 32, 16).cpu(),
                "want": ops,
                "lds": got,
            },
            f"fp4_operand_probe_{mode}.pt",
        )
    print(
        json.dumps(
            {
                "mode": mode,
                "lds_word_mismatch": int(image_bad.shape[0]),
                "first_lds_bad_rc": image_bad[:4].tolist(),
                "pad_words": got[:4, WORDS:].flatten().tolist(),
                "operand_mismatch": int(op_bad.shape[0]),
                "first_operand_bad_twle": op_bad[:4].tolist(),
            }
        ),
        flush=True,
    )
