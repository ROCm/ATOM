"""Preload the exact native specializations observed late in formal A1.

This proves coverage of those misses, not of every reachable BF16 dispatch.
It neither changes a dispatch choice nor removes pointer specialization.
"""

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data"
LATE = HERE / "native_late_specializations.json"
BF16_CONFIG = {
    "block_m": 32,
    "block_n": 128,
    "block_k": 128,
    "stages": 4,
    "split_k": 1,
    "m_waves": 1,
    "n_waves": 4,
    "k_waves": 2,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}


def observed_keys():
    rows = json.loads(LATE.read_text())["late_examples"]
    return {
        "triton": [
            ast.literal_eval(row["detail"])["key"]
            for row in rows
            if row["event"] == "triton_specialization"
        ],
        "flydsl": next(
            ast.literal_eval(row["detail"])["cache_key"]
            for row in rows
            if row["event"] == "flydsl_compile_lock"
        ),
    }


def preload_triton():
    import torch
    from triton.runtime.jit import MockTensor

    from atom.model_ops.attentions.deepseek_v41.indices import _indptr_scan_all
    from atom.model_ops.v4_kernels.state_writes import _swa_write_kernel

    class Pointer(MockTensor):
        def __init__(self, dtype, *, aligned=True, small=True):
            super().__init__(dtype)
            self.address = 16 if aligned else 4
            self.extent = 0 if small else 2**31 + 1024

        def data_ptr(self):
            return self.address

        def ptr_range(self):
            return self.extent

    bf = Pointer(torch.bfloat16)
    i32, i64 = Pointer(torch.int32), Pointer(torch.int64)
    _swa_write_kernel.warmup(
        bf,
        i64,
        i32,
        i32,
        Pointer(torch.bfloat16, small=False),
        512,
        2_100_001,
        512,
        17,
        i32,
        BLOCK_D=512,
        CACHED_UNTIL=True,
        RING_SLOTS=128,
        SLOT_ROWS=5153,
        RING_STRIDE=128,
        RUN_ROWS=128,
        grid=(3, 8),
    )
    for block in (512, 1024):
        _indptr_scan_all.warmup(
            i32,
            i64,
            i32,
            (i32, i32, i32),
            (i32, i32, i32),
            block,
            i32,
            Pointer(torch.int32, aligned=False),
            DECODE=False,
            WINDOW=128,
            RATIOS=(0, 1, 2),
            TOPKS=(0, 512, 512),
            SPLIT=False,
            BLOCK=block,
            grid=(3,),
        )
    device = torch.cuda.current_device()
    expected = observed_keys()["triton"]
    functions = (_swa_write_kernel, _indptr_scan_all, _indptr_scan_all)
    result = []
    for key, function in zip(expected, functions, strict=True):
        if key not in function.device_caches[device][0]:
            raise RuntimeError(
                f"Exact observed key not preloaded: {function.__name__}: {key}"
            )
        result.append(
            {
                "function": function.__name__,
                "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
            }
        )
    return result


def bf16_arguments():
    import torch
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import get_split_k_buffers

    stream = torch.cuda.current_stream()
    a = torch.empty((32, 5120), dtype=torch.bfloat16, device="cuda")
    b = torch.empty((384, 5120), dtype=torch.bfloat16, device="cuda").T
    out = torch.empty((32, 384), dtype=torch.bfloat16, device="cuda")
    # Native persistent zeros are prepared before the no-launch observation.
    get_split_k_buffers(stream, a.device, 1)
    torch.cuda.synchronize()
    return a, b, out


def preload_bf16(arguments):
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16

    from atom.mono.runtime.compile import compile_only

    with compile_only():
        gemm_a16w16(*arguments[:2], out=arguments[2], user_kwargs=BF16_CONFIG)
