# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The LM head GEMM at decode rows as one TDM-streamed launch (gfx1250): ``x[M, K] @ w[N, K]^T``, bf16, M <= 8.

At decode the LM head is a pure weight stream (615 MB of a 200K vocabulary's shard at TP 4). ``tgemm.mm`` runs it
on opus_gemm (128 x 64 x 256 tiles over 391 blocks); here every CU streams its own run of 16-row groups through a
non-temporal TDM ring a wave (16 waves, one group each, 128 K a slot, 3 slots), with x staged once in LDS
(kernels/dspark_tail_gfx1250.hip, ``tail_l_launch``). 36.7 vs 42.9 us per call on gfx1250 (graph of 20, 1 GPU).

Numerics: bit-identical to ``tgemm.mm`` (opus_gemm): per 16-row group one f32 accumulator chained over K in 32-wide
chunks from 0 upwards (``v_wmma_f32_16x16x32_bf16``, the weight in the first operand slot as opus's swap_ab, no
split-K), one round-to-nearest-even cast. Checked op-level at M = 8 / 7 / 1 (normal, wide-range and large-row x,
every logit) and on graph replay.

Served (``mm``) at M <= 8 rows, bf16 x and w, no bias, w contiguous with whole 16-row groups, K a multiple of 128 up
to 6144, gfx1250, and a shape whose groups fit one a wave (the launcher checks it before launching). Anything else
returns None and the caller keeps ``tgemm.mm``. Used by ``ParallelLMHead`` for the target's and the DSpark draft's
LM head alike (the draft shares the target's head).
"""

import ctypes
import fcntl
import hashlib
import logging
import os
import subprocess
import tempfile

import torch

logger = logging.getLogger("atom")

_SRC = os.path.join(os.path.dirname(__file__), "kernels", "dspark_tail_gfx1250.hip")
MT = 8
KMAX = 6144
KB = 128
# The (N, K) weight shapes it was measured on: bit-identical to tgemm.mm and faster at every M = 1..8 (MI455X,
# graphs of 20 calls, interleaved rounds: -6.0 us at M 1 .. -1.7 us at M 8). Every other shape keeps tgemm.mm.
MEASURED_SHAPES = frozenset({(50016, 6144)})
# A/B switch (read at call time; a module attribute, not an env flag): False keeps tgemm.mm.
_STREAM = True
# Calls served, counted per call (a graph recording counts its captures).
LMH_STATS = {"lmh_stream": 0}
_lib = None
_lib_failed = False
_ncta: dict = {}


def _opus_include() -> "str | None":
    import aiter

    inc = os.path.join(os.path.dirname(os.path.dirname(aiter.__file__)), "csrc", "include")
    return inc if os.path.isfile(os.path.join(inc, "opus", "opus.hpp")) else None


def _load():
    """Build (once per source + include, under a file lock: every TP rank imports this) and load the launcher."""
    global _lib, _lib_failed
    if _lib is not None or _lib_failed:
        return _lib
    try:
        inc = _opus_include()
        if inc is None:
            raise RuntimeError("aiter's opus.hpp not found")
        with open(_SRC, "rb") as f:
            src = f.read()
        key = hashlib.sha1(src + inc.encode()).hexdigest()[:16]
        cache = os.path.join(os.path.expanduser("~"), ".cache", "atom", "dspark_tail")
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"tail_{key}.so")
        with open(os.path.join(cache, ".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(so):
                hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
                fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
                os.close(fd)
                cmd = [hipcc, "-O3", "--offload-arch=gfx1250", "-std=c++17", "-fPIC", "-shared", f"-I{inc}", _SRC,
                       "-o", tmp]
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if r.returncode:
                    os.unlink(tmp)
                    raise RuntimeError(f"hipcc failed: {r.stderr[-2000:]}")
                os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
        lib.tail_l_launch.restype = I
        lib.tail_l_launch.argtypes = [P, I, L, P, I, I, P, L, I, P]
        _lib = lib
    except Exception as e:  # noqa: BLE001
        _lib_failed = True
        logger.warning(f"LM head stream: launcher unavailable ({e}); the LM head keeps tgemm.mm")
    return _lib


def _usable(x: torch.Tensor, w: torch.Tensor, bias) -> bool:
    return (
        bias is None
        and x.dim() == 2
        and 0 < x.shape[0] <= MT
        and x.dtype == w.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and w.dim() == 2
        and w.is_contiguous()
        and w.shape[1] == x.shape[1]
        and x.shape[1] <= KMAX
        and x.shape[1] % KB == 0
        and w.shape[0] % 16 == 0
        and tuple(w.shape) in MEASURED_SHAPES
        and x.device.type == "cuda"
        and x.device == w.device
    )


def _cus(dev) -> "int | None":
    n = _ncta.get(dev.index)
    if n is None:
        p = torch.cuda.get_device_properties(dev)
        n = _ncta[dev.index] = p.multi_processor_count if "gfx1250" in p.gcnArchName else 0
    return n or None


def mm(x: torch.Tensor, w: torch.Tensor, bias=None) -> "torch.Tensor | None":
    """``tgemm.mm(x, w, bias)`` bit for bit in one streamed launch, or None when not served."""
    if not _STREAM or not _usable(x, w, bias):
        return None
    ncta = _cus(x.device)
    if ncta is None or _load() is None:
        return None
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    rc = _lib.tail_l_launch(x.data_ptr(), m, x.stride(0), w.data_ptr(), n, k, out.data_ptr(), out.stride(0), ncta,
                            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
    if rc in (-1, -2):  # the shape check refused before any launch
        return None
    if rc != 0:
        raise RuntimeError(f"LM head stream launch failed ({rc}) after its checks passed")
    LMH_STATS["lmh_stream"] += 1
    return out
