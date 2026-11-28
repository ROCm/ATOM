# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The LM head GEMM at decode rows as one TDM-streamed launch on gfx1250:
``x[M, K] @ w[N, K]^T``, bf16, M <= 8.

At decode the LM head is a pure weight stream. Every CU streams its own run of
16-row groups of ``w`` through a non-temporal TDM ring a wave, with ``x``
staged once in LDS (``kernels/lm_head_stream_gfx1250.hip``).

Numerics: the same reduction as ``tgemm.mm``'s opus_gemm without split-K
(per 16-row group one f32 accumulator chained over K in 32-wide chunks from 0
upwards, one round-to-nearest-even cast), so the logits are bit-identical
where ``tgemm.mm`` runs that kernel.

``mm`` serves, on gfx1250 only: M <= 8 rows, bf16 ``x`` and ``w``, no bias,
``w`` contiguous and of a measured (N, K) (``MEASURED_SHAPES``). Anything else
returns None and the caller keeps ``tgemm.mm``.
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

_SRC = os.path.join(os.path.dirname(__file__), "kernels", "lm_head_stream_gfx1250.hip")
MAX_ROWS = 8
MAX_K = 6144  # the kernel's x stage in LDS (KMAX)
K_SLOT = 128  # K columns a TDM slot (KB)
WAVES = 16  # waves a CTA (NW), one 16-row group each
# The (N, K) weight shapes it was measured on (MI455X, M = 1..8, one run, interleaved with tgemm.mm): bit-identical to
# tgemm.mm and faster at every M (-6.0 .. -1.7 us at 50016 x 6144). Every other shape keeps tgemm.mm.
MEASURED_SHAPES = frozenset({(50016, 6144)})
_lib = None
_lib_failed = False
_ctas: dict = {}


def _opus_include() -> "str | None":
    """aiter's ``csrc/include``: the source tree, or the wheel's ``aiter_meta``."""
    from aiter.jit.core import AITER_META_DIR

    inc = os.path.join(AITER_META_DIR, "csrc", "include")
    return inc if os.path.isfile(os.path.join(inc, "opus", "opus.hpp")) else None


def _build_key(inc: str, hipcc: str) -> str:
    """The kernel source, the opus header and the compiler it is built with."""
    h = hashlib.sha1()
    for path in (_SRC, os.path.join(inc, "opus", "opus.hpp"), os.path.realpath(hipcc)):
        h.update(path.encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _load():
    """Build (once per key, under a file lock: every TP rank imports this) and
    load the launcher; None where it cannot be built."""
    global _lib, _lib_failed
    if _lib is not None or _lib_failed:
        return _lib
    try:
        inc = _opus_include()
        if inc is None:
            raise RuntimeError("aiter's opus.hpp not found")
        hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
        cache = os.path.join(
            os.path.expanduser("~"), ".cache", "atom", "lm_head_stream"
        )
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"lm_head_stream_{_build_key(inc, hipcc)}.so")
        with open(os.path.join(cache, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(so):
                fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
                os.close(fd)
                cmd = [hipcc, "-O3", "--offload-arch=gfx1250", "-std=c++17"]
                cmd += ["-fPIC", "-shared", f"-I{inc}", _SRC, "-o", tmp]
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if r.returncode:
                    os.unlink(tmp)
                    raise RuntimeError(f"hipcc failed: {r.stderr[-2000:]}")
                os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
        lib.lm_head_stream_launch.restype = I
        lib.lm_head_stream_launch.argtypes = [P, I, L, P, I, I, P, L, I, P]
        _lib = lib
    except Exception as e:  # noqa: BLE001
        _lib_failed = True
        logger.warning(f"LM head stream unavailable ({e}); the LM head keeps tgemm.mm")
    return _lib


def _gfx1250_ctas(dev: torch.device) -> int:
    """One CTA a CU on gfx1250; 0 on any other arch (cached per device)."""
    n = _ctas.get(dev.index)
    if n is None:
        props = torch.cuda.get_device_properties(dev)
        n = props.multi_processor_count if "gfx1250" in props.gcnArchName else 0
        _ctas[dev.index] = n
    return n


def _served(x: torch.Tensor, w: torch.Tensor, bias, out, ctas: int) -> bool:
    m, k = x.shape if x.dim() == 2 else (0, 0)
    groups = w.shape[0] // 16 if w.dim() == 2 else 0
    return (
        bias is None
        and 0 < m <= MAX_ROWS
        and x.dtype == w.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and x.device == w.device
        and w.is_contiguous()
        and w.shape == (groups * 16, k)
        and (w.shape[0], k) in MEASURED_SHAPES
        and 0 < groups <= ctas * WAVES
        and 0 < k <= MAX_K
        and k % K_SLOT == 0
        and (
            out is None
            or (
                out.dtype == torch.bfloat16
                and out.shape == (m, w.shape[0])
                and out.stride(-1) == 1
                and out.stride(0) % 8 == 0
                and out.data_ptr() % 16 == 0
            )
        )
    )


def mm(x: torch.Tensor, w: torch.Tensor, bias=None, out=None) -> "torch.Tensor | None":
    """``tgemm.mm(x, w, bias, out=out)`` in one streamed launch, or None where
    the call is not served (the caller then runs ``tgemm.mm``)."""
    if x.device.type != "cuda":
        return None
    ctas = _gfx1250_ctas(x.device)
    if not ctas or not _served(x, w, bias, out, ctas) or _load() is None:
        return None
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    args = (x.data_ptr(), m, x.stride(0), w.data_ptr(), n, k)
    args += (out.data_ptr(), out.stride(0), ctas, ctypes.c_void_p(stream))
    rc = _lib.lm_head_stream_launch(*args)
    if rc in (-1, -2, -4):  # refused before launching (shape / LDS attribute)
        return None
    if rc != 0:
        raise RuntimeError(f"LM head stream launch failed ({rc})")
    return out
