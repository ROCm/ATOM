# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The DSpark Markov chain of ONE request in one persistent launch (gfx1250).

``DSparkMarkovHead.sample_next`` runs the block's K sequential steps as K pairs
of ``dspark_markov_argmax`` launches, and every step re-reads the whole
``markov_w2`` table (V x 256 bf16, 102 MB for a 200K vocabulary). Here one
launch runs all K steps: one workgroup per CU, each owning a run of 16-id vocab
tiles; the tiles' W2 rows are copied into LDS (TDM) at the first step and the
rest stay in registers, so later steps read no W2 from memory. After each step
the workgroups exchange their (max, id) through tagged 64-bit slots and every
workgroup reduces them itself.

Numerics: bit-identical to ``dspark_markov_argmax`` -- per tile the same chain
of eight ``v_wmma_f32_16x16x32_bf16`` its Triton stage 1 emits on gfx1250 (the
same operands, the same K split across lanes), ``fp32(base) + acc``, lowest id
on ties.

Served when (``usable``): gfx1250, ONE request (the block's rows are its K
positions), bf16 logits / tables of rank 256, a vocabulary of whole 16-id
tiles, K <= 8, and a tile split that fits the device (the launch checks it).
Anything else keeps the two-stage path.

The exchange waits for every workgroup, which holds when the launch has the GPU
to itself. Its waits are bounded so that a workgroup that never becomes
resident (another process or a long kernel on a side stream holding CUs) cannot
hang the step; such a wait sets a bit in an error word and that step's drafts
are not to be trusted. ``check_errors``, called by the draft on the host every
step, reads the word without waiting on the device and raises, rather than
drafting on.
"""

import ctypes
import fcntl
import glob
import hashlib
import logging
import os
import subprocess
import tempfile
from typing import NamedTuple

import torch

logger = logging.getLogger("atom")

_SRC = os.path.join(
    os.path.dirname(__file__), "kernels", "dspark_markov_chain_gfx1250.hip"
)
RANK = 256
MAX_STEPS = 8
LDS_TILES = 38
# Return codes of `markov_chain_launch` for a tile split or LDS size this
# device cannot hold: the one-launch path is never served on it.
_DOES_NOT_FIT = (-3, -4)
_lib = None
_lib_failed = False
_calls = 0


class _DeviceState(NamedTuple):
    slots: torch.Tensor  # [MAX_STEPS * ncta] int64 exchange slots
    err: torch.Tensor  # [1] int32 error word, bit k: a wait of step k gave up
    ncta: int
    err_host: torch.Tensor  # pinned copy of `err`
    err_copied: torch.cuda.Event  # recorded after each copy into `err_host`


_state: dict[int, _DeviceState] = {}


def _opus_include() -> "str | None":
    import aiter

    inc = os.path.join(
        os.path.dirname(os.path.dirname(aiter.__file__)), "csrc", "include"
    )
    return inc if os.path.isfile(os.path.join(inc, "opus", "opus.hpp")) else None


def _load():
    """Build (once per source + include, under a file lock: every TP rank
    imports this) and load the launcher."""
    global _lib, _lib_failed
    if _lib is not None or _lib_failed:
        return _lib
    try:
        inc = _opus_include()
        if inc is None:
            raise RuntimeError("aiter's opus.hpp not found")
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        # Linked against the HIP runtime file itself: a pip ROCm SDK's runtime
        # directory ships only the versioned soname, which `-lamdhip64` misses.
        runtimes = sorted(glob.glob(os.path.join(rocm, "lib", "libamdhip64.so*")))
        if not runtimes:
            raise RuntimeError(f"no libamdhip64 under {rocm}/lib")
        with open(_SRC, "rb") as f:
            src = f.read()
        key = hashlib.sha1(src + inc.encode() + runtimes[0].encode()).hexdigest()[:16]
        cache = os.path.join(
            os.path.expanduser("~"), ".cache", "atom", "dspark_markov_chain"
        )
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"markov_chain_{key}.so")
        with open(os.path.join(cache, ".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(so):
                hipcc = os.path.join(rocm, "bin", "hipcc")
                fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
                os.close(fd)
                cmd = [
                    hipcc,
                    "-O3",
                    "--offload-arch=gfx1250",
                    "-std=c++17",
                    "-fPIC",
                    "-shared",
                    f"-I{inc}",
                    _SRC,
                    "-o",
                    tmp,
                    "-no-hip-rt",
                    f"-Wl,{runtimes[0]}",
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if r.returncode:
                    os.unlink(tmp)
                    raise RuntimeError(f"hipcc failed: {r.stderr[-2000:]}")
                os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.markov_chain_launch.restype = ctypes.c_int
        lib.markov_chain_launch.argtypes = (
            [ctypes.c_int]
            + [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_long]
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            + [ctypes.c_void_p, ctypes.c_void_p]
            + [ctypes.c_int] * 4
            + [ctypes.c_void_p]
        )
        _lib = lib
    except Exception as e:  # noqa: BLE001 -- any failure keeps the two-stage path
        _lib_failed = True
        logger.warning(
            "DSpark Markov chain kernel unavailable, keeping the two-stage path: %s", e
        )
    return _lib


def usable(
    base: torch.Tensor, ids: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """``base`` [1, K, V] bf16 (one request's block), ``ids`` [1, K + 1]."""
    if base.dim() != 3 or base.shape[0] != 1 or ids.shape != (1, base.shape[1] + 1):
        return False
    K, V = base.shape[1], base.shape[2]
    dev = base.device
    if dev.type != "cuda" or not torch.cuda.get_device_properties(
        dev
    ).gcnArchName.startswith("gfx1250"):
        return False
    if not 1 <= K <= MAX_STEPS or V % 16 or ids.dtype not in (torch.int32, torch.int64):
        return False
    for w in (w1, w2):
        if w.dtype != torch.bfloat16 or w.shape != (V, RANK) or not w.is_contiguous():
            return False
    b = base[0]
    return (
        base.dtype == torch.bfloat16
        and b.stride(1) == 1
        and b.data_ptr() % 16 == 0
        and (b.stride(0) * 2) % 16 == 0
        and _load() is not None
    )


def markov_chain_bs1(base, ids, w1, w2) -> "torch.Tensor | None":
    """Drafted ids into ``ids[0, 1:]`` (``ids[0, 0]`` is the anchor) and the
    ``[K, 256]`` W1 rows of the K steps returned; None when not served."""
    global _lib_failed
    if not usable(base, ids, w1, w2):
        return None
    dev = base.device
    st = _state.get(dev.index)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            # The buffers must outlive the graph pool: allocated on an eager call.
            return None
        ncta = torch.cuda.get_device_properties(dev).multi_processor_count
        st = _state[dev.index] = _DeviceState(
            torch.zeros(MAX_STEPS * ncta, dtype=torch.int64, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
            ncta,
            torch.zeros(1, dtype=torch.int32, pin_memory=True),
            torch.cuda.Event(),
        )
    K = base.shape[1]
    embeds = torch.empty(K, RANK, dtype=w1.dtype, device=dev)
    b = base[0]
    rc = _lib.markov_chain_launch(
        int(ids.dtype == torch.int64),
        b.data_ptr(),
        b.stride(0),
        ids.data_ptr(),
        ids.stride(1),
        w1.data_ptr(),
        w2.data_ptr(),
        embeds.data_ptr(),
        embeds.stride(0),
        st.slots.data_ptr(),
        st.err.data_ptr(),
        b.shape[1],
        K,
        LDS_TILES,
        st.ncta,
        torch.cuda.current_stream(dev).cuda_stream,
    )
    if rc in _DOES_NOT_FIT:
        _lib_failed = True
        logger.warning(
            "DSpark Markov chain kernel declined (rc=%d): two-stage path", rc
        )
        return None
    if rc != 0:
        raise RuntimeError(f"DSpark Markov chain launch failed: rc={rc}")
    return embeds


def check_errors(every: int = 512) -> None:
    """Raise if a bounded wait of the one-launch chain gave up.

    Host side, outside any recording. Every ``every`` calls it reads the
    error word as copied the time before (through a pinned buffer and an
    event, so it never waits on the device), then copies it again.
    """
    global _calls
    _calls += 1
    if _calls % every:
        return
    for st in _state.values():
        if st.err_copied.query() and st.err_host.item():
            raise RuntimeError(
                "DSpark Markov chain: a workgroup of the one-launch chain was not "
                f"resident (steps 0x{st.err_host.item():x} waited it out), so those "
                "steps' drafts are not to be trusted. Something else holds CUs of "
                "this GPU; set ATOM_DSPARK_FUSED_MARKOV_SAMPLE=0 to sample with "
                "the two-stage path."
            )
        st.err_host.copy_(st.err, non_blocking=True)
        st.err_copied.record()
