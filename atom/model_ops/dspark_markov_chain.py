# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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
of eight ``v_wmma_f32_16x16x32_bf16`` its Triton stage 1 emits on gfx1250 (W2 in
the first operand slot, the embedding row in the second, the same K split
across lanes), ``fp32(base) + acc``, lowest id on ties. Checked op-level on the
ids, the W1 rows and every biased logit of all K steps, eager and graph replay.

Served when (``usable``): gfx1250, ONE request (the block's rows are its K
positions), bf16 logits / tables of rank 256, a vocabulary of whole 16-id
tiles, K <= 8, and every CU free for one workgroup (the launch checks its tile
split fits). Anything else keeps the two-stage path.

The waits are bounded. A wait that gives up (a workgroup not resident, which a
dedicated GPU does not hit) sets a bit in an error word and the step continues
on what it has: the drafted ids are then wrong but only drafted -- verification
rejects them. ``poll_errors`` reports such a word to the log.
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

_SRC = os.path.join(
    os.path.dirname(__file__), "kernels", "dspark_markov_chain_gfx1250.hip"
)
RANK = 256
MAX_STEPS = 8
LDS_TILES = 38
_lib = None
_lib_failed = False
_state: dict = {}  # device index -> (slots, err)
_err_reported = 0
_calls = 0


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
        with open(_SRC, "rb") as f:
            src = f.read()
        key = hashlib.sha1(src + inc.encode()).hexdigest()[:16]
        cache = os.path.join(
            os.path.expanduser("~"), ".cache", "atom", "dspark_markov_chain"
        )
        os.makedirs(cache, exist_ok=True)
        so = os.path.join(cache, f"markov_chain_{key}.so")
        with open(os.path.join(cache, ".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not os.path.exists(so):
                hipcc = os.path.join(
                    os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc"
                )
                fd, tmp = tempfile.mkstemp(suffix=".so", dir=cache)
                os.close(fd)
                cmd = [
                    hipcc,
                    "-O3",
                    "--offload-arch=gfx1250",
                    "-std=c++17",
                    "-fPIC",
                    "-shared",
                    "-DFILL_TDM=1",
                    f"-I{inc}",
                    _SRC,
                    "-o",
                    tmp,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if r.returncode:
                    os.unlink(tmp)
                    raise RuntimeError(f"hipcc failed: {r.stderr[-2000:]}")
                os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.markov_chain_launch.restype = ctypes.c_int
        lib.markov_chain_launch.argtypes = (
            [ctypes.c_int] * 4
            + [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_long]
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            + [ctypes.c_int] * 4
            + [ctypes.c_void_p, ctypes.c_void_p]
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
    if not usable(base, ids, w1, w2):
        return None
    dev = base.device
    st = _state.get(dev.index)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return (
                None  # buffers must outlive the graph pool: allocate on an eager call
            )
        ncta = torch.cuda.get_device_properties(dev).multi_processor_count
        st = _state[dev.index] = (
            torch.zeros(MAX_STEPS * ncta + MAX_STEPS, dtype=torch.int64, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
            ncta,
        )
    slots, err, ncta = st
    K = base.shape[1]
    embeds = torch.empty(K, RANK, dtype=w1.dtype, device=dev)
    b = base[0]
    rc = _lib.markov_chain_launch(
        0,
        0,
        int(ids.dtype == torch.int64),
        0,
        b.data_ptr(),
        b.stride(0),
        ids.data_ptr(),
        ids.stride(1),
        w1.data_ptr(),
        w2.data_ptr(),
        embeds.data_ptr(),
        embeds.stride(0),
        slots.data_ptr(),
        err.data_ptr(),
        None,
        b.shape[1],
        K,
        LDS_TILES,
        ncta,
        torch.cuda.current_stream(dev).cuda_stream,
        None,
    )
    if rc != 0:
        if rc in (
            -3,
            -4,
        ):  # tile split / LDS does not fit this device: never served here
            global _lib_failed
            _lib_failed = True
            logger.warning(
                "DSpark Markov chain kernel declined (rc=%d): two-stage path", rc
            )
            return None
        raise RuntimeError(f"DSpark Markov chain launch failed: rc={rc}")
    return embeds


def poll_errors(every: int = 512) -> None:
    """Report a bounded wait that gave up (read every ``every`` calls: a sync)."""
    global _calls, _err_reported
    _calls += 1
    if _calls % every or not _state:
        return
    for slots, err, _ in _state.values():
        v = int(err.item())
        if v:
            _err_reported += 1
            err.zero_()
            logger.error(
                "DSpark Markov chain: a bounded wait gave up (step mask 0x%x, report %d); "
                "drafts of that step were wrong, verification rejected them",
                v,
                _err_reported,
            )
