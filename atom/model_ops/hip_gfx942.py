# SPDX-License-Identifier: MIT
"""gfx942 HIP kernels built with hipcc on first use and launched on torch's stream."""

import ctypes
import hashlib
import os
import subprocess
import tempfile

import torch


def _loaded_hip_library():
    # Bind to the libamdhip64 torch mapped: ROCm 10 wheels carry a second copy,
    # and launches through it on torch's streams fail (hipErrorContextIsDestroyed).
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1]
            if "libamdhip64" in os.path.basename(path):
                return path
    return "libamdhip64.so"


class Hip:
    def __init__(self):
        torch.cuda.current_device()
        lib = ctypes.CDLL(_loaded_hip_library())
        lib.hipModuleLoad.restype = ctypes.c_int
        lib.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.hipModuleGetFunction.restype = ctypes.c_int
        lib.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.hipModuleLaunchKernel.restype = ctypes.c_int
        lib.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib = lib

    def check(self, rc, what):
        if rc != 0:
            raise RuntimeError(
                f"{what} failed: {rc} ({self.lib.hipGetErrorString(rc).decode()})"
            )

    def launch(self, fn, grid, args):
        size = ctypes.c_size_t(ctypes.sizeof(args))
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.byref(args), ctypes.c_void_p).value,
            2,
            ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value,
            3,
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(
            self.lib.hipModuleLaunchKernel(
                fn,
                grid,
                1,
                1,
                256,
                1,
                1,
                0,
                stream,
                None,
                ctypes.cast(extra, ctypes.c_void_p),
            ),
            "hipModuleLaunchKernel",
        )


def _code_object(src, flags):
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read() + " ".join(flags).encode()).hexdigest()[:16]
    cache_dir = os.path.join(tempfile.gettempdir(), "atom_hip_cache")
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    co = os.path.join(cache_dir, f"{name}_{digest}.co")
    if not os.path.exists(co):
        hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
        fd, tmp = tempfile.mkstemp(suffix=".co", dir=cache_dir)
        os.close(fd)
        cmd = [hipcc, "--genco", "--offload-arch=gfx942", "-O3", *flags, "-o", tmp, src]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, co)
    return co


def load(src, names, flags=()):
    """`(hip, {name: function})` for the kernels `names` in HIP source `src`.
    Raises OSError, CalledProcessError or RuntimeError when it cannot build or load."""
    hip = Hip()
    module = ctypes.c_void_p()
    hip.check(
        hip.lib.hipModuleLoad(
            ctypes.byref(module), _code_object(src, tuple(flags)).encode()
        ),
        "hipModuleLoad",
    )
    return hip, {n: _function(hip, module, n) for n in names}


def _function(hip, module, name):
    fn = ctypes.c_void_p()
    hip.check(
        hip.lib.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()), name
    )
    return fn
