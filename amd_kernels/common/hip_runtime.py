# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Binding to the HIP runtime that torch itself uses.

Images ship several ``libamdhip64.so`` copies (/opt/rocm, ``torch/lib``,
triton). A bare ``CDLL("libamdhip64.so")`` resolves through the loader search
path, so once UCX/NIXL (PD KV transfer) has loaded the /opt/rocm copy, ctypes
binds that second, uninitialized runtime instead of the one owning the device
context. The raw HIP calls in this package must all resolve there, as in the
``_C_stable_libtorch`` CMake target.

``allocate_shared_buffer_and_handle`` / ``open_mem_handle`` are vLLM's
``_custom_ops`` of the same names (``csrc/custom_all_reduce.cu``) through ctypes.
"""

import ctypes
import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_HIP_LIB_NAME = "libamdhip64.so"
_HIP_DEVICE_MALLOC_UNCACHED = 0x3  # hip_runtime_api.h: hipDeviceMallocUncached
_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


@functools.cache
def _torch_lib_dir() -> str:
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def hip_runtime_path() -> str:
    """Path of torch's HIP runtime, or the bare soname when torch bundles none.

    A source build against the system ROCm has a single runtime in the process,
    which the bare soname resolves to as before.
    """
    torch_hip = os.path.join(_torch_lib_dir(), _HIP_LIB_NAME)
    if os.path.exists(torch_hip):
        return torch_hip
    logger.debug(
        "PyTorch does not bundle %s; ctypes will bind the first copy on the "
        "loader path, which is an uninitialized runtime when a KV connector has "
        "already loaded the system ROCm copy.",
        _HIP_LIB_NAME,
    )
    return _HIP_LIB_NAME


def load_hip_runtime() -> ctypes.CDLL:
    """A fresh ctypes handle on the HIP runtime to bind.

    Not cached: callers set ``argtypes`` on the function objects of their handle,
    and the all-reduce + norm host, the pull collectives and this module each
    declare ``hipIpcGetMemHandle`` / ``hipIpcOpenMemHandle`` with their own
    ``_IpcHandle`` type. A shared
    ``CDLL`` would share those function objects, so the last caller's argtypes
    would win and the other module's calls would fail ctypes type checks. Each
    ``CDLL`` of the same path reuses the already-loaded library.
    """
    return ctypes.CDLL(hip_runtime_path())


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = load_hip_runtime()
    lib.hipExtMallocWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


def allocate_shared_buffer_and_handle(nbytes: int) -> tuple[int, bytes]:
    """A zeroed ``nbytes`` buffer in uncached device memory on the current device
    (``hipExtMallocWithFlags(hipDeviceMallocUncached)``: the flag words the peers
    poll), never freed -> (its address, the 64 raw bytes of its IPC handle)."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    rc = hip.hipExtMallocWithFlags(
        ctypes.byref(ptr), nbytes, _HIP_DEVICE_MALLOC_UNCACHED
    )
    assert rc == 0, f"hipExtMallocWithFlags({nbytes}) failed: {rc}"
    rc = hip.hipMemset(ptr, 0, nbytes)
    assert rc == 0, f"hipMemset failed: {rc}"
    torch.cuda.synchronize()
    h = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(h), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    # the raw bytes: a c_char field reads back only up to its first NUL
    return ptr.value, ctypes.string_at(ctypes.addressof(h), ctypes.sizeof(h))


def open_mem_handle(raw: bytes) -> int:
    """A peer's buffer mapped here (``hipIpcOpenMemHandle``, lazy peer access)."""
    hip = _hip()
    h = _IpcHandle()
    assert len(raw) == ctypes.sizeof(h)
    ctypes.memmove(ctypes.addressof(h), raw, len(raw))
    ptr = ctypes.c_void_p()
    rc = hip.hipIpcOpenMemHandle(
        ctypes.byref(ptr), h, _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
    )
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value
