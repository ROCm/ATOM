# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""ctypes binding to the HSA runtime HIP itself loaded (the SDMA copies), and to HIP's stream memory operations.

The binding must reach the runtime that owns this process's device context: an image can ship several
``libhsa-runtime64.so`` copies, as it ships several HIP runtimes (``amd_kernels.common.hip_runtime``), so the
library is taken from the process's own mappings after torch has initialized HIP.
"""

import ctypes
import functools
import os

import torch

from amd_kernels.common.hip_runtime import load_hip_runtime

HSA_AGENT_INFO_DEVICE = 17
HSA_DEVICE_TYPE_GPU = 1
HSA_AMD_AGENT_INFO_BDFID = 0xA006
HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1  # the value lives in device memory, where a HIP stream can write it
HSA_SIGNAL_CONDITION_LT = 2
HSA_WAIT_STATE_ACTIVE = 1
HIP_STREAM_WAIT_VALUE_GTE = 0


class Agent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class Signal(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


def _mapped(name: str) -> str | None:
    """The path of a shared library already mapped into this process."""
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1]
            if os.path.basename(path).startswith(name):
                return path
    return None


@functools.cache
def hsa() -> ctypes.CDLL:
    torch.cuda.init()
    lib = ctypes.CDLL(_mapped("libhsa-runtime64.so") or "libhsa-runtime64.so.1")
    rc = lib.hsa_init()  # reference counted: HIP has already initialized it
    assert rc == 0, f"hsa_init failed: {rc:#x}"
    visit = ctypes.CFUNCTYPE(ctypes.c_int, Agent, ctypes.c_void_p)
    lib.hsa_iterate_agents.argtypes = [visit, ctypes.c_void_p]
    lib.hsa_agent_get_info.argtypes = [Agent, ctypes.c_int, ctypes.c_void_p]
    lib.hsa_amd_signal_create.argtypes = [ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
                                          ctypes.POINTER(Signal)]  # fmt: skip
    lib.hsa_signal_store_relaxed.argtypes = [Signal, ctypes.c_int64]
    lib.hsa_signal_load_relaxed.argtypes = [Signal]
    lib.hsa_signal_load_relaxed.restype = ctypes.c_int64
    lib.hsa_signal_wait_scacquire.argtypes = [Signal, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64, ctypes.c_int]
    lib.hsa_signal_wait_scacquire.restype = ctypes.c_int64
    lib.hsa_amd_signal_value_pointer.argtypes = [Signal, ctypes.POINTER(ctypes.c_void_p)]
    lib.hsa_amd_memory_copy_engine_status.argtypes = [Agent, Agent, ctypes.POINTER(ctypes.c_uint32)]
    lib.hsa_amd_memory_async_copy_on_engine.argtypes = [
        ctypes.c_void_p, Agent, ctypes.c_void_p, Agent, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(Signal),
        Signal, ctypes.c_uint32, ctypes.c_bool,
    ]  # fmt: skip
    lib._visit_type = visit
    return lib


@functools.cache
def hip() -> ctypes.CDLL:
    lib = load_hip_runtime()
    lib.hipStreamWriteValue64.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint]
    lib.hipStreamWaitValue64.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint,
                                         ctypes.c_uint64]  # fmt: skip
    return lib


@functools.cache
def gpu_agents() -> dict[int, Agent]:
    """Every GPU agent of the process by its PCI bus / device id (the function bits cleared)."""
    lib, out = hsa(), {}

    def visit(agent, _):
        kind = ctypes.c_uint32()
        lib.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, ctypes.byref(kind))
        if kind.value == HSA_DEVICE_TYPE_GPU:
            bdf = ctypes.c_uint32()
            lib.hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_BDFID, ctypes.byref(bdf))
            out[bdf.value & ~0x7] = Agent(agent.handle)
        return 0

    callback = lib._visit_type(visit)
    rc = lib.hsa_iterate_agents(callback, None)
    assert rc == 0, f"hsa_iterate_agents failed: {rc:#x}"
    return out


def pci_key(device: torch.device) -> int:
    """The key of ``gpu_agents`` for a torch device."""
    p = torch.cuda.get_device_properties(device)
    return (p.pci_bus_id << 8) | (p.pci_device_id << 3)


def signal(value: int = 1) -> Signal:
    s = Signal()
    rc = hsa().hsa_amd_signal_create(value, 0, None, HSA_AMD_SIGNAL_AMD_GPU_ONLY, ctypes.byref(s))
    assert rc == 0, f"hsa_amd_signal_create failed: {rc:#x}"
    return s


def value_pointer(s: Signal) -> int:
    p = ctypes.c_void_p()
    rc = hsa().hsa_amd_signal_value_pointer(s, ctypes.byref(p))
    assert rc == 0, f"hsa_amd_signal_value_pointer failed: {rc:#x}"
    return p.value
