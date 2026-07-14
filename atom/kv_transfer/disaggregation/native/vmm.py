# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""HIP VMM cross-process GPU buffer sharing (scale-up KV transfer).

A producer allocates an exportable VMM buffer and shares its POSIX fd (over a
UNIX socket); a consumer imports it, grants its device peer access, and copies
directly over the fabric. Unlike legacy hipIpc, VMM shareable handles are
reliable cross-process and don't need the source GPU visible to the consumer.
The C++ helper is JIT-compiled lazily (never at import), so importing is cheap
on CPU-only hosts.
"""

from __future__ import annotations

import functools
import os

import torch

__all__ = ["supported", "supported_fabric", "VmmBuffer"]


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    src = os.path.join(os.path.dirname(__file__), "_vmm_ext.cpp")
    return load(
        name="atom_vmm_ext",
        sources=[src],
        extra_include_paths=[os.path.join(rocm, "include")],
        extra_ldflags=[f"-L{os.path.join(rocm, 'lib')}", "-lamdhip64"],
        verbose=False,
    )


def supported(device: int = 0) -> bool:
    """Whether the device supports HIP Virtual Memory Management."""
    return bool(_ext().vmm_supported(device))


def supported_fabric(device: int = 0) -> bool:
    """Whether the device can export VMM buffers as *fabric* handles.

    Fabric handles are node-independent (shared over TCP) and enable the
    cross-node / scale-out (IFOE) path on gfx1250. False on scale-up-only parts
    (e.g. gfx950), where only the POSIX-fd / XGMI path is used.
    """
    return bool(_ext().vmm_supported_fabric(device))


def copy(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    """Device-to-device copy between raw device pointers (peer-mapped ok)."""
    _ext().vmm_copy(dst_ptr, src_ptr, nbytes)


class VmmBuffer:
    """An exportable VMM buffer mapped on one device.

    Create with :meth:`alloc` (producer) or :meth:`import_fd` (consumer). Use
    :meth:`tensor` to get a (non-owning) view for reads/writes/copies.
    """

    def __init__(self, region_id: int, nbytes: int, device: int):
        self._id = region_id
        self.nbytes = nbytes
        self.device = device

    @classmethod
    def alloc(cls, nbytes: int, device: int, fabric: bool = False) -> "VmmBuffer":
        """Allocate an exportable VMM buffer.

        ``fabric=True`` requests a node-independent fabric handle (scale-out /
        IFOE, gfx1250); the default POSIX-fd handle is scale-up / XGMI.
        """
        return cls(_ext().vmm_alloc(nbytes, device, fabric), nbytes, device)

    @classmethod
    def import_fd(cls, fd: int, nbytes: int, device: int) -> "VmmBuffer":
        """Import a producer's exported fd and map it on ``device``.

        ``nbytes`` must match the producer's ``alloc`` size.
        """
        return cls(_ext().vmm_import(fd, nbytes, device), nbytes, device)

    @classmethod
    def import_fabric(cls, handle: bytes, nbytes: int, device: int) -> "VmmBuffer":
        """Import a producer's 64-byte fabric handle (received over TCP)."""
        return cls(_ext().vmm_import_fabric(handle, nbytes, device), nbytes, device)

    def export_fd(self) -> int:
        """POSIX fd to send to a peer (e.g. via ``socket.send_fds``)."""
        return _ext().vmm_export_fd(self._id)

    def export_fabric(self) -> bytes:
        """64-byte fabric handle to send to a remote node (e.g. over TCP)."""
        return _ext().vmm_export_fabric(self._id)

    @property
    def data_ptr(self) -> int:
        """Raw mapped device pointer (for :func:`copy`)."""
        return _ext().vmm_ptr(self._id)

    def tensor(self, dtype: torch.dtype, shape) -> torch.Tensor:
        """View of the buffer as ``dtype`` reshaped to ``shape``.

        The returned tensor keeps this :class:`VmmBuffer` (and therefore the
        underlying VMM mapping) alive for its own lifetime, so callers may
        drop the buffer reference and keep only the tensor.
        """
        flat = _ext().vmm_tensor(self._id, self.nbytes, self.device)  # uint8
        view = flat.view(dtype).view(*shape)
        view._vmm_keepalive = self  # tie mapping lifetime to the tensor
        return view

    def close(self) -> None:
        if self._id is not None:
            _ext().vmm_free(self._id)
            self._id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
