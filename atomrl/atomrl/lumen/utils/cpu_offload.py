# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU offload for autograd saved tensors in Lumen.

Uses ``torch.autograd.graph.saved_tensors_hooks`` to move tensors saved for
backward to pinned CPU memory, freeing GPU memory during forward. Dedicated
CUDA streams overlap D2H work with compute; unpack restores tensors on GPU
during backward.
"""

from contextlib import contextmanager

import torch


class CPUOffloadManager:
    """Offload autograd-saved tensors to pinned CPU memory."""

    def __init__(self, enabled=True, pin_memory=True):
        self.enabled = enabled
        self.pin_memory = pin_memory
        self._d2h_streams = {}
        self._h2d_streams = {}
        self._offloaded_bytes = 0

    def _get_d2h_stream(self, device):
        if device not in self._d2h_streams:
            self._d2h_streams[device] = torch.cuda.Stream(device=device)
        return self._d2h_streams[device]

    def _get_h2d_stream(self, device):
        if device not in self._h2d_streams:
            self._h2d_streams[device] = torch.cuda.Stream(device=device)
        return self._h2d_streams[device]

    @property
    def memory_saved_bytes(self):
        return self._offloaded_bytes

    def _pack(self, tensor):
        if not tensor.is_cuda or not self.enabled:
            return tensor
        if tensor.nelement() * tensor.element_size() < 1024:
            return tensor
        device = tensor.device
        d2h = self._get_d2h_stream(device)
        event = torch.cuda.current_stream(device).record_event()
        with torch.cuda.stream(d2h):
            d2h.wait_event(event)
            cpu = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                pin_memory=self.pin_memory,
            )
            cpu.copy_(tensor, non_blocking=True)
            cpu._d2h_event = d2h.record_event()
            cpu._src_device = device
        self._offloaded_bytes += tensor.nelement() * tensor.element_size()
        return cpu

    def _unpack(self, packed):
        if not isinstance(packed, torch.Tensor) or packed.is_cuda:
            return packed
        if hasattr(packed, "_d2h_event"):
            packed._d2h_event.synchronize()
        device = getattr(packed, "_src_device", torch.device("cuda"))
        h2d = self._get_h2d_stream(device)
        event = torch.cuda.current_stream(device).record_event()
        with torch.cuda.stream(h2d):
            h2d.wait_event(event)
            gpu = packed.to(device, non_blocking=True)
        h2d.synchronize()
        return gpu


@contextmanager
def lumen_cpu_offload_context(enabled=True, pin_memory=True):
    """Context manager wrapping ``saved_tensors_hooks`` for CPU offload.

    Usage::

        with lumen_cpu_offload_context(enabled=True) as mgr:
            output = model(input)
            loss = criterion(output, target)
        loss.backward()

    Args:
        enabled: When False, no hooks are installed.
        pin_memory: Allocate pinned CPU buffers for async copies.
    """
    mgr = CPUOffloadManager(enabled=enabled, pin_memory=pin_memory)
    if not enabled:
        yield mgr
        return
    with torch.autograd.graph.saved_tensors_hooks(mgr._pack, mgr._unpack):
        yield mgr
