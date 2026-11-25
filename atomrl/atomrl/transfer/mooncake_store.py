# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Mooncake distributed store wrapper for hidden state transfer."""

from __future__ import annotations

import ctypes
import logging
import socket
import threading
import time
from typing import Any

import torch

from atomrl.transfer.mooncake_config import MooncakeConfig

logger = logging.getLogger(__name__)

STORAGE_DTYPE = torch.bfloat16

_DTYPE_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


class MooncakeStore:
    """Wrapper around MooncakeDistributedStore for hidden state transfer.

    Provides put/get/remove operations with RDMA or TCP transport.
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store = None
        self._initialized = False
        self._lock = threading.Lock()

    def setup(self, device: torch.device | None = None) -> None:
        if self._initialized:
            return

        from mooncake.store import MooncakeDistributedStore

        local_hostname = self.config.local_hostname
        if not local_hostname:
            local_hostname = socket.gethostname()

        metadata_server = self.config.metadata_server
        if not metadata_server:
            host = self.config.master_server_address.split(":")[0]
            metadata_server = f"http://{host}:8090/metadata"

        self._store = MooncakeDistributedStore()
        result = self._store.setup(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            global_segment_size=self.config.global_segment_size_bytes,
            local_buffer_size=self.config.local_buffer_size_bytes,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result is not None and result != 0:
            raise RuntimeError(
                f"Mooncake setup failed (error={result}). "
                f"Master: {self.config.master_server_address}, "
                f"Metadata: {metadata_server}"
            )

        self._initialized = True
        logger.info(
            "MooncakeStore initialized (protocol=%s, device=%s)",
            self.config.protocol,
            self.config.device_name or "(auto)",
        )

    def put_tensors(
        self,
        key: str,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Store tensors with key suffixes. Returns shapes/dtypes metadata."""
        self._ensure_initialized()

        keys = []
        data_list = []
        shapes = {}
        dtypes = {}

        for name, tensor in tensors.items():
            suffix_key = f"{key}_{name}"
            if tensor.dtype in (torch.float32, torch.float64) and name != "input_ids":
                tensor = tensor.to(STORAGE_DTYPE)
            cpu_tensor = tensor.contiguous().cpu()
            keys.append(suffix_key)
            data_list.append(cpu_tensor)
            shapes[name] = tuple(tensor.shape)
            dtypes[name] = str(tensor.dtype)

        ptrs = [t.data_ptr() for t in data_list]
        sizes = [t.nelement() * t.element_size() for t in data_list]

        results = self._store.batch_put_from(keys, ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            detail = ", ".join(f"{k}(err={r})" for k, r in failures)
            raise RuntimeError(f"batch_put_from failed: {detail}")

        return {"shapes": shapes, "dtypes": dtypes}

    def get_tensors(
        self,
        key: str,
        shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, str],
        device: torch.device,
        max_wait: float = 30.0,
    ) -> dict[str, torch.Tensor]:
        """Retrieve tensors from Mooncake store."""
        self._ensure_initialized()

        keys = []
        specs = []
        for name, shape in shapes.items():
            keys.append(f"{key}_{name}")
            dt = getattr(torch, dtypes[name].replace("torch.", ""))
            specs.append((name, shape, dt))

        start = time.time()
        while True:
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, b in enumerate(buffers) if b is None]
            if not missing:
                break
            if time.time() - start > max_wait:
                mk = ", ".join(keys[i] for i in missing)
                raise RuntimeError(f"Timed out waiting for keys: {mk}")
            time.sleep(0.1)

        result = {}
        for i, (name, shape, dt) in enumerate(specs):
            buf = buffers[i]
            numel = 1
            for d in shape:
                numel *= d
            nbytes = numel * _DTYPE_SIZES[dt]
            c_arr = (ctypes.c_byte * nbytes).from_address(buf.ptr())
            host_t = torch.frombuffer(c_arr, dtype=dt, count=numel).reshape(shape)
            result[name] = host_t.to(device)

        return result

    def remove(self, key: str, suffixes: list[str]) -> None:
        """Remove tensors by key + suffixes."""
        keys = [f"{key}_{s}" for s in suffixes]
        try:
            self._store.batch_remove(keys, force=True)
        except Exception:
            logger.warning("Failed to remove keys for %s", key, exc_info=True)

    def close(self) -> None:
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.setup()
