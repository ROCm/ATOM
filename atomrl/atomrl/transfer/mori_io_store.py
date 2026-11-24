# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""P2P GPU-to-GPU hidden state transfer for Eagle3 SDDD.

Replaces Mooncake TCP with direct GPU P2P copy via CUDA/HIP IPC.
Producer (vLLM, src GPU) packs tensors into a local buffer, then
copies directly into consumer's (trainer, dst GPU) pre-registered
buffer using cudaMemcpyPeer. Completion signaled via /dev/shm.

Both processes share GPU memory handles via CUDA IPC (hipIPC on ROCm).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any

import torch

from atomrl.transfer.eagle_mooncake_store import (
    _DTYPE_ELEMENT_SIZES,
    HIDDEN_STATES_STORAGE_DTYPE,
    Eagle3TargetOutput,
    calculate_eagle3_buffer_size,
)

logger = logging.getLogger(__name__)

_DESC_DIR = "/dev/shm/mori_io"
_READY_DIR = "/dev/shm/mori_io_ready"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_ipc_handle(tensor: torch.Tensor, path: str) -> None:
    """Save a CUDA tensor's IPC handle so another process can open it."""
    storage = tensor.untyped_storage()
    handle = storage._share_cuda_()
    with open(path, "wb") as f:
        pickle.dump(
            {
                "ipc_handle": handle,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "device": tensor.device.index,
                "nbytes": storage.nbytes(),
            },
            f,
        )


def _load_ipc_tensor(path: str) -> torch.Tensor:
    """Reconstruct a CUDA tensor from an IPC handle saved by another process."""
    with open(path, "rb") as f:
        info = pickle.load(f)

    handle = info["ipc_handle"]
    storage = torch.UntypedStorage._new_shared_cuda(*handle)
    tensor = (
        torch.tensor([], dtype=info["dtype"])
        .set_(
            storage.typed(info["dtype"]),
        )
        .reshape(info["shape"])
    )
    return tensor


class MoriIOStore:
    """P2P GPU-to-GPU hidden state transfer via CUDA/HIP IPC.

    Producer and consumer share GPU buffers via IPC handles
    exchanged through /dev/shm. Transfers use cudaMemcpyPeer
    (hipMemcpyPeer on ROCm) — no CPU staging.
    """

    def __init__(
        self,
        role: str,
        src_gpu: int,
        dst_gpu: int,
        host: str = "127.0.0.1",
        port: int = 0,
        qp_per_transfer: int = 2,
        max_seq_len: int = 8192,
        hidden_dim: int = 4096,
        num_aux_layers: int = 3,
    ) -> None:
        assert role in ("producer", "consumer")
        self._role = role
        self._src_gpu = src_gpu
        self._dst_gpu = dst_gpu
        self._max_seq_len = max_seq_len
        self._hidden_dim = hidden_dim
        self._num_aux_layers = num_aux_layers

        self._buffer = None
        self._peer_buffer = None
        self._initialized = False

    @property
    def _my_gpu(self) -> int:
        return self._src_gpu if self._role == "producer" else self._dst_gpu

    @property
    def _peer_role(self) -> str:
        return "consumer" if self._role == "producer" else "producer"

    def _buffer_size(self) -> int:
        return calculate_eagle3_buffer_size(
            max_seq_len=self._max_seq_len,
            batch_size=1,
            hidden_dim=self._hidden_dim,
            num_aux_layers=self._num_aux_layers,
            safety_margin=2.0,
        )

    def setup(self, device=None) -> None:
        if self._initialized:
            return

        _ensure_dir(_DESC_DIR)
        _ensure_dir(_READY_DIR)

        buf_size = self._buffer_size()
        my_device = torch.device(f"cuda:{self._my_gpu}")
        self._buffer = torch.zeros(buf_size, dtype=torch.uint8, device=my_device)
        logger.info(
            "MoriIO %s: allocated %dMB buffer on GPU %d",
            self._role,
            buf_size // (1024 * 1024),
            self._my_gpu,
        )

        # Export our buffer's IPC handle so peer process can access it
        my_ipc_path = os.path.join(_DESC_DIR, f"{self._role}_ipc.bin")
        _save_ipc_handle(self._buffer, my_ipc_path)
        logger.info("MoriIO %s: saved IPC handle to %s", self._role, my_ipc_path)

        # Wait for peer's IPC handle
        peer_ipc_path = os.path.join(_DESC_DIR, f"{self._peer_role}_ipc.bin")
        deadline = time.time() + 120
        while not os.path.exists(peer_ipc_path):
            if time.time() > deadline:
                raise TimeoutError(
                    f"MoriIO {self._role}: timed out waiting for peer IPC at {peer_ipc_path}"
                )
            time.sleep(0.5)

        time.sleep(1.0)
        self._peer_buffer = _load_ipc_tensor(peer_ipc_path)
        logger.info(
            "MoriIO %s: loaded peer buffer via IPC (device=%s, size=%dMB)",
            self._role,
            self._peer_buffer.device,
            self._peer_buffer.nbytes // (1024 * 1024),
        )

        self._initialized = True
        logger.info(
            "MoriIO %s: setup complete (src=GPU%d, dst=GPU%d, P2P IPC)",
            self._role,
            self._src_gpu,
            self._dst_gpu,
        )

    def _pack_tensors(self, tensors: list[torch.Tensor]) -> tuple[list[int], list[int]]:
        offsets, sizes = [], []
        offset = 0
        for t in tensors:
            t = t.contiguous()
            nbytes = t.numel() * t.element_size()
            self._buffer[offset : offset + nbytes].copy_(t.view(torch.uint8).view(-1))
            offsets.append(offset)
            sizes.append(nbytes)
            offset += nbytes
        return offsets, sizes

    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Producer: pack tensors into local buffer, P2P copy to consumer's buffer."""
        assert self._role == "producer", "put() only on producer"
        assert self._initialized, "call setup() first"

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)

        tensors = [hidden_states, input_ids]
        if last_hidden_states is not None:
            tensors.append(last_hidden_states)

        offsets, sizes = self._pack_tensors(tensors)
        total_bytes = sum(sizes)

        # P2P copy: local buffer (src GPU) → peer buffer (dst GPU) via IPC
        self._peer_buffer[:total_bytes].copy_(self._buffer[:total_bytes])
        torch.cuda.synchronize(self._src_gpu)

        meta = {
            "offsets": offsets,
            "sizes": sizes,
            "total_bytes": total_bytes,
            "has_last_hidden_states": last_hidden_states is not None,
        }
        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": str(hidden_states.dtype),
            "input_ids": str(input_ids.dtype),
        }
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = str(last_hidden_states.dtype)

        ready_path = os.path.join(_READY_DIR, f"{key}.json")
        with open(ready_path, "w") as f:
            json.dump(meta, f)

        return {"shapes": shapes, "dtypes": dtypes}

    def get(
        self,
        key: str,
        shapes: dict,
        dtypes: dict,
        device: torch.device,
    ) -> Eagle3TargetOutput:
        """Consumer: wait for transfer, slice tensors from local GPU buffer."""
        assert self._role == "consumer", "get() only on consumer"
        assert self._initialized, "call setup() first"

        ready_path = os.path.join(_READY_DIR, f"{key}.json")
        deadline = time.time() + 60
        while not os.path.exists(ready_path):
            if time.time() > deadline:
                raise TimeoutError(f"MoriIO get: timed out waiting for {key}")
            time.sleep(0.01)

        with open(ready_path, "r") as f:
            meta = json.load(f)

        offsets = meta["offsets"]

        def _slice_tensor(idx, shape, dtype):
            numel = 1
            for d in shape:
                numel *= d
            nbytes = numel * _DTYPE_ELEMENT_SIZES[dtype]
            return (
                self._buffer[offsets[idx] : offsets[idx] + nbytes]
                .view(dtype)[:numel]
                .reshape(shape)
            )

        hs_shape = shapes["hidden_states"]
        ids_shape = shapes["input_ids"]
        hs_dtype = dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE)
        if isinstance(hs_dtype, str):
            hs_dtype = getattr(torch, hs_dtype.replace("torch.", ""))

        hidden_states = _slice_tensor(0, hs_shape, hs_dtype)
        input_ids = _slice_tensor(1, ids_shape, torch.int64)
        input_ids_cpu = input_ids.cpu().clone()

        last_hidden_states = None
        if "last_hidden_states" in shapes and meta.get("has_last_hidden_states"):
            lhs_shape = shapes["last_hidden_states"]
            last_hidden_states = _slice_tensor(2, lhs_shape, hs_dtype)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            last_hidden_states=last_hidden_states,
            input_ids_cpu=input_ids_cpu,
        )

    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
    ) -> None:
        ready_path = os.path.join(_READY_DIR, f"{key}.json")
        try:
            os.remove(ready_path)
        except FileNotFoundError:
            pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._peer_buffer = None
        self._buffer = None
        self._initialized = False

        for f in [
            os.path.join(_DESC_DIR, f"{self._role}_ipc.bin"),
        ]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
