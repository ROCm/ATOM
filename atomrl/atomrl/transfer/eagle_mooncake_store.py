# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Eagle3 Mooncake store for speculative decoding hidden state transfer.

Eagle3 Mooncake store. Provides async RDMA/TCP
transfers of hidden states between inference and FSDP2 training.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
import typing
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16

_DTYPE_ELEMENT_SIZES = {
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


def calculate_eagle3_buffer_size(
    max_seq_len: int,
    batch_size: int,
    hidden_dim: int,
    num_aux_layers: int = 3,
    include_last_hidden_states: bool = True,
    safety_margin: float = 1.1,
) -> int:
    bfloat16_size = 2
    int64_size = 8
    total = (
        batch_size * max_seq_len * hidden_dim * num_aux_layers * bfloat16_size
        + batch_size * max_seq_len * int64_size
    )
    if include_last_hidden_states:
        total += batch_size * max_seq_len * hidden_dim * bfloat16_size
    aligned = (int(total * safety_margin) + 255) // 256 * 256
    return aligned


def _format_bytes(size: int) -> str:
    if size < 0:
        return f"{size}B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024


# ---------------------------------------------------------------------------
# Host / GPU buffer classes
# ---------------------------------------------------------------------------


class HostBuffer:
    def __init__(self, size: int):
        self.size = size
        self._tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if offset + nbytes > self.size:
            raise ValueError(
                f"Buffer overflow: need {offset + nbytes}, have {self.size}"
            )
        self._tensor[offset : offset + nbytes].copy_(tensor.view(torch.uint8).view(-1))
        return nbytes

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


class HostBufferPool:
    def __init__(self, buffer_size: int, pool_size: int = 2):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._buffers: list[HostBuffer] = []
        self._current_idx = 0

    def initialize(self) -> None:
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
        logger.info(
            "Host buffer pool: %d x %.1fGB", self.pool_size, self.buffer_size / 1024**3
        )

    def get_buffer(self) -> HostBuffer:
        if not self._buffers:
            self.initialize()
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()


class AsyncPutManager:
    def __init__(self, store: Any, max_workers: int = 1, replicate_config: Any = None):
        self._store = store
        self._replicate_config = replicate_config
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="async_put"
        )
        self._in_flight: dict[int, Future] = {}
        self._last_error: BaseException | None = None
        self._put_lock = threading.Lock()
        self._flight_lock = threading.Lock()

    def check_last_error(self) -> None:
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise err

    def wait_for_buffer(self, buffer_ptr: int) -> None:
        with self._flight_lock:
            future = self._in_flight.pop(buffer_ptr, None)
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:
            self._last_error = exc
            raise

    def submit(
        self,
        keys,
        buffer_ptrs,
        sizes,
        owner_buffer_ptr,
        wait_event=None,
        device_index=None,
    ):
        future = self._executor.submit(
            self._do_put, keys, buffer_ptrs, sizes, wait_event, device_index
        )
        with self._flight_lock:
            self._in_flight[owner_buffer_ptr] = future

    def _do_put(self, keys, buffer_ptrs, sizes, wait_event=None, device_index=None):
        if wait_event is not None:
            if device_index is not None:
                torch.cuda.set_device(device_index)
            wait_event.synchronize()
        with self._put_lock:
            if self._replicate_config is not None:
                results = self._store.batch_put_from(
                    keys, buffer_ptrs, sizes, config=self._replicate_config
                )
            else:
                results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            try:
                self._store.batch_remove(keys, force=True)
            except (
                OSError,
                RuntimeError,
            ):  # Failed put cleanup must preserve the primary store error.
                pass
            detail = ", ".join(f"{k} (code={r})" for k, r in failures)
            raise RuntimeError(f"async batch_put_from failed: {detail}")

    def drain(self) -> None:
        with self._flight_lock:
            items = list(self._in_flight.items())
        for _, future in items:
            try:
                future.result()
            # Drain records arbitrary executor task failures for the caller.
            except Exception as exc:  # noqa: BLE001 -- boundary
                if self._last_error is None:
                    self._last_error = exc
        with self._flight_lock:
            self._in_flight.clear()

    def shutdown(self) -> None:
        self.drain()
        self._executor.shutdown(wait=True)


class GPUReceiveBuffer:
    def __init__(self, size: int, device=None):
        self.size = size
        self.device = torch.device(device) if device else torch.device("cuda")
        self._tensor = None
        self._ptr = 0

    def initialize(self):
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self):
        return self._ptr

    def get_slice(self, offset, size):
        return self._tensor[offset : offset + size]

    def free(self):
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


class GPUSendBuffer:
    def __init__(self, size: int, device=None):
        self.size = size
        self.device = torch.device(device) if device else torch.device("cuda")
        self._tensor = None
        self._ptr = 0

    def initialize(self):
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()

    @property
    def ptr(self):
        return self._ptr

    def copy_from_tensor(self, tensor, offset=0):
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if offset + nbytes > self.size:
            raise ValueError(
                f"Buffer overflow: need {offset + nbytes}, have {self.size}"
            )
        self._tensor[offset : offset + nbytes].copy_(tensor.view(torch.uint8).view(-1))
        return nbytes

    def free(self):
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0


# ---------------------------------------------------------------------------
# EagleMooncakeStore
# ---------------------------------------------------------------------------


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    target: torch.Tensor | None = None
    last_hidden_states: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    input_ids_cpu: torch.Tensor | None = None


class EagleMooncakeStore:
    """Mooncake store specialized for Eagle3 hidden state transfer.

    Supports async host-buffer RDMA puts and GPU Direct RDMA.
    Each Eagle3 output stored as tensors with key suffixes: _hs, _ids, _tgt, _lhs.
    """

    TENSOR_SUFFIXES: typing.ClassVar = ["_hs", "_tgt", "_ids", "_lhs"]

    def __init__(self, config):
        self.config = config
        self._store = None
        self._initialized = False
        self._init_event = threading.Event()
        self._registered_buffers: dict[int, int] = {}
        self._host_buffer_pool: HostBufferPool | None = None
        self._async_put_manager: AsyncPutManager | None = None
        self._gpu_receive_buffer: GPUReceiveBuffer | None = None
        self._gpu_send_buffer: GPUSendBuffer | None = None
        self._gpu_direct_available = False
        self._copy_stream = None
        self._replicate_config = None

    def setup(self, device=None) -> None:
        if self._initialized:
            return

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        from mooncake.store import MooncakeDistributedStore

        self._store = MooncakeDistributedStore()

        gs = self.config.global_segment_size
        global_segment_size = (
            (self.config.parse_size(gs) if isinstance(gs, str) else int(gs))
            if hasattr(self.config, "parse_size")
            else int(gs)
        )
        lb = self.config.local_buffer_size
        local_buffer_size = (
            (self.config.parse_size(lb) if isinstance(lb, str) else int(lb))
            if hasattr(self.config, "parse_size")
            else int(lb)
        )

        local_hostname = self.config.local_hostname or "localhost"
        metadata_server = self.config.metadata_server or ""
        master_addr = self.config.master_server_address or ""

        logger.info(
            "Connecting to Mooncake: master=%s, metadata=%s",
            master_addr,
            metadata_server,
        )
        result = self._store.setup(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=master_addr,
        )
        if result is not None and result != 0:
            raise RuntimeError(f"Mooncake setup failed (error={result})")

        self._build_replicate_config()

        pool_size = getattr(self.config, "async_put_pool_size", None) or 1
        if pool_size > 0:
            host_buf_size = (
                getattr(self.config, "host_buffer_size", None) or 512 * 1024**2
            )
            self._host_buffer_pool = HostBufferPool(
                buffer_size=host_buf_size, pool_size=pool_size
            )
            self._host_buffer_pool.initialize()
            for buf in self._host_buffer_pool._buffers:
                self._register_buffer(buf.ptr, buf.size)
            self._async_put_manager = AsyncPutManager(
                store=self._store,
                max_workers=pool_size,
                replicate_config=self._replicate_config,
            )

        if (
            getattr(self.config, "enable_gpu_direct", False)
            and torch.cuda.is_available()
        ):
            self._setup_gpu_direct(device)

        if torch.cuda.is_available():
            cuda_device = device if device else torch.device("cuda")
            self._copy_stream = torch.cuda.Stream(device=cuda_device)

        self._initialized = True
        self._init_event.set()
        logger.info(
            "EagleMooncakeStore initialized (protocol=%s, device=%s, gpu_direct=%s)",
            self.config.protocol,
            self.config.device_name or "(auto)",
            self._gpu_direct_available,
        )

    def _setup_gpu_direct(self, device=None):
        try:
            gpu_buf_size = (
                getattr(self.config, "gpu_buffer_size", None) or 512 * 1024**2
            )
            self._gpu_receive_buffer = GPUReceiveBuffer(
                size=gpu_buf_size, device=device
            )
            self._gpu_receive_buffer.initialize()
            if not self._register_buffer(
                self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size
            ):
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
                return
            host_buf_size = (
                getattr(self.config, "host_buffer_size", None) or 512 * 1024**2
            )
            self._gpu_send_buffer = GPUSendBuffer(size=host_buf_size, device=device)
            self._gpu_send_buffer.initialize()
            if not self._register_buffer(
                self._gpu_send_buffer.ptr, self._gpu_send_buffer.size
            ):
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None
            self._gpu_direct_available = True
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("GPU Direct setup failed: %s", e)
            self._gpu_direct_available = False

    def _register_buffer(self, ptr, size):
        if ptr in self._registered_buffers:
            return True
        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(ptr, size)
                if result == 0:
                    self._registered_buffers[ptr] = size
                    return True
        except (OSError, RuntimeError) as e:
            logger.warning("register_buffer failed: %s", e)
        return False

    def _build_replicate_config(self):
        self._replicate_config = None
        if not getattr(self.config, "enable_hard_pin", False):
            return
        try:
            from mooncake.store import ReplicateConfig

            cfg = ReplicateConfig()
            if hasattr(cfg, "with_hard_pin"):
                cfg.with_hard_pin = True
                self._replicate_config = cfg
        except ImportError:
            pass

    def _ensure_initialized(self, timeout=30.0):
        if self._init_event.is_set():
            return
        if not self._init_event.wait(timeout=timeout) and not self._initialized:
            self.setup()

    def warmup_rdma(self) -> None:
        self._ensure_initialized()
        if self._host_buffer_pool is None:
            return
        key = f"_warmup_{uuid.uuid4().hex[:8]}"
        buf = self._host_buffer_pool.get_buffer()
        self._store.batch_put_from([key], [buf.ptr], [4096])
        self._store.batch_remove([key], force=True)
        logger.info("RDMA warmup complete")

    # ---- PUT ----

    def put(self, key, hidden_states, input_ids, last_hidden_states=None, target=None):
        self._ensure_initialized()

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        keys = [f"{key}_hs", f"{key}_ids"]
        tensors = [hidden_states, input_ids]
        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)
        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        if self._gpu_direct_available and self._gpu_send_buffer is not None:
            buffer_ptrs, sizes = self._stage_tensors_into_buffer(
                self._gpu_send_buffer, tensors
            )
            self._do_sync_batch_put(keys, buffer_ptrs, sizes)
        elif self._host_buffer_pool is None or self._async_put_manager is None:
            raise RuntimeError("put() requires host buffer pool or GPU Direct")
        else:
            buf = self._host_buffer_pool.get_buffer()
            self._async_put_manager.check_last_error()
            self._async_put_manager.wait_for_buffer(buf.ptr)

            compute_event = torch.cuda.Event()
            compute_event.record()
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(compute_event)
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                copy_done = torch.cuda.Event()
                copy_done.record()

            for t in tensors:
                if t.is_cuda:
                    t.record_stream(self._copy_stream)

            self._async_put_manager.submit(
                keys,
                buffer_ptrs,
                sizes,
                buf.ptr,
                wait_event=copy_done,
                device_index=self._copy_stream.device.index,
            )

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {"hidden_states": hidden_states.dtype, "input_ids": input_ids.dtype}
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype
        return {"shapes": shapes, "dtypes": dtypes}

    def flush(self):
        self._ensure_initialized()
        if self._async_put_manager is None:
            return
        self._async_put_manager.check_last_error()
        self._async_put_manager.drain()
        self._async_put_manager.check_last_error()

    def _do_sync_batch_put(self, keys, buffer_ptrs, sizes):
        if self._replicate_config is not None:
            results = self._store.batch_put_from(
                keys, buffer_ptrs, sizes, config=self._replicate_config
            )
        else:
            results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            try:
                self._store.batch_remove(keys, force=True)
            except (
                OSError,
                RuntimeError,
            ):  # Failed put cleanup must preserve the primary store error.
                pass
            detail = ", ".join(f"{k} (code={r})" for k, r in failures)
            raise RuntimeError(f"batch_put_from failed: {detail}")

    @staticmethod
    def _stage_tensors_into_buffer(buf, tensors):
        buffer_ptrs, sizes = [], []
        offset = 0
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    # ---- GET ----

    def get(self, key, shapes, dtypes, device) -> Eagle3TargetOutput:
        self._ensure_initialized()

        keys = [f"{key}_hs", f"{key}_ids"]
        tensor_specs = [
            (
                "hidden_states",
                shapes["hidden_states"],
                dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
            ),
            ("input_ids", shapes["input_ids"], torch.int64),
        ]
        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(
                (
                    "target",
                    shapes["target"],
                    dtypes.get("target", HIDDEN_STATES_STORAGE_DTYPE),
                )
            )
        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(
                (
                    "last_hidden_states",
                    shapes["last_hidden_states"],
                    dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
                )
            )

        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)

        if tensor_map is None:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)

        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
        )

    def _get_tensors_gpu_direct(self, keys, tensor_specs, device):
        total_size = sum(self._compute_tensor_size(s, d) for _, s, d in tensor_specs)
        if total_size > self._gpu_receive_buffer.size:
            return None
        buffer_ptrs, sizes, offsets = [], [], []
        offset = 0
        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    return None
        except (OSError, RuntimeError, ValueError):
            return None
        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)
        return tensor_map

    @staticmethod
    def _compute_tensor_size(shape, dtype):
        numel = 1
        for d in shape:
            numel *= d
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    def _get_tensors_via_host_buffer(self, keys, tensor_specs, device):
        wait_seconds = max(getattr(self.config, "get_retry_wait_seconds", 0.5), 0.05)
        max_wait = max(getattr(self.config, "get_retry_max_wait_seconds", 60.0), 0.0)
        start_time = time.time()

        while True:
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break
            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                mk = ", ".join(keys[i] for i in missing)
                raise RuntimeError(
                    f"batch_get_buffer timed out for: {mk} ({elapsed:.1f}s)"
                )
            time.sleep(wait_seconds)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            numel = 1
            for d in shape:
                numel *= d
            nbytes = numel * _DTYPE_ELEMENT_SIZES[dtype]
            c_arr = (ctypes.c_byte * nbytes).from_address(buf.ptr())
            host_t = torch.frombuffer(c_arr, dtype=dtype, count=numel).reshape(shape)
            tensor_map[name] = host_t.to(device)
            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_t.clone()
        return tensor_map

    # ---- REMOVE ----

    def remove_eagle3_tensors(
        self, key, has_last_hidden_states=False, has_target=False
    ):
        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")
        for attempt in range(1, 4):
            try:
                results = self._store.batch_remove(keys, force=True)
            except (OSError, RuntimeError):
                if attempt < 3:
                    time.sleep(0.5)
                continue
            failed = [(k, r) for k, r in zip(keys, results) if r not in (None, 0, -704)]
            if not failed:
                return
            if attempt < 3:
                time.sleep(0.5)
                keys = [k for k, _ in failed]
            else:
                logger.error("Force delete abandoned for %s: %s", key, failed)

    def close(self):
        if self._async_put_manager is not None:
            self._async_put_manager.shutdown()
            self._async_put_manager = None
        if self._gpu_send_buffer is not None:
            self._gpu_send_buffer.free()
            self._gpu_send_buffer = None
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        self._copy_stream = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False
        self._init_event.clear()
        self._gpu_direct_available = False
