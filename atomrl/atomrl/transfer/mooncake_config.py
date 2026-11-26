# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Mooncake distributed store configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MooncakeConfig:
    """Configuration for Mooncake distributed KV store."""

    master_server_address: str | None = None
    metadata_server: str | None = None
    local_hostname: str = ""
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: str | int = "16GB"
    local_buffer_size: str | int = "4GB"
    host_buffer_size: int | None = None
    gpu_buffer_size: int | None = None
    async_put_pool_size: int | None = None
    enable_gpu_direct: bool = False
    enable_hard_pin: bool = False
    kv_lease_ttl_s: float = 5.0
    max_seq_len: int = 8192
    hidden_dim: int = 4096
    get_batch_size: int = 1
    get_retry_wait_seconds: float = 0.5
    get_retry_log_interval_seconds: float = 10.0
    get_retry_max_wait_seconds: float = 60.0
    store_full_wait_seconds: float = 0.5
    store_full_log_interval_seconds: float = 5.0
    store_full_max_wait_seconds: float = 0.0

    def __post_init__(self):
        for field_name in (
            "global_segment_size",
            "local_buffer_size",
            "host_buffer_size",
            "gpu_buffer_size",
        ):
            val = getattr(self, field_name)
            if isinstance(val, str):
                setattr(self, field_name, self.parse_size(val))

        if self.host_buffer_size is None:
            from atomrl.transfer.eagle_mooncake_store import (
                calculate_eagle3_buffer_size,
            )

            self.host_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=1,
                hidden_dim=self.hidden_dim,
                safety_margin=2.0,
            )

        if self.async_put_pool_size is None:
            self.async_put_pool_size = 2

        if self.gpu_buffer_size is None and self.enable_gpu_direct:
            from atomrl.transfer.eagle_mooncake_store import (
                calculate_eagle3_buffer_size,
            )

            self.gpu_buffer_size = calculate_eagle3_buffer_size(
                max_seq_len=self.max_seq_len,
                batch_size=self.get_batch_size,
                hidden_dim=self.hidden_dim,
            )

    @staticmethod
    def parse_size(size_str: str) -> int:
        size_str = size_str.upper().strip()
        multipliers = [
            ("TB", 1024**4),
            ("GB", 1024**3),
            ("MB", 1024**2),
            ("KB", 1024),
            ("T", 1024**4),
            ("G", 1024**3),
            ("M", 1024**2),
            ("K", 1024),
            ("B", 1),
        ]
        for suffix, mult in multipliers:
            if size_str.endswith(suffix):
                return int(float(size_str[: -len(suffix)]) * mult)
        return int(size_str)

    @property
    def global_segment_size_bytes(self) -> int:
        v = self.global_segment_size
        return self.parse_size(v) if isinstance(v, str) else int(v)

    @property
    def local_buffer_size_bytes(self) -> int:
        v = self.local_buffer_size
        return self.parse_size(v) if isinstance(v, str) else int(v)

    def export_env(self) -> None:
        """Export config as environment variables for Mooncake transfer clients."""
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = self.local_hostname
        os.environ["MOONCAKE_METADATA_SERVER"] = self.metadata_server or ""
        os.environ["MOONCAKE_MASTER_SERVER"] = self.master_server_address or ""
        gs = self.global_segment_size
        os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(
            self.parse_size(gs) if isinstance(gs, str) else gs
        )
        lb = self.local_buffer_size
        os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(
            self.parse_size(lb) if isinstance(lb, str) else lb
        )
        if self.host_buffer_size is not None:
            os.environ["MOONCAKE_HOST_BUFFER_SIZE"] = str(self.host_buffer_size)
        os.environ["MOONCAKE_PROTOCOL"] = self.protocol
        os.environ["MOONCAKE_DEVICE_NAME"] = self.device_name
        os.environ["MOONCAKE_ENABLE_GPU_DIRECT"] = (
            "1" if self.enable_gpu_direct else "0"
        )
        if self.async_put_pool_size is not None:
            os.environ["MOONCAKE_ASYNC_PUT_POOL_SIZE"] = str(self.async_put_pool_size)
        os.environ["MOONCAKE_ENABLE_HARD_PIN"] = "1" if self.enable_hard_pin else "0"

    @classmethod
    def from_env(cls) -> MooncakeConfig:
        """Create config from MOONCAKE_* environment variables."""
        master_host = os.getenv("MOONCAKE_MASTER_HOST", "localhost")
        master_port = os.getenv("MOONCAKE_MASTER_PORT", "50051")
        metadata_port = os.getenv("MOONCAKE_METADATA_PORT", "8090")

        host_buffer_env = os.getenv("MOONCAKE_HOST_BUFFER_SIZE")
        host_buffer_size = int(host_buffer_env) if host_buffer_env else None
        pool_size_env = os.getenv("MOONCAKE_ASYNC_PUT_POOL_SIZE")
        async_put_pool_size = int(pool_size_env) if pool_size_env else None

        return cls(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv(
                "MOONCAKE_METADATA_SERVER",
                f"http://{master_host}:{metadata_port}/metadata",
            ),
            master_server_address=os.getenv(
                "MOONCAKE_MASTER_SERVER", f"{master_host}:{master_port}"
            ),
            global_segment_size=int(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(4 * 1024**3))
            ),
            local_buffer_size=int(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(512 * 1024**2))
            ),
            host_buffer_size=host_buffer_size,
            async_put_pool_size=async_put_pool_size,
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            enable_gpu_direct=os.getenv("MOONCAKE_ENABLE_GPU_DIRECT", "0") == "1",
            enable_hard_pin=os.getenv("MOONCAKE_ENABLE_HARD_PIN", "0") == "1",
            get_retry_wait_seconds=float(
                os.getenv("MOONCAKE_GET_RETRY_WAIT_SECONDS", "0.2")
            ),
            get_retry_log_interval_seconds=float(
                os.getenv("MOONCAKE_GET_RETRY_LOG_INTERVAL_SECONDS", "5.0")
            ),
            get_retry_max_wait_seconds=float(
                os.getenv("MOONCAKE_GET_RETRY_MAX_WAIT_SECONDS", "90.0")
            ),
        )
