# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""WideEP topology value object (M-TOPO).

Single source of truth for multi-node EP parallel layout. Pure data + arithmetic;
no processes, sockets, CUDA, or environment reads.
"""

from __future__ import annotations

from dataclasses import dataclass


def parse_dist_init_addr(addr: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` or ``[IPv6]:PORT`` from ``--dist-init-addr``."""
    addr = addr.strip()
    if not addr:
        raise ValueError("dist_init_addr must not be empty")
    if addr.startswith("["):
        end = addr.index("]")
        host = addr[1:end]
        rest = addr[end + 1 :]
        if not rest.startswith(":"):
            raise ValueError(f"Invalid dist_init_addr: {addr!r}")
        port = int(rest[1:])
    else:
        host, _, port_str = addr.rpartition(":")
        if not host or not port_str:
            raise ValueError(f"Invalid dist_init_addr: {addr!r}")
        port = int(port_str)
    if not (0 < port < 65536):
        raise ValueError(f"Invalid port in dist_init_addr: {port}")
    return host, port


@dataclass(frozen=True)
class WideEPTopology:
    # --- inputs (from CLI / Config) ---
    nnodes: int
    node_rank: int
    dp_attention: bool
    raw_tp_size: int
    raw_dp_size: int

    # --- derived at construction ---
    tp_size: int
    global_dp_size: int
    local_engine_count: int

    # --- rendezvous (§4.3); None when nnodes == 1 ---
    dist_init_host: str | None
    dist_init_base_port: int | None

    @classmethod
    def create(
        cls,
        *,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_attention: bool,
        raw_tp_size: int,
        raw_dp_size: int,
        dist_init_addr: str | None = None,
    ) -> WideEPTopology:
        tp_size = 1 if dp_attention else raw_tp_size
        global_dp_size = (
            raw_tp_size * raw_dp_size if dp_attention else raw_dp_size
        )
        local_engine_count = global_dp_size // nnodes

        dist_init_host: str | None = None
        dist_init_base_port: int | None = None
        if nnodes > 1:
            if dist_init_addr is None:
                raise ValueError("nnodes>1 requires dist_init_addr")
            dist_init_host, dist_init_base_port = parse_dist_init_addr(
                dist_init_addr
            )

        topo = cls(
            nnodes=nnodes,
            node_rank=node_rank,
            dp_attention=dp_attention,
            raw_tp_size=raw_tp_size,
            raw_dp_size=raw_dp_size,
            tp_size=tp_size,
            global_dp_size=global_dp_size,
            local_engine_count=local_engine_count,
            dist_init_host=dist_init_host,
            dist_init_base_port=dist_init_base_port,
        )
        topo._validate()
        return topo

    def _validate(self) -> None:
        if self.nnodes < 1:
            raise ValueError(f"nnodes must be >= 1, got {self.nnodes}")
        if not (0 <= self.node_rank < self.nnodes):
            raise ValueError(
                f"node_rank must satisfy 0 <= node_rank < nnodes "
                f"({self.node_rank}, {self.nnodes})"
            )
        if self.nnodes > 1 and not self.dp_attention:
            raise ValueError(
                "nnodes>1 requires dp_attention (TP does not span nodes)"
            )
        if self.global_dp_size % self.nnodes != 0:
            divisors = [
                n
                for n in range(1, self.global_dp_size + 1)
                if self.global_dp_size % n == 0
            ]
            raise ValueError(
                f"global_dp_size={self.global_dp_size} is not divisible by "
                f"nnodes={self.nnodes}. Valid nnodes values: {divisors}"
            )
        if self.ep_size != self.gpu_per_node * self.nnodes:
            raise ValueError(
                f"ep_size invariant violated: ep_size={self.ep_size} != "
                f"gpu_per_node({self.gpu_per_node}) * nnodes({self.nnodes})"
            )
        if self.nnodes > 1 and self.local_engine_count < 1:
            raise ValueError(
                f"nnodes>1 requires local_engine_count >= 1, "
                f"got {self.local_engine_count}"
            )

    @property
    def ep_size(self) -> int:
        return self.global_dp_size

    @property
    def gpu_per_node(self) -> int:
        return self.local_engine_count

    @property
    def is_multinode(self) -> bool:
        return self.nnodes > 1

    def _require_rendezvous_base_port(self) -> int:
        if self.dist_init_base_port is None:
            raise ValueError(
                "rendezvous ports require nnodes>1 and dist_init_addr"
            )
        return self.dist_init_base_port

    @property
    def rendezvous_port_world(self) -> int:
        return self._require_rendezvous_base_port() + 0

    @property
    def rendezvous_port_dp_gloo(self) -> int:
        return self._require_rendezvous_base_port() + 1

    def rendezvous_port_reserved(self, i: int) -> int:
        if not (0 <= i < 6):
            raise ValueError(f"reserved port index must be in [0, 6), got {i}")
        return self._require_rendezvous_base_port() + 2 + i

    def dp_rank(self, engine_index: int, *, pp_size: int = 1) -> int:
        local_dp = engine_index // pp_size
        return self.node_rank * self.local_engine_count + local_dp

    def dp_rank_local(self, engine_index: int, *, pp_size: int = 1) -> int:
        return engine_index // pp_size

    def local_device_rank(
        self,
        engine_index: int,
        tp_rank: int,
        *,
        pp_size: int = 1,
        pcp_size: int = 1,
    ) -> int:
        dp_rank_local = self.dp_rank_local(engine_index, pp_size=pp_size)
        pp_rank = engine_index % pp_size
        stage_span = self.tp_size * pcp_size
        engine_idx = dp_rank_local * pp_size + pp_rank
        return engine_idx * stage_span + tp_rank

    def describe(self) -> str:
        return (
            f"[wideep] nnodes={self.nnodes} node_rank={self.node_rank} | "
            f"ep={self.ep_size} gpu_per_node={self.gpu_per_node} | "
            f"dp: global={self.global_dp_size} local={self.local_engine_count}"
        )
