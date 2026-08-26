# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""WideEP topology view (M-TOPO).

Derived, immutable view of the engine/EP layout. Pure data + arithmetic; no
processes, sockets, CUDA, or environment reads.

This does NOT replace the DP topology fields on ``ParallelConfig`` -- those stay
authoritative and ``CoreManager`` still rewrites them into engine units. What
this adds is a named, testable expression of the quantities the MoE and EPLB
layers need (``nnodes``, ``gpu_per_node``, ``ep_size``) plus assertions that
turn the rewrite's implicit invariants into explicit ones.
"""

from __future__ import annotations

from dataclasses import dataclass


def parse_dist_init_addr(addr: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` or ``[IPv6]:PORT``."""
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


def format_startup_summary(
    topo: WideEPTopology,
    *,
    dp_rank: int,
    dp_rank_local: int,
    device_rank: int,
    visible_devices: str | None = None,
    mori_env: dict[str, str] | None = None,
) -> str:
    """One greppable line per worker: is this rank where it thinks it is?

    Multi-node misconfiguration is diagnosed by comparing ranks across nodes,
    which means reading the same fields out of several logs. Emitting them
    together, once, on a line with a fixed prefix is the difference between
    one grep and a correlation exercise. Everything here is already known at
    worker init; nothing is computed for the message.
    """
    parts = [
        topo.describe(),
        f"rank={dp_rank} local_rank={dp_rank_local}",
        f"device=cuda:{device_rank}",
    ]
    if visible_devices:
        parts.append(f"visible={visible_devices}")
    if mori_env:
        parts.append(
            "mori: " + " ".join(f"{k}={v}" for k, v in sorted(mori_env.items()))
        )
    return " | ".join(parts)


def node_count(config) -> int:
    """How many nodes this Config runs on.

    The one place that answers the question for the simulated case, so callers
    that only need a node count do not each re-derive the rule: ``--fake-eplb``
    runs a single box pretending to be a wider deployment, and the DP fields
    alone cannot distinguish that from being one node of a real split.
    """
    if getattr(config, "dp_logical_size", 0):
        return 1
    return WideEPTopology.from_config(config).nnodes


@dataclass(frozen=True)
class WideEPTopology:
    """Engine-unit view of the parallel layout.

    Deliberately stores no raw ``-tp`` / ``-dp`` values: after CoreManager folds
    TP into DP those are unrecoverable, and a second copy of an authoritative
    field is how the two drift apart.
    """

    nnodes: int
    node_rank: int
    dp_attention: bool
    tp_size: int
    """TP width per engine. 1 under DP-attention, which folds TP into DP."""
    global_dp_size: int
    """Engines across all nodes."""
    local_engine_count: int
    """Engines on this node."""
    dist_init_host: str | None
    dist_init_base_port: int | None

    @classmethod
    def create(
        cls,
        *,
        dp_attention: bool,
        raw_tp_size: int,
        raw_dp_size: int,
        nnodes: int = 1,
        node_rank: int = 0,
        dist_init_addr: str | None = None,
    ) -> WideEPTopology:
        """Build from pre-fold CLI values (``-tp`` / ``-dp``)."""
        tp_size = 1 if dp_attention else raw_tp_size
        global_dp_size = raw_tp_size * raw_dp_size if dp_attention else raw_dp_size
        if nnodes < 1:
            raise ValueError(f"nnodes must be >= 1, got {nnodes}")
        return cls._build(
            nnodes=nnodes,
            node_rank=node_rank,
            dp_attention=dp_attention,
            tp_size=tp_size,
            global_dp_size=global_dp_size,
            local_engine_count=global_dp_size // nnodes,
            dist_init_addr=dist_init_addr,
        )

    @classmethod
    def from_parallel_config(
        cls,
        parallel_config,
        *,
        tensor_parallel_size: int,
        dp_attention: bool,
        dp_logical_size: int = 0,
    ) -> WideEPTopology:
        """Derive from ``ParallelConfig``, before or after CoreManager's fold.

        Both regimes work because the fold is a change of units -- it scales the
        DP quantities by ``tp_size`` and divides TP by the same -- so the ratio
        and the product this reads are invariant under it.

        ``dp_logical_size`` (``Config.dp_logical_size``, non-zero only under
        ``--fake-eplb``) must be passed through: while simulating, the DP fields
        describe a deployment wider than the box, so their ratio is not a node
        count. Refuse rather than report a node count that does not exist.
        """
        if dp_logical_size:
            raise ValueError(
                f"cannot derive a node topology while simulating a "
                f"{dp_logical_size}-wide deployment (--fake-eplb): the DP fields "
                f"describe the simulated width, not this box. Callers that "
                f"support simulation must handle it explicitly."
            )
        global_dp_size = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local
        rank_offset = parallel_config.data_parallel_rank
        if local_engine_count is None:
            local_engine_count = global_dp_size
        if dp_attention:
            # Pre-fold the DP fields still count replicas, not engines.
            if tensor_parallel_size > 1:
                global_dp_size *= tensor_parallel_size
                local_engine_count *= tensor_parallel_size
                rank_offset *= tensor_parallel_size
            tp_size = 1
        else:
            tp_size = tensor_parallel_size

        if local_engine_count < 1:
            raise ValueError(
                f"data_parallel_size_local must be >= 1, got {local_engine_count}"
            )
        # EPLB's hierarchical placement asserts num_gpus % num_nodes == 0
        # (model_ops/eplb.py), so a ragged split has to fail here rather than
        # deep inside a rebalance.
        if global_dp_size % local_engine_count != 0:
            raise ValueError(
                f"data_parallel_size ({global_dp_size}) must be divisible by "
                f"data_parallel_size_local ({local_engine_count}): nodes must "
                f"hold equal slices for EPLB's hierarchical placement"
            )
        nnodes = global_dp_size // local_engine_count
        if rank_offset % local_engine_count != 0:
            raise ValueError(
                f"data_parallel_rank ({rank_offset}) must be a multiple of "
                f"data_parallel_size_local ({local_engine_count}): it names the "
                f"first global rank of this node's slice"
            )

        host = getattr(parallel_config, "data_parallel_master_ip", None)
        port = getattr(parallel_config, "data_parallel_master_port", None)
        return cls._build(
            nnodes=nnodes,
            node_rank=rank_offset // local_engine_count,
            dp_attention=dp_attention,
            tp_size=tp_size,
            global_dp_size=global_dp_size,
            local_engine_count=local_engine_count,
            dist_init_addr=f"{host}:{port}" if nnodes > 1 and host else None,
        )

    @classmethod
    def from_config(cls, config) -> WideEPTopology:
        """Same as ``from_parallel_config``, reading the fields off a Config."""
        return cls.from_parallel_config(
            config.parallel_config,
            tensor_parallel_size=config.tensor_parallel_size,
            dp_attention=config.enable_dp_attention,
            dp_logical_size=getattr(config, "dp_logical_size", 0),
        )

    @classmethod
    def _build(
        cls,
        *,
        nnodes: int,
        node_rank: int,
        dp_attention: bool,
        tp_size: int,
        global_dp_size: int,
        local_engine_count: int,
        dist_init_addr: str | None,
    ) -> WideEPTopology:
        host: str | None = None
        base_port: int | None = None
        if nnodes > 1:
            if dist_init_addr is None:
                raise ValueError(
                    "nnodes>1 requires a rendezvous address "
                    "(dist_init_addr, or data_parallel_master_ip/_port)"
                )
            host, base_port = parse_dist_init_addr(dist_init_addr)
        topo = cls(
            nnodes=nnodes,
            node_rank=node_rank,
            dp_attention=dp_attention,
            tp_size=tp_size,
            global_dp_size=global_dp_size,
            local_engine_count=local_engine_count,
            dist_init_host=host,
            dist_init_base_port=base_port,
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
            raise ValueError("nnodes>1 requires dp_attention (TP does not span nodes)")
        if self.local_engine_count < 1:
            raise ValueError(
                f"local_engine_count must be >= 1, got {self.local_engine_count}"
            )
        if self.local_engine_count * self.nnodes != self.global_dp_size:
            divisors = [
                n
                for n in range(1, self.global_dp_size + 1)
                if self.global_dp_size % n == 0
            ]
            raise ValueError(
                f"global_dp_size={self.global_dp_size} is not divisible by "
                f"nnodes={self.nnodes}. Valid nnodes values: {divisors}"
            )
        # Only DP-attention flattens every rank into one EP group; otherwise EP
        # spans a TP group and there are global_dp_size independent groups, so
        # the identity does not apply.
        if self.dp_attention and self.ep_size != self.gpu_per_node * self.nnodes:
            raise ValueError(
                f"ep_size invariant violated: ep_size={self.ep_size} != "
                f"gpu_per_node({self.gpu_per_node}) * nnodes({self.nnodes})"
            )

    @property
    def ep_size(self) -> int:
        """Ranks in one EP group (``FusedMoEParallelConfig.make``)."""
        if self.dp_attention:
            return self.global_dp_size * self.tp_size
        return self.tp_size

    @property
    def gpu_per_node(self) -> int:
        """GPUs this node contributes; reaches MoRI as ``gpu_per_node``."""
        return self.local_engine_count * self.tp_size

    @property
    def is_multinode(self) -> bool:
        return self.nnodes > 1

    def _require_rendezvous_base_port(self) -> int:
        if self.dist_init_base_port is None:
            raise ValueError(
                "rendezvous ports require nnodes>1 and a rendezvous address"
            )
        return self.dist_init_base_port

    @property
    def rendezvous_port_world(self) -> int:
        return self._require_rendezvous_base_port()

    @property
    def rendezvous_port_dp_gloo(self) -> int:
        return self._require_rendezvous_base_port() + 1

    def rendezvous_port_reserved(self, i: int) -> int:
        if not (0 <= i < 6):
            raise ValueError(f"reserved port index must be in [0, 6), got {i}")
        return self._require_rendezvous_base_port() + 2 + i

    def dp_rank(self, engine_index: int, *, pp_size: int = 1) -> int:
        return self.node_rank * self.local_engine_count + engine_index // pp_size

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
        """This node's physical device index; mirrors ModelRunner's formula."""
        stage_span = self.tp_size * pcp_size
        pp_rank = engine_index % pp_size
        engine_idx = (
            self.dp_rank_local(engine_index, pp_size=pp_size) * pp_size + pp_rank
        )
        return engine_idx * stage_span + tp_rank

    def describe(self) -> str:
        return (
            f"[wideep] nnodes={self.nnodes} node_rank={self.node_rank} | "
            f"ep={self.ep_size} gpu_per_node={self.gpu_per_node} | "
            f"dp: global={self.global_dp_size} local={self.local_engine_count}"
        )
