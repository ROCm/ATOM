# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What an engine and its workers support, so a caller can ask instead of guess.

LumenRL currently probes ATOM with ``hasattr(self.engine, ...)`` in four places
and otherwise hard-codes behaviour per backend. That works only while the two
repos move together; the moment they do not, a missing method is discovered as a
runtime failure mid-rollout rather than at negotiation time.

Two layers, answering different questions:

- :class:`WorkerCapabilities` -- what one rank can do. Collected from every rank
  over ``collective_rpc``, so discovery uses the mechanism it describes.
- :class:`EngineCapabilities` -- the engine-wide answer: static topology plus the
  **intersection** of the per-rank feature sets.

Intersection, not union, is the load-bearing choice. A feature present on some
ranks cannot be driven by a collective: the ranks that have it would block in
one collective while the rest went elsewhere, which deadlocks the group with no
error. Advertising it would be worse than not knowing.

Kept free of heavy imports for the same reason as ``collective_rpc``: the
dispatch and negotiation layers must import on a machine with no GPU build.
"""

from dataclasses import dataclass, field

# Bump when the collective-RPC wire contract changes incompatibly. A consumer
# that pins a version can refuse to negotiate rather than fail mid-collective.
COLLECTIVE_RPC_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class WorkerCapabilities:
    """One rank's view of itself."""

    protocol_version: int
    tp_rank: int
    dp_rank_local: int
    methods: frozenset[str] = frozenset()
    features: frozenset[str] = frozenset()

    @classmethod
    def from_payload(cls, payload: object) -> "WorkerCapabilities":
        """Build from the dict a worker returned over the wire.

        Tolerant of missing keys on purpose: an older worker that predates a
        field should report less, not fail the whole negotiation.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"worker capabilities must be a dict, got {type(payload).__name__}"
            )
        return cls(
            protocol_version=int(payload.get("protocol_version", 0)),
            tp_rank=int(payload.get("tp_rank", -1)),
            dp_rank_local=int(payload.get("dp_rank_local", 0)),
            methods=frozenset(payload.get("methods", ())),
            features=frozenset(payload.get("features", ())),
        )


@dataclass(frozen=True)
class EngineCapabilities:
    """The engine-wide answer a consumer should negotiate against."""

    protocol_version: int
    tp_world_size: int
    data_parallel_size: int
    pipeline_parallel_size: int
    kv_cache_dtype: str
    # Only what every rank reports. See the module docstring.
    methods: frozenset[str] = frozenset()
    features: frozenset[str] = frozenset()
    worker_count: int = 0
    _mismatches: tuple[str, ...] = field(default=())

    @classmethod
    def from_workers(cls, *, config, workers) -> "EngineCapabilities":
        if not workers:
            raise ValueError("no worker capabilities to aggregate")

        versions = {w.protocol_version for w in workers}
        if len(versions) != 1:
            raise RuntimeError(
                f"workers disagree on the RPC protocol version: {sorted(versions)}"
            )

        methods = frozenset.intersection(*(w.methods for w in workers))
        features = frozenset.intersection(*(w.features for w in workers))

        # Record what was dropped, so a caller debugging a "missing" feature can
        # see it was present-but-not-universal rather than absent everywhere.
        union_methods = frozenset.union(*(w.methods for w in workers))
        union_features = frozenset.union(*(w.features for w in workers))
        mismatches = tuple(
            sorted((union_methods - methods) | (union_features - features))
        )

        parallel = getattr(config, "parallel_config", None)
        return cls(
            protocol_version=versions.pop(),
            tp_world_size=int(getattr(config, "tp_world_size", len(workers))),
            data_parallel_size=int(getattr(parallel, "data_parallel_size", 1) or 1),
            pipeline_parallel_size=int(
                getattr(parallel, "pipeline_parallel_size", 1) or 1
            ),
            kv_cache_dtype=str(getattr(config, "kv_cache_dtype", "auto")),
            methods=methods,
            features=features,
            worker_count=len(workers),
            _mismatches=mismatches,
        )

    def supports(self, name: str) -> bool:
        """Whether *name* is usable on every rank, as a method or a feature."""
        return name in self.methods or name in self.features

    def partial(self) -> tuple[str, ...]:
        """Names some ranks reported but not all, hence not advertised."""
        return self._mismatches
