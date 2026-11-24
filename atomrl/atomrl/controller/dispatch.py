# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Dispatch and collect DataProto across workers."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from atomrl.core.protocol import DataProto


class DispatchMode(str, Enum):
    """Supported dispatch semantics for worker calls."""

    RANK_ZERO = "rank_zero"
    DP_COMPUTE_PROTO = "dp_compute_proto"
    DP_COMPUTE = "dp_compute"
    DP_COMPUTE_PROTO_WITH_FUNC = "dp_compute_proto_with_func"
    DP_COMPUTE_METRIC = "dp_compute_metric"
    ONE_TO_ALL = "one_to_all"
    ALL_TO_ALL = "all_to_all"
    DIRECT_ROLLOUT_METHOD = "direct_rollout_method"


def _normalize_mode(mode: DispatchMode | str | None) -> DispatchMode:
    if mode is None:
        return DispatchMode.DP_COMPUTE_PROTO
    if isinstance(mode, DispatchMode):
        return mode
    if mode == "broadcast":
        # Backward-compatible alias used by some older config schemas.
        return DispatchMode.ONE_TO_ALL
    return DispatchMode(mode)


def _round_robin_chunks(chunks: list[DataProto], num_workers: int) -> list[DataProto]:
    if len(chunks) == num_workers:
        return chunks
    if not chunks:
        return [DataProto() for _ in range(num_workers)]
    return [chunks[i % len(chunks)] for i in range(num_workers)]


def _build_nd_dispatch(
    data: DataProto,
    num_workers: int,
    mesh_mapping: list[int] | None,
) -> list[DataProto]:
    if mesh_mapping is None:
        return data.split(num_workers)
    if len(mesh_mapping) != num_workers:
        raise ValueError("mesh_mapping length must equal num_workers.")
    if not mesh_mapping:
        return []

    num_groups = max(mesh_mapping) + 1
    grouped = data.split(num_groups)
    return [grouped[group_id] for group_id in mesh_mapping]


def dispatch_rank_zero(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    del mesh_mapping, lazy_state, lazy_key
    if num_workers <= 0:
        return []
    return [data]


def dispatch_one_to_all(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    del mesh_mapping, lazy_state, lazy_key
    return [data for _ in range(num_workers)]


def dispatch_all_to_all(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    del mesh_mapping, lazy_state, lazy_key
    chunks = data.split(num_workers)
    return _round_robin_chunks(chunks, num_workers)


def dispatch_dp_compute_data_proto(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    mapping = mesh_mapping
    if lazy_state is not None and lazy_key is not None:
        cached = lazy_state.get(lazy_key)
        if cached is not None and isinstance(cached, list):
            mapping = cached
        elif mesh_mapping is not None:
            lazy_state[lazy_key] = mesh_mapping
    return _build_nd_dispatch(data, num_workers, mapping)


def dispatch_dp_compute(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    return dispatch_dp_compute_data_proto(
        data,
        num_workers,
        mesh_mapping=mesh_mapping,
        lazy_state=lazy_state,
        lazy_key=lazy_key,
    )


def dispatch_dp_compute_data_proto_with_func(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    return dispatch_dp_compute_data_proto(
        data,
        num_workers,
        mesh_mapping=mesh_mapping,
        lazy_state=lazy_state,
        lazy_key=lazy_key,
    )


def dispatch_dp_compute_metric(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    return dispatch_dp_compute_data_proto(
        data,
        num_workers,
        mesh_mapping=mesh_mapping,
        lazy_state=lazy_state,
        lazy_key=lazy_key,
    )


def dispatch_direct_rollout_forbidden(
    data: DataProto,
    num_workers: int,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    del data, num_workers, mesh_mapping, lazy_state, lazy_key
    raise RuntimeError("Direct rollout call is forbidden.")


def collect_direct_rollout_forbidden(
    results: list[DataProto],
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    del results, deduplicate_by_identity
    raise RuntimeError("Direct rollout call is forbidden.")


def collect_all_to_all(
    results: list[DataProto],
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    if deduplicate_by_identity:
        unique: list[DataProto] = []
        seen: set[int] = set()
        for result in results:
            marker = id(result)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(result)
        return DataProto.merge(unique)
    return DataProto.merge(results)


def collect_dp_compute_data_proto(
    results: list[DataProto],
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    del deduplicate_by_identity
    return DataProto.merge(results)


def collect_dp_compute(
    results: list[DataProto],
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    del deduplicate_by_identity
    return DataProto.merge(results)


def collect_rank_zero(
    results: list[DataProto],
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    del deduplicate_by_identity
    if not results:
        return DataProto()
    return results[0]


DispatchFn = Callable[..., list[DataProto]]
CollectFn = Callable[..., DataProto]


DISPATCH_MODE_FN_REGISTRY: dict[DispatchMode, dict[str, DispatchFn | CollectFn]] = {
    DispatchMode.RANK_ZERO: {
        "dispatch_fn": dispatch_rank_zero,
        "collect_fn": collect_rank_zero,
    },
    DispatchMode.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    DispatchMode.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    DispatchMode.DP_COMPUTE: {
        "dispatch_fn": dispatch_dp_compute,
        "collect_fn": collect_dp_compute,
    },
    DispatchMode.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    DispatchMode.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    DispatchMode.DP_COMPUTE_METRIC: {
        "dispatch_fn": dispatch_dp_compute_metric,
        "collect_fn": collect_dp_compute,
    },
    DispatchMode.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dispatch_direct_rollout_forbidden,
        "collect_fn": collect_direct_rollout_forbidden,
    },
}


def dispatch_with_dp_ranks(
    data: DataProto,
    num_workers: int,
    dp_rank_mapping: list[int],
) -> list[DataProto]:
    """Split *data* by dp_size unique ranks, route chunk[dp_rank] to each worker.

    verl-aligned: workers with the same dp_rank (e.g. TP peers) receive
    identical data so TP collectives stay consistent.
    """
    if len(dp_rank_mapping) != num_workers:
        raise ValueError(
            f"dp_rank_mapping length ({len(dp_rank_mapping)}) != num_workers ({num_workers})"
        )
    dp_size = len(set(dp_rank_mapping))
    chunks = data.split(dp_size)
    return [chunks[dp_rank_mapping[i]] for i in range(num_workers)]


def collect_with_mask(
    results: list[DataProto],
    collect_mask: list[bool],
) -> DataProto:
    """Merge only the results where collect_mask[i] is True.

    verl-aligned: TP/PP non-source ranks return duplicates; collect_mask
    filters them out so each DP shard contributes exactly once.
    """
    if len(collect_mask) != len(results):
        raise ValueError(
            f"collect_mask length ({len(collect_mask)}) != results length ({len(results)})"
        )
    filtered = [r for r, keep in zip(results, collect_mask) if keep]
    return DataProto.merge(filtered)


def dispatch_proto(
    data: DataProto,
    num_workers: int,
    mode: DispatchMode | str | None = None,
    *,
    mesh_mapping: list[int] | None = None,
    lazy_state: dict[str, Any] | None = None,
    lazy_key: str | None = None,
) -> list[DataProto]:
    """Create per-worker inputs according to dispatch mode.

    `lazy_state` + `lazy_key` allow caller-side caching of expensive ND mapping.
    """
    dispatch_mode = _normalize_mode(mode)
    if num_workers <= 0:
        return []
    dispatch_fn = DISPATCH_MODE_FN_REGISTRY[dispatch_mode]["dispatch_fn"]
    return dispatch_fn(
        data,
        num_workers,
        mesh_mapping=mesh_mapping,
        lazy_state=lazy_state,
        lazy_key=lazy_key,
    )


def collect_proto(
    results: list[DataProto],
    mode: DispatchMode | str | None = None,
    *,
    deduplicate_by_identity: bool = False,
) -> DataProto:
    """Collect worker outputs into a merged DataProto.

    For ONE_TO_ALL, callers can optionally deduplicate repeated shared results.
    """
    dispatch_mode = _normalize_mode(mode)
    if not results:
        return DataProto()
    collect_fn = DISPATCH_MODE_FN_REGISTRY[dispatch_mode]["collect_fn"]
    return collect_fn(results, deduplicate_by_identity=deduplicate_by_identity)
