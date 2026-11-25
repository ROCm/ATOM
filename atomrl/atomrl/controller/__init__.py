# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Ray-based orchestration: cluster, worker groups, dispatch."""

try:
    from atomrl.controller.ray_cluster import RayCluster
    from atomrl.controller.ray_worker_group import RayWorkerGroup
except ModuleNotFoundError:  # Optional in unit test env without ray.
    RayCluster = None
    RayWorkerGroup = None

from atomrl.controller.colocation import (
    create_colocated_worker_cls,
    create_fused_worker_cls,
)
from atomrl.controller.dispatch import (
    DISPATCH_MODE_FN_REGISTRY,
    DispatchMode,
    collect_proto,
    dispatch_proto,
)

__all__ = [
    "DISPATCH_MODE_FN_REGISTRY",
    "DispatchMode",
    "RayCluster",
    "RayWorkerGroup",
    "collect_proto",
    "create_colocated_worker_cls",
    "create_fused_worker_cls",
    "dispatch_proto",
]
