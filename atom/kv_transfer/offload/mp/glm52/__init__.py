# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.2 LMCache multiprocess offload implementation."""

from atom.kv_transfer.offload.mp.registry import MPModelConnectorPlugin


class GLM52MPConnectorPlugin(MPModelConnectorPlugin):
    """Register the GLM-5.2 worker and scheduler without central branching."""

    name = "glm52"
    model_types = frozenset({"glm_moe_dsa"})
    worker_module = "atom.kv_transfer.offload.mp.glm52.connector"
    worker_class = "GLM52LMCacheMPConnector"
    scheduler_module = worker_module
    scheduler_class = "GLM52LMCacheMPConnectorScheduler"


__all__ = ["GLM52MPConnectorPlugin"]
