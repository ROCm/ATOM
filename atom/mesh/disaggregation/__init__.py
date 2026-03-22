# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV cache disaggregation for Prefill/Decode (P/D) separation.

Public API re-exports — engine code should import from this package
rather than reaching into submodules directly.
"""

from atom.mesh.disaggregation.aggregator import KVOutputAggregator
from atom.mesh.disaggregation.base import KVConnectorBase, KVConnectorSchedulerBase
from atom.mesh.disaggregation.factory import KVConnectorFactory
from atom.mesh.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    ReqMeta,
)

__all__ = [
    "KVConnectorBase",
    "KVConnectorSchedulerBase",
    "KVConnectorFactory",
    "KVConnectorOutput",
    "KVOutputAggregator",
    "ConnectorMetadata",
    "ReqMeta",
]
