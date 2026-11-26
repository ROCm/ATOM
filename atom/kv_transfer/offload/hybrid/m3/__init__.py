# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PAGE-only (multi-region, no SLOT) LMCache offload for MiniMax-M3 NSA.

The connector stays lazy. ``codec`` only names the Store namespace, and the
Mooncake Store layout table imports that module from every worker; importing
the connector here would pull the LMCache M3 worker into a process that never
builds one.
"""


def __getattr__(name: str):
    if name in {"M3OffloadConnector", "M3OffloadScheduler"}:
        from atom.kv_transfer.offload.hybrid.m3.connector import (
            M3OffloadConnector,
            M3OffloadScheduler,
        )

        return (
            M3OffloadConnector if name == "M3OffloadConnector" else M3OffloadScheduler
        )
    raise AttributeError(name)


__all__ = ["M3OffloadConnector", "M3OffloadScheduler"]
