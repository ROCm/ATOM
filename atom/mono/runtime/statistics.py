# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only session snapshots; never synchronize or inspect device data."""


def serving_statistics(session, policy: str, rank: int) -> dict:
    result = {
        "rank": rank,
        "policy": policy,
        "installed": session is not None,
        "enabled": session is not None and not session.disabled,
    }
    if session is not None:
        result.update(session.serving_status())
    return result
