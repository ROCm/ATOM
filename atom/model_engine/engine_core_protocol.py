# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight protocol types shared between CoreManager and EngineCore.

Deliberately free of torch/aiter/zmq imports so the CPU-only CoreManager
(request thread) and unit tests can import ``EngineCoreRequestType`` without
dragging in the heavy ``engine_core`` -> ``async_proc`` -> ``aiter`` chain.
"""

import enum


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used within EngineCore.
    SHUTDOWN = b"\x05"
    # Stream output for callbacks
    STREAM = b"\x06"
    # Signal that EngineCore is fully initialized and ready
    READY = b"\x07"
    # Response to a synchronous utility command
    UTILITY_RESPONSE = b"\x08"
    # Unsolicited metrics snapshot, pushed by EngineCore on its own clock.
    METRICS = b"\x09"
    # (seqs ingested so far, prefill tokens still to compute, applied
    # DP_PREFIX_CREDIT updates), pushed by a DP engine core when the first two
    # change or an update was applied; read by `share_cache` DP routing.
    PREFILL_BACKLOG = b"\x0a"
    # [(seq id, prompt tokens reused)] of requests `share_cache` placed on an
    # estimated prefix, sent ahead of their first output.
    DP_REUSE = b"\x0b"
    # Absolute cached-prefix estimate corrections, sent after the seq's ADD.
    DP_PREFIX_CREDIT = b"\x0c"
