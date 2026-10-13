# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mooncake producer endpoint fields shared by discovery and legacy handoff."""

DEFAULT_HANDSHAKE_PORT = 6301


def producer_endpoint(*, host, handshake_port, pp_size, block_size, dcp_size):
    """Connection and page geometry; request flags belong to the caller."""
    return {
        "remote_host": host,
        "remote_handshake_port": handshake_port,
        "remote_pp_size": pp_size,
        "block_size": block_size,
        "dcp_size": dcp_size,
    }
