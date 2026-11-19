# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mooncake producer endpoint fields shared by discovery and legacy handoff."""

DEFAULT_HANDSHAKE_PORT = 6301


def producer_endpoint(
    *, host, handshake_port, pp_size, block_size, dcp_size, pp_hosts=()
):
    """Connection and page geometry; request flags belong to the caller.

    ``pp_hosts`` names the node of each stage of a pipeline that spans nodes
    (``Config.pp_stage_ips``); a consumer reaches stage ``s`` there instead of
    at ``host``. Left out of the endpoint when empty, so a single-node producer
    advertises what it always did.
    """
    endpoint = {
        "remote_host": host,
        "remote_handshake_port": handshake_port,
        "remote_pp_size": pp_size,
        "block_size": block_size,
        "dcp_size": dcp_size,
    }
    if pp_hosts:
        endpoint["remote_pp_hosts"] = list(pp_hosts)
    return endpoint
