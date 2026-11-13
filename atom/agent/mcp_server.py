# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The agent's tools as an MCP server (stdio, newline-delimited JSON-RPC).

    claude mcp add atom -- python -m atom.agent mcp

lets Claude Code (or Hermes, Cursor, ...) plan, launch, watch, diagnose and
stop ATOM deployments with the same tools ``atom-chat`` uses. The MCP client
owns approvals, so tools run when called; the annotations tell it which ones
change the machine.
"""

from __future__ import annotations

import json
import os
import sys

from atom.agent.config import AgentConfig
from atom.agent.policy import Policy
from atom.agent.tools import ToolBox, ToolContext, render_result

PROTOCOL_VERSION = "2025-06-18"


def _version() -> str:
    try:
        from atom._version import __version__

        return __version__
    except ImportError:
        return "0"


def handle(box: ToolBox, msg: dict) -> dict | None:
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:  # notification (notifications/initialized, cancelled, ...)
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        client_version = (msg.get("params") or {}).get("protocolVersion")
        return ok(
            {
                "protocolVersion": client_version or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "atom", "version": _version()},
                "instructions": (
                    "Deploy and operate ATOM LLM servers on this machine's AMD "
                    "GPUs. Typical flow: inspect_system -> plan_deployment -> "
                    "launch_server -> wait_until_ready -> smoke_test; on failure "
                    "call diagnose before changing anything."
                ),
            }
        )
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok(
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.parameters,
                        "annotations": {
                            "readOnlyHint": not t.mutating,
                            "destructiveHint": t.name == "stop_server",
                            "openWorldHint": False,
                        },
                    }
                    for t in box.tools.values()
                ]
            }
        )
    if method == "tools/call":
        params = msg.get("params") or {}
        result = box.call(params.get("name", ""), params.get("arguments") or {})
        return ok(
            {
                "content": [{"type": "text", "text": render_result(result, 60000)}],
                "isError": bool(result.get("error")),
            }
        )
    return {
        "jsonrpc": "2.0",
        "id": mid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def serve(
    cfg: AgentConfig,
    stdin=None,
    stdout=None,
    mode: str = "auto",
    profile: str = "developer",
) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    ctx = ToolContext(
        cfg,
        policy=Policy(mode),
        progress=lambda line: print(line, file=sys.stderr, flush=True),
        session=f"mcp-{os.getpid()}",
        profile=profile,
    )
    # switch_brain only means something inside atom-chat's own loop.
    box = ToolBox(ctx, include_brain_switch=False)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            reply = handle(box, msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
