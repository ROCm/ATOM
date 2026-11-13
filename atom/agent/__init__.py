# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM Chat: a conversational deployment agent for ATOM.

Users describe what they want served ("DeepSeek-V4.1-Flash on the free GPUs,
tuned for agentic coding") and the agent inspects the machine, finds the
matching recipe, launches the server, watches it come up, verifies it and
explains failures -- asking before anything that changes the machine.

The package is standard-library only and never imports torch, AITER or the
engine, so it runs on a bare host that drives ATOM through docker as well as
inside an ATOM container. Its "brain" is any OpenAI-compatible endpoint with
tool calling -- including an ATOM server, so ATOM can deploy ATOM.

Entry points: ``python -m atom.agent`` / ``atom-chat`` (interactive chat),
``atom-chat mcp`` (the same tools over MCP stdio for Claude Code, Hermes, ...).
"""
