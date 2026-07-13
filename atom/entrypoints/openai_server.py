# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backward-compatible entry point for the ATOM OpenAI API server.

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

from atom.utils import envs, set_ulimit

if envs.USE_ATOMESH_ENTRYPOINTS:
    from atom.entrypoints.atomesh.server import main
else:
    from atom.entrypoints.openai.api_server import main

if __name__ == "__main__":
    # Raise the open-file soft limit before the server (and the engine-core
    # subprocesses it spawns) start, so high connection concurrency does not
    # exhaust file descriptors. Inherited by spawned children.
    set_ulimit()
    main()
