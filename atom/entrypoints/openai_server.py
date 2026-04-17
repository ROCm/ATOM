# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backward-compatible entry point for the ATOM OpenAI API server.

Usage:
    python -m atom.entrypoints.openai_server --model <model> [options]
"""

from atom.entrypoints.openai.api_server import main

if __name__ == "__main__":
    main()
