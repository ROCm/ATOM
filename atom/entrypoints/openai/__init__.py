# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM OpenAI-compatible API server package."""


def main():
    """Lazy entry point to avoid importing heavy deps at package level."""
    from .api_server import main as _main

    _main()


__all__ = ["main"]
