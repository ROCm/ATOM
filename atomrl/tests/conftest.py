# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared pytest configuration for AtomRL-owned tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires one ROCm GPU")
    config.addinivalue_line("markers", "multigpu: requires multiple ROCm GPUs")
