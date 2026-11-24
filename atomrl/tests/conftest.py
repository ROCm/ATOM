"""Shared pytest configuration for AtomRL-owned tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires one ROCm GPU")
    config.addinivalue_line("markers", "multigpu: requires multiple ROCm GPUs")
