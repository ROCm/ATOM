"""Fixtures for end-to-end training smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def e2e_config_dir() -> Path:
    """Directory containing checked-in YAML recipes (``configs/``)."""
    return Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Isolated checkpoint directory per test."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
