"""Multi-node orchestration smoke (placeholder)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.multigpu, pytest.mark.slow]


def test_multinode_smoke() -> None:
    """Placeholder until multi-node Ray + cluster configs are exercised in CI."""
    pytest.skip(
        "Multi-node Ray cluster fixture not yet implemented for automated tests."
    )
