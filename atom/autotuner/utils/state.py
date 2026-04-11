"""
State management for crash recovery and session persistence.

The autotuner can be interrupted by:
- User Ctrl+C
- Machine resource contention (someone else grabs GPUs)
- SSH disconnection
- OOM kills

StateManager saves periodic checkpoints and can resume from the last one.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from atom.autotuner.types import TunerState

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages autotuner state persistence for crash recovery.

    Saves checkpoints at configurable intervals.  On resume, loads the
    latest checkpoint and restores the experiment tracker, Pareto frontier,
    and best configuration.
    """

    def __init__(
        self,
        state_dir: Path,
        checkpoint_interval_sec: int = 300,
    ):
        self.state_dir = state_dir
        self.checkpoint_interval_sec = checkpoint_interval_sec
        self._last_checkpoint = 0.0
        state_dir.mkdir(parents=True, exist_ok=True)

    def should_checkpoint(self) -> bool:
        return (time.time() - self._last_checkpoint) >= self.checkpoint_interval_sec

    def save(self, state: TunerState) -> Path:
        """Save a state checkpoint."""
        state.last_checkpoint = time.time()
        path = self.state_dir / f"checkpoint_{state.session_id}.json"
        state.save(path)
        self._last_checkpoint = time.time()

        latest_link = self.state_dir / "latest_checkpoint.json"
        state.save(latest_link)

        logger.info(
            "Checkpoint saved: session=%s, experiments=%d",
            state.session_id, len(state.all_experiments),
        )
        return path

    def load_latest(self) -> Optional[TunerState]:
        """Load the most recent checkpoint."""
        latest = self.state_dir / "latest_checkpoint.json"
        if not latest.exists():
            return None

        try:
            state = TunerState.load(latest)
            logger.info(
                "Loaded checkpoint: session=%s, model=%s",
                state.session_id, state.model,
            )
            return state
        except Exception:
            logger.exception("Failed to load checkpoint from %s", latest)
            return None

    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints sorted by time (newest first)."""
        checkpoints = list(self.state_dir.glob("checkpoint_*.json"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints

    def cleanup_old(self, keep: int = 5) -> int:
        """Remove old checkpoints, keeping the N most recent."""
        checkpoints = self.list_checkpoints()
        removed = 0
        for cp in checkpoints[keep:]:
            cp.unlink()
            removed += 1
        if removed:
            logger.info("Cleaned up %d old checkpoints", removed)
        return removed
