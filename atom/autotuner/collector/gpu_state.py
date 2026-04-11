"""
GPU state management for reproducible benchmarking on AMD GPUs.

Addresses Q4: clock locking, power mode, warm-up strategy.
Uses ``rocm-smi`` to pin performance level and clock frequencies,
ensuring stable measurements across benchmark runs.
"""

from __future__ import annotations

import logging
import subprocess
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUClockState:
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    perf_level: str = "auto"
    power_cap_watts: int = 0


class GPUStateManager:
    """
    Controls AMD GPU state for reproducible kernel benchmarks.

    Lifecycle::

        mgr = GPUStateManager(device_ids=[0, 1, 2, 3])
        with mgr.pinned():
            # clocks are locked, perf level = high
            run_benchmarks()
        # clocks restored to original state
    """

    def __init__(self, device_ids: list[int] | None = None):
        self.device_ids = device_ids or [0]
        self._saved_states: dict[int, GPUClockState] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    class _PinnedCtx:
        def __init__(self, mgr: GPUStateManager):
            self._mgr = mgr

        def __enter__(self):
            self._mgr._save_and_pin()
            return self._mgr

        def __exit__(self, *exc):
            self._mgr._restore()

    def pinned(self) -> _PinnedCtx:
        return self._PinnedCtx(self)

    # ------------------------------------------------------------------
    # rocm-smi wrappers
    # ------------------------------------------------------------------

    def _run_smi(self, args: list[str]) -> str:
        cmd = ["rocm-smi"] + args
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return proc.stdout
        except FileNotFoundError:
            logger.warning("rocm-smi not found — GPU state management disabled")
            return ""
        except subprocess.TimeoutExpired:
            logger.warning("rocm-smi timed out: %s", " ".join(cmd))
            return ""

    def get_gpu_info(self, device_id: int = 0) -> dict:
        """Query basic GPU info via rocm-smi."""
        output = self._run_smi(["-d", str(device_id), "--showproductname"])
        info = {"device_id": device_id, "name": "unknown"}
        for line in output.splitlines():
            if "Card Series" in line or "Card series" in line:
                info["name"] = line.split(":")[-1].strip()
        return info

    def get_memory_usage(self, device_id: int = 0) -> dict:
        """Query VRAM usage."""
        output = self._run_smi(["-d", str(device_id), "--showmemuse"])
        info = {"used_pct": 0.0}
        for line in output.splitlines():
            m = re.search(r"(\d+\.?\d*)%", line)
            if m:
                info["used_pct"] = float(m.group(1))
                break
        return info

    def get_temperature(self, device_id: int = 0) -> float:
        output = self._run_smi(["-d", str(device_id), "--showtemp"])
        for line in output.splitlines():
            m = re.search(r"(\d+\.?\d*)\s*c", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return 0.0

    def _save_and_pin(self) -> None:
        """Save current clock state, then lock to high-perf mode."""
        for dev in self.device_ids:
            state = GPUClockState()
            output = self._run_smi(["-d", str(dev), "--showperflevel"])
            for line in output.splitlines():
                if "Performance Level" in line:
                    state.perf_level = line.split(":")[-1].strip().lower()
            self._saved_states[dev] = state

        for dev in self.device_ids:
            self._run_smi(["-d", str(dev), "--setperflevel", "high"])
        logger.info(
            "GPU clocks pinned to high-perf for devices %s", self.device_ids
        )

    def _restore(self) -> None:
        """Restore original GPU clock state."""
        for dev, state in self._saved_states.items():
            level = state.perf_level if state.perf_level else "auto"
            self._run_smi(["-d", str(dev), "--setperflevel", level])
        logger.info("GPU clocks restored for devices %s", list(self._saved_states))
        self._saved_states.clear()

    def wait_for_cool(self, target_temp_c: float = 70.0, timeout_sec: float = 120.0) -> None:
        """Block until GPU temperature drops below threshold."""
        import time

        start = time.time()
        for dev in self.device_ids:
            while True:
                temp = self.get_temperature(dev)
                if temp <= target_temp_c or temp == 0.0:
                    break
                if time.time() - start > timeout_sec:
                    logger.warning(
                        "GPU %d still at %.1f°C after %.0fs — proceeding anyway",
                        dev, temp, timeout_sec,
                    )
                    break
                time.sleep(2)
