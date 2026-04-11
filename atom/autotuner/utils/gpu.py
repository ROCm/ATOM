"""ROCm GPU utilities for the autotuner."""

from __future__ import annotations

import logging
import re
import subprocess

from atom.autotuner.types import GPUInfo

logger = logging.getLogger(__name__)


class ROCmGPU:
    """Utility class for querying AMD GPU state via rocm-smi."""

    @staticmethod
    def detect() -> GPUInfo:
        """Auto-detect AMD GPU model and create appropriate GPUInfo."""
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            output = proc.stdout.lower()
            num_gpus = ROCmGPU.count_gpus()

            if "mi355" in output:
                info = GPUInfo.mi355x(num_gpus)
            elif "mi325" in output:
                info = GPUInfo.mi325x(num_gpus)
            elif "mi300" in output:
                info = GPUInfo.mi300x(num_gpus)
            else:
                logger.warning("Unknown GPU model, defaulting to MI300X profile")
                info = GPUInfo.mi300x(num_gpus)

            info.rocm_version = ROCmGPU.get_rocm_version()
            info.driver_version = ROCmGPU.get_driver_version()
            return info

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("rocm-smi not available, using default MI300X profile")
            return GPUInfo.mi300x()

    @staticmethod
    def count_gpus() -> int:
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=10,
            )
            return max(proc.stdout.count("GPU"), 1)
        except Exception:
            return 1

    @staticmethod
    def _smi_driver_field(keyword: str) -> str:
        """Extract a field from ``rocm-smi --showdriverversion`` matching *keyword*."""
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showdriverversion"],
                capture_output=True, text=True, timeout=10,
            )
            for line in proc.stdout.splitlines():
                if keyword in line.lower():
                    return line.split(":")[-1].strip()
        except Exception:
            pass
        return "unknown"

    @classmethod
    def get_rocm_version(cls) -> str:
        return cls._smi_driver_field("version")

    @classmethod
    def get_driver_version(cls) -> str:
        return cls._smi_driver_field("driver")

    @staticmethod
    def get_vram_usage() -> dict[int, float]:
        """Return VRAM usage percentage per GPU."""
        usage = {}
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showmemuse"],
                capture_output=True, text=True, timeout=10,
            )
            gpu_id = 0
            for line in proc.stdout.splitlines():
                m = re.search(r"(\d+\.?\d*)%", line)
                if m:
                    usage[gpu_id] = float(m.group(1))
                    gpu_id += 1
        except Exception:
            pass
        return usage

    @staticmethod
    def get_power_draw() -> dict[int, float]:
        """Return current power draw in watts per GPU."""
        power = {}
        try:
            proc = subprocess.run(
                ["rocm-smi", "--showpower"],
                capture_output=True, text=True, timeout=10,
            )
            gpu_id = 0
            for line in proc.stdout.splitlines():
                m = re.search(r"([\d.]+)\s*W", line)
                if m:
                    power[gpu_id] = float(m.group(1))
                    gpu_id += 1
        except Exception:
            pass
        return power

    @staticmethod
    def clear_compile_cache() -> None:
        """Clear ATOM/torch compile cache to avoid stale artifacts."""
        import shutil
        from pathlib import Path

        cache_dirs = [
            Path.home() / ".cache" / "atom",
            Path.home() / ".cache" / "torch_extensions",
            Path("/tmp") / "torchinductor_root",
        ]
        for d in cache_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info("Cleared cache: %s", d)
