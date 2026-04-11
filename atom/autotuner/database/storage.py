"""
Performance data persistence layer.

Stores kernel benchmark results in a lightweight JSON-lines format with
SQLite index for fast querying.  Supports multiple "systems" (mi355x, mi300x)
and multiple framework versions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from atom.autotuner.types import KernelBenchResult, KernelConfig, KernelType

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    system      TEXT    NOT NULL,
    kernel_type TEXT    NOT NULL,
    fingerprint TEXT    NOT NULL,
    params_json TEXT    NOT NULL,
    latency_us  REAL    NOT NULL,
    tflops      REAL    DEFAULT 0,
    mem_bw_gbps REAL    DEFAULT 0,
    power_w     REAL    DEFAULT 0,
    gpu_util    REAL    DEFAULT 0,
    timestamp   REAL    NOT NULL,
    UNIQUE(system, kernel_type, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_system_type ON benchmarks(system, kernel_type);
CREATE INDEX IF NOT EXISTS idx_fingerprint ON benchmarks(fingerprint);
"""


class PerfStorage:
    """
    SQLite-backed performance data store.

    Usage::

        store = PerfStorage(Path("data/perf.db"))
        store.insert("mi355x", result)
        results = store.query("mi355x", KernelType.GEMM, m=4096)
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def insert(self, system: str, result: KernelBenchResult) -> None:
        fp = result.config.fingerprint()
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO benchmarks
                   (system, kernel_type, fingerprint, params_json,
                    latency_us, tflops, mem_bw_gbps, power_w, gpu_util, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    system,
                    result.config.kernel_type.value,
                    fp,
                    json.dumps(result.config.params, sort_keys=True),
                    result.latency_us,
                    result.throughput_tflops,
                    result.memory_bw_gbps,
                    result.power_watts,
                    result.gpu_util_pct,
                    result.timestamp,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to insert benchmark result")

    def insert_batch(self, system: str, results: list[KernelBenchResult]) -> int:
        count = 0
        for r in results:
            try:
                self.insert(system, r)
                count += 1
            except Exception:
                pass
        return count

    def query(
        self,
        system: str,
        kernel_type: KernelType,
        **param_filters: object,
    ) -> list[KernelBenchResult]:
        """Query results, optionally filtering by parameter values."""
        rows = self._conn.execute(
            "SELECT params_json, latency_us, tflops, mem_bw_gbps, power_w, gpu_util, timestamp "
            "FROM benchmarks WHERE system = ? AND kernel_type = ?",
            (system, kernel_type.value),
        ).fetchall()

        results = []
        for params_json, lat, tfl, bw, pw, gu, ts in rows:
            params = json.loads(params_json)
            if param_filters:
                if not all(params.get(k) == v for k, v in param_filters.items()):
                    continue
            results.append(KernelBenchResult(
                config=KernelConfig(kernel_type=kernel_type, params=params),
                latency_us=lat,
                throughput_tflops=tfl,
                memory_bw_gbps=bw,
                power_watts=pw,
                gpu_util_pct=gu,
                timestamp=ts,
            ))
        return results

    def query_all(self, system: str) -> list[KernelBenchResult]:
        rows = self._conn.execute(
            "SELECT kernel_type, params_json, latency_us, tflops, mem_bw_gbps, "
            "power_w, gpu_util, timestamp FROM benchmarks WHERE system = ?",
            (system,),
        ).fetchall()

        return [
            KernelBenchResult(
                config=KernelConfig(
                    kernel_type=KernelType(kt), params=json.loads(pj)
                ),
                latency_us=lat,
                throughput_tflops=tfl,
                memory_bw_gbps=bw,
                power_watts=pw,
                gpu_util_pct=gu,
                timestamp=ts,
            )
            for kt, pj, lat, tfl, bw, pw, gu, ts in rows
        ]

    def count(self, system: str, kernel_type: Optional[KernelType] = None) -> int:
        if kernel_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM benchmarks WHERE system = ? AND kernel_type = ?",
                (system, kernel_type.value),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM benchmarks WHERE system = ?", (system,)
            ).fetchone()
        return row[0] if row else 0

    def import_jsonl(self, system: str, path: Path) -> int:
        """Import benchmark results from JSON-lines file."""
        count = 0
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    config = KernelConfig(
                        kernel_type=KernelType(row["kernel_type"]),
                        params=row["params"],
                    )
                    result = KernelBenchResult(
                        config=config,
                        latency_us=row["latency_us"],
                        throughput_tflops=row.get("throughput_tflops", 0),
                        memory_bw_gbps=row.get("memory_bw_gbps", 0),
                        power_watts=row.get("power_watts", 0),
                        gpu_util_pct=row.get("gpu_util_pct", 0),
                        timestamp=row.get("timestamp", time.time()),
                    )
                    self.insert(system, result)
                    count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        logger.info("Imported %d records from %s", count, path)
        return count

    def export_jsonl(self, system: str, path: Path) -> int:
        results = self.query_all(system)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                row = {
                    "kernel_type": r.config.kernel_type.value,
                    "params": r.config.params,
                    "latency_us": r.latency_us,
                    "throughput_tflops": r.throughput_tflops,
                    "memory_bw_gbps": r.memory_bw_gbps,
                    "power_watts": r.power_watts,
                    "gpu_util_pct": r.gpu_util_pct,
                    "timestamp": r.timestamp,
                }
                f.write(json.dumps(row) + "\n")
        logger.info("Exported %d records to %s", len(results), path)
        return len(results)
