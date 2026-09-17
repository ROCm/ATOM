# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host CPU accounting for this server's process tree.

Why this exists: every latency slice the engine reports is either GPU device
time or a wall-clock interval, so a host that is out of CPU looks exactly like
a host that is waiting on the GPU. `api_preprocess` inflating under load, a
decode step whose wall time exceeds its target forward -- both are consistent
with CPU starvation and with GPU/fabric contention, and nothing in the existing
exposition tells the two apart. These counters do.

What is measured, and by whom: the API process is the parent of every engine
core and GPU worker (`get_mp_context()` spawns them), so it can account for the
whole tree without any engine-side plumbing or IPC. Processes are grouped by
the title `set_process_title` gives them, which is what that function exists
for; the API process itself is reported as ``api``.

Counters, not a utilization gauge, for the same reason the queue gauges are
sampled state and the latency metrics are histograms: a gauge read once per
scrape misses everything between scrapes, and CPU saturation is bursty.
`rate(atom:process_cpu_seconds_total[60s])` is cores consumed, and it is
correct regardless of scrape interval. `atom:process_cpus` is the ceiling those
rates run against.

Cost: rendering runs inline on the event loop that delivers every SSE stream
(see `_gc_metrics`), so this stays a fixed number of small procfs reads. The
descendant set is discovered once and refreshed on a timer, because
`children(recursive=True)` walks all of /proc and the worker set is static
after startup.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable

import psutil
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

logger = logging.getLogger("atom")

# The worker set does not change after startup; a crash takes the server with
# it. Long enough that the /proc walk is amortized to nothing, short enough
# that a late-arriving process still shows up within a benchmark window.
_TREE_REFRESH_SECONDS = 30.0

# What the API process calls itself. Its own title carries the same prefix as
# the workers', which would make the row that owns tokenization and SSE
# indistinguishable from the ones that own the GPU.
_SELF_LABEL = "api"


class _ProcessTree:
    """This process plus its descendants, rediscovered on a timer."""

    def __init__(self, refresh_seconds: float = _TREE_REFRESH_SECONDS):
        self._refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._self_proc: psutil.Process | None = None
        self._children: list[psutil.Process] = []
        self._discovered_at = 0.0

    def _title(self, proc: psutil.Process) -> str:
        """The label a process is grouped under.

        `set_process_title` writes ``<prefix>::<name>``; the prefix is constant
        across the tree and only the name distinguishes a rank, so the prefix is
        dropped. Falls back to the process name when no title was set (the soft
        setproctitle dependency is absent, or the child is not ours).
        """
        try:
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "unknown"
        _, separator, suffix = name.partition("::")
        return (suffix or name) if separator else name

    def snapshot(self) -> list[tuple[str, psutil.Process]]:
        """(label, process) for the whole tree, refreshing the set when due."""
        now = time.monotonic()
        with self._lock:
            if self._self_proc is None or now - self._discovered_at >= (
                self._refresh_seconds
            ):
                try:
                    self._self_proc = self._self_proc or psutil.Process()
                    self._children = self._self_proc.children(recursive=True)
                except psutil.Error as exc:
                    # Never fail a scrape over process accounting. A stale or
                    # empty child list still reports this process correctly.
                    logger.debug("CPU metrics: could not walk process tree: %s", exc)
                self._discovered_at = now
            return [(_SELF_LABEL, self._self_proc)] + [
                (self._title(child), child) for child in self._children
            ]


_TREE = _ProcessTree()


def _families(describe: bool) -> Iterable[Metric]:
    cpu = CounterMetricFamily(
        "atom:process_cpu_seconds",
        "CPU seconds consumed by this server's processes, grouped by process "
        "title and mode. rate() over this is cores in use; compare against "
        "atom:process_cpus. The 'api' row owns tokenization, chat templating "
        "and SSE, so it is the one that bounds api_preprocess.",
        labels=["process", "mode"],
    )
    threads = GaugeMetricFamily(
        "atom:process_threads",
        "Threads per process. The API process runs one event loop plus its "
        "executor pool, so this is the ceiling on concurrent tokenization.",
        labels=["process"],
    )
    cpus = GaugeMetricFamily(
        "atom:process_cpus",
        "Logical CPUs visible to this server, honoring cgroup/affinity limits. "
        "The ceiling that rate(atom:process_cpu_seconds_total) runs against.",
    )
    if describe:
        return (cpu, threads, cpus)

    # Processes sharing a title are summed: two ranks may be titled alike, and
    # emitting the same label set twice is invalid exposition.
    user: dict[str, float] = {}
    system: dict[str, float] = {}
    thread_count: dict[str, int] = {}
    for label, proc in _TREE.snapshot():
        if proc is None:
            continue
        try:
            # One /proc/<pid>/stat read serves both fields.
            with proc.oneshot():
                times = proc.cpu_times()
                count = proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # A process that exited between discovery and this read simply
            # stops contributing; its title's counter flattens rather than
            # dropping out mid-series.
            continue
        user[label] = user.get(label, 0.0) + times.user
        system[label] = system.get(label, 0.0) + times.system
        thread_count[label] = thread_count.get(label, 0) + count

    for label in sorted(thread_count):
        cpu.add_metric([label, "user"], user[label])
        cpu.add_metric([label, "system"], system[label])
        threads.add_metric([label], float(thread_count[label]))
    cpus.add_metric([], float(_visible_cpus()))
    return (cpu, threads, cpus)


def _visible_cpus() -> int:
    """Logical CPUs this server may actually run on.

    `os.cpu_count()` reports the machine, not the share a container was given,
    which would make every utilization ratio read low on a shared host.
    `sched_getaffinity` is the honest ceiling where it exists.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def collect_cpu_metrics(snapshot) -> Iterable[Metric]:
    """Collector entry point; the snapshot argument is unused.

    CPU belongs to this process tree and is read live at scrape time, the way
    `_gc_metrics` reads this interpreter's own collector rather than the
    engine's snapshot.
    """
    yield from _families(describe=snapshot is None)
