# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stack dumps for a serving loop that stops making progress.

A loop arms its watch at the start of each unit of work. If one unit is still
running `ATOM_STALL_DUMP_S` seconds later, a daemon thread logs every thread's
stack in that process, once per unit. The same thread samples the loop
thread's stack on every poll, and when an overrunning unit ends it logs where
those samples landed, so a long stall is attributed rather than caught at one
instant. Python-level rather than `faulthandler` so each line names its
process, which a merged multi-process log needs; the waits worth catching
(device syncs, sockets, locks) release the GIL.
"""

import collections
import logging
import os
import sys
import threading
import time
import traceback

logger = logging.getLogger("atom")

_STACK_DEPTH = 24
_SAMPLE_DEPTH = 12
_SAMPLE_TOP = 8


class _NoStallWatch:
    def arm(self) -> None:
        pass

    def disarm(self) -> None:
        pass


class StallWatch:
    def __init__(self, label: str, seconds: float, poll_s: float | None = None):
        self.label = label
        self.seconds = seconds
        self._armed_at: float | None = None
        self._owner: int | None = None
        self._poll_s = poll_s if poll_s is not None else min(1.0, seconds / 4)
        self._thread = threading.Thread(
            target=self._watch, name=f"stall-watch-{label}", daemon=True
        )
        self._thread.start()

    def arm(self) -> None:
        self._owner = threading.get_ident()
        self._armed_at = time.monotonic()

    def disarm(self) -> None:
        self._armed_at = None

    def _watch(self) -> None:
        unit: float | None = None
        dumped = False
        samples: collections.Counter[str] = collections.Counter()
        while True:
            time.sleep(self._poll_s)
            armed_at = self._armed_at
            now = time.monotonic()
            if armed_at != unit:
                if dumped:
                    self._log_samples(now - unit, samples)
                unit, dumped = armed_at, False
                samples.clear()
            if armed_at is None:
                continue
            try:
                frame = sys._current_frames().get(self._owner)
                if frame is not None:
                    samples[_compact(frame, _SAMPLE_DEPTH, lines=False)] += 1
                if not dumped and now - armed_at >= self.seconds:
                    dumped = True
                    self.dump(now - armed_at)
            except Exception:  # a diagnostic must not end the watch
                logger.exception("[stall] %s: stack dump failed", self.label)

    def dump(self, stalled_s: float) -> None:
        names = {t.ident: t.name for t in threading.enumerate()}
        me = threading.get_ident()
        by_stack: dict[str, list[str]] = {}
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            stack = _compact(frame, _STACK_DEPTH, lines=True)
            by_stack.setdefault(stack, []).append(names.get(ident, str(ident)))
        logger.warning(
            "[stall] %s: one step has run %.0f s; %d threads, %d distinct stacks",
            self.label,
            stalled_s,
            sum(len(v) for v in by_stack.values()),
            len(by_stack),
        )
        for stack, threads in sorted(by_stack.items(), key=lambda kv: -len(kv[1])):
            logger.warning(
                "[stall] %s: n=%d threads=%s stack=%s",
                self.label,
                len(threads),
                ",".join(sorted(threads)[:4]),
                stack,
            )

    def _log_samples(self, ran_s: float, samples: collections.Counter[str]) -> None:
        logger.warning(
            "[stall] %s: step ended after %.0f s; %d samples of its thread, "
            "one per %.2g s",
            self.label,
            ran_s,
            sum(samples.values()),
            self._poll_s,
        )
        for stack, n in samples.most_common(_SAMPLE_TOP):
            logger.warning("[stall] %s: samples=%d stack=%s", self.label, n, stack)


def _compact(frame, depth: int, lines: bool) -> str:
    frames = traceback.extract_stack(frame, limit=depth)
    return "<".join(
        f"{os.path.basename(f.filename)}:{f.name}" + (f":{f.lineno}" if lines else "")
        for f in reversed(frames)
    )


NO_STALL_WATCH = _NoStallWatch()


def stall_watch(label: str) -> StallWatch | _NoStallWatch:
    from atom.utils import envs

    seconds = envs.ATOM_STALL_DUMP_S
    if seconds <= 0:
        return NO_STALL_WATCH
    return StallWatch(label, float(seconds))
