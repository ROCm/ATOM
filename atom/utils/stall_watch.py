# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stack dumps for a serving loop that stops making progress.

A loop arms its watch at the start of each unit of work. If one unit is still
running `ATOM_STALL_DUMP_S` seconds later, a daemon thread logs every thread's
stack in that process, once per unit. Python-level rather than `faulthandler`
so each line names its process, which a merged multi-process log needs; the
waits worth catching (device syncs, sockets, locks) release the GIL.
"""

import logging
import os
import sys
import threading
import time
import traceback

logger = logging.getLogger("atom")

_STACK_DEPTH = 24


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
        self._reported: float | None = None
        self._poll_s = poll_s if poll_s is not None else min(1.0, seconds / 4)
        self._thread = threading.Thread(
            target=self._watch, name=f"stall-watch-{label}", daemon=True
        )
        self._thread.start()

    def arm(self) -> None:
        self._armed_at = time.monotonic()

    def disarm(self) -> None:
        self._armed_at = None

    def _watch(self) -> None:
        while True:
            time.sleep(self._poll_s)
            armed_at = self._armed_at
            if armed_at is None or armed_at == self._reported:
                continue
            stalled_s = time.monotonic() - armed_at
            if stalled_s < self.seconds:
                continue
            self._reported = armed_at
            try:
                self.dump(stalled_s)
            except Exception:  # a diagnostic must not end the watch
                logger.exception("[stall] %s: stack dump failed", self.label)

    def dump(self, stalled_s: float) -> None:
        names = {t.ident: t.name for t in threading.enumerate()}
        me = threading.get_ident()
        by_stack: dict[str, list[str]] = {}
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            frames = traceback.extract_stack(frame, limit=_STACK_DEPTH)
            stack = "<".join(
                f"{os.path.basename(f.filename)}:{f.name}:{f.lineno}"
                for f in reversed(frames)
            )
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


NO_STALL_WATCH = _NoStallWatch()


def stall_watch(label: str) -> StallWatch | _NoStallWatch:
    from atom.utils import envs

    seconds = envs.ATOM_STALL_DUMP_S
    if seconds <= 0:
        return NO_STALL_WATCH
    return StallWatch(label, float(seconds))
