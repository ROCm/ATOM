# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``atom.mono.runtime.compile.compile_only`` on several threads, without a
GPU or a kernel: whether a thread inside its block can find the switch off.

The switch is one process-wide environment variable. A block that found it
unset removes it on exit, also while another thread is still inside its own
block; a launcher called there compiles and then LAUNCHES, with the zero
arguments a compile-only call passes. This is the shape of
``test_the_step_kernel_traces_on_several_threads_at_once``
(tests/test_deepseek_v41_gfx1250.py): four threads, each
``with compile_only(): launcher(0, ...)``. Here the launcher is a stand-in
that records what it would have done.

    python tools/steptail_compile_only_race.py
    env: TRIALS (200), THREADS (4)
"""

import os
import threading
import time

from atom.mono.runtime.compile import compile_only

TRIALS = int(os.environ.get("TRIALS", "200"))
THREADS = int(os.environ.get("THREADS", "4"))


def trial(stagger: float):
    launched, barrier = [], threading.Barrier(THREADS)

    def build(rank):
        barrier.wait()
        with compile_only():
            time.sleep(stagger * rank)  # the trace / the cache load
            if os.environ.get("COMPILE_ONLY") != "1":
                launched.append(rank)  # a real launcher would run here

    threads = [threading.Thread(target=build, args=(r,)) for r in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    left_on = os.environ.pop("COMPILE_ONLY", None)
    return launched, left_on


def main():
    for stagger in (0.0, 0.0005):
        hit = leaked = 0
        for _ in range(TRIALS):
            launched, left_on = trial(stagger)
            hit += bool(launched)
            leaked += left_on is not None
        print(
            f"stagger {stagger * 1e3:.1f} ms: {hit}/{TRIALS} trials had a thread inside its "
            f"block with the switch off; {leaked}/{TRIALS} left the switch on afterwards"
        )


if __name__ == "__main__":
    main()
