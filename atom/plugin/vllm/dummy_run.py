# SPDX-License-Identifier: MIT
"""Whether vLLM is running a batch it made up rather than one it scheduled.

A model whose step carries per-request state has to tell the two apart: a
synthetic batch's rows are not requests, so they must not take the STATE slots
real requests hold, and their cursors mean nothing.

vLLM announces it twice, and only one of the two is general.
`build_for_cudagraph_capture` is called in place of `build`, which is exact --
but only for a FULL graph; a PIECEWISE capture and the warmup forwards come
through `build` like any other step. What covers all of them is the method
vLLM enters to run one: `_dummy_run`.

Inferring it from the batch was tried four times and each inference was wrong
in a way that only appeared under load -- a repeated placeholder request id
(which stops repeating as soon as the request-id pass-through works), a
startup-only window, a capture-active flag, a reserved slot range. This asks
the engine instead.
"""

import functools
import logging
import threading

logger = logging.getLogger("atom")

_local = threading.local()


def in_dummy_run() -> bool:
    """Whether this thread is inside one of vLLM's synthetic forwards."""
    return bool(getattr(_local, "active", False))


def apply_vllm_dummy_run_patch() -> int:
    """Mark the window `_dummy_run` spans. Returns how many classes it covers.

    The count is returned, and logged by the caller, because a patch that
    installs on a class nobody instantiates reads exactly like one that works:
    three ATOM patches were silently covering nothing for that reason.
    """
    from atom.plugin.vllm.gpu_model_runner_targets import gpu_model_runner_classes

    covered = 0
    for runner_cls in gpu_model_runner_classes():
        original = getattr(runner_cls, "_dummy_run", None)
        if original is None or getattr(original, "_atom_dummy_run_patched", False):
            continue

        def make_wrapper(original):
            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                previous = getattr(_local, "active", False)
                _local.active = True
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _local.active = previous

            wrapped._atom_dummy_run_patched = True
            return wrapped

        runner_cls._dummy_run = make_wrapper(original)
        covered += 1
    return covered
