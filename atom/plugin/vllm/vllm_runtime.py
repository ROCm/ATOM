# SPDX-License-Identifier: MIT
"""Three facts about the vLLM a worker is running inside.

Every one of them is a question ATOM code has to ask the frontend rather than
answer for itself: which runner class this deployment selected, whether a CUDA
graph capture may end mid-forward, and whether the batch arriving now is one
vLLM scheduled or one it made up. None is about any particular model -- V4 and
V4.1 both ask all three -- and each is one or two functions, so they live
together here rather than in a file apiece.

What they share beyond that is how they fail. All three replace an inference
that read correctly right up to the deployment where it did not: a patch
assumed to be applied because its name resolved, a fork assumed safe because
the capture usually did not break, a synthetic batch inferred from its rows.
Inert and applied look the same in a log, so each of these asks the engine.
"""

import functools
import logging
import threading

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# Which GPU model runner classes exist
# ---------------------------------------------------------------------------

_RUNNER_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
)


def gpu_model_runner_classes() -> list[type]:
    """Every GPU model runner class a worker might instantiate, newest last.

    vLLM ships two unrelated `GPUModelRunner` classes -- the original in
    `vllm.v1.worker.gpu_model_runner` and the V2 rewrite in
    `vllm.v1.worker.gpu.model_runner` -- and `GPUWorker` picks between them on
    `use_v2_model_runner`. They do not share a base class and do not share
    method objects, so patching one leaves the other untouched. A patch
    written against the first name alone is therefore not "applied unless vLLM
    moved it": it is silently inert for every deployment that selects the
    other, and inert is indistinguishable from applied in the logs.

    Measured on 0.31.1 with the V2 runner selected: `issubclass` is False in
    both directions and `A.initialize_kv_cache is B.initialize_kv_cache` is
    False.

    Missing modules are skipped rather than raised on: which names exist is a
    property of the installed version, and a patch that must cover both should
    not refuse to cover one.
    """
    classes: list[type] = []
    for module_name in _RUNNER_MODULES:
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
            cls = module.GPUModelRunner
        except (ImportError, AttributeError):
            continue
        if cls not in classes:
            classes.append(cls)
    if not classes:
        logger.debug("ATOM plugin: no vLLM GPUModelRunner class found to patch")
    return classes


# ---------------------------------------------------------------------------
# Whether a capture may break mid-forward
# ---------------------------------------------------------------------------


def graphs_enabled(vllm_config) -> bool:
    """Whether this deployment will capture anything at all.

    Distinct from `breakable_capture_enabled`, which answers whether breaks
    are available if a capture happens. vLLM turns breakable capture on by
    itself for some architectures, so the two disagree exactly where it
    matters: an eager deployment, where reading the breakable flag says
    "graphs" and every workaround a capture needs gets installed anyway.
    An eager run then pays for them -- and an eager/graph comparison measures
    them on both arms.
    """
    if getattr(getattr(vllm_config, "model_config", None), "enforce_eager", False):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    from vllm.config import CUDAGraphMode

    return getattr(compilation_config, "cudagraph_mode", None) != CUDAGraphMode.NONE


def breakable_capture_enabled() -> bool:
    """Whether this run may capture graphs around a model's step work at all.

    Asked before a capture starts, not during one: a builder reads it to
    decide whether it has to keep one dummy cache alive across buckets, so it
    has to answer for the whole run rather than for this instant. The
    instantaneous question had its own function here and never acquired a
    caller; `BreakableCUDAGraphCapture.is_active()` answers it directly.
    """
    try:
        from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    except ImportError:
        return False
    return bool(is_breakable_cudagraph_enabled())


# ---------------------------------------------------------------------------
# Whether this batch is one vLLM made up
# ---------------------------------------------------------------------------

_dummy_local = threading.local()


def in_dummy_run() -> bool:
    """Whether this thread is inside one of vLLM's synthetic forwards.

    A model whose step carries per-request state has to tell a synthetic batch
    from a scheduled one: a synthetic batch's rows are not requests, so they
    must not take the STATE slots real requests hold, and their cursors mean
    nothing.

    vLLM announces it twice, and only one of the two is general.
    `build_for_cudagraph_capture` is called in place of `build`, which is exact
    -- but only for a FULL graph; a PIECEWISE capture and the warmup forwards
    come through `build` like any other step. What covers all of them is the
    method vLLM enters to run one: `_dummy_run`.

    Inferring it from the batch was tried four times and each inference was
    wrong in a way that only appeared under load -- a repeated placeholder
    request id (which stops repeating as soon as the request-id pass-through
    works), a startup-only window, a capture-active flag, a reserved slot
    range. This asks the engine instead.
    """
    return bool(getattr(_dummy_local, "active", False))


def apply_vllm_dummy_run_patch() -> int:
    """Mark the window `_dummy_run` spans. Returns how many classes it covers.

    The count is returned, and logged by the caller, because a patch that
    installs on a class nobody instantiates reads exactly like one that works:
    three ATOM patches were silently covering nothing for that reason.

    It counts classes that are covered, not classes this call installed on.
    The two differ because this is deliberately installed from two places
    (`register_platform` can be swallowed whole), so counting installs made
    the second call -- the normal one -- report `covers 0`, which is the exact
    reading the count exists to distinguish.
    """
    covered = 0
    for runner_cls in gpu_model_runner_classes():
        original = getattr(runner_cls, "_dummy_run", None)
        if original is None:
            continue
        if getattr(original, "_atom_dummy_run_patched", False):
            covered += 1
            continue

        def make_wrapper(original):
            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                previous = getattr(_dummy_local, "active", False)
                _dummy_local.active = True
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _dummy_local.active = previous

            wrapped._atom_dummy_run_patched = True
            return wrapped

        runner_cls._dummy_run = make_wrapper(original)
        covered += 1
    return covered
