# SPDX-License-Identifier: MIT
"""Facts about the run that only the frontend driving it knows.

DeepSeek-V4.1 asks two of them -- may a CUDA-graph capture end in the middle
of a forward, and must the stale-state verdict be rendered in the step that
asks for it. Both are properties of whoever is driving the model, not of the
model, and the answers differ between ATOM's own engine and the vLLM plugin.

They are pushed in rather than pulled out. `atom/model_ops/` reaching into
`atom/plugin/vllm/` for them is the shape this replaces: it points the
dependency from the shared half at one particular frontend, so the shared half
only builds where that frontend is importable, and a second frontend wanting a
different answer has to be added to the same `if`. Here core owns the
question and its default, and a frontend states its own answer.

The defaults are ATOM's native behaviour, deliberately. This exists to remove
an import, not to change what anything computes, so a frontend that says
nothing gets exactly what it got before.

`capture_breaks_mid_forward` holds a callable, not a value: it asks whether a
capture is recording *right now*, which changes within a single run.
"""

from __future__ import annotations

import os
from collections.abc import Callable

_capture_breaks_probe: Callable[[], bool] | None = None
_breakable_capture_probe: Callable[[], bool] | None = None
_state_probe_blocking: bool = False


def set_execution_policy(
    *,
    capture_breaks_mid_forward: Callable[[], bool] | None = None,
    breakable_capture_enabled: Callable[[], bool] | None = None,
    state_probe_blocking: bool | None = None,
) -> None:
    """State this frontend's answers. Idempotent; omitted ones are left alone."""
    global _capture_breaks_probe, _breakable_capture_probe, _state_probe_blocking

    if capture_breaks_mid_forward is not None:
        _capture_breaks_probe = capture_breaks_mid_forward
    if breakable_capture_enabled is not None:
        _breakable_capture_probe = breakable_capture_enabled
    if state_probe_blocking is not None:
        _state_probe_blocking = bool(state_probe_blocking)


def capture_breaks_mid_forward() -> bool:
    """Whether the frontend may end a capture segment inside one forward.

    A side stream forked earlier in the forward and joined later is still
    outstanding at such a break, and `capture_end()` refuses it:

        HIP error: capturing stream has unjoined work
        (hipErrorStreamCaptureUnjoined)

    So a fork whose join is not in the same segment must not be opened while
    this is true. False by default: ATOM's native engine records one graph per
    forward and never breaks it.
    """
    return bool(_capture_breaks_probe()) if _capture_breaks_probe else False


def breakable_capture_enabled() -> bool:
    """Whether this run may capture graphs around the model's step work at all.

    Asked before a capture starts, where `capture_breaks_mid_forward` asks
    whether one is recording at this instant. A builder reads it to decide
    whether it must keep one dummy cache alive across buckets, so it has to
    answer for the whole run rather than for this moment.
    """
    return bool(_breakable_capture_probe()) if _breakable_capture_probe else False


def state_probe_blocking() -> bool:
    """Whether to render the stale-slot verdict in the step that asks for it.

    `prepare_state` can ship the cursor rows asynchronously and judge them a
    step later, which takes one blocking D2H (~58 us) off each decode step.
    The deferred path is wrong under concurrency, and wrong in the direction
    that costs most: it refuses correct state and takes the engine down.

    Measured on the vLLM plugin, FULL_AND_PIECEWISE, 64 concurrent GSM8K
    requests, no KV connector of any kind:

        needs state at 952, found 951

    while the state-row dump for that same request, printed beside it, reads
    `computed 953, cursor 953` -- self-consistent. The verdict is rendered
    against a claim from one step and a `_probe` buffer that later steps have
    already reused, so the two describe different moments.

    This is a guard, not a fix. The register slip in the deferred probe is
    still not understood, and a check that fails closed on correct state is
    worse than the microseconds it saves.

    `ATOM_V41_BLOCKING_STATE_PROBE` overrides either way.
    """
    override = os.environ.get("ATOM_V41_BLOCKING_STATE_PROBE")
    if override is not None and override != "":
        return override not in ("0", "false", "False")
    return _state_probe_blocking
