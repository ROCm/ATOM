# SPDX-License-Identifier: MIT
"""Facts about the run that only the frontend driving it knows.

DeepSeek-V4.1 asks one of them: must the stale-state verdict be rendered in
the step that asks for it. It is a property of whoever is driving the model,
not of the model, and ATOM's own engine and the vLLM plugin answer
differently.

They are pushed in rather than pulled out. `atom/model_ops/` reaching into
`atom/plugin/vllm/` for them is the shape this replaces: it points the
dependency from the shared half at one particular frontend, so the shared half
only builds where that frontend is importable, and a second frontend wanting a
different answer has to be added to the same `if`. Here core owns the
question and its default, and a frontend states its own answer.

Only one question is left here. The other two this started with -- whether a
capture may break mid-forward, and whether graphs may be captured at all --
turned out to have a single reader each, in a class a frontend can subclass:
`EngramStaging._forks` and `DeepseekV41MetadataBuilder._dummy_cache`. An
override says the same thing without a flag to keep in sync, so they moved
there.

The defaults are ATOM's native behaviour, deliberately. This exists to remove
an import, not to change what anything computes, so a frontend that says
nothing gets exactly what it got before.
"""

from __future__ import annotations

import os

_state_probe_blocking: bool = False


def set_execution_policy(
    *,
    state_probe_blocking: bool | None = None,
) -> None:
    """State this frontend's answers. Idempotent; omitted ones are left alone."""
    global _state_probe_blocking

    if state_probe_blocking is not None:
        _state_probe_blocking = bool(state_probe_blocking)


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
