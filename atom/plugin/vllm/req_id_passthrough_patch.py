"""Expose the current step's request ids (CPU, batch-ordered) to ATOM builders.

The V4 proxy metadata build needs a per-request key for its state slot. Deriving
one from ``block_table_tensor[:, 0]`` costs a ``.cpu()`` sync per step;
``input_batch.req_ids`` is the same key, already host-resident and already
reordered with every per-request tensor row (``InputBatch.swap_states``).

Each wrapper below snapshots it into a thread-local for the duration of one
call, read back via ``get_current_req_ids()``. Both of vLLM's two unrelated
``GPUModelRunner`` classes need covering: ``use_v2_model_runner`` forces V2 for
dspark, so a V1-only patch would be a silent no-op for every DSpark run.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")

_req_id_local = threading.local()


def get_current_req_ids() -> list[str] | None:
    """Return the current step's batch-ordered request ids, or None.

    Valid only inside an attention metadata builder's ``build()`` for either the
    target or the draft: on V1 that means ``_build_attention_metadata`` or
    ``propose_draft_token_ids`` is on the stack; on V2, ``execute_model`` after
    ``prepare_inputs`` has run, or ``sample_tokens``. Returns the empty list for
    a batch with no real requests (V2 dummy runs and cudagraph capture), and None
    outside any of those scopes or if the pass-through patch was not applied --
    callers must treat None as "fall back to the device-side key".
    """
    return getattr(_req_id_local, "req_ids", None)


def _wrap_with_req_id_snapshot(cls, method_name: str) -> bool:
    """Wrap ``cls.method_name`` to expose batch-ordered req_ids as a thread-local.

    The wrapped method snapshots ``self.input_batch.req_ids`` for the duration of
    the call so ATOM metadata builders invoked transitively can read it via
    ``get_current_req_ids()`` with no device sync. Idempotent.
    """
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        try:
            # Snapshot now: req_ids is already batch-reordered (swap_states ran
            # in _prepare_inputs) so it aligns with the per-request rows the
            # builder sees -- for both the target build and the draft proposal,
            # which reuses this same ordering. A copy keeps it stable even if the
            # batch mutates later in the step.
            _req_id_local.req_ids = list(self.input_batch.req_ids)
        except Exception:
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def _wrap_runner_scope(cls, method_name: str) -> bool:
    """Open a req_id scope for the duration of ``cls.method_name`` (V2 runner).

    The batch order is not known at entry (``prepare_inputs`` computes it), so
    this only *scopes* the thread-local, publishing ``[]`` in and restoring out;
    ``_wrap_v2_prepare_inputs`` fills in the real order. That keeps dummy /
    capture runs, which never call ``prepare_inputs``, from reading the previous
    real step's ids.

    ``[]`` not None: to the V4 builder None means "patch not installed" (a
    fail-fast contract violation) while ``[]`` means "no real requests".
    """
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        _req_id_local.req_ids = []
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def _wrap_v2_prepare_inputs(cls) -> bool:
    """Publish the batch order as soon as the V2 runner computes it.

    The returned ``InputBatch.req_ids`` line up with row ``i`` of every
    per-request tensor, as on V1. Every reader runs after this point inside the
    same ``execute_model`` call, whose wrapper bounds the scope.
    """
    original = getattr(cls, "prepare_inputs", None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        input_batch = original(self, *args, **kwargs)
        try:
            _req_id_local.req_ids = list(input_batch.req_ids)
        except Exception:  # noqa: BLE001 - unknown order, fall back to None
            _req_id_local.req_ids = None
        return input_batch

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    cls.prepare_inputs = wrapped
    return True


def _wrap_v2_sample_tokens(cls) -> bool:
    """Re-publish the batch order for the draft proposal (V2 runner).

    ``sample_tokens`` is a separate call from ``execute_model``, so outside its
    scope, and it drives ``speculator.propose()`` (V2's
    ``propose_draft_token_ids``), which builds metadata through the V4 bridge.
    """
    original = getattr(cls, "sample_tokens", None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        try:
            state = getattr(self, "execute_model_state", None)
            _req_id_local.req_ids = (
                list(state.input_batch.req_ids) if state is not None else []
            )
        except Exception:  # noqa: BLE001 - unknown order, fall back to None
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    cls.sample_tokens = wrapped
    return True


def _apply_v2_patch() -> bool:
    """Patch the V2 runner (``vllm.v1.worker.gpu.model_runner``), if present."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    except ImportError as e:  # pragma: no cover - vLLM version guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: V2 GPUModelRunner unavailable "
            "(%s), skip",
            e,
        )
        return False

    # capture_model builds metadata from prepare_inputs_to_capture -- no
    # execute_model, no prepare_inputs -- and for PIECEWISE through the plain
    # `build()` (for_capture=False), so the builder cannot tell it is a capture
    # from its arguments alone. Scope it.
    scoped = _wrap_runner_scope(GPUModelRunnerV2, "execute_model")
    scoped_capture = _wrap_runner_scope(GPUModelRunnerV2, "capture_model")
    scoped_dummy = _wrap_runner_scope(GPUModelRunnerV2, "_dummy_run")
    published = _wrap_v2_prepare_inputs(GPUModelRunnerV2)
    drafted = _wrap_v2_sample_tokens(GPUModelRunnerV2)

    patched = scoped or scoped_capture or scoped_dummy or published or drafted
    if patched:
        logger.info(
            "ATOM plugin: patched vLLM V2 GPUModelRunner (execute_model=%s, "
            "capture_model=%s, _dummy_run=%s, prepare_inputs=%s, sample_tokens=%s) "
            "to expose batch-ordered req_ids to ATOM metadata builders",
            scoped,
            scoped_capture,
            scoped_dummy,
            published,
            drafted,
        )
    return patched


def apply_vllm_req_id_passthrough_patch() -> bool:
    patched_v2 = _apply_v2_patch()

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:  # pragma: no cover - vLLM version guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: GPUModelRunner unavailable (%s), "
            "skip",
            e,
        )
        return patched_v2

    # Target attention metadata build.
    patched_target = _wrap_with_req_id_snapshot(
        GPUModelRunner, "_build_attention_metadata"
    )
    # MTP/Eagle draft proposal: the drafter builds its own attention metadata
    # (through the ATOM V4 bridge) here, outside _build_attention_metadata.
    patched_draft = _wrap_with_req_id_snapshot(
        GPUModelRunner, "propose_draft_token_ids"
    )
    # V1's cudagraph memory profiling reaches _dummy_run without builder
    # metadata, so the bridge takes its inline fallback, reads None from
    # get_current_req_ids() -- "patch not installed" -- and fails fast on a batch
    # with no real requests. Scope it to [] as on V2. Hit by mtp and friends.
    patched_dummy = _wrap_runner_scope(GPUModelRunner, "_dummy_run")

    if patched_target or patched_draft or patched_dummy:
        logger.info(
            "ATOM plugin: patched vLLM GPUModelRunner "
            "(_build_attention_metadata=%s, propose_draft_token_ids=%s, "
            "_dummy_run=%s) to expose "
            "batch-ordered req_ids to ATOM metadata builders (removes the "
            "block-table D2H in DeepSeek-V4 slot assignment; covers the MTP draft "
            "path)",
            patched_target,
            patched_draft,
            patched_dummy,
        )
    return patched_target or patched_draft or patched_dummy or patched_v2
