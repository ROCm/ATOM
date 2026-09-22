"""Expose the current step's request ids (CPU, batch-ordered) to ATOM builders.

The DeepSeek-V4 proxy metadata build needs a stable per-request key to assign a
state slot (its SWA ring + compressor state). Previously it derived that key
from ``block_table_tensor[:, 0]`` with a ``.cpu()`` copy, which forces a host<->
device sync and leaves a large bubble on the decode stream even though the copy
itself is tiny.

vLLM already has the canonical, host-resident key: ``input_batch.req_ids``. By
the time attention metadata is built it has been reordered together with the
block table / seq_lens rows (``InputBatch.swap_states``), so ``req_ids[i]``
lines up with row ``i`` of every per-request tensor.

This patch wraps two GPUModelRunner methods to snapshot ``req_ids`` into a
thread-local for the duration of each call:

* ``_build_attention_metadata`` -- constructs ``CommonAttentionMetadata`` *and*
  drives the target ``builder.build()`` in one synchronous call.
* ``propose_draft_token_ids`` -- drives the MTP/Eagle drafter, which (in current
  vLLM) builds its *own* attention metadata via
  ``SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata`` ->
  ``build_for_drafting`` -> the ATOM V4 bridge, entirely outside
  ``_build_attention_metadata``. Without this second wrap the thread-local is
  unset during drafting and the V4 slot allocator's fail-fast contract trips.

The drafter reuses the target step's ``input_batch`` ordering (pure decodes were
already pulled to the front and the batch is not re-reordered before the draft
forward), so ``req_ids[i]`` still aligns with row ``i`` of the draft metadata --
the same invariant the target build relies on. ATOM's V4 metadata builder reads
the snapshot via ``get_current_req_ids()`` and keys slot allocation on it, with
no D2H. All of this lives in ATOM; no vLLM source is modified.

The same wrappers expose the ``InputBatch`` itself through
``get_current_input_batch()``. DeepSeek-V4.1 hands vLLM's block table straight
to its own PAGE table (its PAGE size and vLLM's block size are both 256, so the
ids are the same ids) and needs the host mirror of it, plus
``num_computed_tokens_cpu`` -- neither of which ``CommonAttentionMetadata``
carries without a device sync.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")

_req_id_local = threading.local()


def get_current_input_batch():
    """Return the current step's vLLM ``InputBatch``, or None.

    Valid over the same window as :func:`get_current_req_ids` -- the batch is
    the object ``req_ids`` was snapshotted from, handed over live rather than
    copied. DeepSeek-V4.1 needs more of it than the ids: its PAGE table is the
    request's own, so the builder reads the host block-table mirror
    (``input_batch.block_table[group].block_table.np``) and
    ``num_computed_tokens_cpu`` from here instead of paying the D2H that
    ``CommonAttentionMetadata``'s deprecated CPU properties force.

    Callers must treat None as "fall back to the device-side tensors", and must
    not hold the reference past the call that exposed it.
    """
    return getattr(_req_id_local, "input_batch", None)


def get_current_req_ids() -> list[str] | None:
    """Return the current step's batch-ordered request ids, or None.

    Valid only while ``GPUModelRunner._build_attention_metadata`` (target) or
    ``GPUModelRunner.propose_draft_token_ids`` (MTP/Eagle draft) is on the stack
    -- i.e. inside an attention metadata builder's ``build()`` for either the
    target or the draft. Returns None otherwise, or if the pass-through patch was
    not applied -- callers must treat None as "fall back to the device-side key".
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
        prev_batch = getattr(_req_id_local, "input_batch", None)
        _req_id_local.input_batch = getattr(self, "input_batch", None)
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
            _req_id_local.input_batch = prev_batch

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)
    return True


def apply_vllm_req_id_passthrough_patch() -> bool:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: GPUModelRunner unavailable (%s), "
            "skip",
            e,
        )
        return False

    # Target attention metadata build.
    patched_target = _wrap_with_req_id_snapshot(
        GPUModelRunner, "_build_attention_metadata"
    )
    # MTP/Eagle draft proposal: the drafter builds its own attention metadata
    # (through the ATOM V4 bridge) here, outside _build_attention_metadata.
    patched_draft = _wrap_with_req_id_snapshot(
        GPUModelRunner, "propose_draft_token_ids"
    )

    if patched_target or patched_draft:
        logger.info(
            "ATOM plugin: patched vLLM GPUModelRunner "
            "(_build_attention_metadata=%s, propose_draft_token_ids=%s) to expose "
            "batch-ordered req_ids to ATOM metadata builders (removes the "
            "block-table D2H in DeepSeek-V4 slot assignment; covers the MTP draft "
            "path)",
            patched_target,
            patched_draft,
        )
    return patched_target or patched_draft
