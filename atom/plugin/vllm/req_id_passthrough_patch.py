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
``get_current_input_batch()``, for two readers that want different things from
it. DeepSeek-V4's state-checkpoint ops read the host mirror of the state
checkpoint groups' block tables, which ``CommonAttentionMetadata`` does not
carry without a device sync.

What DeepSeek-V4.1 takes from it is the
persistent, host-resident request mapping that ``CommonAttentionMetadata`` does
not carry -- how many requests this step actually scheduled, their batch order,
and the input ids -- which is what its STATE slots are keyed on.

Not a host mirror of the PAGE table, and not a way to avoid that sync: the
V2 runner's ``InputBatch`` has no full host block table, and its host sequence
lengths can be optimistic under async speculation, so V4.1 still reads the
exact lengths and PAGE rows with blocking copies (``snapshot_v41_batch``).
The checkpoint groups' rows, read per element, come from the host
(``install_v2_host_block_rows``).
Removing the D2H is what this patch does for V4's slot assignment, above; it is
not what V4.1 uses it for.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")

_req_id_local = threading.local()


def get_current_cudagraph_mode():
    """The mode `prepare_attn` was called for, or None outside it."""
    return getattr(_req_id_local, "cudagraph_mode", None)


class _HostMirror:
    __slots__ = ("np",)

    def __init__(self, array):
        self.np = array


class _HostRows:
    """The V2 runner's block-table rows on the host, by request id.

    V2 keeps its tables on the device only, so reading one row on the host
    cost a blocking copy of the pool-wide table on every step with a
    checkpoint store or restore -- most decode steps, once any row's keep can
    cross a boundary. The rows are rebuilt instead from the scheduler output
    the runner applies, by the runner's own rules (`finish_requests`,
    `add_requests`, `update_requests`): finished and preempted requests are
    dropped, new requests and `block_table_updates` overwrite, cached requests
    append. A resumed request is overwritten too, as V1 does: vLLM 0.31 sends
    it to V2 as a new request, so this only matters if that changes.
    """

    def __init__(self) -> None:
        self.rows: dict[str, list[list[int]]] = {}

    @staticmethod
    def _expand(block_ids, per_kv_block: int) -> list[int]:
        if per_kv_block > 1:
            return [
                b * per_kv_block + k for b in block_ids for k in range(per_kv_block)
            ]
        return list(block_ids)

    def apply(self, scheduler_output, blocks_per_kv_block) -> None:
        def expand(groups):
            return [
                self._expand(ids, blocks_per_kv_block[g] if blocks_per_kv_block else 1)
                for g, ids in enumerate(groups)
            ]

        rows = self.rows
        gone = set(getattr(scheduler_output, "finished_req_ids", None) or ())
        gone |= set(getattr(scheduler_output, "preempted_req_ids", None) or ())
        for req_id in gone:
            rows.pop(req_id, None)
        for new in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
            rows[new.req_id] = expand(new.block_ids)
        updates = getattr(scheduler_output, "block_table_updates", None) or {}
        for req_id, block_ids in updates.items():
            if req_id in rows:
                rows[req_id] = expand(block_ids)
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None:
            return
        resumed = set(getattr(cached, "resumed_req_ids", None) or ())
        for req_id, new_block_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_block_ids is None or req_id in updates:
                continue
            if req_id in resumed:
                # Its blocks were freed at preemption: the ids are the whole row.
                rows[req_id] = expand(new_block_ids)
                continue
            row = rows.get(req_id)
            if row is None:
                continue
            for group, ids in zip(row, expand(new_block_ids)):
                group.extend(ids)


class _HostRowIndex:
    """`[row, col]` of one group's table, read from `_HostRows`.

    Rows are batch rows (`req_ids` order); past a row's end reads 0, as the
    gathered device table does.
    """

    __slots__ = ("_group", "_req_ids", "_rows")

    def __init__(self, rows, req_ids, group: int) -> None:
        self._rows = rows
        self._req_ids = req_ids
        self._group = group

    def __getitem__(self, key) -> int:
        row, col = key
        groups = self._rows.get(self._req_ids[row])
        if groups is None:
            return 0
        ids = groups[self._group]
        return ids[col] if 0 <= col < len(ids) else 0


#: The rows `install_v2_host_block_rows` keeps current.
_HOST_ROWS_KEPT = _HostRows()
#: `_HOST_ROWS_KEPT` once a runner class keeps it current; None leaves V2 reads
#: on the device copy, which is right rather than all zeros.
_V2_HOST_ROWS: _HostRows | None = None


#: Which runners keep `_HOST_ROWS_KEPT`; set by `install_v2_host_block_rows`.
_HOST_ROWS_WANTED = None


def install_v2_host_block_rows(wanted) -> int:
    """Keep `_HostRows` current on every runner class that has V2's
    `update_requests`, and read from it only once one does. Returns how many
    classes this call patched.

    `wanted(runner)` names the runners whose model reads the rows; it is
    asked once per runner, and every other runner's step skips the
    bookkeeping. Installed from the plugin's entry point, which runs for
    every model.
    """
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    global _V2_HOST_ROWS, _HOST_ROWS_WANTED
    _HOST_ROWS_WANTED = wanted
    host_rows = _HOST_ROWS_KEPT
    classes = gpu_model_runner_classes()
    patched = 0
    for cls in classes:
        original = getattr(cls, "update_requests", None)
        if original is None or getattr(original, "_atom_v2_host_rows", False):
            continue

        @functools.wraps(original)
        def update_requests(self, scheduler_output, *args, _orig=original, **kwargs):
            keeps = self.__dict__.get("_atom_keeps_host_rows")
            if keeps is None:
                keeps = bool(_HOST_ROWS_WANTED is not None and _HOST_ROWS_WANTED(self))
                self._atom_keeps_host_rows = keeps
            if keeps:
                tables = getattr(self, "block_tables", None)
                host_rows.apply(
                    scheduler_output, getattr(tables, "blocks_per_kv_block", None)
                )
            return _orig(self, scheduler_output, *args, **kwargs)

        update_requests._atom_v2_host_rows = True
        cls.update_requests = update_requests
        patched += 1
    if any(
        getattr(getattr(cls, "update_requests", None), "_atom_v2_host_rows", False)
        for cls in classes
    ):
        _V2_HOST_ROWS = host_rows
    return patched


class _BlockTableGroupView:
    """`batch.block_table[g]` for a runner that keeps no host mirror."""

    __slots__ = ("block_table",)

    def __init__(self, host_mirror):
        self.block_table = host_mirror


class _V2InputBatchView:
    """The V2 ``InputBatch`` plus the ``block_table`` the V1 one carries.

    The V2 runner keeps its block tables outside the batch, as
    ``StagedWriteTensor``s that have only a `.gpu` side -- there is no host
    mirror to read, so `batch.block_table[g].block_table.np` (what the shared
    checkpoint mechanism uses to turn a (row, column) into a block id) has nothing
    to resolve against. With `install_v2_host_block_rows` it reads the rows
    kept on the host from the scheduler output (`_HostRows`); without it, the
    group's device table is copied down once per step, on first use -- a sync
    of the whole pool-wide table. Readers of every other attribute get the
    real batch.

    Either way `.np` is not the trimmed host array `snapshot_v41_batch` reads
    every step, so it takes its own (`copies_whole_table`).
    """

    __slots__ = ("_batch", "_cache", "_computed", "_tables")

    #: `block_table` is not a host array of the step's rows (see above).
    copies_whole_table = True

    def __init__(self, batch, tables):
        self._batch = batch
        self._tables = tables
        self._cache = {}
        self._computed = None

    def __getattr__(self, name):
        return getattr(self._batch, name)

    #: This runner carries no second, independently ordered count to check
    #: the step's row order against -- see `num_computed_tokens_cpu`.
    atom_row_order_unverifiable = True

    @property
    def num_computed_tokens_cpu(self):
        """V1's name for the per-row computed counts; V2 calls it `_np`.

        Batch-row order, NOT the block table's order. On V2 this is the
        batch's own count and is NOT independent of the step: measured, it
        disagrees with `seq_lens - query_lens` on some batches (13 vs 9) and
        agrees on others, and no field of the V2 batch reproduces the step's
        number (`num_draft_tokens_per_req` corrects one case and breaks the
        other). V4.1 therefore takes its decode-keep starts from the step's own
        `seq_lens - query_lens` (`snapshot_v41_batch`), not from here; the
        class declares `atom_row_order_unverifiable`, and the row-order
        cross-check skips rather than fire on a difference that says nothing
        about row order. The two index spaces are
        the same object on V1's persistent batch and different on V2, so this
        one is left in row order deliberately: its readers are row-order
        readers -- `_check_row_alignment` compares it against the step's own
        per-row counts (and says so loudly if they diverge), and
        `_exact_starts` pairs it with `req_ids`. Scattering it into slot order
        to match the block table made that check fail at row 0 immediately.
        """
        if self._computed is None:
            import numpy as np

            batch = self._batch
            n = int(batch.num_reqs)
            out = np.asarray(batch.num_computed_tokens_np[:n], dtype=np.int64)
            self._computed = out
        return self._computed

    @property
    def block_table(self):
        return self

    def __getitem__(self, group_id):
        view = self._cache.get(group_id)
        if view is None and _V2_HOST_ROWS is not None:
            view = _BlockTableGroupView(
                _HostMirror(
                    _HostRowIndex(_V2_HOST_ROWS.rows, self._batch.req_ids, group_id)
                )
            )
            self._cache[group_id] = view
        if view is None:
            table = self._tables[group_id]
            # Rows are already batch rows: `gather_block_tables` builds the
            # table in batch order, the same order the callers index.
            #
            # Gathering it again with `idx_mapping_np` permutes it twice, which
            # is what was measured: a concurrent restore asked for (row 4,
            # col 2) while the blocks sat at (row 6, col 2), and the mechanism
            # refused every such hit with null unit ids.
            view = _BlockTableGroupView(_HostMirror(table.cpu().numpy()))
            self._cache[group_id] = view
        return view


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
    batch = getattr(_req_id_local, "input_batch", None)
    if batch is None or hasattr(batch, "block_table"):
        return batch
    tables = getattr(_req_id_local, "block_tables", None)
    if tables is None:
        return batch
    view = getattr(_req_id_local, "input_batch_view", None)
    if view is None or view._batch is not batch or view._tables is not tables:
        view = _V2InputBatchView(batch, tables)
        _req_id_local.input_batch_view = view
    return view


def get_current_req_ids() -> list[str] | None:
    """Return the current step's batch-ordered request ids, or None.

    Valid only while ``GPUModelRunner._build_attention_metadata`` (target) or
    ``GPUModelRunner.propose_draft_token_ids`` (MTP/Eagle draft) is on the stack
    -- i.e. inside an attention metadata builder's ``build()`` for either the
    target or the draft. Returns None otherwise, or if the pass-through patch was
    not applied -- callers must treat None as "fall back to the device-side key".
    """
    return getattr(_req_id_local, "req_ids", None)


def _wrap_with_req_id_snapshot(cls, method_name: str, batch_from_arg=False) -> bool:
    """Wrap ``cls.method_name`` to expose batch-ordered req_ids as a thread-local.

    The wrapped method snapshots the step's ``InputBatch`` for the duration of
    the call so ATOM metadata builders invoked transitively can read it via
    ``get_current_req_ids()`` with no device sync. Idempotent.

    ``batch_from_arg`` says where the batch is. The V1 runner keeps it on the
    runner (``self.input_batch``); the V2 runner hands it to
    ``ModelState.prepare_attn`` as the first argument and keeps no such
    attribute, so reading ``self`` there yields None and the pass-through goes
    silently dead -- which is what it did, because this patch only ever
    targeted a method the V2 runner does not have.
    """
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        prev_batch = getattr(_req_id_local, "input_batch", None)
        batch = (
            args[0] if (batch_from_arg and args) else getattr(self, "input_batch", None)
        )
        _req_id_local.input_batch = batch
        # V2 `prepare_attn(input_batch, cudagraph_mode, block_tables, ...)`.
        # The V1 runner carries these inside the batch, so there is nothing to
        # take from the arguments there. Without them `get_current_input_batch`
        # hands back the raw V2 batch, which has no `block_table` at all, and
        # the first state-checkpoint restore dies on the attribute.
        prev_tables = getattr(_req_id_local, "block_tables", None)
        _req_id_local.block_tables = (
            args[2] if (batch_from_arg and len(args) > 2) else None
        )
        _req_id_local.input_batch_view = None
        prev_mode = getattr(_req_id_local, "cudagraph_mode", None)
        # `prepare_attn` is handed the mode it is preparing for. A model whose
        # step work runs out here needs to know which one, because a FULL
        # decode graph and a PIECEWISE one do not reach the forward the same
        # way.
        _req_id_local.cudagraph_mode = (
            args[1] if (batch_from_arg and len(args) > 1) else None
        )
        try:
            # Snapshot now: req_ids is already batch-reordered (swap_states ran
            # in _prepare_inputs) so it aligns with the per-request rows the
            # builder sees -- for both the target build and the draft proposal,
            # which reuses this same ordering. A copy keeps it stable even if the
            # batch mutates later in the step.
            _req_id_local.req_ids = list(batch.req_ids)
        except Exception:  # noqa: BLE001 - best effort; None is a valid answer
            # Deliberately blind. This is an optimisation -- callers read None
            # as "fall back to the device-side tensors" -- so no shape a future
            # vLLM gives `req_ids` should be able to take the engine down here.
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev
            _req_id_local.input_batch = prev_batch
            _req_id_local.block_tables = prev_tables
            _req_id_local.input_batch_view = None
            _req_id_local.cudagraph_mode = prev_mode

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

    # The V2 runner has no `_build_attention_metadata` at all: it prepares the
    # batch and calls the builders from `ModelState.prepare_attn`, which takes
    # the `InputBatch` as its first argument rather than keeping it on the
    # runner. Patched here too, and by the same measurement that found the
    # profile-cache patch on the wrong class -- a patch written against one
    # runner's names is not "applied unless vLLM moved it", it is silently
    # inert for every deployment that selects the other, and inert reads the
    # same as applied in the logs.
    patched_states = _patch_model_state_factory()

    # `_patch_model_state_factory` reports its outcome by name, and the names
    # for its failures -- "unavailable", "absent" -- are as truthy as the ones
    # for success. Taking `bool` of it therefore read every failure as an
    # install, and said so in the log: the one place someone checks.
    states_installed = patched_states in _STATE_FACTORY_INSTALLED
    if patched_target or patched_draft or states_installed:
        logger.info(
            "ATOM plugin: patched vLLM GPUModelRunner "
            "(_build_attention_metadata=%s, propose_draft_token_ids=%s) and model "
            "states %s to expose batch-ordered req_ids to ATOM metadata builders "
            "(removes the block-table D2H in DeepSeek-V4 slot assignment; covers "
            "the MTP draft path and the V2 runner)",
            patched_target,
            patched_draft,
            patched_states,
        )
    return bool(patched_target or patched_draft or states_installed)


# The outcomes of `_patch_model_state_factory` that leave the hook in place.
# "already" counts: an earlier call installed it.
_STATE_FACTORY_INSTALLED = ("factory", "already")


def _patch_model_state_factory() -> str:
    """Wrap `prepare_attn` on whatever model state the runner actually builds.

    Patching the classes this module can see at import time does not work, and
    the way it fails is silent: the wrapper installs on `DefaultModelState`,
    the log reports it, and it is never called, because the class the V2
    runner instantiates is resolved later -- `resolve_model_state_cls` even
    lets the model hand back a class of its own. Measured here: four classes
    reported as patched, zero hook entries.

    So the hook goes where the choice is made. `init_model_state` is wrapped
    in the model runner's own namespace, because that module imported the name
    at import time and would not see a change made on the defining module, and
    the state's class is wrapped on the way out -- once, whatever it is.
    """
    try:
        import vllm.v1.worker.gpu.model_runner as runner_mod
    except ImportError:
        return "unavailable"
    original = getattr(runner_mod, "init_model_state", None)
    if original is None:
        return "absent"
    if getattr(original, "_atom_req_id_passthrough_patched", False):
        return "already"

    @functools.wraps(original)
    def wrapped_init_model_state(*args, **kwargs):
        state = original(*args, **kwargs)
        if _wrap_with_req_id_snapshot(type(state), "prepare_attn", batch_from_arg=True):
            logger.info(
                "ATOM plugin: req_id pass-through now covers %s.prepare_attn",
                type(state).__name__,
            )
        return state

    wrapped_init_model_state._atom_req_id_passthrough_patched = True
    runner_mod.init_model_state = wrapped_init_model_state
    return "factory"
