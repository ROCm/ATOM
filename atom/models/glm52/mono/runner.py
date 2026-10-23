# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The runner: owns the per-(layer, width) GLM-5.2 mono kernel builds and runs
the forward pass over the layers mono owns.

``prepare`` is called by ``LazyRunner.ready`` once per step to make sure a
build for that step's row count exists before ``forward`` is dispatched; the
build itself (allocation, the TP peer handshake) cannot happen inside a CUDA
graph capture, which is why ``prepare`` refuses to build while capturing, and
why a width that was refused once must stay refused instead of retrying every
step -- the kernels wait in-kernel on every peer rank, so a rank that rebuilds
while its peers stay refused leaves them spinning on the GPU forever.
"""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.attention_mla import (
    triton_convert_req_index_to_global_index,
    triton_gather_kv_indices_sparse,
)
from atom.models.glm52.mono.config import (
    fused_index_capacity,
    fused_indexer_refusal,
    indexer_cp_refusal,
    native_config,
)
from atom.models.glm52.mono.contract import mono_layers
from atom.models.glm52.mono.contract import (
    shared_sparse_buffer as _shared_sparse_buffer,
)
from atom.models.glm52.mono.kernels.geometry_config import (
    GLM5_CONFIG,
    AttentionWeight,
    KvCacheLayout,
    glm5_kernel_samples,
)
from atom.models.glm52.mono.weights import bf16_vector, layer_weights
from atom.mono.runtime.consensus import MonoUnsupported, tp_uniform_local_validation
from atom.mono.weights.layer import LayerWeights
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
def _is_full_indexer_layer(layer) -> bool:
    """A layer that actually selects, as opposed to sharing a preceding full
    layer's selection -- the same predicate ``_refresh_indexer`` already uses
    to no-op on "shared" layers (``indexer is None`` there)."""

    attn = layer.self_attn
    return attn.indexer is not None and not attn.skip_topk


class _GlmLayerOp:
    """The build for any layer mono owns on the external-indexer (base) path.

    ``launches_per_step`` is how many times *this runtime* is launched in one
    decode step, and it is a kernel build constant, not a hint: the kernel
    double-buffers its peer mailboxes on
    ``peer_slot = (step * launches_per_step + layer) & 1``
    (``glm/kernel.py``). That alternates between consecutive launches only when
    ``layer`` runs ``0 .. launches_per_step - 1`` and ``step`` advances exactly
    once per step -- at the step boundary the last launch is
    ``s*N + (N-1)`` and the next is ``(s+1)*N + 0 = s*N + N``, consecutive
    integers, so the parity flips. Pass anything else (``LAYER_SLOTS``, say,
    with ``layer = layer_idx``) and two launches either side of a step boundary
    can land on the same slot, which is a silent peer-buffer overwrite or a
    four-rank spin, not an error. ``Glm52MonoRunner.forward`` keeps the counter
    and asserts the slots it consumed match this number.
    """

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        topk: int,
        kv_cache_dtype: str,
        prepared_weights,
        runtime,
        launches_per_step: int,
    ) -> None:
        from atom.models.glm52.mono.kernels.op import Glm5MonoKernel

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        self.op = Glm5MonoKernel(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=get_tp_group().cpu_group,
            topk=topk,
            launches_per_step=launches_per_step,
            with_indexer=False,
            attention_weight=AttentionWeight.FP8_PTPC,
            kv_cache_layout=KvCacheLayout.ATOM,
            kv_cache_dtype=kv_cache_dtype,
            prepared_weights=prepared_weights,
            runtime=runtime,
            native_fp4_mfma=True,
            # ``Glm5MonoKernel`` defaults to FP32 RoPE tables and validates the
            # dtype it is handed (``glm/op.py``); the tables ATOM passes to
            # ``forward`` are ``attn.rotary_emb.{cos,sin}_cache``, which are
            # BF16. ``Glm5TP4MonoKernel`` pins the same value itself and takes
            # no ``rope_dtype`` argument, so only this site sets it.
            rope_dtype="bf16",
        )

    def close(self) -> None:
        self.op.close()


class _GlmFusedLayerOp:
    """The build for a "full" layer when the fused TP4 indexer is on.

    Index Q/K/W projection, paged FP8 index-cache update, scoring and top-k
    selection all run inside this one launch (``kernels/{kernel,layout,op}.py``,
    vendored into this repo and maintained here since -- ``kernel.py``'s
    index-cache writes were changed in this PR; ``tp4_op.py`` is the thin
    subclass that pins the TP4 positional API). Never built
    unless ``envs.ATOM_GLM52_FUSED_INDEXER`` is on and
    ``atom.models.glm52.mono.config.fused_indexer_refusal`` returns None --
    see ``Glm52MonoRunner.__init__``. "Shared" layers always build
    ``_GlmLayerOp`` instead, regardless of this switch.

    Deliberately does NOT share ``runtime`` (scratch / peer buffer / step)
    with ``_GlmLayerOp``: ``with_indexer=True`` is a different kernel binary
    built from a different scratch layout (``layout(..., with_indexer=True)``
    vs the non-fused kernel's), and ``Glm5MonoKernel.__init__`` itself refuses
    a ``runtime`` whose ``scr_layout``/``sym_layout`` disagree with its own --
    cross-wiring the two would either raise there or, worse, pass by
    coincidence at one width and alias scratch at another. Fused layers of
    the same chunk width share a runtime only among themselves (see
    ``Glm52MonoRunner._fused_runtimes``).

    ``index_request_width`` and ``index_block_table_stride`` are passed
    explicitly, never left to default: ``Glm5TP4MonoKernel`` derives
    ``request = s // index_request_width`` (``glm/kernel.py``) to decide
    which request each row of the step belongs to, so an unpassed (defaulted)
    width is only coincidentally right for a batch of one request and wrong
    for any wider one -- it would collapse every request in the step into
    request 0, cross-contaminating their KV/index caches. ``index_block_table_stride``
    is the column count of ``block_tables`` the kernel indexes with; ATOM's
    real column count is ``max_model_len // kv_cache_block_size``, not the
    kernel's own ``index_max_seq // index_page_size`` default.

    Because it does not share a runtime, it does not share a ``step`` counter
    either, so it needs its OWN ``0 .. launches_per_step - 1`` slot numbering
    and its own ``launches_per_step`` -- see ``_GlmLayerOp``'s docstring for
    why the numbering has to be exactly that. With the fused indexer on, the
    18 full layers launch here and the 57 shared layers launch on the base
    runtime, so the two counts are 18 and 57, not 75 and 75.

    ``indexer_cp`` shards index scoring and selection across the TP ranks
    (``envs.ATOM_GLM52_INDEXER_CP``, default off, refusals in
    ``config.indexer_cp_refusal``). It is a compile-time build parameter, not a
    runtime mode: CP-on and CP-off are different kernel binaries with different
    scratch and symmetric layouts, so the value must be the same for every
    fused op sharing a ``runtime`` -- which it is, being read once per
    deployment in ``Glm52MonoRunner.__init__``.
    """

    def __init__(
        self,
        weights: LayerWeights,
        samples: int,
        topk: int,
        kv_cache_dtype: str,
        index_page_size: int,
        index_request_width: int,
        index_block_table_stride: int,
        indexer_cp: bool,
        index_max_seq: int,
        prepared_weights,
        runtime,
        launches_per_step: int,
    ) -> None:
        from atom.models.glm52.mono.kernels.tp4_op import Glm5TP4MonoKernel

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        self.op = Glm5TP4MonoKernel(
            weights,
            samples,
            rank=rank,
            npes=npes,
            group=get_tp_group().cpu_group,
            topk=topk,
            launches_per_step=launches_per_step,
            with_indexer=True,
            index_max_seq=index_max_seq,
            index_request_width=index_request_width,
            index_page_size=index_page_size,
            index_block_table_stride=index_block_table_stride,
            index_cache_dtype="fp8",
            attention_weight=AttentionWeight.FP8_PTPC,
            kv_cache_layout=KvCacheLayout.ATOM,
            kv_cache_dtype=kv_cache_dtype,
            prepared_weights=prepared_weights,
            runtime=runtime,
            native_fp4_mfma=True,
            indexer_cp=indexer_cp,
        )

    def close(self) -> None:
        self.op.close()


def _publish_selected_indices(
    op,
    shared_indices: torch.Tensor,
    sparse_indptr: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    samples: int,
    max_seqlen_q: int,
    batch_id_per_q_token: torch.Tensor | None,
) -> None:
    """Translate one fused layer's internal top-k selection into ATOM's
    shared ``sparse_kv_indices_buffer``, at the offsets ``sparse_kv_indptr``
    already reserves for each sample.

    Required so the first three (dense) layers and every "shared" mono layer
    -- which all read ``sparse_kv_indices_buffer``/``sparse_kv_indptr``
    directly, never the fused kernel's own scratch -- see the SAME selection
    a fused "full" layer made, rather than a stale or absent one. Two sources
    of indices diverging would silently produce wrong (non-crashing) output.

    This is not a copy, but a COORDINATE TRANSLATION. ``op.scr_layout
    ["indices"]`` (a dense ``(samples, topk)`` int32 scratch: real selected
    indices packed at the front of each row, an explicit zero sentinel beyond
    the valid count -- ``glm/kernel.py``'s index_select stage) holds LOGICAL
    per-request token positions: the kernel's own attention-gather branch
    translates these same values through ``index_page_cell``/
    ``index_page_slot`` before using them as a cache row (``glm/kernel.py``'s
    ``with_indexer`` branch). ``sparse_kv_indices_buffer`` is read by every
    OTHER consumer -- the dense layers, every "shared" mono layer, and this
    same kernel's own non-fused/ATOM-layout attention branch -- as an
    already-PHYSICAL cache row, with no translation of its own. Publishing
    the logical scratch here unmodified would make every one of those
    readers attend to the wrong (past the first request, another REQUEST's)
    KV: fluent, silently wrong output.

    ATOM's own producer (``deepseek_v2.py``'s ``sparse_attn_indexer``)
    branches on ``max_seqlen_q`` because ``sparse_kv_indptr`` itself changes
    indexing domain with it (``aiter_mla.py``): per-REQUEST
    (``running_bs + 1`` entries) at ``max_seqlen_q == 1`` -- this step's
    "virtual sequences each with q_len=1" framing -- and per-TOKEN
    (``running_tokens + 1`` entries) at any MTP width. ``cu_seqlens_q``/
    ``kv_indptr`` are always per-request. Mirror that branch exactly, not a
    generalization of either side alone:

    - ``max_seqlen_q == 1``: ``triton_convert_req_index_to_global_index``, a
      per-request GATHER through ``kv_indices`` (ATOM's flat,
      per-request-concatenated physical-row table) driven by
      ``cu_seqlens_q``/``kv_indptr``, matching ``sparse_kv_indptr``'s
      per-request shape at this width.
    - ``max_seqlen_q > 1`` (MTP): ``triton_gather_kv_indices_sparse``, a
      per-token GATHER driven by ``batch_id_per_q_token`` (published by
      decode metadata exactly when ``max_seqlen_q > 1``), matching
      ``sparse_kv_indptr``'s per-token shape at these widths. Calling the
      per-request converter here instead silently aliases every row of a
      multi-row request onto the same ``sparse_kv_indptr`` slice (last
      write wins) and misaligns every request after the first.

    Both are reused rather than reimplemented, so this also inherits their
    capture-safety: each kernel body is ``tl.load``/``tl.store`` under a
    fixed 2D grid, with no ``nonzero()`` or other data-dependent control
    flow -- unlike the boolean-mask ``index_copy_`` this replaces, which is
    unsound under CUDA graph replay.
    """

    off = op.scr_layout["indices"]
    topk = op.topk
    selected = (
        op.scratch[off : off + samples * topk * 4].view(torch.int32).view(samples, topk)
    )
    if max_seqlen_q > 1:
        triton_gather_kv_indices_sparse(
            sparse_indptr,
            batch_id_per_q_token,
            selected,
            kv_indices,
            kv_indptr,
            NUM_TOPK_TOKENS=topk,
            out=shared_indices,
        )
    else:
        triton_convert_req_index_to_global_index(
            cu_seqlens_q,
            kv_indptr,
            sparse_indptr,
            kv_indices,
            selected,
            NUM_TOPK_TOKENS=topk,
            out=shared_indices,
        )


class Glm52MonoRunner:
    """Owns the per-(layer, width) GLM-5.2 mono kernel builds and the forward."""

    def __init__(self, causal_lm, atom_config) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        spec = atom_config.speculative_config
        self._query_length = 1 if spec is None else spec.num_speculative_tokens + 1
        is_mtp = spec is not None and spec.method == "mtp"
        self._shard = native_config(
            samples=self._query_length if self._query_length > 1 else 4,
            tp_size=atom_config.tensor_parallel_size,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            mtp=is_mtp,
            query_length=self._query_length,
        )
        self._ops: dict[tuple[int, int], _GlmLayerOp] = {}
        # Full layers ALSO get an entry here when the fused switch is on,
        # alongside (not instead of) their `_ops` entry above: the fused and
        # non-fused builds for the same layer/width are separate kernel
        # binaries with separate scratch, and the base `_ops` entry stays the
        # deployment-wide fallback a failed fused build drops back to. A
        # layer with no entry here (every "shared" layer, and every "full"
        # layer when the switch is off) only ever runs its `_ops` entry.
        self._fused_ops: dict[tuple[int, int], _GlmFusedLayerOp] = {}
        self._weights: dict[int, LayerWeights] = {}
        self._prepared: dict[int, dict[str, torch.Tensor]] = {}
        # The fused (with_indexer=True) weight mapping, kept separate from
        # `_weights` above: a layer's base mapping must always succeed (it is
        # `_GlmLayerOp`'s only source), while its fused mapping is allowed to
        # fail and fall back -- see `_build`'s two independent
        # `tp_uniform_local_validation` passes.
        #
        # There is no fused counterpart to `_prepared`. `with_indexer` only ADDS
        # index tensors to the mapping, and `prepare_glm5_weights` reads none of
        # them (only w_qkv_a / w_q_b / w_uk / w_uv / w_o / w_r / w_ug), so a
        # second pack would allocate a byte-for-byte equivalent copy of every
        # attention and router tensor and pin it for all 18 full layers.
        # `Glm5TP4MonoKernel` does `dict(prepared_weights)` before adding its
        # three index-only packs, so handing it `_prepared`'s mapping is safe:
        # it never mutates what it was given.
        self._fused_weights: dict[int, LayerWeights] = {}
        self._runtimes: dict[int, object] = {}
        # A separate runtime pool for `_GlmFusedLayerOp`: `Glm5TP4MonoKernel`
        # is a different kernel binary, built from a different scratch layout
        # (`layout(..., with_indexer=True)` vs the non-fused kernel's), so it must never
        # share scratch/peer-buffer/step with `_GlmLayerOp` -- see
        # `_GlmFusedLayerOp`'s docstring.
        self._fused_runtimes: dict[int, object] = {}
        # Launches per decode step on each runtime, per chunk width -- the
        # number the ops of that width were BUILT with, which `forward` must
        # then consume exactly. Kept per width because the fused indexer can
        # be on for one width and off for a later one (a fused build may fail
        # and clear `_fused_indexer_on` after an earlier width already built
        # its fused ops), and the split between the two runtimes moves with it.
        self._base_launches: dict[int, int] = {}
        self._fused_launches: dict[int, int] = {}
        self._refused: set[int] = set()
        self._announced: set[int] = set()
        # The `cur_pos` argument `Glm5MonoKernel.forward` demands: contiguous
        # int32 with exactly one element. Under `KvCacheLayout.ATOM` -- the
        # only layout ATOM builds -- the kernel does not read a position from
        # it. It loads `pos0` once and every use of `pos0` sits in a
        # `not use_atom_kv_cache` branch (`glm/kernel.py`: `row_position`,
        # `row_slot`, `split_keys`'s `kv_len`, the `is_new`/`sn` pair); the
        # real positions come from `positions`/`slot_mapping`, and with the
        # fused indexer on, slot 3 carries `block_tables` instead and `pos0`
        # is pinned to 0. So a permanent zero satisfies the contract exactly.
        # Allocated once and reused: the kernel reads it on device, so a
        # per-call tensor would have its address baked into the decode
        # CUDAGraph and then be freed.
        self._cur_pos: torch.Tensor | None = None
        # The starting value is decided once, from static deployment config --
        # never per-step, never data-dependent -- so every TP rank reaches the
        # same answer without needing `tp_uniform_local_validation` here. It
        # is NOT immutable after that, though: `_build` may later turn it off
        # (and only off, never back on) if a fused-only weight mapping or
        # kernel build fails -- always through `tp_uniform_local_validation`
        # at that point, since a build-time failure is not guaranteed
        # symmetric across ranks the way this deployment-config read is.
        self._fused_indexer_on = (
            envs.ATOM_GLM52_FUSED_INDEXER and fused_indexer_refusal(atom_config) is None
        )
        # Narrower than `_fused_indexer_on` and read the same way: once, from
        # static deployment config, so every rank agrees without a collective.
        # `indexer_cp_refusal` already requires `fused_indexer_refusal` to pass,
        # so this can only ever be True where the fused indexer is also on; the
        # `and` is belt-and-braces against `_build` later clearing
        # `_fused_indexer_on` on its own. With it False the kernel is built
        # `indexer_cp=False`, which is the same build, routing and numerics as
        # before this switch existed -- see
        # tests/models/glm52/test_indexer_cp_switch.py.
        self._indexer_cp = (
            self._fused_indexer_on and indexer_cp_refusal(atom_config) is None
        )

    @property
    def rank(self) -> int:
        return get_tensor_model_parallel_rank()

    def layers(self):
        """The MoE layers mono owns (see ``contract.mono_layers``)."""
        return mono_layers(self._lm.model)

    def prepare(self, rows: int) -> bool:
        # `_refused` is keyed by ROWS, and the reuse test below by kernel chunk.
        # Asking the chunk first answered for a `rows` the chunk cannot speak
        # for: a refused width, or a chunked one that `_build` refuses because
        # its launches would share a mailbox epoch. `samples` includes cudagraph
        # pad rows while `supports` gates on `scheduled_bs`, so a `prepare(15)`
        # really does arrive while a chunk-5 op exists, and answering True there
        # runs 15 rows on the chunk-5 build.
        if rows in self._refused:
            return False
        chunk_samples = glm5_kernel_samples(rows, self._query_length)
        if rows == chunk_samples and any(
            chunk == chunk_samples for _, chunk in self._ops
        ):
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        return self._build(rows)

    def _build(self, rows: int) -> bool:
        chunk_samples = glm5_kernel_samples(rows, self._query_length)
        layers = self.layers()
        if rows != chunk_samples:
            # A chunked launch is `rows // chunk_samples` kernel launches from
            # one `op.forward` call, all sharing the `layer` argument and
            # separated by a `step` bump (`glm/op.py`). This runner's slot
            # scheme is the opposite -- one `step` bump per decode step, a
            # distinct `layer` per launch -- and the two cannot be mixed: the
            # chunks would share a mailbox epoch. No width mono actually routes
            # is chunked (`AGENTX_BATCHES` is (1, 2) and every supported
            # `QUERY_LENGTHS` entry divides a kernel width exactly), so this is
            # unreachable today and refuses loudly rather than silently
            # corrupting a mailbox if that ever changes.
            self._refused.add(rows)
            logger.warning(
                "GLM-5.2 MonoKernel fallback before launch: chunked launches "
                "(rows=%d, kernel chunk=%d) are not supported by the per-step "
                "mailbox epoch scheme",
                rows,
                chunk_samples,
            )
            return False

        mapped = []
        validation_error = None
        rank = self.rank
        npes = get_tensor_model_parallel_world_size()
        try:
            # Imported inside the agreed-refusal path, not above it: an absent
            # FlyDSL raises ImportError, and `_build`'s caller `prepare` is
            # invoked outside `LazyRunner.ready`'s try, so raising from above
            # here would abort the decode warmup instead of leaving the
            # deployment on the production path. In here it becomes
            # `validation_error` like any other local failure, is agreed across
            # ranks below, and every rank refuses at the same point.
            from atom.models.glm52.mono.kernels.op import prepare_glm5_weights

            bf16_vector(
                self._lm.model.norm.weight, "model.norm.weight", self._shard.hidden
            )
            native = native_config(
                samples=rows,
                tp_size=npes,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                mtp=(
                    self._atom_config.speculative_config is not None
                    and self._atom_config.speculative_config.method == "mtp"
                ),
                query_length=self._query_length,
            )
            if native is None:
                raise MonoUnsupported("layer backend")
            for layer in layers:
                if layer.layer_idx not in self._weights:
                    # Always the base (with_indexer=False) mapping: every
                    # layer needs this to build its `_GlmLayerOp`, whether or
                    # not the fused indexer is on. The fused-only
                    # (with_indexer=True) mapping is a separate, independently
                    # validated pass below -- see `_GlmFusedLayerOp`'s
                    # docstring and the "falls back, does not refuse `rows`"
                    # contract that pass must honor.
                    weights = layer_weights(layer, rank, npes, with_indexer=False)
                    mapped.append(
                        (
                            layer.layer_idx,
                            weights,
                            prepare_glm5_weights(weights, AttentionWeight.FP8_PTPC),
                        )
                    )
        # `Exception`, not (MonoUnsupported, ValueError): every one of these four
        # blocks hands its verdict to `tp_uniform_local_validation`, which is
        # COLLECTIVE. A failure the narrow tuple missed -- `torch.OutOfMemoryError`
        # from the dense allocations in `prepare_glm5_weights`, realistically --
        # left this rank before that collective while its peers blocked inside it,
        # so the group hung instead of falling back together. Same shape
        # `bind_agreed` uses. The original type survives: the helper reports each
        # rank's failure as `f"{type(error).__name__}: {error}"`.
        except Exception as error:  # noqa: BLE001 - collective below
            validation_error = error
        try:
            tp_uniform_local_validation(
                validation_error,
                group=get_tp_group().cpu_group,
                world_size=npes,
                context="GLM-5.2 weight mapping failed",
            )
        except MonoUnsupported as error:
            self._refused.add(rows)
            logger.warning("GLM-5.2 MonoKernel fallback before launch: %s", error)
            return False
        for layer_idx, weights, prepared in mapped:
            self._weights[layer_idx] = weights
            self._prepared[layer_idx] = prepared

        # A second, independent weight-mapping pass for the fused indexer's
        # own (with_indexer=True) tensors -- full layers only. Validated
        # through its OWN `tp_uniform_local_validation`, separate from the
        # pass above: a failure here (e.g. one of `indexer_weights`'s own
        # `need()` checks) must disable only the fused path, TP-uniformly,
        # and fall back to the external-indexer mono path for the rest of the
        # deployment -- never refuse `rows` outright the way a genuine base
        # mapping failure still does above.
        if self._fused_indexer_on:
            fused_mapped = []
            fused_validation_error = None
            try:
                for layer in layers:
                    if (
                        _is_full_indexer_layer(layer)
                        and layer.layer_idx not in self._fused_weights
                    ):
                        fused_mapped.append(
                            (
                                layer.layer_idx,
                                layer_weights(layer, rank, npes, with_indexer=True),
                            )
                        )
            except Exception as error:  # noqa: BLE001 - collective below
                fused_validation_error = error
            try:
                tp_uniform_local_validation(
                    fused_validation_error,
                    group=get_tp_group().cpu_group,
                    world_size=npes,
                    context="GLM-5.2 fused indexer weight mapping failed",
                )
            except MonoUnsupported as error:
                logger.warning(
                    "GLM-5.2 fused indexer disabled, falling back to the "
                    "external-indexer mono path: %s",
                    error,
                )
                self._fused_indexer_on = False
            else:
                for layer_idx, weights in fused_mapped:
                    self._fused_weights[layer_idx] = weights

        # The fused ops build FIRST, and the base ops after it, because the
        # base runtime's `launches_per_step` is "every layer the fused runtime
        # did not take" -- which is only settled once the fused build has
        # either succeeded or disabled itself. Built the other way round the
        # base ops would have to be torn down and rebuilt whenever a fused
        # build failed.
        #
        # A fused failure is agreed TP-uniformly and disables only the fused
        # path, never `rows` as a whole: every full layer also gets a
        # `_GlmLayerOp` from the loop below, which is the deployment-wide
        # fallback. See Important #6 in the design review.
        #
        # The TP-collective ordering invariant is unchanged by the swap: every
        # rank still reaches every `tp_uniform_local_validation` in the same
        # order, and the only early return is after the LAST of them, so no
        # rank can leave while a peer is still waiting on it.
        made_fused = []
        fused_runtime_created = False
        fused_launches = 0
        if self._fused_indexer_on:
            fused_build_error = None
            fused_runtime = self._fused_runtimes.get(chunk_samples)
            fused_runtime_created = fused_runtime is None
            fused_layer_count = sum(
                1 for layer in layers if _is_full_indexer_layer(layer)
            )
            try:
                for layer in layers:
                    if not _is_full_indexer_layer(layer):
                        continue
                    key = (layer.layer_idx, chunk_samples)
                    if key not in self._fused_ops:
                        fused_owned = _GlmFusedLayerOp(
                            self._fused_weights[layer.layer_idx],
                            chunk_samples,
                            self._atom_config.hf_config.index_topk,
                            self._atom_config.kv_cache_dtype,
                            self._atom_config.kv_cache_block_size,
                            self._query_length,
                            self._atom_config.max_model_len
                            // self._atom_config.kv_cache_block_size,
                            self._indexer_cp,
                            fused_index_capacity(self._atom_config.max_model_len),
                            self._prepared[layer.layer_idx],
                            fused_runtime,
                            launches_per_step=fused_layer_count,
                        )
                        if fused_runtime is None:
                            fused_runtime = fused_owned.op
                            self._fused_runtimes[chunk_samples] = fused_runtime
                        self._fused_ops[key] = fused_owned
                        made_fused.append(key)
                # The number the fused ops were actually BUILT with, which is
                # what `forward` has to consume; recomputing it from
                # `_fused_ops` later would silently paper over a disagreement.
                fused_launches = fused_layer_count
            except Exception as error:  # noqa: BLE001 - collective below
                fused_build_error = error
            try:
                tp_uniform_local_validation(
                    fused_build_error,
                    group=get_tp_group().cpu_group,
                    world_size=npes,
                    context="GLM-5.2 fused indexer build failed",
                )
            except MonoUnsupported as error:
                for key in made_fused:
                    self._fused_ops.pop(key).close()
                made_fused = []
                fused_launches = 0
                if fused_runtime_created:
                    self._fused_runtimes.pop(chunk_samples, None)
                    fused_runtime_created = False
                logger.warning(
                    "GLM-5.2 fused indexer disabled, falling back to the "
                    "external-indexer mono path: %s",
                    error,
                )
                self._fused_indexer_on = False

        # Which layers each runtime will launch for THIS width, now that the
        # fused phase has settled. A full layer runs its fused op and not its
        # base one (`forward` prefers `_fused_ops`), so the two counts
        # partition the mono layers rather than overlapping.
        base_launches = len(layers) - fused_launches

        made = []
        runtime = self._runtimes.get(chunk_samples)
        runtime_created = runtime is None
        build_error = None
        try:
            if base_launches <= 0:
                raise MonoUnsupported(
                    "every mono layer is a fused-indexer layer, leaving the "
                    "base runtime with no launches to number"
                )
            for layer in layers:
                key = (layer.layer_idx, chunk_samples)
                if key not in self._ops:
                    owned = _GlmLayerOp(
                        self._weights[layer.layer_idx],
                        chunk_samples,
                        self._atom_config.hf_config.index_topk,
                        self._atom_config.kv_cache_dtype,
                        self._prepared[layer.layer_idx],
                        runtime,
                        launches_per_step=base_launches,
                    )
                    if runtime is None:
                        runtime = owned.op
                        self._runtimes[chunk_samples] = runtime
                    self._ops[key] = owned
                    made.append(key)
        except Exception as error:  # noqa: BLE001 - collective below
            build_error = error
        # Agreed TP-uniformly, like the weight-mapping failures above: every
        # rank must reach this same collective before any rank may return, or
        # a rank whose local build failed leaves early while its peers block
        # forever on a participant that already left (a silent gloo hang, no
        # error).
        try:
            tp_uniform_local_validation(
                build_error,
                group=get_tp_group().cpu_group,
                world_size=npes,
                context="GLM-5.2 ops build failed",
            )
        except MonoUnsupported as error:
            for key in made:
                self._ops.pop(key).close()
            if runtime_created:
                self._runtimes.pop(chunk_samples, None)
            # The fused ops for this width are unreachable without their base
            # siblings -- `forward` only consults `_fused_ops` for a layer that
            # also has an `_ops` entry -- so they would be leaked GPU memory.
            for key in made_fused:
                self._fused_ops.pop(key).close()
            if fused_runtime_created:
                self._fused_runtimes.pop(chunk_samples, None)
            self._refused.add(rows)
            logger.warning("GLM-5.2 MonoKernel fallback before launch: %s", error)
            return False

        self._base_launches[chunk_samples] = base_launches
        self._fused_launches[chunk_samples] = fused_launches
        if rank == 0 and rows not in self._announced:
            # Which indexer this width actually ended up on is named here, not
            # inferred. The fused path has several ways to turn itself off --
            # `fused_indexer_refusal` at construction, a weight-mapping
            # failure, a build failure rolled back TP-uniformly above -- and
            # every one of them leaves a server that boots, serves, and looks
            # identical to a fused run from the outside. A run that silently
            # fell back is not evidence about the fused kernel, so say which
            # one ran.
            fused_layers = sum(
                1 for _, chunk in self._fused_ops if chunk == chunk_samples
            )
            logger.info(
                "GLM-5.2 MonoKernel on: S=%d chunk=%d indexer=%s",
                rows,
                chunk_samples,
                (
                    f"FUSED on {fused_layers} full layers"
                    f"{' +CP' if self._indexer_cp else ''}"
                    if fused_layers
                    else "external"
                ),
            )
            self._announced.add(rows)
        return True

    # Re-runs input RMSNorm and fused_qkv_a_proj outside the kernel to feed the
    # indexer its q_a, duplicating work the kernel also does for its own q path.
    # The kernel keeps that q_a in scratch it does not export, and the indexer
    # must run BEFORE the launch that consumes its indices -- so the duplicate
    # is load-bearing, not an oversight. Removing it leaves the kernel selecting
    # over a stale index buffer, which is a different computation, not a faster
    # one.
    @staticmethod
    def _refresh_indexer(layer, state: torch.Tensor, positions: torch.Tensor) -> None:
        from atom.model_ops.layernorm import rmsnorm2d_fwd_

        attn = layer.self_attn
        indexer = attn.indexer
        if indexer is None or attn.skip_topk:
            return
        normalized = rmsnorm2d_fwd_(
            state,
            layer.input_layernorm.weight,
            layer.input_layernorm.eps,
            GLM5_CONFIG.hidden,
        )
        qkv = attn.fused_qkv_a_proj(normalized)
        q_c = qkv[:, : GLM5_CONFIG.q_lora]
        qr = rmsnorm2d_fwd_(
            q_c,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.eps,
            GLM5_CONFIG.q_lora,
        )
        indexer(normalized, qr, None, positions, attn.indexer_rope_emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self._lm.model
        fwd = get_forward_context()
        metadata = fwd.attn_metadata
        samples = input_ids.numel()
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        residual = None
        if self._cur_pos is None:
            self._cur_pos = torch.zeros(1, dtype=torch.int32, device=positions.device)
        cur_pos = self._cur_pos
        shared_indices = _shared_sparse_buffer(model.layers)
        slot_mapping = metadata.slot_mapping[:samples]
        sparse_indptr = metadata.sparse_kv_indptr[: samples + 1]
        # One mailbox epoch per decode step, not per layer. `step` is a device
        # tensor, so bumping it once per layer cost 75 one-element int32 adds
        # per step -- 485 us of pure launch overhead on a serialized stream,
        # measured in `mono-profile-attribution.md` group D. The kernel's own
        # scheme (`glm/kernel.py`: tag = step * LAYER_SLOTS + layer + 1) already
        # distinguishes launches within a step by `layer`, so the counters below
        # do the distinguishing and `advance=True` fires once, on each runtime's
        # last launch.
        #
        # These are plain Python ints on the host, incremented in a loop whose
        # trip count and branch structure depend only on the module tree and on
        # `chunk_samples` -- never on tensor data -- so every value is fixed at
        # CUDA-graph capture and replays identically.
        step_chunk = glm5_kernel_samples(samples, metadata.max_seqlen_q)
        base_launches = self._base_launches.get(step_chunk, 0)
        fused_launches = self._fused_launches.get(step_chunk, 0)
        base_slot = 0
        fused_slot = 0
        for layer in model.layers[model.start_layer : model.end_layer]:
            key = (layer.layer_idx, step_chunk)
            owned = self._ops.get(key)
            if owned is None:
                hidden, residual = layer(positions, hidden, residual)
                continue
            fused_owned = self._fused_ops.get(key)
            if residual is not None and layer.input_layernorm.fused_allreduce:
                hidden = get_tp_group().all_reduce(hidden, ca_fp8_quant=False)
            state = hidden if residual is None else hidden + residual
            attn = layer.self_attn
            cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"].k_cache
            fused = fused_owned is not None
            owned = fused_owned if fused else owned
            if fused:
                # The fused kernel projects/scores/selects its own indices
                # inside the launch -- _refresh_indexer's external compute is
                # for the non-fused Glm5MonoKernel path only, and is dead
                # weight (worse: a second, divergent, source of indices) here.
                index_cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"].index_cache
                hidden = owned.op.forward(
                    state,
                    cur_pos,
                    cache,
                    cache,
                    shared_indices,
                    attn.rotary_emb.cos_cache,
                    attn.rotary_emb.sin_cache,
                    layer=fused_slot,
                    advance=fused_slot + 1 == fused_launches,
                    positions=positions,
                    slot_mapping=slot_mapping,
                    sparse_kv_indptr=sparse_indptr,
                    index_cache=index_cache,
                    block_tables=metadata.block_tables,
                )
                fused_slot += 1
                # Publish into ATOM's production buffers: the first three
                # (dense) layers and every "shared" mono layer read
                # sparse_kv_indices_buffer/sparse_kv_indptr directly, never
                # this kernel's own scratch -- see _publish_selected_indices.
                _publish_selected_indices(
                    owned.op,
                    shared_indices,
                    sparse_indptr,
                    metadata.cu_seqlens_q,
                    metadata.kv_indptr,
                    metadata.kv_indices,
                    samples,
                    metadata.max_seqlen_q,
                    metadata.batch_id_per_q_token,
                )
            else:
                self._refresh_indexer(layer, state, positions)
                hidden = owned.op.forward(
                    state,
                    cur_pos,
                    cache,
                    cache,
                    shared_indices,
                    attn.rotary_emb.cos_cache,
                    attn.rotary_emb.sin_cache,
                    layer=base_slot,
                    advance=base_slot + 1 == base_launches,
                    positions=positions,
                    slot_mapping=slot_mapping,
                    sparse_kv_indptr=sparse_indptr,
                )
                base_slot += 1
            residual = None
        if base_slot != base_launches or fused_slot != fused_launches:
            # The epoch scheme this loop implements is only correct when each
            # runtime's slots run 0 .. launches_per_step-1 and `step` advances
            # exactly once. If the loop consumed a different number of slots
            # than the ops were built with, `advance` either never fired (every
            # later step reuses this step's epochs) or the step-boundary parity
            # of `peer_slot = (step * launches_per_step + layer) & 1` stopped
            # alternating -- a peer writing over a buffer a peer is still
            # reading. Both are silent: wrong numbers, or four ranks spinning
            # in-kernel with no error and no bounded-wait report (GLM-5.2 has
            # no `ATOM_MONO_DEBUG`). So this fails loudly instead.
            raise RuntimeError(
                "GLM-5.2 mono: mailbox slot accounting is wrong -- consumed "
                f"{base_slot}/{base_launches} base and "
                f"{fused_slot}/{fused_launches} fused launches at chunk width "
                f"{step_chunk}. The ops were built for the other number, so "
                "the peer-buffer double buffer is no longer alternating."
            )
        state = hidden if residual is None else hidden + residual
        from atom.model_ops.layernorm import rmsnorm2d_fwd_

        return rmsnorm2d_fwd_(
            state,
            model.norm.weight,
            model.norm.eps,
            GLM5_CONFIG.hidden,
        )

    def close(self) -> None:
        for owned in self._ops.values():
            owned.close()
        for owned in self._fused_ops.values():
            owned.close()
        self._ops.clear()
        self._fused_ops.clear()
        self._weights.clear()
        self._prepared.clear()
        self._fused_weights.clear()
        self._runtimes.clear()
        self._fused_runtimes.clear()
        self._base_launches.clear()
        self._fused_launches.clear()
        self._refused.clear()
        self._announced.clear()
        self._cur_pos = None
