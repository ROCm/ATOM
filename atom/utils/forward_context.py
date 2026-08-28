# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Union

import numpy as np
import torch

from atom.config import Config, CUDAGraphMode, KVCacheTensor, ParallelConfig


class AttnState(Enum):
    """Attention dispatch state — controls which kv-indices buffers are built
    and which forward branch fires.

    Backends that distinguish only "decode vs prefill" can treat any
    ``PREFILL_*`` value as prefill. Backends with chunked-prefill awareness
    (e.g. V4) further distinguish ``PREFILL_NATIVE`` from ``PREFILL_PREFIX``.

    - ``DECODE``: 1+K tokens/seq uniformly (decode + spec). Per-token decode
      kv-indices buffers are valid; prefill prefix buffers may be stale.
    - ``PREFILL_NATIVE``: fresh prefill — every seq starts at position 0 in
      this fwd. No prior-chunk KV history to read; the prefix region is
      empty per token.
    - ``PREFILL_PREFIX``: chunked prefill — at least one seq has
      ``chunk_start > 0`` and therefore reads its prior chunk's KV from
      the paged history (e.g. V4 SWA ring via ``kv_indices_prefix_swa``).
    """

    DECODE = "decode"
    PREFILL_NATIVE = "prefill_native"
    PREFILL_PREFIX = "prefill_prefix"


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: list[int], max_num_tokens: int, chunk_idx: int
) -> list[int]:
    dp_size = len(num_tokens_across_dp_cpu)

    local_size = [-1] * dp_size
    for i in range(dp_size):
        dp_tokens = num_tokens_across_dp_cpu[i]
        local_size[i] = min(max_num_tokens, dp_tokens - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp: int  # Pre-computed int value for cudagraph compatibility
    local_sizes: list[int] | None = None

    @staticmethod
    def num_tokens_across_dp(
        num_tokens: int, dp_size: int, dp_rank: int
    ) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(
            num_tokens_across_dp, device="cpu", dtype=torch.int32
        )
        import torch.distributed as dist
        from aiter.dist.parallel_state import get_dp_group

        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        # attn_metadata: Any,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None = None,
    ) -> "DPMetadata":

        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (
            num_tokens_across_dp is None or num_tokens_across_dp[dp_rank] == batchsize
        )
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank
            )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        max_tokens_across_dp = (
            max_tokens_across_dp_cpu.item()
        )  # Pre-compute int for cudagraph
        return DPMetadata(
            max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu, max_tokens_across_dp
        )

    @contextmanager
    def chunked_sizes(self, max_chunk_size_per_rank: int, chunk_idx: int):
        """
        Context manager to compute and temporarily set the per-rank local token
        sizes for a specific chunk during chunked forward execution.
        This is necessary to ensure each DP (data parallel) rank processes its
        designated portion of tokens in lockstep with others, even when the
        token counts are uneven or some ranks have completed their input early.
        For chunked execution, we break up the total tokens on each rank into
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given
        `chunk_idx`, this context manager sets `self.local_sizes` to the number
        of tokens to process in that chunk on each rank.
        It uses cumulative sizes (`cu_tokens_across_dp_cpu`) to derive the
        number of tokens per rank, and calls `_compute_chunked_local_num_tokens`
        to determine the chunk-wise split.
        `self.local_sizes` is only valid inside the context.
        Args:
            max_chunk_size_per_rank: The max number of tokens each rank is
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        cu_sizes = self.cu_tokens_across_dp_cpu
        num_tokens_across_dp_cpu = [
            (cu_sizes[i] - cu_sizes[i - 1]).item() if i > 0 else cu_sizes[0].item()
            for i in range(len(cu_sizes))
        ]
        self.local_sizes = _compute_chunked_local_num_tokens(
            num_tokens_across_dp_cpu, max_chunk_size_per_rank, chunk_idx
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        return self.local_sizes

    def get_sizes_across_dp(self) -> list[int]:
        """Per-rank token counts derived from cumulative tensor."""
        cu = self.cu_tokens_across_dp_cpu
        return [(cu[i] - (cu[i - 1] if i > 0 else 0)).item() for i in range(len(cu))]


@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor
    num_spec_steps: int
    num_draft_tokens_np: np.ndarray
    cu_num_draft_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor


@dataclass(frozen=True)
class ForwardMode:
    """Per-step dispatch, and the two batch sizes ATOM names.

    ``scheduled_bs`` is what this rank was handed; ``running_bs`` is what the
    GPU is made to run. The second is DP-group-UNIFIED and that is its whole
    point: MoE pads to it and a captured graph holds its collective at that
    width, so a rank arriving with a different one hangs the group.

    An op reads whichever its own tensors carry. They differ in one corner --
    uniform decode running eager -- where MoE pads while attention keeps its
    unpadded rows for aiter's ``t == t_slot``.
    """

    use_cudagraph: bool
    scheduled_bs: int
    running_bs: int
    is_prefill: bool

    @classmethod
    def decide(
        cls,
        *,
        is_prefill: bool,
        scheduled_bs: int,
        running_tokens: int,
        dp_uniform_decode: bool,
        enforce_eager: bool,
        capture_sizes: list[int],
        mtp_step: int = 1,
    ) -> "ForwardMode":
        """Pick dispatch and the step's ``running_bs``. Any new force-eager
        condition belongs here, not in caller-side checks."""
        if is_prefill:
            # Prefill is always eager, so nothing is padded and the two are the
            # same number. Per-token sums ride cu_seqlens_q, not this.
            return cls(
                use_cudagraph=False,
                scheduled_bs=scheduled_bs,
                running_bs=scheduled_bs,
                is_prefill=True,
            )

        # The sequences the step will run, recovered from the token count:
        # DP-unified in uniform mode (`running_tokens` is the group max, set in
        # ModelRunner._preprocess), local-equivalent otherwise.
        unified_bs = (running_tokens + mtp_step - 1) // mtp_step

        if not dp_uniform_decode:
            # MoE goes through all_gatherv (variable-length, per-rank sizes),
            # so pad_for_all_gather is never reached and there is nothing to
            # unify. Running at the scheduled batch pads nothing.
            return cls(
                use_cudagraph=False,
                scheduled_bs=scheduled_bs,
                running_bs=scheduled_bs,
                is_prefill=False,
            )

        # From here on uniform decode: MoE WILL pad_for_all_gather, so
        # running_bs MUST be the same number on every rank.
        if enforce_eager or unified_bs > capture_sizes[-1]:
            # Eager under uniform decode, either forced or above the largest
            # captured size. This is the one corner where the two differ:
            # attention keeps its own scheduled batch while MoE pads to the
            # unified one.
            return cls(
                use_cudagraph=False,
                scheduled_bs=scheduled_bs,
                running_bs=unified_bs,
                is_prefill=False,
            )

        # CUDAGraph: the smallest captured size that fits. Captured sizes are
        # a static list every rank shares, so this is unified by construction.
        return cls(
            use_cudagraph=True,
            scheduled_bs=scheduled_bs,
            running_bs=next((x for x in capture_sizes if x >= unified_bs), unified_bs),
            is_prefill=False,
        )

    @property
    def attn_tensors_are_padded(self) -> bool:
        """True iff per-token attention tensors carry padding this step
        (cudagraph today; update if other padded layouts are added)."""
        return self.use_cudagraph

    def assert_shape_contract(
        self,
        input_ids: "torch.Tensor",
        attn_metadata: "AttentionMetaData",
    ) -> None:
        """Validate ``input_ids`` / ``slot_mapping`` against this ForwardMode.

        Skips prefill (variable-length) and cudagraph (graph-internal shape);
        callers invoke unconditionally to keep dispatch decisions centralised.
        """
        if self.is_prefill or self.attn_tensors_are_padded:
            return
        if (
            input_ids is None
            or attn_metadata is None
            or attn_metadata.slot_mapping is None
        ):
            return
        # Past the early-return nothing is padded, so the rows attention runs
        # are exactly the scheduled ones.
        max_q = attn_metadata.max_seqlen_q
        expected = self.scheduled_bs * max_q
        actual_in = input_ids.shape[0]
        actual_slot = attn_metadata.slot_mapping.shape[0]
        assert actual_in == expected, (
            f"eager input_ids length {actual_in} != scheduled_bs*max_q="
            f"{expected} ({self})"
        )
        assert actual_slot == expected, (
            f"eager slot_mapping length {actual_slot} != scheduled_bs*max_q="
            f"{expected} ({self}); attn_metadata_builder used a stale bs"
        )


def running_tokens_from_bs(bs: int, *, is_prefill: bool, attn_metadata) -> int:
    """``running_tokens`` for a bridge that only knows a request count.

    Prefill returns that count verbatim, which is NOT a height: prompt lengths
    are ragged, so no multiplier recovers one, and a number below the real
    height just leaves the batch unpadded -- correct there, since a prefill
    all_gather is variable-length.
    """
    if is_prefill or attn_metadata is None:
        return bs
    return bs * int(attn_metadata.max_seqlen_q)


@dataclass
class Context:
    # This context is used to store the basic context of the forward.
    positions: torch.Tensor
    is_prefill: bool = False
    is_dummy_run: bool = False
    # What this rank was handed. Duplicated from `forward_mode` because a
    # capture context has none; `scheduled_tokens` is what an eager forward
    # actually runs.
    scheduled_bs: int = 0
    scheduled_tokens: int = 0
    # The step's DP-unified padded shape. `running_bs` counts SEQUENCES (graph
    # identity, the draft's pad width), `running_tokens` the hidden_states rows
    # MoE pads to. Both stored, because the ratio is not always max_seqlen_q --
    # a DSpark ragged step runs a flat bucket no rectangular bs*q recovers.
    running_bs: int = 0
    running_tokens: int = 0
    is_draft: bool = False
    # True iff all DP ranks are running pure decode this step (DP-disabled
    # case is treated as True). Mirrors vLLM's `uniform_decode` flag and
    # gates DP-specific variable-length all_gather/scatter paths.
    dp_uniform_decode: bool = True
    # Single source of truth for cudagraph vs eager dispatch + the step's
    # scheduled_bs / running_bs.
    # Set by prepare_inputs via ForwardMode.decide(). None only on legacy
    # paths that haven't been routed through it (run_model falls back to
    # the original four-OR derivation in that case for back-compat).
    forward_mode: ForwardMode | None = None
    # Optional flat token ids for the current forward. Read by callbacks
    # invoked inside Dynamo-opaque custom ops (e.g. V4 MoE hash routing)
    # that need the token ids but cannot receive them as a function arg
    # (the op signature is fixed by the consumer's plugin contract).
    input_ids: torch.Tensor | None = None
    # Row offset of this (micro-)batch's tokens on the full forward's token
    # axis. 0 except inside a TBO micro-batch. Anything writing into a
    # whole-forward buffer from inside the model must add it, or the ubatches
    # land on each other's rows. Read per-thread (the context is thread-local).
    ubatch_token_offset: int = 0

    # Optional speculative-decoding inputs staged alongside the target
    # forward's other host-to-device copies. Keeping these on the per-forward
    # context avoids launching pinned-buffer H2Ds later from postprocess.
    draft_anchor_overrides: torch.Tensor | None = None
    draft_ragged_lens: torch.Tensor | None = None

    def __init__(
        self,
        positions: torch.Tensor,
        is_prefill: bool = False,
        is_dummy_run: bool = False,
        scheduled_bs: int = 0,
        scheduled_tokens: int = 0,
        running_bs: int = 0,
        running_tokens: int = 0,
        is_draft: bool = False,
        dp_uniform_decode: bool = True,
        forward_mode: ForwardMode | None = None,
        input_ids: torch.Tensor | None = None,
        ubatch_token_offset: int = 0,
        draft_anchor_overrides: torch.Tensor | None = None,
        draft_ragged_lens: torch.Tensor | None = None,
    ):
        self.positions = positions
        self.is_prefill = is_prefill
        self.is_dummy_run = is_dummy_run
        self.scheduled_bs = scheduled_bs
        self.scheduled_tokens = scheduled_tokens
        self.running_bs = running_bs
        self.running_tokens = running_tokens
        self.is_draft = is_draft
        self.dp_uniform_decode = dp_uniform_decode
        self.forward_mode = forward_mode
        self.input_ids = input_ids
        self.ubatch_token_offset = ubatch_token_offset
        self.draft_anchor_overrides = draft_anchor_overrides
        self.draft_ragged_lens = draft_ragged_lens


@dataclass
class AttentionMetaData:
    """Attention metadata for prefill and decode batched together."""

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    dropout_p: float = 0.0
    # True for standard causal attention; False only for DSpark's bidirectional
    # draft block. The MLA asm decode kernel selects a different .co by this flag.
    causal: bool = True

    state: AttnState = AttnState.PREFILL_NATIVE
    """One of `DECODE / PREFILL_NATIVE / PREFILL_PREFIX` — controls which
    kv-indices buffers downstream forward branches read. Default is
    `PREFILL_NATIVE`; every `prepare_*` path overrides explicitly.
    Backends that don't need the NATIVE/PREFIX distinction can treat
    `any PREFILL_*` as prefill. See ``AttnState`` for full semantics."""

    kv_indptr: torch.Tensor | None = None
    kv_indices: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    kv_last_page_lens: torch.Tensor | None = None
    cu_seqlen_ks: torch.Tensor | None = None
    cu_seqlen_ke: torch.Tensor | None = None
    sparse_kv_indptr: torch.Tensor | None = None
    # Last-page lens for sparse (DSA) attention: all 1s, one per query token in
    # prefill/MTP-verify and per seq in decode. Separate from kv_last_page_lens
    # (the dense per-seq buffer) so the two never clobber each other.
    sparse_kv_last_page_lens: torch.Tensor | None = None

    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # for prefix cache
    has_cached: bool = False
    total_kv: int | None = None
    num_cached_tokens: torch.Tensor | None = None
    seq_starts: torch.Tensor | None = None

    def __init__(
        self,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        min_seqlen_q: int = 0,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = True,
        state: AttnState = AttnState.PREFILL_NATIVE,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        qo_indptr: torch.Tensor | None = None,
        kv_last_page_lens: torch.Tensor | None = None,
        cu_seqlen_ks: torch.Tensor | None = None,
        cu_seqlen_ke: torch.Tensor | None = None,
        sparse_kv_indptr: torch.Tensor | None = None,
        work_meta_data: torch.Tensor | None = None,
        work_indptr: torch.Tensor | None = None,
        work_info_set: torch.Tensor | None = None,
        reduce_indptr: torch.Tensor | None = None,
        reduce_final_map: torch.Tensor | None = None,
        reduce_partial_map: torch.Tensor | None = None,
        sparse_cu_seqlens_q: torch.Tensor | None = None,
        token_to_seq_idxs: torch.Tensor | None = None,
        has_cached: bool = False,
        total_kv: int | None = None,
        num_cached_tokens: torch.Tensor | None = None,
        seq_starts: torch.Tensor | None = None,
    ):
        self.has_cached = has_cached
        self.total_kv = total_kv
        self.num_cached_tokens = num_cached_tokens
        self.seq_starts = seq_starts
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.min_seqlen_q = min_seqlen_q
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.dropout_p = dropout_p
        self.causal = causal
        self.state = state
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.kv_last_page_lens = kv_last_page_lens
        self.cu_seqlen_ks = cu_seqlen_ks
        self.cu_seqlen_ke = cu_seqlen_ke
        self.sparse_kv_indptr = sparse_kv_indptr
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        self.sparse_cu_seqlens_q = sparse_cu_seqlens_q
        self.token_to_seq_idxs = token_to_seq_idxs

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in skip_fields
        }


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any] = field(default_factory=dict)

    attn_metadata: Union["AttentionMetaData", dict[str, "AttentionMetaData"]] | None = (
        None
    )

    kv_cache_data: dict[str, KVCacheTensor] = None

    context: Context | None = None

    dp_metadata: DPMetadata | None = None

    spec_decode_metadata: SpecDecodeMetadata | None = None

    ubatch_slices: list[Any] | None = None

    # Cross-DP MAX of per-ubatch token counts, precomputed in
    # ``ModelRunner._preprocess`` and propagated here so
    # ``UBatchWrapper._compute_ub_graph_bs`` no longer needs its own
    # per-ubatch all_reduce. Shape: tuple of length N == len(ubatch_slices).
    # None when DP is off or when TBO is not active this step.
    ub_max_tokens_across_dp: tuple | None = None

    # Cached current_stream() captured at set_forward_context() time, so
    # downstream code (V4 attention / MoE / metadata builder) doesn't have
    # to query torch.cuda.current_stream() repeatedly during a forward —
    # multiple call sites caching independent Stream handles was widening
    # the hipStream handle pool and complicating reasoning about which
    # logical stream each wait_stream() refers to. CG capture / TBO
    # threads each call set_forward_context() inside their own stream
    # context, so the cached value is correct for the captured graph or
    # active thread.
    main_stream: torch.cuda.Stream | None = None

    # True only while the model forward runs inside a CUDAGraph capture
    # block (model_runner.capture_model loop). Components that gate
    # multi-stream side-launches (V4 main Compressor on alt_stream,
    # indexer.compressor on indexer_stream) check this flag: side-stream
    # work is safe to emit inside a captured graph (graph records the
    # fork-join edges and replay re-uses the same stream layout) but
    # racy in eager mode where launches accumulate across layers and
    # deadlock the hipStream queue. Replay does not re-execute Python
    # forward, so it ignores the flag entirely.
    in_hipgraph: bool = False

    # Piecewise-cudagraph dispatch, read per forward by CUDAGraphWrapper:
    # cudagraph_runtime_mode picks the capture/replay mode, batch_descriptor is
    # the key (num_tokens). None defaults keep wrappers inert until model_runner
    # sets them. Typed Any to dodge a CUDAGraphMode circular import.
    cudagraph_runtime_mode: Any = None
    batch_descriptor: Any | None = None

    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}
        if self.attn_metadata is None:
            self.attn_metadata = {}


_forward_context: ForwardContext | None = ForwardContext()
_forward_kv_cache_context: ForwardContext | None = ForwardContext()

# Cached once at module import — CUDA availability does not change at
# runtime, so we don't pay torch.cuda.is_available() per set_forward_context().
_CUDA_AVAILABLE: bool = torch.cuda.is_available()

# Thread-local storage for TBO dual-thread execution

_forward_context_local = threading.local()


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # Check thread-local first (used by TBO threads)
    ctx = getattr(_forward_context_local, "ctx", None)
    if ctx is not None:
        return ctx

    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def _normalize_cudagraph_runtime_mode(mode: Any) -> CUDAGraphMode | None:
    """Normalize a frontend runtime mode to ATOM's concrete enum.

    Frontends own their graph dispatch and therefore use distinct enum
    classes.  Match by name rather than value so their enum layouts can evolve
    independently.  Composite configuration modes are deliberately rejected:
    a forward context must describe the concrete NONE/PIECEWISE/FULL decision
    for the current batch.
    """
    name = mode if isinstance(mode, str) else getattr(mode, "name", None)
    if name not in {"NONE", "PIECEWISE", "FULL"}:
        return None
    return CUDAGraphMode[name]


def get_current_cudagraph_runtime_mode() -> CUDAGraphMode:
    """Return the concrete graph mode for the active model forward.

    In vLLM plugin mode graph capture/replay is owned by vLLM, so its forward
    context is authoritative.  Native ATOM records the same decision on its
    own ForwardContext.  An unavailable/unknown context is treated as NONE:
    eager dual-stream execution is valid, and some vLLM runners expose NONE
    while a whole-model FULL graph is being captured.  Replay does not execute
    this Python dispatcher.
    """
    from atom.plugin import is_vllm

    if is_vllm():
        try:
            from vllm.forward_context import (
                get_forward_context as get_vllm_forward_context,
            )
            from vllm.forward_context import (
                is_forward_context_available,
            )

            if is_forward_context_available():
                mode = _normalize_cudagraph_runtime_mode(
                    get_vllm_forward_context().cudagraph_runtime_mode
                )
                if mode is not None:
                    return mode
        except (ImportError, AttributeError, AssertionError):
            pass

    mode = _normalize_cudagraph_runtime_mode(
        getattr(get_forward_context(), "cudagraph_runtime_mode", None)
    )
    return mode if mode is not None else CUDAGraphMode.NONE


def set_forward_context(
    attn_metadata: AttentionMetaData,
    atom_config: Config,
    context: Context,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    spec_decode_metadata: SpecDecodeMetadata | None = None,
    ubatch_slices: list[Any] | None = None,
    in_hipgraph: bool = False,
    ub_max_tokens_across_dp: tuple | None = None,
) -> None:
    global _forward_context
    dp_metadata: DPMetadata | None = None
    if atom_config.parallel_config.data_parallel_size > 1 and num_tokens is not None:
        dp_metadata = DPMetadata.make(
            atom_config.parallel_config,
            # attn_metadata,
            num_tokens or 0,
            num_tokens_across_dp,
        )

    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        kv_cache_data=_forward_kv_cache_context.kv_cache_data,
        context=context,
        dp_metadata=dp_metadata,
        spec_decode_metadata=spec_decode_metadata,
        ubatch_slices=ubatch_slices,
        ub_max_tokens_across_dp=ub_max_tokens_across_dp,
        main_stream=(torch.cuda.current_stream() if _CUDA_AVAILABLE else None),
        in_hipgraph=in_hipgraph,
    )  # _forward_context.attn_metadata = attn_metadata
    # _forward_context.no_compile_layers = atom_config.compilation_config.static_forward_context
    # _forward_context = ForwardContext(no_compile_layers=atom_config.compilation_config.static_forward_context, attn_metadata=attn_metadata)


def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


# ---------------------------------------------------------------------------
# KV Connector global instances (lazy initialization)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("atom")

_global_kvconnector: Any | None = None
_global_kvconnector_scheduler: Any | None = None


def get_kvconnector(role: str = "worker", config: Config | None = None) -> Any:
    """Get or lazily initialize the global KV connector instance.

    The connector is role-dependent:
      - ``"worker"``: Returns a :class:`KVConnectorBase` (worker-side, per TP rank).
      - ``"scheduler"``: Returns a :class:`KVConnectorSchedulerBase` (scheduler-side).

    The concrete backend is selected by :class:`KVConnectorFactory` based on
    ``config.kv_transfer_config["kv_connector"]`` (default: ``"moriio"``).

    Args:
        role: Either ``"worker"`` or ``"scheduler"``.
        config: Engine config; required on first call to trigger initialization.

    Returns:
        The KV connector instance, or ``None`` if KV transfer is not configured.
    """
    global _global_kvconnector, _global_kvconnector_scheduler

    if not (hasattr(config, "kv_transfer_config") and config.kv_transfer_config):
        return _global_kvconnector

    if role == "worker":
        from aiter.dist.parallel_state import get_tp_group

        try:
            tp_rank = get_tp_group().rank_in_group
        except Exception:
            _logger.warning(
                "get_tp_group() failed (dist not initialized?), returning None"
            )
            return None

        if _global_kvconnector is None:
            from atom.kv_transfer.disaggregation import KVConnectorFactory

            _global_kvconnector = KVConnectorFactory.create_connector(
                config, role="worker"
            )
            _logger.debug("Initialized global KVConnector at tp_rank %d", tp_rank)

    elif role == "scheduler":
        from atom.kv_transfer.disaggregation import KVConnectorFactory

        _global_kvconnector_scheduler = KVConnectorFactory.create_connector(
            config, role="scheduler"
        )
        _logger.debug("Initialized global KVConnectorScheduler")
        return _global_kvconnector_scheduler

    else:
        raise ValueError(f"Unknown KV connector role: {role!r}")

    return _global_kvconnector


def set_kv_cache_data(
    kv_cache_data: dict[int, KVCacheTensor],
    config: Config | None = None,
    transfer_tensors: Any = None,
    num_blocks: int | None = None,
) -> None:
    """Register KV cache data globally and with the KV connector if enabled.

    ``num_blocks`` is the scheduler-visible KV block count (the ID space used
    by request block tables). The offload connector needs it to byte-slice
    MLA's token-major latent cache, where tensor.shape[0] is the page-size-1
    physical row count rather than the scheduler block count.
    """
    if hasattr(config, "kv_transfer_config") and config.kv_transfer_config:
        connector = get_kvconnector(config=config)
        if connector is not None:
            connector.register_kv_caches(
                kv_cache_data, transfer_tensors, num_blocks=num_blocks
            )

    _forward_kv_cache_context.kv_cache_data = kv_cache_data
