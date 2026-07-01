"""Scoped runtime adapter from SGLang batches to ATOM core."""

from __future__ import annotations

import copy
import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from atom.plugin.sglang.runtime.context import bind_current_forward_batch

logger = logging.getLogger("atom.plugin.sglang.runtime.forward_context")


def _is_dummy_forward(forward_batch: ForwardBatch) -> bool:
    """Return whether an SGLang batch represents an empty/idle dummy run."""

    forward_mode = getattr(forward_batch, "forward_mode", None)
    return bool(
        forward_mode is not None
        and hasattr(forward_mode, "is_idle")
        and forward_mode.is_idle()
    )


def _pad_dummy_like(
    tensor: Optional[torch.Tensor],
    *,
    length: int,
    fill_value: int | float = 0,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    shape = (length, *tensor.shape[1:])
    return torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)


def _materialize_atom_dummy_forward(
    input_ids: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    input_embeds: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    ForwardBatch,
]:
    """Convert an empty SGLang IDLE batch into ATOM-style dummy inputs."""

    if positions is None:
        raise RuntimeError("SGLang dummy forward materialization requires positions")
    if input_ids is None:
        raise RuntimeError("SGLang dummy forward materialization requires input_ids")

    dummy_positions = positions.new_zeros((1,))
    dummy_input_ids = input_ids.new_zeros((1,))
    dummy_input_embeds = _pad_dummy_like(input_embeds, length=1, fill_value=0)

    model_forward_batch = copy.copy(forward_batch)
    model_forward_batch.positions = dummy_positions
    model_forward_batch.batch_size = 1
    model_forward_batch.seq_lens_sum = 1
    model_forward_batch.seq_lens = forward_batch.seq_lens.new_ones((1,))
    model_forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu.new_ones((1,))

    return dummy_input_ids, dummy_positions, dummy_input_embeds, model_forward_batch


def _trim_hidden_states_for_output(hidden_states, num_tokens: int):
    if torch.is_tensor(hidden_states):
        return hidden_states[:num_tokens]
    if isinstance(hidden_states, tuple):
        return tuple(
            tensor[:num_tokens] if torch.is_tensor(tensor) else tensor
            for tensor in hidden_states
        )
    return hidden_states


def _resolve_num_tokens_across_dp(
    atom_config: Any,
    forward_batch: ForwardBatch,
    num_tokens: int,
    is_dummy_run: bool,
) -> torch.Tensor:
    """Resolve per-DP token counts for ATOM's CPU-side DPMetadata."""

    global_num_tokens_cpu = getattr(forward_batch, "global_num_tokens_cpu", None)
    if global_num_tokens_cpu is not None:
        num_tokens_across_dp = torch.tensor(
            global_num_tokens_cpu, dtype=torch.int32, device="cpu"
        )
    else:
        dp_size = atom_config.parallel_config.data_parallel_size
        global_num_tokens_gpu = getattr(forward_batch, "global_num_tokens_gpu", None)
        global_dp_buffer_len = getattr(forward_batch, "global_dp_buffer_len", None)
        is_static_same_shape_batch = (
            global_num_tokens_gpu is not None
            and global_dp_buffer_len == num_tokens * dp_size
        )
        if not is_static_same_shape_batch:
            raise RuntimeError(
                "[SGL+ATOM] SGLang dp-attention requires "
                "forward_batch.global_num_tokens_cpu unless the batch uses static "
                "same-shape DP metadata."
            )

        # Static batches, such as CUDA graph capture batches, may only keep
        # global token counts on GPU. Avoid GPU-to-CPU reads here and mirror
        # their same-shape layout directly for ATOM's CPU DPMetadata.
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens, dtype=torch.int32, device="cpu"
        )

    if is_dummy_run:
        # SGLang reports idle ranks as 0 tokens, but ATOM materializes them
        # as one local dummy token so collectives and DPMetadata stay aligned.
        dp_rank = atom_config.parallel_config.data_parallel_rank
        num_tokens_across_dp[dp_rank] = num_tokens
    return num_tokens_across_dp


def _slice_v4_graph_metadata_for_capture(
    attn_metadata: Any, *, num_tokens: int, bs: int
):
    """Narrow reusable V4 graph metadata to this capture bucket.

    The DSV4 fallback metadata is initialized at max graph size.  SGLang then
    captures smaller buckets (e.g. bs=248, tokens=496), so per-token arrays must
    be narrowed before model code reads them.
    """

    if attn_metadata is None:
        return None

    md = copy.copy(attn_metadata)

    def _slice_attr(name: str, n: int):
        value = getattr(md, name, None)
        if torch.is_tensor(value):
            setattr(md, name, value[:n])
        elif value is not None:
            try:
                setattr(md, name, value[:n])
            except Exception:
                pass

    for name in (
        "batch_id_per_token",
        "batch_id_per_token_cpu",
        "slot_mapping",
        "kv_indices_swa",
        "kv_indices_csa",
        "kv_indices_hca",
        "kv_indices_extend",
        "kv_indices_prefix_swa",
        "kv_indices_prefix_csa",
        "kv_indices_prefix_hca",
        "skip_prefix_len_csa",
    ):
        _slice_attr(name, num_tokens)

    for name in (
        "kv_indptr_swa",
        "kv_indptr_csa",
        "kv_indptr_hca",
        "kv_indptr_extend",
        "kv_indptr_prefix_swa",
        "kv_indptr_prefix_csa",
        "kv_indptr_prefix_hca",
    ):
        _slice_attr(name, num_tokens + 1)

    for name in (
        "state_slot_mapping",
        "state_slot_mapping_cpu",
        "n_committed_csa_per_seq",
        "n_committed_csa_per_seq_cpu",
        "n_committed_hca_per_seq",
        "n_committed_hca_per_seq_cpu",
        "context_lens",
    ):
        _slice_attr(name, bs)

    block_tables = getattr(md, "block_tables", None)
    if torch.is_tensor(block_tables):
        md.block_tables = block_tables[:bs]

    for name in ("cu_seqlens_q", "cu_seqlens_k"):
        _slice_attr(name, bs + 1)

    indexer_meta = getattr(md, "indexer_meta", None)
    if isinstance(indexer_meta, dict):
        indexer_meta = dict(indexer_meta)
        for key in (
            "batch_id_per_token_gpu",
            "seq_base_per_token_gpu",
            "cu_starts_gpu",
            "cu_ends_gpu",
        ):
            value = indexer_meta.get(key)
            if torch.is_tensor(value):
                indexer_meta[key] = value[:num_tokens]
        value = indexer_meta.get("n_committed_per_seq_gpu")
        if torch.is_tensor(value):
            indexer_meta["n_committed_per_seq_gpu"] = value[:bs]
        md.indexer_meta = indexer_meta

    return md


def _set_atom_forward_context(
    atom_config: Any,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
) -> None:
    """Bridge SGLang batch metadata into ATOM's global forward context."""

    from atom.utils.forward_context import (
        AttentionMetaData,
        Context,
        set_forward_context,
    )

    forward_mode = forward_batch.forward_mode
    # This value is only used by ATOM-side MoE padding in the SGLang wrapper.
    max_seqlen_q = 1 if forward_mode.is_decode_or_idle() else 0
    attn_metadata = None
    try:
        attn_metadata = getattr(forward_batch, "atom_v4_graph_metadata", None)
        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_attention_metadata_from_sglang,
            maybe_get_proxy_pool_from_sglang_backend,
        )

        if attn_metadata is None:
            try:
                from sglang.srt.model_executor.forward_context import get_attn_backend

                backend = get_attn_backend()
                attn_metadata = getattr(backend, "atom_v4_graph_metadata", None)
            except Exception:
                attn_metadata = None

        if attn_metadata is None:
            backend = getattr(forward_batch, "attn_backend", None)
            attn_metadata = getattr(backend, "atom_v4_graph_metadata", None)

        if attn_metadata is None and backend is not None:
            backend_forward_batch = getattr(backend, "forward_metadata", None)
            attn_metadata = getattr(
                backend_forward_batch, "atom_v4_graph_metadata", None
            )

        proxy_pool, req_to_token_pool = maybe_get_proxy_pool_from_sglang_backend()
        try:
            is_capture_batch = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            is_capture_batch = False
        if attn_metadata is None and is_capture_batch:
            try:
                from atom.plugin.sglang.attention_backend.deepseek_v4_backend import (
                    ATOMDeepseekV4BackendForSgl,
                )

                attn_metadata = ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata
                if attn_metadata is not None:
                    attn_metadata = _slice_v4_graph_metadata_for_capture(
                        attn_metadata,
                        num_tokens=int(positions.shape[0]),
                        bs=int(forward_batch.batch_size),
                    )
            except Exception:
                attn_metadata = None
        if attn_metadata is None and getattr(
            proxy_pool, "is_atom_v4_proxy_pool", False
        ):
            if is_capture_batch:
                raise RuntimeError(
                    "ATOM DeepSeek-V4 CUDA graph metadata was not initialized before capture"
                )
            else:
                attn_metadata = build_atom_v4_attention_metadata_from_sglang(
                    forward_batch,
                    positions,
                    proxy_pool=proxy_pool,
                    req_to_token_pool=req_to_token_pool,
                )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build ATOM DeepSeek-V4 metadata for SGLang"
        ) from exc

    if attn_metadata is None:
        attn_metadata = AttentionMetaData(max_seqlen_q=max_seqlen_q)
    batch_size = int(forward_batch.batch_size)
    is_dummy_run = _is_dummy_forward(forward_batch)
    is_prefill = forward_mode.is_prefill()
    num_tokens = int(positions.shape[0])

    if bool(atom_config.enable_dp_attention):
        num_tokens_across_dp = _resolve_num_tokens_across_dp(
            atom_config, forward_batch, num_tokens, is_dummy_run
        )
        graph_bs = int(torch.max(num_tokens_across_dp).item())
    else:
        num_tokens_across_dp = None
        graph_bs = num_tokens if is_prefill else batch_size

    context = Context(
        positions=positions,
        is_prefill=is_prefill,
        is_dummy_run=is_dummy_run,
        batch_size=batch_size,
        graph_bs=graph_bs,
    )
    set_forward_context(
        attn_metadata=attn_metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def _reset_atom_forward_context() -> None:
    from atom.utils.forward_context import reset_forward_context

    reset_forward_context()


@dataclass
class SGLangPluginRuntime:
    """Scoped adapter for running ATOM model code under SGLang plugin runtime.

    The adapter owns the temporary translation from SGLang's ``ForwardBatch`` to
    ATOM's process-local runtime state.  Callers should use the normalized
    ``input_ids``, ``positions``, ``input_embeds``, and ``forward_batch`` exposed
    by this object while inside the context.
    """

    atom_config: Any
    forward_batch: ForwardBatch
    positions: torch.Tensor
    input_ids: Optional[torch.Tensor] = None
    input_embeds: Optional[torch.Tensor] = None
    set_forward_context: bool = True
    _original_forward_batch: ForwardBatch = field(init=False, repr=False)
    _is_dummy_run: bool = field(init=False, default=False)
    _exit_stack: ExitStack = field(init=False, repr=False)

    def __enter__(self) -> "SGLangPluginRuntime":
        self._original_forward_batch = self.forward_batch
        self._is_dummy_run = _is_dummy_forward(self.forward_batch)

        if self._is_dummy_run:
            (
                self.input_ids,
                self.positions,
                self.input_embeds,
                self.forward_batch,
            ) = _materialize_atom_dummy_forward(
                self.input_ids,
                self.positions,
                self.input_embeds,
                self.forward_batch,
            )

        self._exit_stack = ExitStack()
        self._exit_stack.enter_context(bind_current_forward_batch(self.forward_batch))
        if self.set_forward_context:
            _set_atom_forward_context(
                self.atom_config,
                self.forward_batch,
                self.positions,
            )
            self._exit_stack.callback(_reset_atom_forward_context)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._exit_stack.close()

    def trim_output(self, hidden_states):
        """Map ATOM-visible outputs back to SGLang-visible token count."""

        if self._is_dummy_run:
            return _trim_hidden_states_for_output(hidden_states, 0)
        return hidden_states
