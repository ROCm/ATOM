from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import torch

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.paged_attention import Attention as PagedAttention
from atom.model_ops.attentions.gdn_attn import (
    GDNAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    _forward_kv_cache_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)


@dataclass
class AiterFlashAttentionPhaseMetadata:
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor


AiterFlashAttentionDecodeMetadata = AiterFlashAttentionPhaseMetadata
AiterFlashAttentionPrefillMetadata = AiterFlashAttentionPhaseMetadata


@dataclass
class AiterFlashAttentionMetadataForPluginMode:
    num_actual_tokens: int
    num_actual_kv_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    num_extends: int
    num_extend_tokens: int
    decode_metadata: AiterFlashAttentionPhaseMetadata | None = None
    prefill_metadata: AiterFlashAttentionPhaseMetadata | None = None
    extend_metadata: Any = None
    use_cascade: bool = False
    common_prefix_len: int = 0
    total_tokens: int = 0
    context: Any = None


@dataclass(frozen=True)
class RTPForwardContext:
    gdn_metadata: GDNAttentionMetadata
    attn_metadata: AttentionMetaData
    rtp_attn_inputs: Any
    rtp_seq_size_per_block: int
    rtp_kernel_seq_size_per_block: int
    kv_cache_data: Dict[str, KVCacheTensor]
    state_indices_cache: Dict[tuple[int, bool], torch.Tensor]
    layer_group_map: Dict[int, int]
    context: Context
    num_tokens: int
    LayerMaps = tuple[Dict[int, GatedDeltaNet], Dict[int, Any]]

    @staticmethod
    def _non_empty_int32(
        tensor: torch.Tensor | None, *, device: torch.device | None = None
    ) -> torch.Tensor | None:
        if tensor is None or tensor.numel() == 0:
            return None
        kwargs = {"dtype": torch.int32, "non_blocking": True}
        if device is not None:
            kwargs["device"] = device
        return tensor.to(**kwargs).contiguous()

    @staticmethod
    def _query_start_loc(attn_inputs: Any, *, device: torch.device) -> torch.Tensor:
        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        cu_seqlens = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "cu_seqlens", None),
            device=device,
        )
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            # Decode steps may carry placeholder [0, 0] cu_seqlens from upper layers.
            # Only trust cu_seqlens when it represents non-empty query tokens.
            # In cuda-graph capture the .item() host-sync would abort capture
            # (see rtp+atom_graph.md §2.4); under capture we always fall through
            # to the input_lengths-based path below.
            if not torch.cuda.is_current_stream_capturing() and bool(
                (cu_seqlens[-1] > 0).item()
            ):
                if (
                    input_lengths is not None
                    and cu_seqlens.numel() >= input_lengths.numel() + 1
                ):
                    return cu_seqlens[: input_lengths.numel() + 1]
                return cu_seqlens

        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            if input_lengths is None:
                raise ValueError(
                    "RTP plugin requires attention_inputs.cu_seqlens or input_lengths "
                    "to build GDN query_start_loc."
                )
            prefix = torch.zeros((1,), dtype=torch.int32, device=input_lengths.device)
            return torch.cat([prefix, input_lengths.cumsum(dim=0)], dim=0)

        # Decode: query length is runtime step token count (usually 1 per sequence),
        # not prompt input_lengths.
        sequence_lengths_plus_1 = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
            device=device,
        )
        sequence_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "sequence_lengths", None),
            device=device,
        )
        if (
            sequence_lengths_plus_1 is not None
            and sequence_lengths is not None
            and int(sequence_lengths_plus_1.numel()) == int(sequence_lengths.numel())
        ):
            q_lens = (sequence_lengths_plus_1 - sequence_lengths).contiguous()
            q_lens = torch.clamp(q_lens, min=1)
            prefix = torch.zeros((1,), dtype=torch.int32, device=q_lens.device)
            return torch.cat([prefix, q_lens.cumsum(dim=0)], dim=0)

        if input_lengths is None:
            raise ValueError(
                "RTP decode requires sequence_lengths(+1) or input_lengths "
                "to build GDN query_start_loc."
            )
        q_lens = torch.ones_like(
            input_lengths, dtype=torch.int32, device=input_lengths.device
        )
        prefix = torch.zeros((1,), dtype=torch.int32, device=input_lengths.device)
        return torch.cat([prefix, q_lens.cumsum(dim=0)], dim=0)

    @staticmethod
    def _state_indices(
        attn_inputs: Any,
        is_prefill: bool,
        *,
        device: torch.device,
        seq_size_per_block: int,
        group_id: int | None = None,
    ) -> torch.Tensor:
        block_table = RTPForwardContext._select_block_table_for_layer(
            attn_inputs=attn_inputs,
            group_id=group_id,
        )
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_kernel_block_id_device for GDN metadata."
            )
        if block_table.dim() == 1:
            block_table = block_table.unsqueeze(0)
        base = block_table.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if base.dim() != 2:
            raise ValueError(
                "RTP plugin produced invalid GDN state indices shape "
                f"(state_indices_shape={tuple(base.shape)})."
            )

        if seq_size_per_block <= 0:
            raise ValueError(
                f"RTP plugin got invalid seq_size_per_block={seq_size_per_block}."
            )
        if int(base.shape[0]) == 0 or int(base.shape[1]) == 0:
            raise ValueError("RTP decode requires non-empty GDN state indices.")

        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        if input_lengths is None:
            raise ValueError(
                "RTP plugin requires attention_inputs.input_lengths for GDN state indices."
            )
        if int(input_lengths.numel()) != int(base.shape[0]):
            raise ValueError(
                "RTP plugin input_lengths/block_table batch mismatch "
                f"(input_lengths={int(input_lengths.numel())}, block_table={int(base.shape[0])})."
            )

        if is_prefill:
            prefix_lengths = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "prefix_lengths_d", None),
                device=device,
            )
            if prefix_lengths is None:
                prefix_lengths = RTPForwardContext._non_empty_int32(
                    getattr(attn_inputs, "prefix_lengths", None),
                    device=device,
                )
            if prefix_lengths is None:
                raise ValueError(
                    "RTP prefill requires attention_inputs.prefix_lengths for GDN state indices."
                )
            if int(prefix_lengths.numel()) != int(base.shape[0]):
                raise ValueError(
                    "RTP plugin prefix_lengths/block_table batch mismatch "
                    f"(prefix_lengths={int(prefix_lengths.numel())}, block_table={int(base.shape[0])})."
                )
            last_token_idx = prefix_lengths + input_lengths - 1
        else:
            # RTP decode kernels use sequence_lengths_plus_1_d as canonical runtime value.
            sequence_lengths_plus_1 = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
                device=device,
            )
            if sequence_lengths_plus_1 is not None:
                if int(sequence_lengths_plus_1.numel()) != int(base.shape[0]):
                    raise ValueError(
                        "RTP plugin sequence_lengths_plus_1_d/block_table batch mismatch "
                        f"(sequence_lengths_plus_1_d={int(sequence_lengths_plus_1.numel())}, "
                        f"block_table={int(base.shape[0])})."
                    )
                last_token_idx = sequence_lengths_plus_1 - 1
            else:
                sequence_lengths = RTPForwardContext._non_empty_int32(
                    getattr(attn_inputs, "sequence_lengths", None),
                    device=device,
                )
                if sequence_lengths is None:
                    raise ValueError(
                        "RTP decode requires attention_inputs.sequence_lengths for GDN state indices."
                    )
                if int(sequence_lengths.numel()) != int(base.shape[0]):
                    raise ValueError(
                        "RTP plugin sequence_lengths/block_table batch mismatch "
                        f"(sequence_lengths={int(sequence_lengths.numel())}, block_table={int(base.shape[0])})."
                    )
                # Legacy fallback when sequence_lengths_plus_1_d is unavailable.
                last_token_idx = sequence_lengths + input_lengths - 1

        # Keep eager semantics strict (fail fast on malformed metadata).
        # CUDA-graph warmup/replay may temporarily feed placeholder
        # sequence_lengths_plus_1_d=0, so only graph-mode relaxes by clamping.
        in_capture = torch.cuda.is_current_stream_capturing()
        graph_mode = bool(getattr(attn_inputs, "is_cuda_graph", False))
        relaxed_validation = in_capture or graph_mode
        if relaxed_validation:
            last_token_idx = torch.clamp(last_token_idx, min=0)
        if not relaxed_validation and torch.any(last_token_idx < 0):
            raise ValueError(
                "RTP plugin produced negative token index for GDN state mapping."
            )
        block_col = torch.div(
            last_token_idx,
            int(seq_size_per_block),
            rounding_mode="floor",
        )
        # Only graph mode clamps out-of-range columns for warmup/replay safety.
        if relaxed_validation:
            block_col = torch.clamp(block_col, max=max(int(base.shape[1]) - 1, 0))
        if not relaxed_validation and (
            torch.any(block_col < 0) or torch.any(block_col >= base.shape[1])
        ):
            raise ValueError(
                "RTP plugin block-table index out of range for GDN state mapping "
                f"(max_col={int(base.shape[1]) - 1})."
            )
        row_idx = torch.arange(base.shape[0], device=device, dtype=torch.int64)
        slot_ids = base[row_idx, block_col.to(dtype=torch.int64)]
        if not relaxed_validation and torch.any(slot_ids < 0):
            raise ValueError(
                "RTP plugin resolved padded/invalid (-1) block slot for GDN state mapping."
            )
        return slot_ids.contiguous()

    @staticmethod
    def _select_block_table_for_layer(
        attn_inputs: Any,
        group_id: int | None = None,
    ) -> torch.Tensor | None:
        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is not None and len(by_group):
            gid = int(group_id) if group_id is not None else 0
            if gid < 0 or gid >= len(by_group):
                raise ValueError(
                    f"RTP plugin resolved invalid kv-cache group id {gid}."
                )
            return by_group[gid]
        return getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)

    @staticmethod
    def _build_layer_group_map(attn_inputs: Any) -> Dict[int, int]:
        layer_to_group = getattr(attn_inputs, "kv_cache_layer_to_group", None)
        if layer_to_group is None or int(layer_to_group.numel()) == 0:
            return {}
        layer_to_group_cpu = layer_to_group.detach().to(device="cpu")
        return {idx: int(gid) for idx, gid in enumerate(layer_to_group_cpu.tolist())}

    @staticmethod
    def _layer_group_map_signature(attn_inputs: Any) -> tuple[Any, ...]:
        layer_to_group = getattr(attn_inputs, "kv_cache_layer_to_group", None)
        if layer_to_group is None:
            return ("no_layer_to_group",)
        return (
            int(layer_to_group.data_ptr()),
            int(layer_to_group.numel()),
        )

    @staticmethod
    def _resolve_group_id(
        *,
        attn_inputs: Any,
        layer_num: int | None,
        layer_group_map: Dict[int, int] | None = None,
    ) -> int:
        by_group = getattr(
            attn_inputs, "kv_cache_kernel_block_id_device_by_group", None
        )
        if by_group is None or not len(by_group):
            return 0
        if layer_num is None:
            return 0
        if layer_group_map is not None and layer_num in layer_group_map:
            return int(layer_group_map[layer_num])
        return 0

    @staticmethod
    def state_indices_for_layer(
        *,
        attn_inputs: Any,
        is_prefill: bool,
        device: torch.device,
        seq_size_per_block: int,
        layer_num: int,
        state_indices_cache: Dict[tuple[int, bool], torch.Tensor] | None = None,
        layer_group_map: Dict[int, int] | None = None,
    ) -> torch.Tensor:
        group_id = RTPForwardContext._resolve_group_id(
            attn_inputs=attn_inputs,
            layer_num=layer_num,
            layer_group_map=layer_group_map,
        )
        cache_key = (int(group_id), bool(is_prefill))
        if state_indices_cache is not None:
            cached = state_indices_cache.get(cache_key)
            if cached is not None:
                return cached
        state_indices = RTPForwardContext._state_indices(
            attn_inputs=attn_inputs,
            is_prefill=is_prefill,
            device=device,
            seq_size_per_block=seq_size_per_block,
            group_id=group_id,
        )
        if state_indices_cache is not None:
            state_indices_cache[cache_key] = state_indices
        return state_indices

    @staticmethod
    def _build_gdn_metadata(
        attn_inputs: Any,
        *,
        seq_size_per_block: int,
        num_tokens: int,
        state_indices_cache: Dict[tuple[int, bool], torch.Tensor] | None = None,
        layer_group_map: Dict[int, int] | None = None,
    ) -> GDNAttentionMetadata:
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_kernel_block_id_device for GDN metadata."
            )
        target_device = block_table.device
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        query_start_loc = RTPForwardContext._query_start_loc(
            attn_inputs, device=target_device
        )
        state_indices = RTPForwardContext._state_indices(
            attn_inputs=attn_inputs,
            is_prefill=is_prefill,
            device=target_device,
            seq_size_per_block=seq_size_per_block,
        )
        if state_indices_cache is not None:
            group_id = RTPForwardContext._resolve_group_id(
                attn_inputs=attn_inputs,
                layer_num=None,
                layer_group_map=layer_group_map,
            )
            state_indices_cache[(int(group_id), bool(is_prefill))] = state_indices

        if is_prefill:
            prefix_lengths = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "prefix_lengths", None),
                device=target_device,
            )
            if prefix_lengths is None:
                raise ValueError(
                    "RTP prefill requires attention_inputs.prefix_lengths for GDN metadata."
                )
            has_initial_state = prefix_lengths > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc)
            )
            return GDNAttentionMetadata(
                num_prefills=int(prefix_lengths.numel()),
                num_prefill_tokens=num_tokens,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=num_tokens,
                has_initial_state=has_initial_state,
                spec_query_start_loc=None,
                non_spec_query_start_loc=query_start_loc,
                spec_state_indices_tensor=None,
                non_spec_state_indices_tensor=state_indices,
                spec_sequence_masks=None,
                spec_token_indx=None,
                non_spec_token_indx=None,
                num_accepted_tokens=None,
                nums_dict=nums_dict,
                batch_ptr=batch_ptr,
                token_chunk_offset_ptr=token_chunk_offset_ptr,
            )

        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=target_device,
        )
        if input_lengths is None:
            raise ValueError(
                "RTP decode requires attention_inputs.input_lengths to derive batch size."
            )
        batch_size = int(input_lengths.numel())
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=batch_size,
            num_decode_tokens=num_tokens,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_tokens,
            has_initial_state=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=state_indices,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=None,
            nums_dict=None,
            batch_ptr=None,
            token_chunk_offset_ptr=None,
        )

    @staticmethod
    def _build_seq_lens(attn_inputs: Any, *, device: torch.device) -> torch.Tensor:
        """Build kernel seq_lens using RTP-native field priority.

        Non-cuda-graph decode keeps the pre-cuda-graph field priority:
        sequence_lengths_plus_1_d first, then sequence_lengths + input_lengths.
        Cuda-graph warmup/replay keeps the graph-safe priority introduced for
        dummy inputs.
        """
        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        if input_lengths is None:
            raise ValueError(
                "RTP plugin requires attention_inputs.input_lengths for seq_lens."
            )
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if is_prefill:
            prefix_lengths = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "prefix_lengths_d", None),
                device=device,
            )
            if prefix_lengths is None:
                prefix_lengths = RTPForwardContext._non_empty_int32(
                    getattr(attn_inputs, "prefix_lengths", None),
                    device=device,
                )
            if prefix_lengths is None:
                raise ValueError(
                    "RTP prefill requires attention_inputs.prefix_lengths for seq_lens."
                )
            if int(prefix_lengths.numel()) != int(input_lengths.numel()):
                raise ValueError(
                    "RTP plugin prefix_lengths/input_lengths batch mismatch "
                    f"(prefix_lengths={int(prefix_lengths.numel())}, "
                    f"input_lengths={int(input_lengths.numel())})."
                )
            return (prefix_lengths + input_lengths).contiguous()

        non_cuda_graph_mode = not torch.cuda.is_current_stream_capturing() and not bool(
            getattr(attn_inputs, "is_cuda_graph", False)
        )
        if non_cuda_graph_mode:
            sequence_lengths_plus_1 = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
                device=device,
            )
            if sequence_lengths_plus_1 is not None:
                if int(sequence_lengths_plus_1.numel()) != int(input_lengths.numel()):
                    raise ValueError(
                        "RTP plugin sequence_lengths_plus_1_d/input_lengths batch mismatch "
                        f"(sequence_lengths_plus_1_d={int(sequence_lengths_plus_1.numel())}, "
                        f"input_lengths={int(input_lengths.numel())})."
                    )
                return sequence_lengths_plus_1.contiguous()

        sequence_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "sequence_lengths", None),
            device=device,
        )
        if sequence_lengths is not None:
            if int(sequence_lengths.numel()) != int(input_lengths.numel()):
                raise ValueError(
                    "RTP plugin sequence_lengths/input_lengths batch mismatch "
                    f"(sequence_lengths={int(sequence_lengths.numel())}, "
                    f"input_lengths={int(input_lengths.numel())})."
                )
            # Keep decode seq_lens semantics aligned with pure RTP/aiter path:
            # real context length is sequence_lengths + input_lengths.
            return (sequence_lengths + input_lengths).contiguous()

        if not non_cuda_graph_mode:
            sequence_lengths_plus_1 = RTPForwardContext._non_empty_int32(
                getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
                device=device,
            )
            if sequence_lengths_plus_1 is not None:
                if int(sequence_lengths_plus_1.numel()) != int(input_lengths.numel()):
                    raise ValueError(
                        "RTP plugin sequence_lengths_plus_1_d/input_lengths batch mismatch "
                        f"(sequence_lengths_plus_1_d={int(sequence_lengths_plus_1.numel())}, "
                        f"input_lengths={int(input_lengths.numel())})."
                    )
                return sequence_lengths_plus_1.contiguous()

        raise ValueError(
            "RTP decode requires attention_inputs.sequence_lengths_plus_1_d or "
            "sequence_lengths for seq_lens."
        )

    @staticmethod
    def _build_slot_mapping(
        *,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        seq_size_per_block: int,
        cg_bufs: dict | None = None,
    ) -> torch.Tensor:
        if positions is None or positions.numel() == 0:
            raise ValueError(
                "RTP plugin requires non-empty positions for slot_mapping."
            )
        if query_start_loc is None or query_start_loc.numel() < 2:
            raise ValueError(
                "RTP plugin requires valid query_start_loc for slot_mapping."
            )
        if block_table is None or block_table.numel() == 0:
            raise ValueError("RTP plugin requires block_table for slot_mapping.")
        if block_table.dim() == 1:
            block_table = block_table.unsqueeze(0)
        if block_table.dim() != 2:
            raise ValueError(
                f"RTP plugin invalid block_table shape for slot_mapping: {tuple(block_table.shape)}"
            )
        if seq_size_per_block <= 0:
            raise ValueError(
                f"RTP plugin got invalid seq_size_per_block={seq_size_per_block}."
            )

        device = positions.device
        dtype = torch.int32
        in_capture = torch.cuda.is_current_stream_capturing()

        # Capture path must not silently allocate via .to(...)/.contiguous().
        if in_capture and cg_bufs is not None:
            if positions.device != device or positions.dtype != dtype:
                raise RuntimeError(
                    "RTP plugin capture requires positions to already be int32 on model device."
                )
            if not positions.is_contiguous():
                raise RuntimeError(
                    "RTP plugin capture requires positions to be contiguous to avoid allocation."
                )
            if query_start_loc.device != device or query_start_loc.dtype != dtype:
                raise RuntimeError(
                    "RTP plugin capture requires query_start_loc to already be int32 on model device."
                )
            if not query_start_loc.is_contiguous():
                raise RuntimeError(
                    "RTP plugin capture requires query_start_loc to be contiguous to avoid allocation."
                )
            if block_table.device != device or block_table.dtype != dtype:
                raise RuntimeError(
                    "RTP plugin capture requires block_table to already be int32 on model device."
                )
            if not block_table.is_contiguous():
                raise RuntimeError(
                    "RTP plugin capture requires block_table to be contiguous to avoid allocation."
                )
            pos_i32 = positions
            qsl = query_start_loc
            bt = block_table
        else:
            pos_i32 = positions.to(
                device=device, dtype=dtype, non_blocking=True
            ).contiguous()
            qsl = query_start_loc.to(
                device=device, dtype=dtype, non_blocking=True
            ).contiguous()
            bt = block_table.to(
                device=device, dtype=dtype, non_blocking=True
            ).contiguous()

        batch_size = int(qsl.numel()) - 1
        num_tokens = int(pos_i32.numel())
        if batch_size <= 0:
            raise ValueError("RTP plugin query_start_loc produced empty batch.")
        if int(bt.shape[0]) != batch_size:
            raise ValueError(
                "RTP plugin block_table/query_start_loc batch mismatch "
                f"(block_table={int(bt.shape[0])}, batch={batch_size})."
            )
        validate_slot_mapping = os.getenv("ATOM_VALIDATE_SLOT_MAPPING", "0") == "1"
        if validate_slot_mapping and int(qsl[-1].item()) != num_tokens:
            raise ValueError(
                "RTP plugin query_start_loc/positions token mismatch "
                f"(query_start_loc[-1]={int(qsl[-1].item())}, positions={num_tokens})."
            )

        lengths = qsl[1:] - qsl[:-1]
        if validate_slot_mapping and torch.any(lengths <= 0):
            raise ValueError(
                "RTP plugin query_start_loc contains non-positive sequence length."
            )
        if in_capture and cg_bufs is not None:
            # Zero-alloc path: use pre-allocated buffers so captured GPU ops
            # reference stable addresses that stay alive through replay.
            # For decode (1 token/seq): seq_id[i] == i, pre-computed as arange.
            seq_id = cg_bufs["seq_id"][:num_tokens]
            block_col_buf = cg_bufs["block_col"][:num_tokens]
            torch.div(
                pos_i32,
                int(seq_size_per_block),
                rounding_mode="floor",
                out=block_col_buf,
            )
            block_col_i64_buf = cg_bufs["block_col_i64"][:num_tokens]
            block_col_i64_buf.copy_(block_col_buf)
            slot_base_buf = cg_bufs["slot_base"][:num_tokens]
            slot_base_buf.copy_(bt[seq_id, block_col_i64_buf])
            token_offset_buf = cg_bufs["token_offset"][:num_tokens]
            torch.remainder(pos_i32, int(seq_size_per_block), out=token_offset_buf)
            slot_mapping_buf = cg_bufs["slot_mapping"][:num_tokens]
            torch.add(
                slot_base_buf * int(seq_size_per_block),
                token_offset_buf,
                out=slot_mapping_buf,
            )
            return slot_mapping_buf
        elif in_capture:
            # cg_bufs not provided: fall back to searchsorted (capture-safe but
            # allocates transient tensors — may cause replay fault if GC'd).
            raise RuntimeError(
                "RTP plugin capture requires prewarmed cg_bufs; fallback allocation path is disabled."
            )
        else:
            seq_id = torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.int64),
                lengths.to(dtype=torch.int64),
            )
        if validate_slot_mapping and int(seq_id.numel()) != num_tokens:
            raise ValueError(
                "RTP plugin internal seq_id construction mismatch for slot_mapping."
            )

        block_col = torch.div(
            pos_i32,
            int(seq_size_per_block),
            rounding_mode="floor",
        )
        if validate_slot_mapping and (
            torch.any(block_col < 0) or torch.any(block_col >= bt.shape[1])
        ):
            raise ValueError(
                "RTP plugin block-table index out of range for full-attn slot_mapping "
                f"(max_col={int(bt.shape[1]) - 1})."
            )

        slot_base = bt[seq_id, block_col.to(dtype=torch.int64)]
        if validate_slot_mapping and torch.any(slot_base < 0):
            raise ValueError(
                "RTP plugin resolved padded/invalid (-1) block slot for full-attn slot_mapping."
            )
        token_offset = torch.remainder(pos_i32, int(seq_size_per_block))
        slot_mapping = slot_base * int(seq_size_per_block) + token_offset
        return slot_mapping.to(dtype=torch.int64).contiguous()

    @staticmethod
    def _build_query_start_loc_for_plugin(
        *,
        attn_inputs: Any,
        seq_lens: torch.Tensor,
        num_tokens: int,
        device: torch.device,
        cg_bufs: dict | None = None,
    ) -> torch.Tensor:
        batch_size = int(seq_lens.numel())
        if batch_size <= 0:
            raise ValueError(
                "RTP plugin cannot build query_start_loc with empty seq_lens."
            )

        in_capture = torch.cuda.is_current_stream_capturing()

        # In cuda-graph capture mode, every .tolist()/.item() blocks capture.
        # Decode-only capture path (Qwen3.5-MoE) always has num_tokens==batch_size
        # (1 token/seq), so query_start_loc == arange(0, bs+1).
        if in_capture and cg_bufs is not None:
            # Zero-alloc path: return a pre-allocated slice (stable address).
            return cg_bufs["query_start_loc"][: batch_size + 1]

        if in_capture:
            raise ValueError(
                "RTP plugin capture requires prewarmed cg_bufs for query_start_loc "
                f"(batch={batch_size}, num_tokens={int(num_tokens)})."
            )

        # Eager-mode validations (host sync allowed): keep prior semantics for
        # safety so the eager path catches malformed metadata early.
        qsl = RTPForwardContext._query_start_loc(attn_inputs, device=device)
        if qsl is not None and qsl.numel() == batch_size + 1:
            lengths = qsl[1:] - qsl[:-1]
            qsl_stats = torch.stack([qsl[-1], torch.min(lengths)], dim=0).to(
                device="cpu"
            )
            qsl_total_tokens, qsl_min_len = [int(v) for v in qsl_stats.tolist()]
            if qsl_total_tokens == int(num_tokens) and qsl_min_len > 0:
                return qsl.contiguous()

        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        if input_lengths is not None and int(input_lengths.numel()) == batch_size:
            input_stats = torch.stack(
                [torch.min(input_lengths), torch.sum(input_lengths)],
                dim=0,
            ).to(device="cpu")
            min_input_len, total_input_len = [int(v) for v in input_stats.tolist()]
            if min_input_len > 0 and total_input_len == int(num_tokens):
                prefix = torch.zeros((1,), dtype=torch.int32, device=device)
                return torch.cat(
                    [prefix, input_lengths.cumsum(dim=0)], dim=0
                ).contiguous()

        if int(num_tokens) == batch_size:
            prefix = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
            return prefix.contiguous()
        if batch_size == 1:
            return torch.tensor([0, int(num_tokens)], dtype=torch.int32, device=device)

        raise ValueError(
            "RTP plugin failed to build valid query_start_loc for plugin attention "
            f"(batch={batch_size}, num_tokens={int(num_tokens)})."
        )

    @staticmethod
    def _build_plugin_attention_metadata(
        *,
        attn_inputs: Any,
        positions: torch.Tensor,
        seq_size_per_block: int,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> AttentionMetaData:
        block_table = RTPForwardContext._select_block_table_for_layer(
            attn_inputs=attn_inputs,
        )
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_kernel_block_id_device for plugin attention metadata."
            )
        device = positions.device
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture and cg_bufs is None:
            raise RuntimeError(
                "RTP plugin capture requires prewarmed cg_bufs; metadata fallback path is disabled."
            )
        seq_lens = RTPForwardContext._build_seq_lens(attn_inputs, device=device)
        if in_capture and cg_bufs is not None:
            bs_now = int(seq_lens.shape[0])
            seq_lens_buf = cg_bufs["seq_lens_i32"]
            if int(seq_lens_buf.shape[0]) < bs_now:
                raise RuntimeError(
                    "RTP plugin prewarmed seq_lens_i32 buffer is too small "
                    f"(buffer={int(seq_lens_buf.shape[0])}, required={bs_now})."
                )
            seq_lens_view = seq_lens_buf[:bs_now]
            seq_lens_view.copy_(seq_lens, non_blocking=True)
            seq_lens = seq_lens_view
        else:
            seq_lens = seq_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            ).contiguous()
        batch_size = int(seq_lens.numel())

        # During RTP CUDA graph capture, positions is the full preallocated
        # buffer (CONCURRENCY_LIMIT * MAX_SEQ_LEN elements). For decode (1
        # token per seq) only the first batch_size positions are active —
        # slice here so slot_mapping and num_actual_tokens are correctly sized.
        if in_capture and not is_prefill:
            positions = positions[:batch_size]
        num_actual_tokens = int(positions.numel())

        query_start_loc = RTPForwardContext._build_query_start_loc_for_plugin(
            attn_inputs=attn_inputs,
            seq_lens=seq_lens,
            num_tokens=num_actual_tokens,
            device=device,
            cg_bufs=cg_bufs,
        )
        slot_mapping = RTPForwardContext._build_slot_mapping(
            positions=positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            seq_size_per_block=seq_size_per_block,
            cg_bufs=cg_bufs,
        )

        is_dummy_warmup = False
        if in_capture:
            # Cuda-graph capture path: cannot host-sync. Decode capture (Qwen3.5-MoE
            # decode-only graph, num_tokens_per_bs=1) has fixed per-step query
            # length = 1. max_seq_len comes from the runtime prewarm budget so
            # the kernel-side max_num_partitions = (max_seq_len + 255) // 256
            # matches what RTPFullAttention.prewarm_for_cuda_graph allocated.
            # num_actual_kv_tokens is informational; an upper bound is fine.
            max_query_len = 1
            if cg_max_seq_len <= 0:
                raise RuntimeError(
                    "RTP plugin cuda-graph capture requires cg_max_seq_len; "
                    "did you forget to thread it through RTPForwardContext.bind?"
                )
            max_seq_len = int(cg_max_seq_len)
            num_actual_kv_tokens = max_seq_len * batch_size
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            stats = torch.stack(
                [
                    torch.max(query_lens),
                    torch.max(seq_lens),
                    torch.sum(seq_lens),
                ],
                dim=0,
            ).to(device="cpu")
            max_query_len, max_seq_len, num_actual_kv_tokens = [
                int(v) for v in stats.tolist()
            ]
            # RTP's `initCapture forward for output datatype` probe feeds dummy
            # seq_lens=[0,...] / block_tables=[0,...]. The probe's only purpose
            # is to discover the output dtype — it never reads valid KV history,
            # so running a real attention kernel on those zeros is meaningless
            # and unsafe (aiter.paged_attention_rocm pre-fetches block_tables /
            # KV slots before bounds-checking context_len, → page fault). Mark
            # the metadata so RTPFullAttention can short-circuit to zeros.
            if max_seq_len <= 0:
                is_dummy_warmup = True
                if cg_max_seq_len > 0:
                    max_seq_len = int(cg_max_seq_len)
                else:
                    max_seq_len = 1
            if max_query_len <= 0:
                max_query_len = 1

        decode_md = None
        prefill_md = None
        if is_prefill:
            prefill_md = AiterFlashAttentionPrefillMetadata(
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                query_start_loc=query_start_loc,
            )
        else:
            decode_md = AiterFlashAttentionDecodeMetadata(
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                query_start_loc=query_start_loc,
            )

        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture and cg_bufs is not None:
            # Zero-alloc capture path: always route through prewarmed block_table_i32.
            bt_buf = cg_bufs["block_table_i32"]
            bs_now = int(block_table.shape[0])
            cols_now = int(block_table.shape[1])
            if int(bt_buf.shape[0]) < bs_now or int(bt_buf.shape[1]) < cols_now:
                raise RuntimeError(
                    "RTP plugin prewarmed block_table_i32 buffer is too small "
                    f"(buffer={tuple(bt_buf.shape)}, required=({bs_now}, {cols_now}))."
                )
            bt_view = bt_buf[:bs_now, :cols_now]
            bt_view.copy_(block_table, non_blocking=True)
            block_table_i32 = bt_view
        else:
            block_table_i32 = block_table.to(
                device=device, dtype=torch.int32, non_blocking=True
            ).contiguous()
        plugin_md = AiterFlashAttentionMetadataForPluginMode(
            num_actual_tokens=num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            block_table=block_table_i32,
            num_decodes=0 if is_prefill else batch_size,
            num_decode_tokens=0 if is_prefill else num_actual_tokens,
            num_prefills=batch_size if is_prefill else 0,
            num_prefill_tokens=num_actual_tokens if is_prefill else 0,
            num_extends=0,
            num_extend_tokens=0,
            decode_metadata=decode_md,
            prefill_metadata=prefill_md,
            extend_metadata=None,
            use_cascade=False,
            common_prefix_len=0,
            total_tokens=0,
            context=None,
        )
        # Prefill-only fields shared across all full-attn layers in the step.
        plugin_md.rtp_cu_seqlens_q = query_start_loc
        # Mark dummy probe (RTP initCapture's "forward for output datatype" feeds
        # all-zero seq_lens/block_tables); RTPFullAttention short-circuits to zeros.
        plugin_md.is_dummy_warmup = bool(is_dummy_warmup)
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if (
            prefix_lengths is not None
            and int(prefix_lengths.numel()) > 0
            and not in_capture
        ):
            # .item() is host-sync; skip during capture. rtp_has_prefix is only
            # consulted on the prefill branch and Qwen3.5-MoE decode-graph capture
            # never hits has_prefix=True (decode never has fresh prefix tokens).
            plugin_md.rtp_has_prefix = bool((prefix_lengths > 0).any().item())
        else:
            plugin_md.rtp_has_prefix = False
        attn_metadata = AttentionMetaData(
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            block_tables=plugin_md.block_table,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
        )
        attn_metadata.plugin_metadata = plugin_md
        return attn_metadata

    @staticmethod
    def collect_layer_maps(model: Any) -> LayerMaps:
        gdn_layer_map: Dict[int, GatedDeltaNet] = {}
        full_attn_layer_map: Dict[int, Any] = {}
        rtp_attention_cls: type[Any] | None = None
        try:
            from atom.plugin.rtpllm.attention_backend import AttentionForRTPLLM

            rtp_attention_cls = AttentionForRTPLLM
        except (ImportError, ModuleNotFoundError):
            rtp_attention_cls = None

        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                gdn_layer_map[int(module.layer_num)] = module
            elif isinstance(module, (PagedAttention, PagedAttentionImpl)) or (
                rtp_attention_cls is not None and isinstance(module, rtp_attention_cls)
            ):
                impl = getattr(module, "impl", None)
                layer_num = getattr(impl, "layer_num", None)
                if layer_num is None:
                    layer_num = getattr(module, "layer_num", None)
                if layer_num is not None:
                    full_attn_layer_map[int(layer_num)] = module
        return gdn_layer_map, full_attn_layer_map

    @staticmethod
    def _build_kv_cache_tensors(
        runtime: Any,
        layer_maps: LayerMaps,
    ) -> Dict[str, KVCacheTensor]:
        if runtime.kv_cache is None:
            raise ValueError("RTP plugin requires initialized kv_cache for ATOM model.")

        gdn_layer_map, full_attn_layer_map = layer_maps

        if not gdn_layer_map and not full_attn_layer_map:
            return {}

        cache_tensors: Dict[str, KVCacheTensor] = {}

        # Build GDN cache views from RTP LayerKVCache flat buffers.
        for layer_num, gdn_layer in gdn_layer_map.items():
            layer_cache = runtime.kv_cache.get_layer_cache(layer_num)
            kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
            if kv_cache_base is None:
                raise ValueError(f"Layer {layer_num} kv_cache_base is missing.")

            cache_base = kv_cache_base.reshape(kv_cache_base.shape[0], -1)
            # IMPORTANT: derive GDN cache layout from sharded ATOM module tensors.
            # This keeps RTP plugin aligned with the actual per-rank runtime shape.
            conv_kernel = int(gdn_layer.conv1d.weight.size(2))
            qkv_size = int(gdn_layer.conv1d.weight.size(0))
            local_num_v_heads = int(gdn_layer.dt_bias.numel())
            ssm_state_size = int(
                local_num_v_heads * gdn_layer.head_v_dim * gdn_layer.head_k_dim
            )
            conv_state_size = int((conv_kernel - 1) * qkv_size)
            total_needed = ssm_state_size + conv_state_size
            if cache_base.shape[1] < total_needed:
                raise ValueError(
                    f"Layer {layer_num} kv cache shape is invalid for GDN "
                    f"(have={cache_base.shape[1]}, need={total_needed}, "
                    f"qkv={qkv_size}, conv_kernel={conv_kernel}, "
                    f"local_v_heads={local_num_v_heads}, head_v_dim={gdn_layer.head_v_dim}, "
                    f"head_k_dim={gdn_layer.head_k_dim})."
                )

            conv_state = torch.as_strided(
                cache_base,
                (cache_base.shape[0], qkv_size, conv_kernel - 1),
                (cache_base.stride()[0], 1, qkv_size),
                storage_offset=ssm_state_size + cache_base.storage_offset(),
            )
            ssm_state = torch.as_strided(
                cache_base,
                (
                    cache_base.shape[0],
                    local_num_v_heads,
                    gdn_layer.head_v_dim,
                    gdn_layer.head_k_dim,
                ),
                (
                    cache_base.stride()[0],
                    gdn_layer.head_k_dim * gdn_layer.head_v_dim,
                    gdn_layer.head_k_dim,
                    1,
                ),
                storage_offset=cache_base.storage_offset(),
            )

            cache_tensors[f"layer_{layer_num}"] = KVCacheTensor(
                layer_num=layer_num,
                k_cache=conv_state,
                v_cache=ssm_state,
                k_scale=None,
                v_scale=None,
            )

        # Build full-attn cache references from RTP LayerKVCache.
        # Keep raw RTP layout here (no reshape/repack) and normalize layout
        # in the rtpllm attention patch at call time.
        for layer_num in full_attn_layer_map.keys():
            layer_key = f"layer_{layer_num}"
            if layer_key in cache_tensors:
                continue

            layer_cache = runtime.kv_cache.get_layer_cache(layer_num)
            kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
            if kv_cache_base is None:
                raise ValueError(
                    f"Layer {layer_num} kv_cache_base is missing for full-attn cache."
                )
            if kv_cache_base.dim() < 1:
                raise ValueError(
                    f"Layer {layer_num} full-attn kv_cache_base has invalid shape "
                    f"{tuple(kv_cache_base.shape)}."
                )
            cache_tensors[layer_key] = KVCacheTensor(
                layer_num=layer_num,
                # Keep full LayerKVCache object so the attention bridge can
                # call RTP-native paths without rebuilding pseudo caches.
                k_cache=layer_cache,
                v_cache=None,
                k_scale=None,
                v_scale=None,
            )
        return cache_tensors

    @staticmethod
    def _kv_cache_signature(
        runtime: Any,
        layer_maps: LayerMaps,
    ) -> Tuple[Any, ...]:
        if runtime.kv_cache is None:
            return ("no_kv_cache",)
        gdn_layer_map, full_attn_layer_map = layer_maps
        signature: list[Any] = [id(runtime.kv_cache)]
        all_layer_nums = sorted(
            set(gdn_layer_map.keys()) | set(full_attn_layer_map.keys())
        )
        for layer_num in all_layer_nums:
            layer_cache = runtime.kv_cache.get_layer_cache(layer_num)
            kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
            if kv_cache_base is None:
                signature.append((int(layer_num), None))
                continue
            signature.append(
                (
                    int(layer_num),
                    int(kv_cache_base.data_ptr()),
                    int(kv_cache_base.numel()),
                )
            )
        return tuple(signature)

    @classmethod
    def build(
        cls,
        model: Any,
        runtime: Any,
        inputs: Any,
        positions: torch.Tensor,
        layer_maps: LayerMaps | None = None,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> "RTPForwardContext":
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError(
                "RTP plugin requires inputs.attention_inputs for forward context."
            )

        if runtime.kv_cache is None:
            raise ValueError(
                "RTP plugin requires initialized kv_cache for forward context."
            )
        seq_size_per_block = int(getattr(runtime.kv_cache, "seq_size_per_block", 0))
        kernel_seq_size_per_block = int(
            getattr(runtime.kv_cache, "kernel_seq_size_per_block", 0)
        )
        if kernel_seq_size_per_block <= 0:
            kernel_seq_size_per_block = int(seq_size_per_block)
        state_indices_cache: Dict[tuple[int, bool], torch.Tensor] = {}
        layer_group_map_signature = cls._layer_group_map_signature(attn_inputs)
        layer_group_map = getattr(runtime, "_rtp_layer_group_map", None)
        cached_layer_group_map_signature = getattr(
            runtime, "_rtp_layer_group_map_signature", None
        )
        if (
            layer_group_map is None
            or cached_layer_group_map_signature != layer_group_map_signature
        ):
            layer_group_map = cls._build_layer_group_map(attn_inputs)
            runtime._rtp_layer_group_map = layer_group_map
            runtime._rtp_layer_group_map_signature = layer_group_map_signature
        gdn_metadata = cls._build_gdn_metadata(
            attn_inputs,
            seq_size_per_block=seq_size_per_block,
            num_tokens=int(positions.numel()),
            state_indices_cache=state_indices_cache,
            layer_group_map=layer_group_map,
        )
        # Keep raw RTP attention inputs in metadata so GDN can resolve per-layer
        # block-map/state-index semantics (same idea as RTP's select_block_map_for_layer).
        gdn_metadata.rtp_attn_inputs = attn_inputs
        gdn_metadata.rtp_seq_size_per_block = int(seq_size_per_block)
        gdn_metadata.rtp_state_indices_cache = state_indices_cache
        gdn_metadata.rtp_layer_group_map = layer_group_map
        attn_metadata = cls._build_plugin_attention_metadata(
            attn_inputs=attn_inputs,
            positions=positions,
            seq_size_per_block=kernel_seq_size_per_block,
            cg_max_seq_len=int(cg_max_seq_len),
            cg_bufs=cg_bufs,
        )
        resolved_layer_maps = layer_maps or cls.collect_layer_maps(model)
        kv_cache_signature = cls._kv_cache_signature(
            runtime=runtime,
            layer_maps=resolved_layer_maps,
        )
        kv_cache_data = getattr(runtime, "_rtp_kv_cache_data", None)
        cached_signature = getattr(runtime, "_rtp_kv_cache_signature", None)
        if kv_cache_data is None or cached_signature != kv_cache_signature:
            kv_cache_data = cls._build_kv_cache_tensors(
                runtime=runtime,
                layer_maps=resolved_layer_maps,
            )
            runtime._rtp_kv_cache_data = kv_cache_data
            runtime._rtp_kv_cache_signature = kv_cache_signature
        batch_size = int(attn_metadata.plugin_metadata.num_prefills)
        if batch_size <= 0:
            batch_size = int(attn_metadata.plugin_metadata.num_decodes)
        if batch_size <= 0:
            raise ValueError("RTP plugin failed to derive non-zero batch size.")
        context = Context(
            positions=positions,
            is_prefill=bool(getattr(attn_inputs, "is_prefill", False)),
            batch_size=batch_size,
            graph_bs=batch_size,
        )
        return cls(
            gdn_metadata=gdn_metadata,
            attn_metadata=attn_metadata,
            rtp_attn_inputs=attn_inputs,
            rtp_seq_size_per_block=int(seq_size_per_block),
            rtp_kernel_seq_size_per_block=int(kernel_seq_size_per_block),
            kv_cache_data=kv_cache_data,
            state_indices_cache=state_indices_cache,
            layer_group_map=layer_group_map,
            context=context,
            num_tokens=int(positions.numel()),
        )

    @classmethod
    @contextmanager
    def bind(
        cls,
        *,
        model: Any,
        runtime: Any,
        inputs: Any,
        positions: torch.Tensor,
        layer_maps: LayerMaps | None = None,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> Iterator[None]:
        forward_context = cls.build(
            model=model,
            runtime=runtime,
            inputs=inputs,
            positions=positions,
            layer_maps=layer_maps,
            cg_max_seq_len=cg_max_seq_len,
            cg_bufs=cg_bufs,
        )
        prev_kv = _forward_kv_cache_context.kv_cache_data
        attn_md = forward_context.attn_metadata
        attn_md.gdn_metadata = forward_context.gdn_metadata
        attn_md.rtp_attn_inputs = forward_context.rtp_attn_inputs
        attn_md.rtp_kernel_seq_size_per_block = (
            forward_context.rtp_kernel_seq_size_per_block
        )
        attn_md.rtp_layer_group_map = forward_context.layer_group_map
        try:
            set_kv_cache_data(forward_context.kv_cache_data)
            set_forward_context(
                attn_metadata=attn_md,
                atom_config=get_current_atom_config(),
                context=forward_context.context,
                num_tokens=forward_context.num_tokens,
            )
            yield
        finally:
            reset_forward_context()
            set_kv_cache_data(prev_kv if prev_kv is not None else {})
