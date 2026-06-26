from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

import torch
from aiter import dtypes

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attention_gdn import GatedDeltaNet

try:
    from atom.model_ops.attention_mha import PagedAttentionImpl
except (ImportError, ModuleNotFoundError):
    PagedAttentionImpl = type("PagedAttentionImpl", (), {})
try:
    from atom.model_ops.paged_attention import Attention as PagedAttention
except (ImportError, ModuleNotFoundError):
    try:
        from atom.model_ops.paged_attention import PagedAttention
    except (ImportError, ModuleNotFoundError):
        PagedAttention = type("PagedAttention", (), {})
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


if triton is not None:

    @triton.jit
    def _expand_block_table_for_atom_indexer_kernel(
        block_table,
        output,
        num_cols: tl.constexpr,
        output_cols: tl.constexpr,
        block_ratio: tl.constexpr,
        BLOCK_RATIO: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_RATIO)
        value = tl.load(block_table + row * num_cols + col)
        expanded = value * block_ratio + offsets
        expanded = tl.where(value >= 0, expanded, -1)
        tl.store(output + row * output_cols + col * block_ratio + offsets, expanded)

    @triton.jit
    def _recover_physical_block_table_from_kernel_kernel(
        kernel_block_table,
        output,
        kernel_cols: tl.constexpr,
        physical_cols: tl.constexpr,
        block_ratio: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1)
        kernel_col = col * block_ratio
        value = tl.load(
            kernel_block_table + row * kernel_cols + kernel_col,
            mask=kernel_col < kernel_cols,
            other=-1,
        )
        physical = value // block_ratio
        physical = tl.where(value >= 0, physical, -1)
        tl.store(output + row * physical_cols + col, physical)


@dataclass(frozen=True)
class RTPForwardContext:
    gdn_metadata: GDNAttentionMetadata | None
    attn_metadata: AttentionMetaData
    rtp_attn_inputs: Any
    rtp_seq_size_per_block: int
    rtp_kernel_seq_size_per_block: int
    kv_cache_data: Dict[str, KVCacheTensor]
    state_indices_cache: Dict[tuple[int, bool], torch.Tensor]
    layer_group_map: Dict[int, int]
    context: Context
    num_tokens: int
    mla_layer_map: Dict[int, Any]
    LayerMaps = tuple[Dict[int, GatedDeltaNet], Dict[int, Any], Dict[int, Any]]

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
    def _recover_physical_block_table_from_kernel(
        kernel_block_table: torch.Tensor,
        *,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_bufs: dict | None = None,
    ) -> torch.Tensor:
        if (
            kernel_seq_size_per_block <= 0
            or seq_size_per_block <= 0
            or seq_size_per_block == kernel_seq_size_per_block
        ):
            return kernel_block_table
        if seq_size_per_block % kernel_seq_size_per_block != 0:
            raise ValueError(
                "RTP plugin cannot recover physical block_table from kernel block_table: "
                f"seq_size_per_block={seq_size_per_block}, "
                f"kernel_seq_size_per_block={kernel_seq_size_per_block}."
            )
        if kernel_block_table.dim() == 1:
            kernel_block_table = kernel_block_table.unsqueeze(0)
        if kernel_block_table.dim() != 2:
            raise ValueError(
                "RTP plugin invalid kernel block_table shape for physical recovery: "
                f"{tuple(kernel_block_table.shape)}"
            )
        block_ratio = int(seq_size_per_block // kernel_seq_size_per_block)
        bs_now = int(kernel_block_table.shape[0])
        kernel_cols = int(kernel_block_table.shape[1])
        if kernel_cols < block_ratio or kernel_cols % block_ratio != 0:
            return kernel_block_table.to(
                device=kernel_block_table.device, dtype=torch.int32, non_blocking=True
            ).contiguous()
        physical_cols = (kernel_cols + block_ratio - 1) // block_ratio
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture and cg_bufs is not None:
            if triton is None:
                raise RuntimeError(
                    "RTP plugin cuda-graph capture requires Triton for capture-safe "
                    "physical block_table recovery."
                )
            out_buf = cg_bufs.get("physical_block_table_i32")
            if not isinstance(out_buf, torch.Tensor):
                raise RuntimeError(
                    "RTP plugin capture requires prewarmed physical_block_table_i32."
                )
            if int(out_buf.shape[0]) < bs_now or int(out_buf.shape[1]) < physical_cols:
                raise RuntimeError(
                    "RTP plugin prewarmed block_table_i32 buffer is too small for "
                    "physical recovery "
                    f"(buffer={tuple(out_buf.shape)}, required=({bs_now}, {physical_cols}))."
                )
            out_view = out_buf[:bs_now, :physical_cols]
            _recover_physical_block_table_from_kernel_kernel[(bs_now, physical_cols)](
                kernel_block_table,
                out_view,
                kernel_cols,
                physical_cols,
                block_ratio,
            )
            return out_view

        sampled = kernel_block_table[:, : physical_cols * block_ratio : block_ratio]
        recovered = torch.div(sampled, block_ratio, rounding_mode="floor")
        recovered = torch.where(sampled >= 0, recovered, sampled)
        return recovered.to(
            device=kernel_block_table.device, dtype=torch.int32, non_blocking=True
        ).contiguous()

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

        Decode uses RTP's canonical sequence_lengths_plus_1_d first in both
        eager and CUDA-graph paths. This keeps context_lens aligned with the
        block-table slot/state-index calculation during graph replay.
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
            # For chunked prefill, prefix_lengths can remain per-chunk while
            # sequence_lengths_plus_1_d tracks the true cumulative context length.
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
        lengths = qsl[1:] - qsl[:-1]
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

        block_col = torch.div(
            pos_i32,
            int(seq_size_per_block),
            rounding_mode="floor",
        )

        slot_base = bt[seq_id, block_col.to(dtype=torch.int64)]
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
    def _build_req_id_per_token(
        *,
        query_start_loc: torch.Tensor,
        num_tokens: int,
        device: torch.device,
        cg_bufs: dict | None = None,
    ) -> torch.Tensor:
        batch_size = int(query_start_loc.numel()) - 1
        if batch_size <= 0:
            raise ValueError(
                "RTP plugin cannot build req_id_per_token for empty batch."
            )
        in_capture = torch.cuda.is_current_stream_capturing()
        if cg_bufs is not None and "seq_id_i32" in cg_bufs:
            seq_id_i32 = cg_bufs["seq_id_i32"]
            if not isinstance(seq_id_i32, torch.Tensor):
                raise RuntimeError(
                    "RTP plugin capture requires prewarmed seq_id_i32 tensor."
                )
            if int(seq_id_i32.shape[0]) < int(num_tokens):
                raise RuntimeError(
                    "RTP plugin prewarmed seq_id_i32 buffer is too small "
                    f"(buffer={int(seq_id_i32.shape[0])}, required={int(num_tokens)})."
                )
            if seq_id_i32.device != device or seq_id_i32.dtype != torch.int32:
                raise RuntimeError(
                    "RTP plugin capture requires seq_id_i32 to be int32 on model device."
                )
            if not seq_id_i32.is_contiguous():
                raise RuntimeError(
                    "RTP plugin capture requires seq_id_i32 to be contiguous."
                )
            return seq_id_i32[:num_tokens]
        if in_capture:
            raise RuntimeError(
                "RTP plugin capture requires prewarmed seq_id_i32 for req_id_per_token."
            )
        if int(num_tokens) == 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(dtype=torch.int64)
        if not torch.cuda.is_current_stream_capturing() and int(
            lengths.sum().item()
        ) != int(num_tokens):
            raise ValueError(
                "RTP plugin query_start_loc/num_tokens mismatch for req_id_per_token "
                f"(query_start_loc[-1]={int(query_start_loc[-1].item())}, "
                f"num_tokens={int(num_tokens)})."
            )
        return torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.int32),
            lengths,
        ).contiguous()

    @staticmethod
    def _expand_block_table_for_atom_indexer(
        block_table: torch.Tensor,
        *,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
    ) -> torch.Tensor:
        if (
            kernel_seq_size_per_block <= 0
            or seq_size_per_block <= 0
            or seq_size_per_block == kernel_seq_size_per_block
        ):
            return block_table
        if seq_size_per_block % kernel_seq_size_per_block != 0:
            raise ValueError(
                "RTP plugin cannot expand block_table for ATOM indexer: "
                f"seq_size_per_block={seq_size_per_block}, "
                f"kernel_seq_size_per_block={kernel_seq_size_per_block}."
            )
        block_ratio = int(seq_size_per_block // kernel_seq_size_per_block)
        offsets = torch.arange(
            block_ratio, device=block_table.device, dtype=torch.int32
        )
        base = block_table.to(dtype=torch.int32)
        expanded = base.unsqueeze(-1) * block_ratio + offsets
        expanded = torch.where(base.unsqueeze(-1) >= 0, expanded, -1)
        return expanded.reshape(base.shape[0], base.shape[1] * block_ratio).contiguous()

    @staticmethod
    def _expand_block_table_for_atom_indexer_capture(
        block_table: torch.Tensor,
        *,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_bufs: dict,
    ) -> torch.Tensor:
        if (
            kernel_seq_size_per_block <= 0
            or seq_size_per_block <= 0
            or seq_size_per_block == kernel_seq_size_per_block
        ):
            return block_table
        if seq_size_per_block % kernel_seq_size_per_block != 0:
            raise ValueError(
                "RTP plugin cannot expand block_table for ATOM indexer: "
                f"seq_size_per_block={seq_size_per_block}, "
                f"kernel_seq_size_per_block={kernel_seq_size_per_block}."
            )
        if triton is None:
            raise RuntimeError(
                "RTP plugin cuda-graph capture requires Triton for capture-safe "
                "ATOM indexer block_table expansion."
            )
        out_buf = cg_bufs.get("indexer_block_table_i32")
        if not isinstance(out_buf, torch.Tensor):
            raise RuntimeError(
                "RTP plugin capture requires prewarmed indexer_block_table_i32."
            )
        block_ratio = int(seq_size_per_block // kernel_seq_size_per_block)
        bs_now = int(block_table.shape[0])
        cols_now = int(block_table.shape[1])
        expanded_cols = cols_now * block_ratio
        if int(out_buf.shape[0]) < bs_now or int(out_buf.shape[1]) < expanded_cols:
            raise RuntimeError(
                "RTP plugin prewarmed indexer_block_table_i32 buffer is too small "
                f"(buffer={tuple(out_buf.shape)}, required=({bs_now}, {expanded_cols}))."
            )
        out_view = out_buf[:bs_now, :expanded_cols]
        _expand_block_table_for_atom_indexer_kernel[(bs_now, cols_now)](
            block_table,
            out_view,
            cols_now,
            expanded_cols,
            block_ratio,
            BLOCK_RATIO=block_ratio,
        )
        return out_view

    @classmethod
    def _build_indexer_block_tables(
        cls,
        *,
        block_table_i32: torch.Tensor,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_max_seq_len: int,
        in_capture: bool,
        cg_bufs: dict | None,
    ) -> torch.Tensor:
        del (
            cls,
            seq_size_per_block,
            kernel_seq_size_per_block,
            cg_max_seq_len,
            in_capture,
            cg_bufs,
        )
        # Base path (e.g. Qwen3.5): keep compact physical table layout and do not
        # expand to indexer granularity.
        return block_table_i32

    @classmethod
    def _resolve_plugin_block_table(
        cls,
        *,
        attn_inputs: Any,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_bufs: dict | None,
        in_capture: bool,
    ) -> torch.Tensor | None:
        physical_block_table = getattr(attn_inputs, "kv_cache_block_id_device", None)
        if physical_block_table is not None and physical_block_table.numel() > 0:
            return physical_block_table
        kernel_block_table = cls._select_block_table_for_layer(attn_inputs=attn_inputs)
        if kernel_block_table is None or kernel_block_table.numel() == 0:
            return None
        return cls._recover_physical_block_table_from_kernel(
            kernel_block_table,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=int(kernel_seq_size_per_block),
            cg_bufs=cg_bufs,
        )

    @classmethod
    def _build_plugin_attention_metadata(
        cls,
        *,
        attn_inputs: Any,
        positions: torch.Tensor,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int = 0,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> AttentionMetaData:
        in_capture = torch.cuda.is_current_stream_capturing()
        block_table = cls._resolve_plugin_block_table(
            attn_inputs=attn_inputs,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=int(kernel_seq_size_per_block),
            cg_bufs=cg_bufs,
            in_capture=in_capture,
        )
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_block_id_device for plugin attention metadata."
            )
        device = positions.device
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        if in_capture and cg_bufs is None:
            raise RuntimeError(
                "RTP plugin capture requires prewarmed cg_bufs; metadata fallback path is disabled."
            )
        seq_lens = cls._build_seq_lens(attn_inputs, device=device)
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
            if positions.dtype != torch.int32:
                positions_i32_buf = cg_bufs.get("positions_i32")
                if not isinstance(positions_i32_buf, torch.Tensor):
                    raise RuntimeError(
                        "RTP plugin capture requires prewarmed positions_i32 buffer."
                    )
                if int(positions_i32_buf.shape[0]) < batch_size:
                    raise RuntimeError(
                        "RTP plugin prewarmed positions_i32 buffer is too small "
                        f"(buffer={int(positions_i32_buf.shape[0])}, required={batch_size})."
                    )
                positions_i32 = positions_i32_buf[:batch_size]
                positions_i32.copy_(positions, non_blocking=True)
                positions = positions_i32
        num_actual_tokens = int(positions.numel())

        query_start_loc = cls._build_query_start_loc_for_plugin(
            attn_inputs=attn_inputs,
            seq_lens=seq_lens,
            num_tokens=num_actual_tokens,
            device=device,
            cg_bufs=cg_bufs,
        )
        slot_mapping = cls._build_slot_mapping(
            positions=positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            seq_size_per_block=seq_size_per_block,
            cg_bufs=cg_bufs,
        )
        req_id_per_token = cls._build_req_id_per_token(
            query_start_loc=query_start_loc,
            num_tokens=num_actual_tokens,
            device=device,
            cg_bufs=cg_bufs if in_capture else None,
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
            # Capture must keep the compact physical table layout. Copying into a
            # wider prewarmed table and slicing columns would create a strided view
            # that the downstream Triton expand kernel does not understand.
            if block_table.dtype != torch.int32:
                raise RuntimeError(
                    "RTP plugin capture requires block_table to be int32 to avoid allocation."
                )
            if not block_table.is_contiguous():
                raise RuntimeError(
                    "RTP plugin capture requires block_table to be contiguous to avoid allocation."
                )
            block_table_i32 = block_table
        else:
            block_table_i32 = block_table.to(
                device=device, dtype=torch.int32, non_blocking=True
            ).contiguous()
        indexer_block_table_i32 = cls._build_indexer_block_tables(
            block_table_i32=block_table_i32,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=int(kernel_seq_size_per_block),
            cg_max_seq_len=int(cg_max_seq_len),
            in_capture=in_capture,
            cg_bufs=cg_bufs,
        )
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
        plugin_md.req_id_per_token = req_id_per_token
        plugin_md.topk_tokens = 0
        plugin_md.sparse_block_size = int(seq_size_per_block)
        plugin_md.cg_bufs = cg_bufs
        cu_seqlen_ks = None
        cu_seqlen_ke = None
        if is_prefill:
            prefill_lengths = (query_start_loc[1:] - query_start_loc[:-1]).to(
                dtype=torch.int64
            )
            if in_capture and cg_bufs is not None and "seq_id" in cg_bufs:
                seq_id_for_span = cg_bufs["seq_id"][:num_actual_tokens]
            else:
                seq_id_for_span = torch.repeat_interleave(
                    torch.arange(batch_size, device=device, dtype=torch.int64),
                    prefill_lengths,
                )
            cu_seqlen_ks = (
                query_start_loc[:-1][seq_id_for_span].to(dtype=torch.int32).contiguous()
            )
            cu_seqlen_ke = (
                torch.arange(num_actual_tokens, device=device, dtype=torch.int32) + 1
            ).contiguous()
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
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=query_start_loc,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            block_tables=indexer_block_table_i32,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
            has_cached=False,
            total_kv=int(num_actual_kv_tokens),
        )
        attn_metadata.plugin_metadata = plugin_md
        return attn_metadata

    @staticmethod
    def collect_layer_maps(model: Any) -> LayerMaps:
        gdn_layer_map: Dict[int, GatedDeltaNet] = {}
        full_attn_layer_map: Dict[int, Any] = {}
        mla_layer_map: Dict[int, Any] = {}
        rtp_attention_cls: type[Any] | None = None
        rtp_mla_attention_cls: type[Any] | None = None
        try:
            from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import (
                RTPMLAAttention,
            )

            rtp_mla_attention_cls = RTPMLAAttention
        except (ImportError, ModuleNotFoundError):
            rtp_mla_attention_cls = None
        try:
            from atom.plugin.rtpllm.attention_backend import AttentionForRTPLLM

            rtp_attention_cls = AttentionForRTPLLM
        except (ImportError, ModuleNotFoundError):
            rtp_attention_cls = None

        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                gdn_layer_map[int(module.layer_num)] = module
            elif (
                getattr(module, "indexer", None) is not None
                and getattr(module, "mla_attn", None) is not None
                and getattr(module, "layer_num", None) is not None
            ):
                mla_layer_map[int(module.layer_num)] = module
            elif rtp_mla_attention_cls is not None and isinstance(
                module, rtp_mla_attention_cls
            ):
                layer_num = getattr(module, "layer_id", None)
                if layer_num is None:
                    layer_num = getattr(module, "layer_num", None)
                if layer_num is not None and int(layer_num) not in mla_layer_map:
                    mla_layer_map[int(layer_num)] = module
            elif isinstance(module, (PagedAttention, PagedAttentionImpl)) or (
                rtp_attention_cls is not None and isinstance(module, rtp_attention_cls)
            ):
                impl = getattr(module, "impl", None)
                layer_num = getattr(impl, "layer_num", None)
                if layer_num is None:
                    layer_num = getattr(module, "layer_num", None)
                if layer_num is not None:
                    full_attn_layer_map[int(layer_num)] = module
        return gdn_layer_map, full_attn_layer_map, mla_layer_map

    @staticmethod
    def _build_kv_cache_tensors(
        runtime: Any,
        layer_maps: LayerMaps,
    ) -> Dict[str, KVCacheTensor]:
        if runtime.kv_cache is None:
            raise ValueError("RTP plugin requires initialized kv_cache for ATOM model.")

        gdn_layer_map, full_attn_layer_map, mla_layer_map = layer_maps

        if not gdn_layer_map and not full_attn_layer_map and not mla_layer_map:
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
        # Build MLA cache references separately from full attention. MLA adapters
        # own their kv_cache pointer and refresh it in bind() for every forward.
        for layer_num in mla_layer_map.keys():
            layer_key = f"layer_{layer_num}"
            if layer_key in cache_tensors:
                continue

            layer_cache = runtime.kv_cache.get_layer_cache(layer_num)
            kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
            if kv_cache_base is None:
                raise ValueError(
                    f"Layer {layer_num} kv_cache_base is missing for MLA cache."
                )
            if kv_cache_base.dim() < 1:
                raise ValueError(
                    f"Layer {layer_num} MLA kv_cache_base has invalid shape "
                    f"{tuple(kv_cache_base.shape)}."
                )
            cache_tensors[layer_key] = KVCacheTensor(
                layer_num=layer_num,
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
        gdn_layer_map, full_attn_layer_map, mla_layer_map = layer_maps
        signature: list[Any] = [id(runtime.kv_cache)]
        all_layer_nums = sorted(
            set(gdn_layer_map.keys())
            | set(full_attn_layer_map.keys())
            | set(mla_layer_map.keys())
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
            kv_scale_base = getattr(layer_cache, "kv_scale_base", None)
            if kv_scale_base is not None and kv_scale_base.numel() > 0:
                signature.append(
                    (
                        int(layer_num),
                        "scale",
                        int(kv_scale_base.data_ptr()),
                        int(kv_scale_base.numel()),
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
        resolved_layer_maps = layer_maps or cls.collect_layer_maps(model)
        gdn_layer_map, _, _ = resolved_layer_maps
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
        gdn_metadata = None
        if gdn_layer_map:
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
            seq_size_per_block=seq_size_per_block,
            kernel_seq_size_per_block=kernel_seq_size_per_block,
            cg_max_seq_len=int(cg_max_seq_len),
            cg_bufs=cg_bufs,
        )
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
            mla_layer_map=cls._resolve_mla_layer_map(resolved_layer_maps),
        )

    @classmethod
    def _resolve_mla_layer_map(cls, layer_maps: LayerMaps) -> Dict[int, Any]:
        del cls, layer_maps
        return {}

    @staticmethod
    def _build_fallback_indexer_cache(
        *,
        cache_owner: Any,
        layer_cache: Any,
        indexer: Any,
        block_size: int,
    ) -> torch.Tensor | None:
        kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
        if kv_cache_base is None or kv_cache_base.dim() == 0:
            return None
        index_dim = int(getattr(indexer, "head_dim", 0) or 0) + 4
        if index_dim <= 4:
            return None
        aligned_dim = ((index_dim + 15) // 16) * 16
        num_tokens = int(kv_cache_base.shape[0]) * block_size
        cached = getattr(cache_owner, "_rtp_indexer_kv_cache", None)
        expected_shape = (num_tokens, 1, aligned_dim)
        if (
            cached is None
            or tuple(cached.shape) != expected_shape
            or cached.device != kv_cache_base.device
            or cached.dtype != dtypes.fp8
        ):
            cached = torch.empty(
                expected_shape,
                device=kv_cache_base.device,
                dtype=dtypes.fp8,
            )
            setattr(cache_owner, "_rtp_indexer_kv_cache", cached)
        return cached

    @staticmethod
    def _attach_mla_layer_caches(
        forward_context: "RTPForwardContext",
    ) -> tuple[list[tuple[Any, str, Any]], list[tuple[list[Any], int, Any]]]:
        restore_attrs: list[tuple[Any, str, Any]] = []
        restore_indices: list[tuple[list[Any], int, Any]] = []
        for layer_num, layer in forward_context.mla_layer_map.items():
            cache_tensor = forward_context.kv_cache_data.get(f"layer_{layer_num}")
            if cache_tensor is None:
                continue
            cache_owner = getattr(layer, "mla_attn", layer)
            restore_attrs.append(
                (cache_owner, "kv_cache", getattr(cache_owner, "kv_cache", None))
            )
            cache_owner.kv_cache = cache_tensor.k_cache
            indexer = getattr(layer, "indexer", None)
            if indexer is None:
                indexer = getattr(cache_owner, "indexer", None)
            indexer_cache = getattr(indexer, "k_cache", None)
            indexer_kv_cache = getattr(indexer_cache, "kv_cache", None)
            if not isinstance(indexer_kv_cache, list) or not indexer_kv_cache:
                continue
            layer_cache = cache_tensor.k_cache
            kv_cache_base = getattr(layer_cache, "kv_cache_base", None)
            if kv_cache_base is None or kv_cache_base.dim() == 0:
                continue
            block_size = int(
                getattr(forward_context, "rtp_seq_size_per_block", 0)
                or getattr(forward_context, "rtp_kernel_seq_size_per_block", 0)
                or getattr(get_current_atom_config(), "kv_cache_block_size", 0)
            )
            if block_size <= 0:
                raise ValueError(
                    "RTP plugin requires positive block_size for MLA indexer cache "
                    f"(layer={layer_num}, rtp_seq_size_per_block="
                    f"{getattr(forward_context, 'rtp_seq_size_per_block', 0)}, "
                    "rtp_kernel_seq_size_per_block="
                    f"{getattr(forward_context, 'rtp_kernel_seq_size_per_block', 0)})."
                )
            indexer_cache_tensor = RTPForwardContext._build_fallback_indexer_cache(
                cache_owner=cache_owner,
                layer_cache=layer_cache,
                indexer=indexer,
                block_size=block_size,
            )
            if indexer_cache_tensor is None:
                continue
            restore_indices.append((indexer_kv_cache, 0, indexer_kv_cache[0]))
            indexer_kv_cache[0] = indexer_cache_tensor
        return restore_attrs, restore_indices

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
        attn_md.rtp_seq_size_per_block = getattr(
            forward_context, "rtp_seq_size_per_block", 0
        )
        attn_md.rtp_layer_group_map = forward_context.layer_group_map
        restore_mla_attrs: list[tuple[Any, str, Any]] = []
        restore_mla_indices: list[tuple[list[Any], int, Any]] = []
        try:
            restore_mla_attrs, restore_mla_indices = cls._attach_mla_layer_caches(
                forward_context
            )
            set_kv_cache_data(forward_context.kv_cache_data)
            set_forward_context(
                attn_metadata=attn_md,
                atom_config=get_current_atom_config(),
                context=forward_context.context,
                num_tokens=forward_context.num_tokens,
            )
            yield
        finally:
            for target, index, old_cache in reversed(restore_mla_indices):
                target[index] = old_cache
            for target, attr, old_cache in reversed(restore_mla_attrs):
                setattr(target, attr, old_cache)
            reset_forward_context()
            set_kv_cache_data(prev_kv if prev_kv is not None else {})


@dataclass(frozen=True)
class RTPForwardMLAContext(RTPForwardContext):
    @classmethod
    def _resolve_plugin_block_table(
        cls,
        *,
        attn_inputs: Any,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_bufs: dict | None,
        in_capture: bool,
    ) -> torch.Tensor | None:
        physical_block_table = getattr(attn_inputs, "kv_cache_block_id_device", None)
        if physical_block_table is not None and physical_block_table.numel() > 0:
            return physical_block_table
        kernel_block_table = cls._select_block_table_for_layer(attn_inputs=attn_inputs)
        if kernel_block_table is None or kernel_block_table.numel() == 0:
            return None
        return cls._recover_physical_block_table_from_kernel(
            kernel_block_table,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=int(kernel_seq_size_per_block),
            cg_bufs=cg_bufs if in_capture else None,
        )

    @classmethod
    def _build_indexer_block_tables(
        cls,
        *,
        block_table_i32: torch.Tensor,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_max_seq_len: int,
        in_capture: bool,
        cg_bufs: dict | None,
    ) -> torch.Tensor:
        if in_capture:
            expected_kernel_cols = 0
            if cg_max_seq_len > 0 and int(kernel_seq_size_per_block) > 0:
                expected_kernel_cols = (
                    int(cg_max_seq_len) + int(kernel_seq_size_per_block) - 1
                ) // int(kernel_seq_size_per_block)
            if (
                expected_kernel_cols > 0
                and int(block_table_i32.shape[1]) >= expected_kernel_cols
            ):
                return block_table_i32
            return cls._expand_block_table_for_atom_indexer_capture(
                block_table_i32,
                seq_size_per_block=int(seq_size_per_block),
                kernel_seq_size_per_block=int(kernel_seq_size_per_block),
                cg_bufs=cg_bufs,
            )
        return cls._expand_block_table_for_atom_indexer(
            block_table_i32,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=int(kernel_seq_size_per_block),
        )

    @classmethod
    def _resolve_mla_layer_map(
        cls, layer_maps: RTPForwardContext.LayerMaps
    ) -> Dict[int, Any]:
        del cls
        return layer_maps[2]


@dataclass(frozen=True)
class RTPForwardQwen35HybridContext(RTPForwardContext):
    @staticmethod
    def _build_seq_lens(attn_inputs: Any, *, device: torch.device) -> torch.Tensor:
        """Qwen3.5 decode-cudagraph compatible seq_lens priority.

        Keep the validated sequence_lengths_plus_1_d ordering from
        `develop/rtp_atom_0526_qwen35_cuda_graph_ok`.
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

    @classmethod
    def _resolve_plugin_block_table(
        cls,
        *,
        attn_inputs: Any,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int,
        cg_bufs: dict | None,
        in_capture: bool,
    ) -> torch.Tensor | None:
        del cls, seq_size_per_block, kernel_seq_size_per_block, cg_bufs, in_capture
        return RTPForwardContext._select_block_table_for_layer(attn_inputs=attn_inputs)

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
        if in_capture and cg_bufs is not None:
            return cg_bufs["query_start_loc"][: batch_size + 1]

        if in_capture:
            raise ValueError(
                "RTP plugin capture requires prewarmed cg_bufs for query_start_loc "
                f"(batch={batch_size}, num_tokens={int(num_tokens)})."
            )

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

    @classmethod
    def _build_plugin_attention_metadata(
        cls,
        *,
        attn_inputs: Any,
        positions: torch.Tensor,
        seq_size_per_block: int,
        kernel_seq_size_per_block: int = 0,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> AttentionMetaData:
        del kernel_seq_size_per_block
        block_table = cls._resolve_plugin_block_table(
            attn_inputs=attn_inputs,
            seq_size_per_block=int(seq_size_per_block),
            kernel_seq_size_per_block=0,
            cg_bufs=cg_bufs,
            in_capture=torch.cuda.is_current_stream_capturing(),
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
        seq_lens = cls._build_seq_lens(attn_inputs, device=device)
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

        if in_capture and not is_prefill:
            positions = positions[:batch_size]
        num_actual_tokens = int(positions.numel())

        query_start_loc = cls._build_query_start_loc_for_plugin(
            attn_inputs=attn_inputs,
            seq_lens=seq_lens,
            num_tokens=num_actual_tokens,
            device=device,
            cg_bufs=cg_bufs,
        )
        slot_mapping = cls._build_slot_mapping(
            positions=positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            seq_size_per_block=seq_size_per_block,
            cg_bufs=cg_bufs,
        )

        is_dummy_warmup = False
        if in_capture:
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
            if max_seq_len <= 0:
                is_dummy_warmup = True
                max_seq_len = int(cg_max_seq_len) if cg_max_seq_len > 0 else 1
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

        if in_capture and cg_bufs is not None:
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
        plugin_md.rtp_cu_seqlens_q = query_start_loc
        plugin_md.is_dummy_warmup = bool(is_dummy_warmup)
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if (
            prefix_lengths is not None
            and int(prefix_lengths.numel()) > 0
            and not in_capture
        ):
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
