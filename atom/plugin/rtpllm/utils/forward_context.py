from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import torch

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.model_ops.paged_attention import PagedAttention
from atom.model_ops.attentions.gdn_attn import (
    GDNAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from atom.plugin.attention import (
    AiterFlashAttentionDecodeMetadata,
    AiterFlashAttentionMetadataForPluginMode,
    AiterFlashAttentionPrefillMetadata,
)
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    _forward_kv_cache_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)


@dataclass(frozen=True)
class RTPForwardContext:
    gdn_metadata: GDNAttentionMetadata
    attn_metadata: AttentionMetaData
    rtp_attn_inputs: Any
    rtp_seq_size_per_block: int
    rtp_kernel_seq_size_per_block: int
    kv_cache_data: Dict[str, KVCacheTensor]
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
    def _num_tokens_from_inputs(attn_inputs: Any, *, device: torch.device) -> int:
        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        if input_lengths is None:
            raise ValueError(
                "RTP plugin requires attention_inputs.input_lengths for GDN metadata."
            )
        return int(input_lengths.sum().item())

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
            if int(cu_seqlens[-1].item()) > 0:
                if input_lengths is not None and cu_seqlens.numel() >= input_lengths.numel() + 1:
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
        q_lens = torch.ones_like(input_lengths, dtype=torch.int32, device=input_lengths.device)
        prefix = torch.zeros((1,), dtype=torch.int32, device=input_lengths.device)
        return torch.cat([prefix, q_lens.cumsum(dim=0)], dim=0)

    @staticmethod
    def _state_indices(
        attn_inputs: Any,
        is_prefill: bool,
        *,
        device: torch.device,
        seq_size_per_block: int,
        layer_num: int | None = None,
    ) -> torch.Tensor:
        block_table = RTPForwardContext._select_block_table_for_layer(
            attn_inputs=attn_inputs,
            layer_num=layer_num,
        )
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_kernel_block_id_device for GDN metadata."
            )
        if block_table.dim() == 1:
            block_table = block_table.unsqueeze(0)
        base = block_table.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
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

        if torch.any(last_token_idx < 0):
            raise ValueError(
                "RTP plugin produced negative token index for GDN state mapping."
            )
        block_col = torch.div(
            last_token_idx,
            int(seq_size_per_block),
            rounding_mode="floor",
        )
        if torch.any(block_col < 0) or torch.any(block_col >= base.shape[1]):
            raise ValueError(
                "RTP plugin block-table index out of range for GDN state mapping "
                f"(max_col={int(base.shape[1]) - 1})."
            )
        row_idx = torch.arange(base.shape[0], device=device, dtype=torch.int64)
        slot_ids = base[row_idx, block_col.to(dtype=torch.int64)]
        if torch.any(slot_ids < 0):
            raise ValueError(
                "RTP plugin resolved padded/invalid (-1) block slot for GDN state mapping."
            )
        return slot_ids.contiguous()

    @staticmethod
    def _select_block_table_for_layer(
        attn_inputs: Any,
        layer_num: int | None,
    ) -> torch.Tensor | None:
        by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)
        if by_group is not None and len(by_group):
            gid = 0
            if layer_num is not None:
                layer_to_group = getattr(attn_inputs, "kv_cache_layer_to_group", None)
                if layer_to_group is not None and int(layer_to_group.numel()) > layer_num:
                    gid = int(layer_to_group[layer_num].item())
            if gid < 0 or gid >= len(by_group):
                raise ValueError(
                    f"RTP plugin resolved invalid kv-cache group id {gid} for layer {layer_num}."
                )
            return by_group[gid]
        return getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)

    @staticmethod
    def state_indices_for_layer(
        *,
        attn_inputs: Any,
        is_prefill: bool,
        device: torch.device,
        seq_size_per_block: int,
        layer_num: int,
    ) -> torch.Tensor:
        return RTPForwardContext._state_indices(
            attn_inputs=attn_inputs,
            is_prefill=is_prefill,
            device=device,
            seq_size_per_block=seq_size_per_block,
            layer_num=layer_num,
        )

    @staticmethod
    def _build_gdn_metadata(
        attn_inputs: Any, *, seq_size_per_block: int
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
        num_tokens = int(query_start_loc[-1].item())
        state_indices = RTPForwardContext._state_indices(
            attn_inputs=attn_inputs,
            is_prefill=is_prefill,
            device=target_device,
            seq_size_per_block=seq_size_per_block,
        )

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
            nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                query_start_loc
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
        if sequence_lengths is None:
            raise ValueError(
                "RTP decode requires attention_inputs.sequence_lengths for seq_lens."
            )
        if int(sequence_lengths.numel()) != int(input_lengths.numel()):
            raise ValueError(
                "RTP plugin sequence_lengths/input_lengths batch mismatch "
                f"(sequence_lengths={int(sequence_lengths.numel())}, "
                f"input_lengths={int(input_lengths.numel())})."
            )
        return (sequence_lengths + input_lengths).contiguous()

    @staticmethod
    def _build_slot_mapping(
        *,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        seq_size_per_block: int,
    ) -> torch.Tensor:
        if positions is None or positions.numel() == 0:
            raise ValueError("RTP plugin requires non-empty positions for slot_mapping.")
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
        pos_i32 = positions.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        qsl = query_start_loc.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        bt = block_table.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        batch_size = int(qsl.numel()) - 1
        num_tokens = int(pos_i32.numel())
        if batch_size <= 0:
            raise ValueError("RTP plugin query_start_loc produced empty batch.")
        if int(bt.shape[0]) != batch_size:
            raise ValueError(
                "RTP plugin block_table/query_start_loc batch mismatch "
                f"(block_table={int(bt.shape[0])}, batch={batch_size})."
            )
        if int(qsl[-1].item()) != num_tokens:
            raise ValueError(
                "RTP plugin query_start_loc/positions token mismatch "
                f"(query_start_loc[-1]={int(qsl[-1].item())}, positions={num_tokens})."
            )

        lengths = qsl[1:] - qsl[:-1]
        if torch.any(lengths <= 0):
            raise ValueError(
                "RTP plugin query_start_loc contains non-positive sequence length."
            )
        seq_id = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.int64),
            lengths.to(dtype=torch.int64),
        )
        if int(seq_id.numel()) != num_tokens:
            raise ValueError(
                "RTP plugin internal seq_id construction mismatch for slot_mapping."
            )

        block_col = torch.div(
            pos_i32,
            int(seq_size_per_block),
            rounding_mode="floor",
        )
        if torch.any(block_col < 0) or torch.any(block_col >= bt.shape[1]):
            raise ValueError(
                "RTP plugin block-table index out of range for full-attn slot_mapping "
                f"(max_col={int(bt.shape[1]) - 1})."
            )

        slot_base = bt[seq_id, block_col.to(dtype=torch.int64)]
        if torch.any(slot_base < 0):
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
    ) -> torch.Tensor:
        batch_size = int(seq_lens.numel())
        if batch_size <= 0:
            raise ValueError("RTP plugin cannot build query_start_loc with empty seq_lens.")

        # First try RTP's native path.
        qsl = RTPForwardContext._query_start_loc(attn_inputs, device=device)
        if qsl is not None and qsl.numel() == batch_size + 1:
            lengths = qsl[1:] - qsl[:-1]
            if (
                int(qsl[-1].item()) == int(num_tokens)
                and bool(torch.all(lengths > 0))
            ):
                return qsl.contiguous()

        # Fallback: derive from input_lengths when it is valid for this step.
        input_lengths = RTPForwardContext._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=device,
        )
        if input_lengths is not None and int(input_lengths.numel()) == batch_size:
            if (
                bool(torch.all(input_lengths > 0))
                and int(input_lengths.sum().item()) == int(num_tokens)
            ):
                prefix = torch.zeros((1,), dtype=torch.int32, device=device)
                return torch.cat([prefix, input_lengths.cumsum(dim=0)], dim=0).contiguous()

        # Final fallback for decode-style step where each sequence contributes 1 token.
        if int(num_tokens) == batch_size:
            prefix = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
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
    ) -> AttentionMetaData:
        block_table = RTPForwardContext._select_block_table_for_layer(
            attn_inputs=attn_inputs,
            layer_num=None,
        )
        if block_table is None or block_table.numel() == 0:
            raise ValueError(
                "RTP plugin requires kv_cache_kernel_block_id_device for plugin attention metadata."
            )
        device = positions.device
        seq_lens = RTPForwardContext._build_seq_lens(attn_inputs, device=device)
        query_start_loc = RTPForwardContext._build_query_start_loc_for_plugin(
            attn_inputs=attn_inputs,
            seq_lens=seq_lens,
            num_tokens=int(positions.numel()),
            device=device,
        )
        slot_mapping = RTPForwardContext._build_slot_mapping(
            positions=positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            seq_size_per_block=seq_size_per_block,
        )

        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        batch_size = int(seq_lens.numel())
        num_actual_tokens = int(positions.numel())
        max_query_len = int(torch.max(query_start_loc[1:] - query_start_loc[:-1]).item())
        max_seq_len = int(torch.max(seq_lens).item())
        num_actual_kv_tokens = int(seq_lens.sum().item())

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

        plugin_md = AiterFlashAttentionMetadataForPluginMode(
            num_actual_tokens=num_actual_tokens,
            num_actual_kv_tokens=num_actual_kv_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            block_table=block_table.to(
                device=device, dtype=torch.int32, non_blocking=True
            ).contiguous(),
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
        return AttentionMetaData(
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            block_tables=plugin_md.block_table,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            plugin_metadata=plugin_md,
        )

    @staticmethod
    def collect_layer_maps(model: Any) -> LayerMaps:
        gdn_layer_map: Dict[int, GatedDeltaNet] = {}
        full_attn_layer_map: Dict[int, Any] = {}
        rtp_attention_cls: type[Any] | None = None
        try:
            from atom.plugin.rtpllm.attention_backend import RTPAttention

            rtp_attention_cls = RTPAttention
        except Exception:  # noqa: BLE001
            rtp_attention_cls = None

        for module in model.modules():
            if isinstance(module, GatedDeltaNet):
                gdn_layer_map[int(module.layer_num)] = module
            elif isinstance(module, PagedAttention) or (
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
            ssm_state_size = int(local_num_v_heads * gdn_layer.head_v_dim * gdn_layer.head_k_dim)
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
                (cache_base.shape[0], conv_kernel - 1, qkv_size),
                (cache_base.stride()[0], qkv_size, 1),
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

    @classmethod
    def build(
        cls,
        model: Any,
        runtime: Any,
        inputs: Any,
        positions: torch.Tensor,
        layer_maps: LayerMaps | None = None,
    ) -> "RTPForwardContext":
        attn_inputs = getattr(inputs, "attention_inputs", None)
        if attn_inputs is None:
            raise ValueError("RTP plugin requires inputs.attention_inputs for forward context.")

        if runtime.kv_cache is None:
            raise ValueError("RTP plugin requires initialized kv_cache for forward context.")
        seq_size_per_block = int(getattr(runtime.kv_cache, "seq_size_per_block", 0))
        kernel_seq_size_per_block = int(
            getattr(runtime.kv_cache, "kernel_seq_size_per_block", 0)
        )
        if kernel_seq_size_per_block <= 0:
            kernel_seq_size_per_block = int(seq_size_per_block)
        gdn_metadata = cls._build_gdn_metadata(
            attn_inputs,
            seq_size_per_block=seq_size_per_block,
        )
        # Keep raw RTP attention inputs in metadata so GDN can resolve per-layer
        # block-map/state-index semantics (same idea as RTP's select_block_map_for_layer).
        gdn_metadata.rtp_attn_inputs = attn_inputs
        gdn_metadata.rtp_seq_size_per_block = int(seq_size_per_block)
        attn_metadata = cls._build_plugin_attention_metadata(
            attn_inputs=attn_inputs,
            positions=positions,
            seq_size_per_block=kernel_seq_size_per_block,
        )
        kv_cache_data = getattr(runtime, "_rtp_kv_cache_data", None)
        if kv_cache_data is None:
            kv_cache_data = cls._build_kv_cache_tensors(
                runtime=runtime,
                layer_maps=layer_maps or cls.collect_layer_maps(model),
            )
            runtime._rtp_kv_cache_data = kv_cache_data
        input_lengths = cls._non_empty_int32(
            getattr(attn_inputs, "input_lengths", None),
            device=positions.device,
        )
        if input_lengths is None:
            raise ValueError(
                "RTP plugin requires attention_inputs.input_lengths for forward context."
            )
        batch_size = int(input_lengths.numel())
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
    ) -> Iterator[None]:
        forward_context = cls.build(
            model=model,
            runtime=runtime,
            inputs=inputs,
            positions=positions,
            layer_maps=layer_maps,
        )
        prev_kv = _forward_kv_cache_context.kv_cache_data
        attn_md = forward_context.attn_metadata
        attn_md.gdn_metadata = forward_context.gdn_metadata
        attn_md.rtp_attn_inputs = forward_context.rtp_attn_inputs
        attn_md.rtp_seq_size_per_block = forward_context.rtp_seq_size_per_block
        attn_md.rtp_kernel_seq_size_per_block = (
            forward_context.rtp_kernel_seq_size_per_block
        )
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
