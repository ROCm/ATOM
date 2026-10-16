"""Qwen3.8 QSA/PLE metadata on top of RTP's hybrid-attention cache."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from atom.model_ops.attentions.qwen4_exp_attn import (
    Qwen4ExpPLEMetadata,
    Qwen4ExpQSAMetadata,
)
from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention
from atom.plugin.rtpllm.utils.forward_context import (
    RTPForwardContext,
    RTPForwardQwen35HybridContext,
    _target_verify_state_slots_graph_kernel,
)
from atom.utils.forward_context import get_forward_context


@triton.jit
def _qsa_graph_slots_kernel(
    block_table,
    positions,
    token_to_request,
    slots,
    compressed_slots,
    rows,
    table_stride_row,
    table_stride_col,
    table_cols: tl.constexpr,
    block_size: tl.constexpr,
    ratio: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    live = row < rows
    position = tl.load(positions + row, live, -1)
    column = tl.minimum(tl.maximum(position, 0) // block_size, table_cols - 1)
    request = tl.load(token_to_request + row, live, 0)
    page = tl.load(
        block_table + request * table_stride_row + column * table_stride_col,
        live & (position >= 0),
        -1,
    )
    slot = tl.where(page >= 0, page * block_size + position % block_size, -1)
    compressed = tl.where(
        (slot >= 0) & ((position + 1) % ratio == 0), slot // ratio, -1
    )
    tl.store(slots + row, slot, live)
    tl.store(compressed_slots + row, compressed, live)


class RTPQwen4ExpContext(RTPForwardQwen35HybridContext):
    """Bind the QSA side caches and PLE windows to RTP's request/page IDs."""

    @classmethod
    def _build_gdn_metadata(
        cls,
        attn_inputs: Any,
        *,
        seq_size_per_block: int,
        num_tokens: int,
        state_indices_cache: dict | None = None,
        layer_group_map: dict | None = None,
        cg_bufs: dict | None = None,
    ) -> GDNAttentionMetadata:
        if not getattr(attn_inputs, "is_target_verify", False):
            return super()._build_gdn_metadata(
                attn_inputs,
                seq_size_per_block=seq_size_per_block,
                num_tokens=num_tokens,
                state_indices_cache=state_indices_cache,
                layer_group_map=layer_group_map,
                cg_bufs=cg_bufs,
            )
        # RTP labels verification as prefill, but the Q candidate states are
        # speculative decode states stored in separate reserved blocks.
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if block_table is None or block_table.ndim != 2 or block_table.numel() == 0:
            raise ValueError("invalid Qwen4Exp MTP target-verify state table")
        batch_size = int(attn_inputs.input_lengths.numel())
        if batch_size <= 0 or num_tokens % batch_size:
            raise ValueError("invalid Qwen4Exp MTP target-verify token count")
        query_len = num_tokens // batch_size
        if query_len not in (2, 3, 4) or any(
            int(length) != query_len for length in attn_inputs.input_lengths.tolist()
        ):
            raise ValueError(
                "Qwen4Exp MTP requires 2, 3, or 4 verify tokens per request "
                f"(tokens={num_tokens}, batch={batch_size}, query_len={query_len})"
            )
        seq_lens = attn_inputs.sequence_lengths_plus_1_device
        if seq_lens is None or seq_lens.numel() != batch_size:
            raise ValueError("invalid Qwen4Exp MTP target-verify sequence lengths")
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture:
            if (
                triton is None
                or cg_bufs is None
                or "target_read_slots_i32" not in cg_bufs
            ):
                raise RuntimeError(
                    "Qwen4Exp MTP target Graph requires prewarmed buffers"
                )
            query_start_loc = cg_bufs["query_start_loc"][: batch_size + 1]
            read_slots = cg_bufs["target_read_slots_i32"][:batch_size]
            write_slots = cg_bufs["target_write_slots_i32"][:batch_size]
            _target_verify_state_slots_graph_kernel[(triton.cdiv(batch_size, 128),)](
                block_table,
                seq_lens,
                read_slots,
                write_slots,
                batch_size,
                block_table.stride(0),
                block_table.stride(1),
                block_table.shape[1],
                seq_size_per_block,
                query_len,
                128,
            )
            sequence_masks = cg_bufs["target_sequence_masks"][:batch_size]
            accepted_tokens = cg_bufs["target_accepted_tokens"][:batch_size]
        else:
            query_start_loc = RTPForwardContext._query_start_loc(
                attn_inputs, device=block_table.device
            )
            rows = torch.arange(batch_size, device=block_table.device)[:, None]
            read_col = torch.div(
                seq_lens.to(torch.int64) - 2,
                seq_size_per_block,
                rounding_mode="floor",
            )
            write_cols = torch.div(
                seq_lens.to(torch.int64) - 1,
                seq_size_per_block,
                rounding_mode="floor",
            )[:, None] + torch.arange(query_len, device=block_table.device)
            if bool(
                torch.any((read_col < 0) | (read_col >= block_table.shape[1])).item()
            ) or bool(torch.any(write_cols >= block_table.shape[1]).item()):
                raise ValueError("Qwen4Exp MTP target-verify block table is too narrow")
            read_slots = block_table[rows[:, 0], read_col]
            write_slots = block_table[rows, write_cols]
            if bool(torch.any(read_slots < 0).item()) or bool(
                torch.any(write_slots < 0).item()
            ):
                raise ValueError("Qwen4Exp MTP target-verify state block is invalid")
            sequence_masks = torch.ones(
                batch_size, device=block_table.device, dtype=torch.bool
            )
            accepted_tokens = torch.ones(
                batch_size, device=block_table.device, dtype=torch.int32
            )
        metadata = GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=batch_size,
            num_spec_decode_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            has_initial_state=None,
            spec_query_start_loc=query_start_loc,
            non_spec_query_start_loc=None,
            spec_state_indices_tensor=write_slots.to(torch.int32),
            spec_state_indices_in_tensor=(
                read_slots
                if in_capture
                else read_slots[:, None]
                .expand_as(write_slots)
                .to(torch.int32)
                .contiguous()
            ),
            non_spec_state_indices_tensor=None,
            spec_sequence_masks=sequence_masks,
            spec_token_indx=None,
            non_spec_token_indx=None,
            num_accepted_tokens=accepted_tokens,
        )
        metadata.rtp_target_verify_block_map = block_table
        metadata.rtp_target_verify_sequence_lengths = seq_lens.to(torch.int32)
        metadata.rtp_seq_size_per_block = seq_size_per_block
        return metadata

    @staticmethod
    def collect_layer_maps(model: Any):
        gdn, full, mla = RTPForwardQwen35HybridContext.collect_layer_maps(model)
        for module in model.modules():
            if isinstance(module, Qwen4ExpAttention):
                full[int(module.layer_num)] = module
        return gdn, full, mla

    @classmethod
    def _resolve_qsa_block_table(
        cls,
        full_inputs: Any,
        block_size: int,
        kernel_block_size: int,
        graph_buffers: dict | None,
    ) -> torch.Tensor:
        table = getattr(full_inputs, "kv_cache_block_id_device", None)
        if table is None or table.numel() == 0:
            kernel = cls._select_block_table_for_layer(attn_inputs=full_inputs)
            if kernel is not None and kernel.numel() > 0:
                table = cls._recover_physical_block_table_from_kernel(
                    kernel,
                    seq_size_per_block=block_size,
                    kernel_seq_size_per_block=kernel_block_size,
                    cg_bufs=graph_buffers,
                )
        if table is None or table.numel() == 0:
            raise ValueError("Qwen4Exp full-attention physical block table is empty")
        return table

    @staticmethod
    def _bind_qsa_caches(runtime: Any, qsa_layers: list[Any], block_size: int) -> None:
        from rtp_llm.models_py.modules.factory.attention.common import (
            reshape_paged_kv_cache,
        )

        for module in qsa_layers:
            ratio = int(module.indexer.compress_ratio)
            if block_size <= 0 or ratio <= 0 or block_size % ratio:
                raise ValueError(
                    f"QSA block size {block_size} must be divisible by compression ratio {ratio}"
                )
            layer_cache = runtime.kv_cache.get_layer_cache(module.layer_num)
            raw = layer_cache.kv_cache_base
            if raw is None:
                raise ValueError(f"QSA layer {module.layer_num} has no RTP KV cache")
            if raw.dtype != torch.bfloat16:
                raise ValueError("QSA RTP KV cache requires BF16 storage")
            num_blocks = int(raw.shape[0])
            signature = (raw.data_ptr(), num_blocks, block_size, str(raw.dtype))
            if getattr(module, "_rtp_qsa_cache_signature", None) == signature:
                continue
            # The QSA module is already TP-sharded; RTP allocates this layer's
            # cache for its local KV heads and may pad the raw hybrid stride.
            stored_heads = int(module.num_kv_heads)
            paged = reshape_paged_kv_cache(
                raw, stored_heads, block_size, module.head_dim
            )
            if paged.shape[1:] != (2, stored_heads, block_size, module.head_dim):
                raise ValueError(
                    f"QSA layer {module.layer_num} invalid RTP KV shape {tuple(paged.shape)}"
                )
            if int(paged.shape[0]) != num_blocks:
                raise ValueError("QSA RTP KV cache block count changed during reshape")
            # RTP owns the block storage, but only QSA reads/writes this layer's
            # bytes. AITER reshape_and_cache_flash writes token-major offsets;
            # a permute of RTP's head-major-shaped view changes strides without
            # changing those bytes. Reinterpret the same storage as token-major
            # so the AITER writer and the QSA reader agree for H > 1 as well.
            token_major = paged.view(
                num_blocks, 2, block_size, stored_heads, module.head_dim
            )
            key = token_major[:, 0]
            value = token_major[:, 1]
            device = raw.device
            index_dim = int(module.indexer.index_head_dim)
            side = (
                torch.zeros(
                    (num_blocks, block_size, 1, index_dim),
                    device=device,
                    dtype=raw.dtype,
                ),
                torch.zeros(
                    (
                        num_blocks,
                        block_size // ratio,
                        1,
                        index_dim,
                    ),
                    device=device,
                    dtype=raw.dtype,
                ),
            )
            module.bind_caches(key, value, *side)
            module._rtp_qsa_cache_signature = signature

    @staticmethod
    def _ple_metadata(
        runtime: Any, model: Any, attn_metadata: Any, gdn_layers: dict[int, Any]
    ) -> Qwen4ExpPLEMetadata | None:
        ple_layers = [
            layer.ple for layer in model.model.layers if layer.ple is not None
        ]
        if not ple_layers:
            return None
        if len(ple_layers) != 1:
            raise ValueError("RTP Qwen4Exp currently supports one PLE layer")
        gdn = attn_metadata.gdn_metadata
        if gdn is None or (
            gdn.non_spec_state_indices_tensor is None
            and gdn.spec_state_indices_tensor is None
        ):
            raise ValueError("PLE requires GDN state-slot metadata")
        ple = ple_layers[0]
        if not gdn_layers:
            raise ValueError("PLE requires a GDN layer cache")
        raw = runtime.kv_cache.get_layer_cache(min(gdn_layers)).kv_cache_base
        slots = int(raw.shape[0])
        signature = (raw.data_ptr(), slots, ple.hc_hidden_size, ple.conv_state_len)
        if getattr(runtime, "_rtp_ple_cache_signature", None) != signature:
            runtime._rtp_ple_ngram = torch.full(
                (slots, ple.ple_embedding.ngram_size - 1),
                int(model.config.eos_token_id),
                device=raw.device,
                dtype=torch.int64,
            )
            runtime._rtp_ple_conv = torch.zeros(
                (slots, ple.hc_hidden_size, ple.conv_state_len),
                device=raw.device,
                dtype=model.atom_config.torch_dtype,
            )
            runtime._rtp_ple_cache_signature = signature
        spec_indices = (
            gdn.spec_state_indices_tensor
            if getattr(gdn, "rtp_target_verify_block_map", None) is not None
            else None
        )
        if spec_indices is not None:
            indices = spec_indices[:, -1]
            input_indices = gdn.spec_state_indices_in_tensor[:, 0]
        else:
            indices = gdn.non_spec_state_indices_tensor
            input_indices = gdn.non_spec_state_indices_in_tensor
        has_initial = gdn.has_initial_state
        if has_initial is None:
            if torch.cuda.is_current_stream_capturing():
                graph_buffers = attn_metadata.plugin_metadata.cg_bufs
                has_initial = graph_buffers["qwen4_has_initial"][: indices.numel()]
                logical_positions = attn_metadata.qsa_metadata.logical_positions
                if spec_indices is not None:
                    logical_positions = logical_positions[0 :: spec_indices.shape[1]]
                torch.ge(
                    logical_positions,
                    0,
                    out=has_initial,
                )
            else:
                has_initial = torch.ones_like(indices, dtype=torch.bool)
        metadata = Qwen4ExpPLEMetadata(
            query_start_loc=attn_metadata.plugin_metadata.rtp_cu_seqlens_q,
            ngram_state=runtime._rtp_ple_ngram,
            state_indices_in=input_indices,
            state_indices_out=indices,
            has_initial_state=has_initial,
            conv_state=runtime._rtp_ple_conv,
        )
        # RTP-only per-candidate slots are deliberately not part of the
        # framework-neutral Qwen4ExpPLEMetadata dataclass.
        metadata.rtp_spec_state_indices = spec_indices
        return metadata

    @classmethod
    @contextmanager
    def bind(
        cls,
        *,
        model: Any,
        runtime: Any,
        inputs: Any,
        positions: torch.Tensor,
        layer_maps=None,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> Iterator[None]:
        resolved_layer_maps = layer_maps or cls.collect_layer_maps(model)
        with super().bind(
            model=model,
            runtime=runtime,
            inputs=inputs,
            positions=positions,
            layer_maps=resolved_layer_maps,
            cg_max_seq_len=cg_max_seq_len,
            cg_bufs=cg_bufs,
        ):
            forward_context = get_forward_context()
            metadata = forward_context.attn_metadata
            full_block_size = int(runtime.kv_cache.get_seq_size_per_block("full"))
            qsa_layers = [
                module
                for module in resolved_layer_maps[1].values()
                if isinstance(module, Qwen4ExpAttention)
            ]
            if not qsa_layers:
                raise ValueError("Qwen4Exp requires a QSA attention layer")
            ratio = int(qsa_layers[0].indexer.compress_ratio)
            if any(
                int(module.indexer.compress_ratio) != ratio for module in qsa_layers
            ):
                raise ValueError("Qwen4Exp QSA layers require one compression ratio")
            cls._bind_qsa_caches(runtime, qsa_layers, full_block_size)
            plugin = metadata.plugin_metadata
            logical = positions.reshape(-1).to(torch.int64)
            full_inputs = getattr(inputs, "attention_inputs_by_tag", {}).get("full")
            if full_inputs is None:
                raise ValueError("Qwen4Exp requires RTP's full-attention cache tag")
            in_capture = torch.cuda.is_current_stream_capturing()
            graph_buffers = cg_bufs if in_capture else None
            kernel_block_size = int(
                runtime.kv_cache.get_kernel_seq_size_per_block("full")
            )
            if kernel_block_size <= 0:
                kernel_block_size = full_block_size
            full_blocks = cls._resolve_qsa_block_table(
                full_inputs, full_block_size, kernel_block_size, graph_buffers
            )
            if in_capture:
                if graph_buffers is None:
                    raise RuntimeError("Qwen4Exp Graph requires prewarmed QSA buffers")
                slots = graph_buffers["qsa_slots_i64"][: positions.numel()]
                compressed = graph_buffers["qsa_compressed_slots"][: slots.numel()]
                _qsa_graph_slots_kernel[(triton.cdiv(slots.numel(), 128),)](
                    full_blocks,
                    positions,
                    graph_buffers["seq_id_i32"][: slots.numel()],
                    slots,
                    compressed,
                    slots.numel(),
                    full_blocks.stride(0),
                    full_blocks.stride(1),
                    full_blocks.shape[1],
                    full_block_size,
                    ratio,
                    128,
                )
                logical = graph_buffers["qsa_positions_i64"][: positions.numel()]
                logical.copy_(positions.reshape(-1))
            else:
                slots = cls._build_slot_mapping(
                    positions=positions.reshape(-1),
                    query_start_loc=plugin.rtp_cu_seqlens_q,
                    block_table=full_blocks,
                    seq_size_per_block=full_block_size,
                )
                logical = positions.reshape(-1).to(torch.int64)
                compressed = torch.empty_like(slots)
                qsa_compressed_slots(
                    slots,
                    logical,
                    ratio,
                    compressed,
                )
            metadata.qsa_metadata = Qwen4ExpQSAMetadata(
                block_tables=full_blocks,
                slot_mapping=slots,
                compressed_slot_mapping=compressed,
                token_to_req=cls._build_batch_id_per_q_token(
                    query_start_loc=plugin.rtp_cu_seqlens_q,
                    num_tokens=int(logical.numel()),
                    device=logical.device,
                    cg_bufs=graph_buffers,
                ),
                logical_positions=logical,
                seq_lens=metadata.context_lens,
                max_seq_len=int(metadata.max_seqlen_k),
            )
            metadata.ple_metadata = cls._ple_metadata(
                runtime, model, metadata, resolved_layer_maps[0]
            )
            yield
