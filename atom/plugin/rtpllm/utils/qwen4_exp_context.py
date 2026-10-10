"""Qwen3.8 QSA/PLE metadata on top of RTP's hybrid-attention cache."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.qwen4_exp_attn import (
    Qwen4ExpPLEMetadata,
    Qwen4ExpQSAMetadata,
)
from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention
from atom.plugin.rtpllm.utils.forward_context import RTPForwardQwen35HybridContext
from atom.utils.forward_context import get_forward_context


@triton.jit
def _previous_state_slots_graph_kernel(
    block_table,
    positions,
    output_slots,
    input_slots,
    rows,
    table_stride_row,
    table_stride_col,
    table_cols: tl.constexpr,
    block_size: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    live = row < rows
    position = tl.load(positions + row, live, -1)
    output_slot = tl.load(output_slots + row, live, -1)
    column = tl.minimum(tl.maximum(position - 1, 0) // block_size, table_cols - 1)
    previous = tl.load(
        block_table + row * table_stride_row + column * table_stride_col,
        live & (position > 0),
        -1,
    )
    input_slot = tl.where(position > 0, previous, output_slot)
    tl.store(input_slots + row, tl.where(position >= 0, input_slot, -1), live)


@triton.jit
def _qsa_graph_slots_kernel(
    block_table,
    positions,
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
    page = tl.load(
        block_table + row * table_stride_row + column * table_stride_col,
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

    @staticmethod
    def _previous_state_slots(
        block_table: torch.Tensor,
        positions: torch.Tensor,
        output_slots: torch.Tensor,
        block_size: int,
        graph_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if block_size <= 0 or block_table.ndim != 2:
            raise ValueError("Qwen4Exp requires a two-dimensional state block table")
        positions = positions.reshape(-1)
        if (
            positions.numel() != output_slots.numel()
            or block_table.shape[0] != positions.numel()
        ):
            raise ValueError(
                "Qwen4Exp decode requires one token per request "
                f"(positions={positions.numel()}, outputs={output_slots.numel()}, "
                f"block_rows={block_table.shape[0]})"
            )
        if graph_output is not None:
            if (
                graph_output.shape != output_slots.shape
                or graph_output.dtype != torch.int32
            ):
                raise ValueError("Qwen4Exp Graph previous-state buffer has wrong shape")
            _previous_state_slots_graph_kernel[(triton.cdiv(positions.numel(), 128),)](
                block_table,
                positions,
                output_slots,
                graph_output,
                positions.numel(),
                block_table.stride(0),
                block_table.stride(1),
                block_table.shape[1],
                block_size,
                128,
            )
            return graph_output
        positions = positions.to(torch.int64)
        previous_columns = torch.div(
            torch.clamp(positions - 1, min=0), block_size, rounding_mode="floor"
        )
        in_capture = torch.cuda.is_current_stream_capturing()
        if not in_capture and torch.any(previous_columns >= block_table.shape[1]):
            raise ValueError("Qwen4Exp previous state block is out of range")
        rows = torch.arange(output_slots.numel(), device=block_table.device)
        previous_slots = block_table[
            rows, previous_columns.clamp(max=block_table.shape[1] - 1)
        ]
        if not in_capture and torch.any((positions > 0) & (previous_slots < 0)):
            raise ValueError("Qwen4Exp previous state block is invalid")
        return torch.where(positions > 0, previous_slots, output_slots).to(torch.int32)

    @staticmethod
    def collect_layer_maps(model: Any):
        gdn, full, mla = RTPForwardQwen35HybridContext.collect_layer_maps(model)
        for module in model.modules():
            if isinstance(module, Qwen4ExpAttention):
                full[int(module.layer_num)] = module
        return gdn, full, mla

    @staticmethod
    def _bind_qsa_caches(runtime: Any, model: Any, block_size: int) -> None:
        from rtp_llm.models_py.modules.factory.attention.common import (
            reshape_paged_kv_cache,
        )

        if block_size <= 0 or block_size % 4:
            raise ValueError(
                f"QSA requires a block size divisible by 4, got {block_size}"
            )
        for module in model.modules():
            if not isinstance(module, Qwen4ExpAttention):
                continue
            layer_cache = runtime.kv_cache.get_layer_cache(module.layer_num)
            raw = layer_cache.kv_cache_base
            if raw is None:
                raise ValueError(f"QSA layer {module.layer_num} has no RTP KV cache")
            # The QSA module is already TP-sharded; RTP allocates this layer's
            # cache for its local KV heads and may pad the raw hybrid stride.
            stored_heads = int(module.num_kv_heads)
            paged = reshape_paged_kv_cache(
                raw, stored_heads, block_size, module.head_dim
            )
            if paged.shape[1] != 2 or paged.shape[2] < module.num_kv_heads:
                raise ValueError(
                    f"QSA layer {module.layer_num} invalid RTP KV shape {tuple(paged.shape)}"
                )
            key = paged[:, 0, : module.num_kv_heads].permute(0, 2, 1, 3)
            value = paged[:, 1, : module.num_kv_heads].permute(0, 2, 1, 3)
            num_blocks = int(paged.shape[0])
            signature = (raw.data_ptr(), num_blocks, block_size, str(raw.dtype))
            if getattr(module, "_rtp_qsa_cache_signature", None) == signature:
                continue
            device = raw.device
            index_dim = int(module.indexer.index_head_dim)
            side = (
                torch.empty(
                    (num_blocks, block_size, 1, index_dim),
                    device=device,
                    dtype=raw.dtype,
                ),
                torch.empty(
                    (
                        num_blocks,
                        block_size // module.indexer.compress_ratio,
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
        runtime: Any, model: Any, attn_metadata: Any
    ) -> Qwen4ExpPLEMetadata | None:
        ple_layers = [
            layer.ple for layer in model.model.layers if layer.ple is not None
        ]
        if not ple_layers:
            return None
        if len(ple_layers) != 1:
            raise ValueError("RTP Qwen4Exp currently supports one PLE layer")
        gdn = attn_metadata.gdn_metadata
        if gdn is None or gdn.non_spec_state_indices_tensor is None:
            raise ValueError("PLE requires GDN state-slot metadata")
        ple = ple_layers[0]
        raw = runtime.kv_cache.get_layer_cache(1).kv_cache_base
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
        indices = gdn.non_spec_state_indices_tensor
        input_indices = gdn.non_spec_state_indices_in_tensor
        has_initial = gdn.has_initial_state
        if has_initial is None:
            if torch.cuda.is_current_stream_capturing():
                graph_buffers = runtime._cg_meta_bufs
                has_initial = graph_buffers["qwen4_has_initial"][: indices.numel()]
                torch.ge(
                    attn_metadata.qsa_metadata.logical_positions,
                    0,
                    out=has_initial,
                )
            else:
                has_initial = torch.ones_like(indices, dtype=torch.bool)
        return Qwen4ExpPLEMetadata(
            query_start_loc=attn_metadata.plugin_metadata.rtp_cu_seqlens_q,
            ngram_state=runtime._rtp_ple_ngram,
            state_indices_in=input_indices,
            state_indices_out=indices,
            has_initial_state=has_initial,
            conv_state=runtime._rtp_ple_conv,
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
        layer_maps=None,
        cg_max_seq_len: int = 0,
        cg_bufs: dict | None = None,
    ) -> Iterator[None]:
        with super().bind(
            model=model,
            runtime=runtime,
            inputs=inputs,
            positions=positions,
            layer_maps=layer_maps,
            cg_max_seq_len=cg_max_seq_len,
            cg_bufs=cg_bufs,
        ):
            forward_context = get_forward_context()
            metadata = forward_context.attn_metadata
            gdn = metadata.gdn_metadata
            if gdn is not None and gdn.non_spec_state_indices_in_tensor is None:
                output_slots = gdn.non_spec_state_indices_tensor
                if gdn.num_decodes:
                    # At a block boundary, decode reads the previous block's
                    # recurrent state and writes the new block's state.
                    linear_inputs = inputs.attention_inputs
                    block_table = linear_inputs.kv_cache_kernel_block_id_device
                    block_size = metadata.rtp_seq_size_per_block
                    graph_output = None
                    if torch.cuda.is_current_stream_capturing():
                        if cg_bufs is None:
                            raise RuntimeError(
                                "Qwen4Exp Graph requires prewarmed state-slot buffer"
                            )
                        graph_output = cg_bufs["qwen4_previous_slots_i32"][
                            : output_slots.numel()
                        ]
                    gdn.non_spec_state_indices_in_tensor = cls._previous_state_slots(
                        block_table,
                        positions,
                        output_slots,
                        block_size,
                        graph_output=graph_output,
                    )
                else:
                    gdn.non_spec_state_indices_in_tensor = output_slots
            full_block_size = int(runtime.kv_cache.get_seq_size_per_block("full"))
            cls._bind_qsa_caches(runtime, model, full_block_size)
            plugin = metadata.plugin_metadata
            logical = positions.reshape(-1).to(torch.int64)
            full_inputs = getattr(inputs, "attention_inputs_by_tag", {}).get("full")
            if full_inputs is None:
                raise ValueError("Qwen4Exp requires RTP's full-attention cache tag")
            full_blocks = full_inputs.kv_cache_kernel_block_id_device
            if full_blocks is None or full_blocks.numel() == 0:
                raise ValueError("Qwen4Exp full-attention block table is empty")
            in_capture = torch.cuda.is_current_stream_capturing()
            graph_buffers = cg_bufs if in_capture else None
            if in_capture:
                if graph_buffers is None:
                    raise RuntimeError("Qwen4Exp Graph requires prewarmed QSA buffers")
                slots = graph_buffers["qsa_slots_i64"][: positions.numel()]
                compressed = graph_buffers["qsa_compressed_slots"][: slots.numel()]
                _qsa_graph_slots_kernel[(triton.cdiv(slots.numel(), 128),)](
                    full_blocks,
                    positions,
                    slots,
                    compressed,
                    slots.numel(),
                    full_blocks.stride(0),
                    full_blocks.stride(1),
                    full_blocks.shape[1],
                    full_block_size,
                    int(model.model.layers[3].self_attn.indexer.compress_ratio),
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
                    int(model.model.layers[3].self_attn.indexer.compress_ratio),
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
            metadata.ple_metadata = cls._ple_metadata(runtime, model, metadata)
            yield
