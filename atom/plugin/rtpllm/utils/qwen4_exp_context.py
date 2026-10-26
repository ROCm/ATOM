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
        if gdn is None or gdn.non_spec_state_indices_tensor is None:
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
