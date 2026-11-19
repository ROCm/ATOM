"""Qwen3.8 QSA/PLE metadata on top of RTP's hybrid-attention cache."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

import torch
import triton
import triton.language as tl
from rtp_llm.ops import DataType

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from atom.model_ops.attentions.qwen4_exp_attn import (
    Qwen4ExpPLEMetadata,
    Qwen4ExpQSAMetadata,
)
from atom.model_ops.fla_ops.gdn_flydsl import (
    GDNFlyDSLPolicy,
    build_prefill_metadata,
    select_policy,
)
from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention
from atom.plugin.rtpllm.utils.forward_context import (
    RTPForwardContext,
    RTPForwardQwen35HybridContext,
    _target_verify_state_slots_graph_kernel,
)
from atom.utils import envs
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
    slot_page_size: tl.constexpr,
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
    slot = tl.where(page >= 0, page * slot_page_size + position % slot_page_size, -1)
    compressed = tl.where(
        (slot >= 0) & ((position + 1) % ratio == 0), slot // ratio, -1
    )
    tl.store(slots + row, slot, live)
    tl.store(compressed_slots + row, compressed, live)


class RTPQwen4ExpContext(RTPForwardQwen35HybridContext):
    """Bind the QSA side caches and PLE windows to RTP's request/page IDs."""

    @staticmethod
    def _prefill_state_checkpoints(
        attn_inputs: Any,
        metadata: GDNAttentionMetadata,
        block_size: int,
        kernel_block_size: int,
    ) -> dict[str, torch.Tensor] | None:
        """Snapshot every cacheable interior block, not just the prefill tail.

        RTP publishes completed hash blocks even when one prefill continues
        past them. The GDN/PLE state for each published block must therefore
        be materialized at that exact token boundary before a prefix hit can
        safely skip the preceding tokens.
        """
        from atom.model_ops.fla_ops.chunk import CHUNK_SIZE
        from atom.model_ops.fla_ops.index import prepare_chunk_offsets

        if block_size <= 0 or block_size % CHUNK_SIZE:
            raise ValueError("prefix cache requires a chunk-aligned LINEAR block")
        # RTP's kernel block width lives on the tagged KV cache, not on the
        # attention input. A missing/zero kernel width means the base width.
        kernel_block_size = int(kernel_block_size) or block_size
        if kernel_block_size != block_size:
            raise ValueError("prefix cache requires matching LINEAR kernel blocks")
        lengths = attn_inputs.input_lengths
        prefixes = attn_inputs.prefix_lengths
        table = attn_inputs.kv_cache_block_id
        if (
            lengths.device.type != "cpu"
            or prefixes.device.type != "cpu"
            or table.device.type != "cpu"
            or table.ndim != 2
            or table.shape[0] != lengths.numel()
            or prefixes.numel() != lengths.numel()
        ):
            raise ValueError("prefix checkpoints require host LINEAR block metadata")
        rows, slots, offsets = [], [], []
        for row, (start, length) in enumerate(zip(prefixes.tolist(), lengths.tolist())):
            if start < 0 or length < 0:
                raise ValueError("prefix checkpoint lengths must be non-negative")
            if length == 0:
                # Graph padding can have no tokens and no block-table columns.
                continue
            end = start + length
            # RTP's computeChunkGrant aligns every non-final chunk to the KV
            # block, and a prefix hit is block-aligned. Hence an interior
            # publishable boundary is also on this step's 64-token h grid.
            first_boundary = ((start // block_size) + 1) * block_size
            positions = range(first_boundary, end, block_size)
            if not positions:
                continue
            first_column = (first_boundary - 1) // block_size
            last_column = first_column + len(positions)
            if last_column > table.shape[1]:
                raise ValueError("prefix checkpoint exceeds LINEAR block table")
            # A single host slice avoids a torch scalar extraction for every
            # published block in a long cold prompt.
            row_slots = table[row, first_column:last_column].tolist()
            for position, slot in zip(positions, row_slots):
                offset = position - start
                if offset % CHUNK_SIZE:
                    raise ValueError("prefix checkpoint is not on the GDN chunk grid")
                if slot < 0:
                    continue  # RTP does not publish an unmaterialized LINEAR block.
                rows.append(row)
                slots.append(slot)
                offsets.append(offset)
        if not rows:
            return None
        device = metadata.non_spec_query_start_loc.device

        metadata.ssm_chunk_offsets = prepare_chunk_offsets(
            metadata.non_spec_query_start_loc, CHUNK_SIZE
        )
        # Upload all checkpoint indices together; separate pageable H2D copies
        # here otherwise add four host launches for every chunked prefill.
        packed = torch.tensor(
            [rows, slots, offsets, [0] * len(rows)],
            dtype=torch.int32,
            pin_memory=device.type == "cuda",
        ).to(device, non_blocking=True)
        gpu_slots = packed[1]
        return {
            "rows": packed[0],
            "slots": gpu_slots,
            "offs": packed[2],
            "is_end": packed[3],
            # RTP only snapshots interior boundaries; the kernel never reads
            # runtime slots when is_end=0.
            "runtime": gpu_slots,
        }

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
        # RTP's device-first MTP prepare may supply GPU input lengths. Avoid
        # tolist() on that path (and especially during Graph capture).
        if query_len not in (2, 3, 4):
            raise ValueError(
                "Qwen4Exp MTP requires 2, 3, or 4 verify tokens per request "
                f"(tokens={num_tokens}, batch={batch_size}, query_len={query_len})"
            )
        lengths_on_cpu = attn_inputs.input_lengths.device.type == "cpu"
        if lengths_on_cpu and any(
            int(length) != query_len for length in attn_inputs.input_lengths.tolist()
        ):
            raise ValueError("Qwen4Exp MTP requires a fixed verify window")
        seq_lens = attn_inputs.sequence_lengths_plus_1_device
        if seq_lens is None or seq_lens.numel() != batch_size:
            raise ValueError("invalid Qwen4Exp MTP target-verify sequence lengths")
        in_capture = torch.cuda.is_current_stream_capturing()
        split_conv_slots = None
        if in_capture:
            if (
                triton is None
                or cg_bufs is None
                or "target_read_slots_i32" not in cg_bufs
            ):
                raise RuntimeError(
                    "Qwen4Exp MTP target Graph requires prewarmed buffers"
                )
            if (
                batch_size > cg_bufs["target_read_slots_i32"].shape[0]
                or batch_size > cg_bufs["target_write_slots_i32"].shape[0]
                or query_len > cg_bufs["target_read_slots_i32"].shape[1]
                or query_len > cg_bufs["target_write_slots_i32"].shape[1]
                or batch_size + 1 > cg_bufs["query_start_loc"].numel()
                or triton.cdiv(batch_size, 128)
                > cg_bufs["target_invalid_live_rows"].numel()
            ):
                raise ValueError("Qwen4Exp MTP target Graph buffers are too small")
            query_start_loc = cg_bufs["query_start_loc"][: batch_size + 1]
            read_slots = cg_bufs["target_read_slots_i32"][:batch_size]
            write_slots = cg_bufs["target_write_slots_i32"][:batch_size]
            invalid_live_rows = cg_bufs["target_invalid_live_rows"][
                : triton.cdiv(batch_size, 128)
            ]
            prefix_lengths = getattr(attn_inputs, "prefix_lengths_device", None)
            if (
                prefix_lengths is None
                or prefix_lengths.device != block_table.device
                or prefix_lengths.dtype != torch.int32
                or prefix_lengths.numel() != batch_size
            ):
                raise RuntimeError(
                    "Qwen4Exp MTP target Graph requires device prefix lengths"
                )
            from atom.plugin.rtpllm.utils.split_ssm_storage import target_slots
            split_conv_slots = target_slots(
                attn_inputs, block_table, seq_lens, prefix_lengths, read_slots,
                write_slots, invalid_live_rows, batch_size, seq_size_per_block, query_len,
            )
            if split_conv_slots is None:
                _target_verify_state_slots_graph_kernel[(triton.cdiv(batch_size, 128),)](
                    block_table,
                    seq_lens,
                    prefix_lengths,
                    read_slots,
                    write_slots,
                    invalid_live_rows,
                    batch_size,
                    block_table.stride(0),
                    block_table.stride(1),
                    block_table.shape[1],
                    seq_size_per_block,
                    query_len,
                    128,
                )
            # The index kernel overwrites every active tile's flag on each
            # replay. A bad live row must fail before GDN touches block 0.
            valid_slots = (
                invalid_live_rows[0] == 0
                if invalid_live_rows.numel() == 1
                else torch.all(invalid_live_rows == 0)
            )
            torch._assert_async(
                valid_slots, "Qwen4Exp MTP target-verify state block is invalid"
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
            invalid_columns = (read_col < 0) | (read_col >= block_table.shape[1])
            invalid_columns |= torch.any(
                (write_cols < 0) | (write_cols >= block_table.shape[1]), dim=1
            )
            read_slots = block_table[
                rows[:, 0], read_col.clamp(0, block_table.shape[1] - 1)
            ]
            write_slots = block_table[
                rows, write_cols.clamp(0, block_table.shape[1] - 1)
            ]
            invalid_slots = (read_slots < 0) | torch.any(write_slots < 0, dim=1)
            # One host check is required for eager fail-fast; do not sync once
            # for the table width and again for the block IDs.
            invalid_rows = invalid_columns | invalid_slots
            if not lengths_on_cpu:
                invalid_rows |= attn_inputs.input_lengths != query_len
            if bool(torch.any(invalid_rows).item()):
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
        from atom.plugin.rtpllm.utils.split_ssm_storage import attach_target_conv_metadata
        attach_target_conv_metadata(metadata, split_conv_slots)
        return metadata

    @staticmethod
    def collect_layer_maps(model: Any):
        gdn, full, mla = RTPForwardQwen35HybridContext.collect_layer_maps(model)
        for module in model.modules():
            if isinstance(module, Qwen4ExpAttention):
                full[int(module.layer_num)] = module
        return gdn, full, mla

    @staticmethod
    def _kv_cache_signature(runtime, layer_maps):
        from atom.plugin.rtpllm.utils.linear_double_buffer import pool_signature

        return RTPForwardQwen35HybridContext._kv_cache_signature(runtime, layer_maps) + pool_signature(runtime, layer_maps[0])

    @staticmethod
    def _build_kv_cache_tensors(runtime: Any, layer_maps: Any) -> dict[str, Any]:
        from atom.plugin.rtpllm.utils.split_ssm_storage import enabled, build_caches
        parent = RTPForwardQwen35HybridContext._build_kv_cache_tensors
        from atom.plugin.rtpllm.utils.linear_double_buffer import enabled as request_pool_enabled, build_cache_views
        if request_pool_enabled(runtime):
            return build_cache_views(runtime, layer_maps, parent)
        caches = build_caches(runtime, layer_maps, parent) if enabled() else parent(runtime, layer_maps)
        model_config = getattr(runtime, "_rtp_model_config", None)
        if model_config is None:
            model_config = getattr(runtime, "model_config", None)
        linear_config = getattr(model_config, "linear_attention_config", None)
        if not enabled() and getattr(linear_config, "ssm_state_dtype", None) == DataType.TYPE_FP32:
            for layer_num, layer in layer_maps[0].items():
                raw = runtime.kv_cache.get_layer_cache(layer_num, "linear").kv_cache_base
                base = raw.reshape(raw.shape[0], -1)
                if base.dtype != torch.bfloat16 or not base.is_contiguous():
                    raise ValueError(
                        "Qwen4Exp FP32 GDN state requires BF16 byte storage"
                    )
                heads = int(layer.dt_bias.numel())
                k_dim, v_dim = int(layer.head_k_dim), int(layer.head_v_dim)
                state_elems = heads * k_dim * v_dim
                conv_dim = int(layer.conv1d.weight.size(0))
                conv_length = int(layer.conv1d.weight.size(2)) - 1
                if base.shape[1] < 2 * state_elems + conv_dim * conv_length:
                    raise ValueError(
                        "Qwen4Exp FP32 GDN cache is smaller than its layout"
                    )
                fp32_base = base.view(torch.float32)
                state = torch.as_strided(
                    fp32_base,
                    (base.shape[0], heads, k_dim, v_dim),
                    (fp32_base.stride(0), k_dim * v_dim, v_dim, 1),
                    storage_offset=fp32_base.storage_offset(),
                )
                conv = torch.as_strided(
                    base,
                    (base.shape[0], conv_dim, conv_length),
                    (base.stride(0), 1, conv_dim),
                    storage_offset=base.storage_offset() + 2 * state_elems,
                )
                caches[f"layer_{layer_num}"].v_cache = state
                caches[f"layer_{layer_num}"].k_cache = conv

        # AITER decode expects physical VK storage exposed as a logical KV
        # view. Bind that view before the first state write; the prefill path
        # and Triton decode fallback both honor its inner strides. MTP target
        # verification uses physical VK storage for FlyDSL MTP.
        from atom.plugin.rtpllm.utils.replay_policy import flydsl_mtp_enabled
        if (not getattr(model_config, "atom_qwen4_exp_mtp_target", False)
            and not getattr(model_config, "is_mtp", False)) or (
                flydsl_mtp_enabled()
                and getattr(model_config, "atom_qwen4_exp_mtp_target", False)):
            for layer_num, layer in layer_maps[0].items():
                state = caches[f"layer_{layer_num}"].v_cache
                k_dim, v_dim = state.shape[-2:]
                policy = select_policy(
                    allowed=layer.allow_aiter_flydsl,
                    replayssm=False,
                    lossy_decode=False,
                    state=state,
                    activation_dtype=layer.dt_bias.dtype,
                )
                if policy.decode and state.stride()[1:] == (
                    k_dim * v_dim,
                    v_dim,
                    1,
                ):
                    caches[f"layer_{layer_num}"].v_cache = state.transpose(-1, -2)
        return caches

    @staticmethod
    def _configure_gdn_backends(
        metadata: GDNAttentionMetadata | None,
        gdn_layers: dict[int, Any],
        kv_cache_data: dict[str, Any],
        attn_inputs: Any,
    ) -> None:
        """Match standalone GDN backends when the state layout supports them."""
        if metadata is None or not gdn_layers:
            return
        enabled = False
        for layer_num, layer in gdn_layers.items():
            cache = kv_cache_data.get(f"layer_{layer_num}")
            state = getattr(cache, "v_cache", None)
            if (
                state is None
                or state.ndim != 4
                or state.dtype not in (torch.bfloat16, torch.float32)
            ):
                layer.gdn_flydsl_policy = GDNFlyDSLPolicy()
                continue
            policy = select_policy(
                allowed=layer.allow_aiter_flydsl,
                replayssm=metadata.replayssm,
                lossy_decode=False,
                state=state,
                activation_dtype=layer.dt_bias.dtype,
            )
            # RTP-LLM MTP prefill has no FlyDSL accuracy or perf data under
            # ReplaySSM yet, so it stays on Triton there.
            prefill = policy.prefill and not metadata.replayssm
            layer.gdn_flydsl_policy = GDNFlyDSLPolicy(
                prefill=prefill,
                decode=(
                    policy.decode
                    and metadata.num_spec_decodes == 0
                    and state.stride()[1:]
                    == (state.shape[-2] * state.shape[-1], 1, state.shape[-2])
                ),
            )
            enabled |= prefill
        if not enabled or metadata.num_prefills == 0:
            return
        lengths = getattr(attn_inputs, "input_lengths", None)
        if lengths is None or lengths.device.type != "cpu":
            # A device-to-host read here would synchronize every prefill (and
            # fail during Graph capture); keep the original Triton path.
            return
        query_lengths = tuple(int(n) for n in lengths.tolist())
        if sum(query_lengths) != metadata.num_prefill_tokens:
            raise ValueError("Qwen4Exp prefill lengths do not match token count")
        metadata.flydsl_prefill_metadata = build_prefill_metadata(
            query_lengths, metadata.non_spec_query_start_loc
        )

    @classmethod
    def _resolve_qsa_block_table(
        cls,
        full_inputs: Any,
        block_size: int,
        kernel_block_size: int,
        graph_buffers: dict | None,
    ) -> torch.Tensor:
        # RTP refreshes the kernel block table on every Graph replay, but not
        # kv_cache_block_id_device (that table is only used by cache store).
        # Derive QSA's physical pages from the live kernel table in Graph mode.
        table = (
            None
            if graph_buffers is not None
            else getattr(full_inputs, "kv_cache_block_id_device", None)
        )
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

    @classmethod
    def _qsa_state_slots(cls, tag, prefix, runtime, inputs, logical, query_starts,
                         page_size, ratio, graph_buffers):
        tagged = inputs.attention_inputs_by_tag[tag]
        if int(runtime.kv_cache.get_seq_size_per_block(tag)) != page_size:
            raise ValueError(f"{tag} and full cache page sizes differ")
        buffers = None
        verify = getattr(tagged, "is_target_verify", False)
        if graph_buffers is not None:
            buffers = dict(graph_buffers)
            buffers["physical_block_table_i32"] = buffers[prefix + "raw_physical_block_table_i32"]
            if verify:
                for key, name in (("target_read_slots_i32", "read_slots_i32"),
                                  ("target_write_slots_i32", "write_slots_i32"),
                                  ("target_invalid_live_rows", "invalid_live_rows")):
                    buffers[key] = buffers[prefix + "qsa_state_" + name]
        blocks = cls._resolve_qsa_block_table(
            tagged, page_size, int(runtime.kv_cache.get_kernel_seq_size_per_block(tag)), buffers
        )
        read = candidates = None
        if verify:
            meta = cls._build_gdn_metadata(tagged, seq_size_per_block=page_size,
                                           num_tokens=logical.numel(), cg_bufs=buffers)
            read = meta.spec_state_indices_in_tensor[:, 0]
            candidates = meta.spec_state_indices_tensor
            pages = candidates.reshape(-1).to(torch.int64)
            slots = torch.where(pages > 0, pages * (2 * ratio) + logical.remainder(2 * ratio), -1)
        elif buffers is not None:
            slots = buffers[prefix + "qsa_raw_slots_i64"][:logical.numel()]
            _qsa_graph_slots_kernel[(triton.cdiv(slots.numel(), 128),)](
                blocks, logical, buffers["seq_id_i32"], slots,
                buffers["qsa_raw_compressed_slots"], slots.numel(),
                blocks.stride(0), blocks.stride(1), blocks.shape[1], page_size, 2 * ratio, ratio, 128,
            )
        else:
            slots = cls._build_slot_mapping(positions=logical, query_start_loc=query_starts,
                                            block_table=blocks, seq_size_per_block=page_size)
            slots = torch.where(slots >= 0, (slots // page_size) * (2 * ratio) + logical.remainder(2 * ratio), slots)
        return blocks, slots, read, candidates

    @staticmethod
    def _bind_qsa_caches(runtime: Any, qsa_layers: list[Any], block_size: int) -> None:
        from rtp_llm.models_py.modules.factory.attention.common import (
            reshape_paged_kv_cache,
        )

        cfg = runtime._rtp_model_config
        has_positions = not cfg.is_mtp and cfg.mm_model_config.is_multimodal
        for module in qsa_layers:
            ratio = int(module.indexer.compress_ratio)
            if block_size <= 0 or ratio <= 0 or block_size % ratio:
                raise ValueError(
                    f"QSA block size {block_size} must be divisible by compression ratio {ratio}"
                )
            layer_cache = runtime.kv_cache.get_layer_cache(module.layer_num, "full")
            raw = layer_cache.kv_cache_base
            if raw is None:
                raise ValueError(f"QSA layer {module.layer_num} has no RTP KV cache")
            is_fp8 = raw.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            if raw.dtype != torch.bfloat16 and not is_fp8:
                raise ValueError("QSA RTP KV cache requires BF16 or FP8 storage")
            scale = layer_cache.kv_scale_base
            if is_fp8 and (scale is None or scale.numel() == 0):
                raise ValueError("QSA FP8 KV cache requires per-token FP32 scales")
            num_blocks = int(raw.shape[0])
            index_base = runtime.kv_cache.get_layer_cache(module.layer_num, "indexer_kv").kv_cache_base
            raw_base = runtime.kv_cache.get_layer_cache(module.layer_num, "indexer_state").kv_cache_base
            if index_base is None or index_base.dtype != torch.uint8 or index_base.ndim != 2:
                raise ValueError("QSA indexer_kv pool is missing or has invalid storage")
            if raw_base is None or raw_base.dtype != torch.float32 or raw_base.ndim != 2:
                raise ValueError("QSA indexer_state pool is missing or has invalid storage")
            index_blocks = int(index_base.shape[0])
            position_base = None
            if has_positions:
                position_base = runtime.kv_cache.get_layer_cache(
                    module.layer_num, "indexer_rope_positions"
                ).kv_cache_base
                if (position_base is None or position_base.dtype != torch.int32
                        or position_base.ndim != 2 or position_base.size(1) < 2 * ratio * 3):
                    raise ValueError("invalid indexer_rope_positions pool")
            signature = (raw.data_ptr(), num_blocks, block_size, str(raw.dtype),
                         scale.data_ptr() if is_fp8 else 0, index_base.data_ptr(), raw_base.data_ptr(),
                         position_base.data_ptr() if has_positions else 0)
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
            k_scale = v_scale = None
            if is_fp8:
                if scale.dtype != torch.float32 or scale.numel() != num_blocks * 2 * block_size * stored_heads:
                    raise ValueError("QSA FP8 KV scale layout does not match RTP cache")
                token_scale = scale.view(num_blocks, 2, block_size, stored_heads)
                k_scale, v_scale = token_scale[:, 0], token_scale[:, 1]
            device = raw.device
            index_dim = int(module.indexer.index_head_dim)
            state_rows = 2 * ratio
            state_bytes = state_rows * index_dim * 4
            compressed_bytes = block_size // ratio * index_dim * 2
            if raw_base.shape[1] * raw_base.element_size() < state_bytes or index_base.shape[1] < compressed_bytes:
                raise ValueError("QSA side pool row is too small")
            side = (
                raw_base[:, :state_rows * index_dim].reshape(
                    raw_base.shape[0], state_rows, 1, index_dim
                ),
                index_base[:, :compressed_bytes]
                .view(torch.bfloat16).reshape(index_blocks, block_size // ratio, 1, index_dim),
            )
            rope_positions = None if position_base is None else position_base[:, :state_rows * 3].view(
                position_base.size(0), state_rows, 1, 3
            )
            module.bind_caches(key, value, *side, rope_position_cache=rope_positions,
                               k_scale_cache=k_scale, v_scale_cache=v_scale)
            module._rtp_qsa_cache_signature = signature

    @staticmethod
    def _bind_ple_caches(runtime: Any, model: Any, gdn_layers: dict[int, Any]):
        ple_layers = [
            layer.ple for layer in model.model.layers if layer.ple is not None
        ]
        if not ple_layers:
            return None
        if len(ple_layers) != 1:
            raise ValueError("RTP Qwen4Exp currently supports one PLE layer")
        ple = ple_layers[0]
        if not gdn_layers:
            raise ValueError("PLE requires a GDN layer cache")
        ple_layer = next(
            idx for idx, layer in enumerate(model.model.layers) if layer.ple is not None
        )
        conv_raw = runtime.kv_cache.get_layer_cache(ple_layer, "ple_conv_state").kv_cache_base
        ngram_raw = runtime.kv_cache.get_layer_cache(ple_layer, "ple_ngram_ctx").kv_cache_base
        if conv_raw is None or ngram_raw is None:
            raise ValueError("PLE state pools are missing")
        conv_slots = int(conv_raw.shape[0])
        ngram_slots = int(ngram_raw.shape[0])
        signature = (conv_raw.data_ptr(), ngram_raw.data_ptr(), conv_slots,
                     ngram_slots, ple.hc_hidden_size, ple.conv_state_len)
        if getattr(runtime, "_rtp_ple_cache_signature", None) != signature:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Qwen4Exp PLE caches must be bound before Graph capture"
                )
            from rtp_llm.models_py.utils.cache_payload_view import (
                CachePayloadGeometry, payload_view,
            )
            ngram_width = ple.ple_embedding.ngram_size - 1
            runtime._rtp_ple_ngram = payload_view(
                ngram_raw,
                CachePayloadGeometry(torch.int64, torch.int64, ngram_width, 1),
                "PLE ngram context",
            ).squeeze(-1)
            runtime._rtp_ple_conv = payload_view(
                conv_raw,
                CachePayloadGeometry(torch.bfloat16, torch.bfloat16,
                                     ple.conv_state_len, ple.hc_hidden_size),
                "PLE convolution state",
            ).transpose(1, 2)
            runtime._rtp_ple_ngram.fill_(int(model.config.eos_token_id))
            runtime._rtp_ple_conv.zero_()
            runtime._rtp_ple_cache_signature = signature
        return ple

    @classmethod
    def _ple_metadata(
        cls,
        runtime: Any,
        attn_metadata: Any,
        ple_state_metadata: GDNAttentionMetadata | None = None,
        ngram_state_metadata: GDNAttentionMetadata | None = None,
    ) -> Qwen4ExpPLEMetadata | None:
        gdn = ple_state_metadata
        if gdn is None or (
            gdn.non_spec_state_indices_tensor is None
            and gdn.spec_state_indices_tensor is None
        ):
            raise ValueError("PLE requires GDN state-slot metadata")
        ngram = ngram_state_metadata
        if ngram is None or (
            ngram.non_spec_state_indices_tensor is None
            and ngram.spec_state_indices_tensor is None
        ):
            raise ValueError("PLE requires ngram state-slot metadata")
        spec_indices = (
            gdn.spec_state_indices_tensor
            if getattr(gdn, "rtp_target_verify_block_map", None) is not None
            else None
        )
        if spec_indices is not None:
            indices = spec_indices[:, -1]
            input_indices = gdn.spec_state_indices_in_tensor[:, 0]
            ngram_indices = ngram.spec_state_indices_tensor[:, -1]
            ngram_input_indices = ngram.spec_state_indices_in_tensor[:, 0]
        else:
            indices = gdn.non_spec_state_indices_tensor
            input_indices = gdn.non_spec_state_indices_in_tensor
            ngram_indices = ngram.non_spec_state_indices_tensor
            ngram_input_indices = ngram.non_spec_state_indices_in_tensor
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
            ngram_indices_in=ngram_input_indices,
            ngram_indices_out=ngram_indices,
        )
        # RTP-only per-candidate slots are deliberately not part of the
        # framework-neutral Qwen4ExpPLEMetadata dataclass.
        metadata.rtp_spec_state_indices = spec_indices
        metadata.rtp_ngram_spec_state_indices = (
            ngram.spec_state_indices_tensor if spec_indices is not None else None
        )
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
        from atom.plugin.rtpllm.utils.split_ssm_storage import target_metadata_scope
        with target_metadata_scope(runtime, inputs, cg_bufs), super().bind(
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
            gdn_metadata = metadata.gdn_metadata
            if (
                getattr(
                    runtime._rtp_model_config,
                    "atom_qwen4_exp_prefix_cache_enabled",
                    False,
                )
                and gdn_metadata is not None
                and gdn_metadata.num_prefills
            ):
                gdn_metadata.ssm_checkpoints = cls._prefill_state_checkpoints(
                    inputs.attention_inputs,
                    gdn_metadata,
                    int(runtime.kv_cache.get_seq_size_per_block("linear")),
                    int(runtime.kv_cache.get_kernel_seq_size_per_block("linear")),
                )
                gdn_metadata.ssm_conv_state_len = (
                    int(model.config.linear_conv_kernel_dim) - 1
                )
            from atom.plugin.rtpllm.utils.replay_policy import flydsl_mtp_enabled
            from atom.plugin.rtpllm.utils.replay_hybrid import prepare_metadata

            if gdn_metadata is not None:
                gdn_metadata.rtp_flydsl_mtp = (
                    flydsl_mtp_enabled()
                    and bool(resolved_layer_maps[0])
                    and all(
                        layer.allow_aiter_flydsl
                        and forward_context.kv_cache_data[f"layer_{number}"].v_cache.stride(-2) == 1
                        for number, layer in resolved_layer_maps[0].items()
                    )
                    and gdn_metadata.num_spec_decodes > 0
                    and gdn_metadata.num_prefills == 0
                    and gdn_metadata.num_decodes == 0
                )
                if gdn_metadata.rtp_flydsl_mtp:
                    gdn_metadata.rtp_flydsl_read_indices = (
                        gdn_metadata.spec_state_indices_in_tensor[:, 0].contiguous()
                    )
            from atom.plugin.rtpllm.utils.split_ssm_storage import bind_conv
            bind_conv(cls, runtime, inputs, positions, gdn_metadata, cg_bufs)
            from atom.plugin.rtpllm.utils.linear_double_buffer import enabled as request_pool_enabled, prepare_metadata as prepare_request_pool
            if request_pool_enabled(runtime):
                prepare_request_pool(runtime, gdn_metadata, inputs)
            else:
                prepare_metadata(runtime, gdn_metadata, forward_context.kv_cache_data, resolved_layer_maps[0])
            cls._configure_gdn_backends(
                metadata.gdn_metadata,
                resolved_layer_maps[0],
                forward_context.kv_cache_data,
                inputs.attention_inputs,
            )
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
            index_inputs = getattr(inputs, "attention_inputs_by_tag", {}).get("indexer_kv")
            if index_inputs is None:
                raise ValueError("Qwen4Exp requires RTP's indexer_kv cache tag")
            index_graph_buffers = graph_buffers
            if graph_buffers is not None:
                index_graph_buffers = dict(graph_buffers)
                index_graph_buffers["physical_block_table_i32"] = graph_buffers[
                    "index_physical_block_table_i32"
                ]
            index_blocks = cls._resolve_qsa_block_table(
                index_inputs,
                int(runtime.kv_cache.get_seq_size_per_block("indexer_kv")),
                int(runtime.kv_cache.get_kernel_seq_size_per_block("indexer_kv")),
                index_graph_buffers,
            )
            if int(runtime.kv_cache.get_seq_size_per_block("indexer_kv")) != full_block_size:
                raise ValueError("QSA indexer_kv and full cache page sizes differ")
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
                    full_block_size,
                    ratio,
                    128,
                )
                index_slots = graph_buffers["qsa_index_slots_i64"][: slots.numel()]
                index_compressed = graph_buffers["qsa_index_compressed_slots"][: slots.numel()]
                _qsa_graph_slots_kernel[(triton.cdiv(slots.numel(), 128),)](
                    index_blocks,
                    positions,
                    graph_buffers["seq_id_i32"][: slots.numel()],
                    index_slots,
                    index_compressed,
                    slots.numel(),
                    index_blocks.stride(0),
                    index_blocks.stride(1),
                    index_blocks.shape[1],
                    full_block_size,
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
                index_slots = cls._build_slot_mapping(
                    positions=positions.reshape(-1),
                    query_start_loc=plugin.rtp_cu_seqlens_q,
                    block_table=index_blocks,
                    seq_size_per_block=full_block_size,
                )
                index_compressed = torch.empty_like(index_slots)
                qsa_compressed_slots(index_slots, logical, ratio, index_compressed)
            state_args = (runtime, inputs, logical, plugin.rtp_cu_seqlens_q,
                          full_block_size, ratio, graph_buffers)
            raw_blocks, raw_slots, state_read_pages, state_candidate_pages = cls._qsa_state_slots(
                "indexer_state", "", *state_args
            )
            position_blocks = position_slots = position_read = position_candidates = None
            if qsa_layers[0].rope_position_cache is not None:
                position_blocks, position_slots, position_read, position_candidates = cls._qsa_state_slots(
                    "indexer_rope_positions", "position_", *state_args
                )
            host_lens = context_slots = None
            if (
                envs.ATOM_QSA_FLYDSL_ATTN
                and forward_context.context.is_prefill
                and not runtime._rtp_model_config.is_mtp
                and not getattr(full_inputs, "is_target_verify", False)
                and not in_capture
                and full_inputs.input_lengths.device.type == "cpu"
                and full_inputs.prefix_lengths.device.type == "cpu"
            ):
                query_lens = tuple(int(v) for v in full_inputs.input_lengths.tolist())
                prefix_lens = tuple(int(v) for v in full_inputs.prefix_lengths.tolist())
                seq_lens = tuple(q + p for q, p in zip(query_lens, prefix_lens))
                if (len(query_lens) == full_blocks.shape[0]
                    and len(query_lens) == len(prefix_lens)
                    and sum(query_lens) == int(logical.numel())
                    and all(q >= 0 and p >= 0 for q, p in zip(query_lens, prefix_lens))):
                    host_lens = (query_lens, seq_lens)
                    if any(prefix_lens):
                        lengths = np.asarray(seq_lens, dtype=np.int64)
                        requests = np.repeat(np.arange(len(seq_lens), dtype=np.int64), lengths)
                        tokens = np.arange(len(requests), dtype=np.int64) - np.repeat(
                            np.cumsum(lengths) - lengths, lengths
                        )
                        host = torch.from_numpy(np.stack((requests, tokens))).pin_memory()
                        requests, tokens = host.to(full_blocks.device, non_blocking=True)
                        pages = full_blocks[requests, tokens // full_block_size].to(torch.int64)
                        context_slots = pages * full_block_size + tokens % full_block_size
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
                index_block_tables=index_blocks,
                index_slot_mapping=index_slots,
                index_compressed_slot_mapping=index_compressed,
                raw_index_block_tables=raw_blocks,
                raw_index_slot_mapping=raw_slots,
                state_page_size=int(runtime.kv_cache.get_seq_size_per_block("indexer_state")),
                position_block_tables=position_blocks,
                position_slot_mapping=position_slots,
                position_read_pages=position_read,
                position_candidate_pages=position_candidates,
                state_read_pages=state_read_pages,
                state_candidate_pages=state_candidate_pages,
                host_lens=host_lens,
                context_slots=context_slots,
            )
            # The MTP draft is a single QSA layer with no PLE cache groups.
            # Keep its QSA metadata live through forward without binding PLE.
            if runtime._rtp_model_config.is_mtp or cls._bind_ple_caches(
                runtime, model, resolved_layer_maps[0]
            ) is None:
                metadata.ple_metadata = None
                yield
                return
            tagged_inputs = getattr(inputs, "attention_inputs_by_tag", {})
            ple_inputs = tagged_inputs.get("ple_conv_state")
            ngram_inputs = tagged_inputs.get("ple_ngram_ctx")
            ple_state_metadata = ngram_state_metadata = None
            for tag, tagged, buffer_key in (
                ("ple_conv_state", ple_inputs, "ple_previous_state_slots_i32"),
                ("ple_ngram_ctx", ngram_inputs, "ple_ngram_previous_state_slots_i32"),
            ):
                if tagged is None:
                    raise ValueError(f"Qwen4Exp requires RTP's {tag} cache tag")
                tagged_buffers = graph_buffers
                if graph_buffers is not None:
                    tagged_buffers = dict(graph_buffers)
                    tagged_buffers["previous_state_slots_i32"] = graph_buffers[buffer_key]
                    if getattr(tagged, "is_target_verify", False):
                        for name in (
                            "target_read_slots_i32",
                            "target_write_slots_i32",
                            "target_invalid_live_rows",
                        ):
                            tagged_buffers[name] = graph_buffers[f"{tag}_{name}"]
                state_metadata = cls._build_gdn_metadata(
                    tagged,
                    seq_size_per_block=int(runtime.kv_cache.get_seq_size_per_block(tag)),
                    num_tokens=int(logical.numel()),
                    cg_bufs=tagged_buffers,
                )
                if tag == "ple_conv_state":
                    ple_state_metadata = state_metadata
                else:
                    ngram_state_metadata = state_metadata
            metadata.ple_metadata = cls._ple_metadata(
                runtime, metadata, ple_state_metadata, ngram_state_metadata,
            )
            if metadata.ple_metadata is not None and gdn_metadata is not None:
                ple_checkpoints = None
                if (
                    getattr(runtime._rtp_model_config, "atom_qwen4_exp_prefix_cache_enabled", False)
                    and ple_state_metadata is not None
                    and ple_state_metadata.num_prefills
                ):
                    ple_checkpoints = {
                        "conv": cls._prefill_state_checkpoints(
                            ple_inputs, ple_state_metadata,
                            int(runtime.kv_cache.get_seq_size_per_block("ple_conv_state")),
                            int(runtime.kv_cache.get_kernel_seq_size_per_block("ple_conv_state")),
                        ),
                        "ngram": cls._prefill_state_checkpoints(
                            ngram_inputs, ngram_state_metadata,
                            int(runtime.kv_cache.get_seq_size_per_block("ple_ngram_ctx")),
                            int(runtime.kv_cache.get_kernel_seq_size_per_block("ple_ngram_ctx")),
                        ),
                    }
                metadata.ple_metadata.rtp_checkpoint_targets = ple_checkpoints
            yield
