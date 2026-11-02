# SPDX-License-Identifier: MIT
from dataclasses import dataclass

from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry


def geometry(config, block=32):
    """A PAGE of 32 because the index plane is FP8.

    A block id names 16 rows and a ratio-2 owner halves the PAGE before that
    count is taken, so 32 tokens is the floor -- the same floor production
    rounds up to 256 for block-table reasons.
    """
    return V41PoolGeometry(
        config.num_hidden_layers,
        tuple(
            (owner, config.compress_ratios[owner])
            for owner in config.kv_source_layer_ids
        ),
        block,
        config.sliding_window,
        config.head_dim,
        config.index_head_dim,
        layer_ratios=tuple(
            sorted(set(config.compress_ratios[: config.num_hidden_layers]))
        ),
        index_topk=config.index_topk,
    )


def metadata_buffers(batch_size, tokens, blocks, device="cpu", geometry=None):
    """The `forward_vars` a hand-assembled builder needs, declared once.

    `geometry` adds the per-ratio buffers that geometry owns: the compressor
    plans and the indexer visibility. Pass the same object the builder gets --
    the names are keyed by ratio, so a second geometry would declare buffers
    no forward looks up, and `begin_step` asks for the ratios ITS geometry has.
    """
    import torch

    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.utils import CpuGpuBuffer

    buffers = {
        name: CpuGpuBuffer(
            *shape,
            dtype=torch.int64 if name == "positions" else torch.int32,
            device=device,
            pin_memory=torch.device(device).type != "cpu",
        )
        for name, shape in {
            "positions": (tokens,),
            "cu_seqlens_q": (batch_size + 1,),
            "batch_id_per_q_token": (tokens,),
            "block_tables": (batch_size, blocks),
            "input_ids": (tokens,),
        }.items()
    }
    buffers.update(
        DeepseekV4AttentionMetadataBuilder._state_slot_buffers(
            batch_size,
            device,
            read_side=False,
        )
    )
    buffers.update(DeepseekV41MetadataBuilder._engram_rows_buffer(tokens, device))
    if geometry is not None:
        buffers.update(
            DeepseekV41MetadataBuilder._compress_plan_buffers(
                geometry,
                tokens,
                batch_size,
                device,
            )
        )
        buffers.update(
            DeepseekV41MetadataBuilder._visible_buffers(
                geometry,
                tokens,
                device,
            )
        )
    return buffers


# Test cases bundle a request's span and its page mapping for readability.
# Production metadata receives them separately and retains only RequestSpan.


@dataclass(frozen=True)
class PagedRequest(RequestSpan):
    block_ids: tuple[int, ...]

    @property
    def span(self):
        return RequestSpan(
            self.request_id, self.position, self.offset, self.length, self.slot
        )


def begin_step(cache, requests, **kwargs):
    requests = tuple(requests)
    return cache.begin_step(
        [request.span for request in requests],
        block_tables=[request.block_ids for request in requests],
        **kwargs,
    )


def prepare_step(requests, device, **kwargs):
    from atom.model_ops.attentions.deepseek_v41.metadata import prepare_batch_step

    return prepare_batch_step(
        [request.span for request in requests],
        device,
        block_tables=[request.block_ids for request in requests],
        **kwargs,
    )


def publish_tables(buffer, requests, running_bs):
    from atom.utils.block_tables import block_table_state

    return (
        block_table_state(buffer)
        .prepare([request.block_ids for request in requests], pad_to=running_bs)
        .publish(running_bs)
    )


def decoder_publisher(cache, running_bs, capacity, blocks, device):
    """Publish a step's decoder plan as a serving builder does
    (`DeepseekV41MetadataBuilder._publish_decoder`) at ``running_bs``, from
    one builder assembled around ``cache`` whose token buffers hold
    ``capacity`` rows and whose block tables ``blocks`` PAGEs: call it with a
    step and its requests' PAGEs."""
    from types import SimpleNamespace

    from atom.model_ops.attentions.deepseek_v41.backend import (
        DECODER_ROWS,
        DeepseekV41MetadataBuilder,
    )

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.device, builder.geometry = device, cache.geometry
    builder.max_bs, builder.max_num_batched_tokens = running_bs, capacity
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(
            running_bs, capacity, blocks, device, cache.geometry
        )
    )
    builder.model_runner.forward_vars.update(
        builder._step_buffer_copies(DECODER_ROWS, builder._step_buffer_names())
        | builder._decoder_layout_buffers()
    )

    def publish(step, block_tables):
        step.decoder = builder._publish_decoder(step, block_tables, cache, running_bs)
        return step

    return publish


def publish_decoder(cache, step, block_tables, running_bs=None):
    """``step`` with its decoder plan at ``running_bs`` (its requests by
    default), from a builder whose buffers hold every request's window."""
    running_bs = running_bs or max(len(block_tables), 1)
    capacity = max(step.width, running_bs * cache.geometry.window_size)
    blocks = max((len(row) for row in block_tables), default=1)
    publish = decoder_publisher(
        cache, running_bs, capacity, blocks, step.positions.device
    )
    return publish(step, block_tables)
