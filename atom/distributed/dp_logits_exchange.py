# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Same-node, four-GPU logits exchange through AITER's gfx1250 VMM transport."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.dist.device_communicators.vmm_allocator import VMMBuffer, vmm_exchange


@triton.jit
def _stage(Input, Peers, Rank: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    target = tl.multiple_of(tl.load(Peers + Rank), 16).to(
        tl.pointer_type(Input.dtype.element_ty)
    )
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Input + offsets, offsets < N, other=0)
    tl.store(target + offsets, values, offsets < N)


@triton.jit
def _gather(
    Peers,
    Output,
    Rank: tl.constexpr,
    Rows: tl.constexpr,
    LocalRows: tl.constexpr,
    Vocab: tl.constexpr,
    BLOCK: tl.constexpr,
):
    source = tl.program_id(1)
    peer = tl.multiple_of(tl.load(Peers + source), 16).to(
        tl.pointer_type(Output.dtype.element_ty)
    )
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        peer + Rank * Rows * Vocab + offsets,
        offsets < LocalRows * Vocab,
        other=0,
        cache_modifier=".cg",
    )
    row = offsets // Vocab
    column = offsets % Vocab
    output_offsets = row * (4 * Vocab) + source * Vocab + column
    # Vocab is a multiple of eight words, so each aligned group stays in a row.
    output_offsets = tl.max_contiguous(tl.multiple_of(output_offsets, 8), 8)
    tl.store(Output + output_offsets, values, offsets < LocalRows * Vocab)


class PeerLogitsExchange:
    def __init__(
        self,
        group,
        device,
        max_rows,
        vocab_shard,
        dtype=torch.bfloat16,
        *,
        barrier_group=None,
        barrier_communicator=None,
        namespace="dp_logits_input_v1",
    ):
        if dist.get_world_size(group) != 4:
            raise ValueError("Peer logits exchange requires four available local GPUs")
        if vocab_shard % 32 or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "Expected 16-bit logits with shards aligned to 32 elements"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize the peer pool before graph capture")
        self.rank = dist.get_rank(group)
        self.device = torch.device(device)
        self.barrier_group = barrier_group
        self.barrier_communicator = barrier_communicator
        if barrier_communicator is not None:
            if (
                barrier_communicator.disabled
                or barrier_communicator.rank != self.rank
                or barrier_communicator.world_size != 4
                or barrier_communicator.device != self.device
            ):
                raise ValueError("Barrier communicator does not match the peer pool")
        elif barrier_group is None:
            raise ValueError("A stream-ordered barrier communicator is required")
        self.max_rows = max_rows
        self.vocab = vocab_shard
        self.dtype = dtype
        width = torch.empty((), dtype=dtype).element_size()
        self.buffer = VMMBuffer(4 * max_rows * vocab_shard * width, self.device.index)
        devices = [None] * 4
        dist.all_gather_object(devices, self.device.index, group=group)
        ranks = dist.get_process_group_ranks(group)
        tag = "_".join(map(str, ranks))
        pointers, self.imports = vmm_exchange(
            self.rank,
            4,
            namespace,
            self.buffer,
            dist.distributed_c10d._get_default_store(),
            tag,
            devices,
        )
        if len(pointers) != 4 or not all(pointers):
            raise RuntimeError("Every peer must map the logits pool")
        self.peers = torch.tensor(pointers, dtype=torch.int64, device=self.device)
        self.token = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.barrier_output = torch.empty_like(self.token)
        torch.cuda.synchronize()

    def _barrier(self):
        # Stream-ordered all-reduce: all peers finish the preceding
        # reads before reuse, then publish their input copies before peer reads.
        if self.barrier_communicator is not None:
            self.barrier_communicator.all_reduce(
                self.token,
                out=self.barrier_output,
                registered_input=False,
            )
        else:
            dist.all_reduce(self.token, group=self.barrier_group)

    def __call__(self, x, local_rows):
        if (
            x.ndim != 2
            or x.shape[0] % 4
            or x.shape[1] != self.vocab
            or x.device != self.device
            or x.dtype != self.dtype
        ):
            raise ValueError("Input does not match the peer pool geometry")
        rows = x.shape[0] // 4
        if rows > self.max_rows or not 0 <= local_rows <= rows:
            raise ValueError("Input rows exceed the peer pool capacity")
        x = x.contiguous()
        output = torch.empty(
            (local_rows, 4 * self.vocab), device=x.device, dtype=x.dtype
        )
        # Copy raw 64-bit words. An integer pointer obtained from the peer table
        # can lose alignment metadata in lowering; word-sized transport avoids
        # scalar 16-bit XGMI transactions without changing any logits bits.
        words = x.view(torch.int64)
        output_words = output.view(torch.int64)
        vocab_words = self.vocab // 4
        self._barrier()
        _stage[(triton.cdiv(words.numel(), 8192),)](
            words,
            self.peers,
            self.rank,
            words.numel(),
            8192,
            num_warps=8,
        )
        self._barrier()
        if local_rows:
            _gather[(triton.cdiv(local_rows * vocab_words, 4096), 4)](
                self.peers,
                output_words,
                self.rank,
                rows,
                local_rows,
                vocab_words,
                4096,
                num_warps=4,
            )
        return output

    def close(self):
        torch.cuda.synchronize()
        for imported in self.imports:
            imported.close()
        self.imports.clear()
        self.buffer.close()
