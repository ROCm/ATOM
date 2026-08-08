# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Ulysses sequence parallelism.

Diffusion transformers process one very long sequence per request (~37.7k
packed tokens for MiniMax-H3 at 1344x768), so the parallel axis that matters is
the sequence, not the batch. Ulysses splits the sequence across ranks for the
linear layers, then trades sequence for heads with an all-to-all so each rank
can run attention over the *whole* sequence for a *subset* of heads:

    linears   : [S/W, H,   D]   -- every rank holds W-th of the tokens
      all-to-all
    attention : [S,   H/W, D]   -- every rank holds all tokens, W-th of heads
      all-to-all
    linears   : [S/W, H,   D]

Measured on 8x MI308X at H3's real geometry: 0.272 ms per q/k/v tensor at
202.7 GB/s effective, ~1.9 ms per layer including relayout -- about 2% of a
denoise step, so this is not a bottleneck at this scale.

One caveat worth knowing before tuning: attention efficiency falls as heads per
rank shrinks. At fixed total work the same aiter kernel reaches 123 TFLOP/s
with 56 heads but only 96 TFLOP/s with 7 (Ulysses-8), because 7 heads
under-fill a 80-CU GPU. Higher Ulysses degree still wins on wall clock, just
sub-linearly.
"""

import torch
import torch.distributed as dist


class UlyssesGroup:
    """Sequence<->head all-to-all over a torch.distributed process group.

    A ``world_size`` of 1 degenerates to identity, which keeps single-GPU and
    CPU-only test paths free of distributed setup.
    """

    def __init__(self, group: "dist.ProcessGroup | None" = None) -> None:
        self._group = group
        if dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size(group)
            self._rank = dist.get_rank(group)
        else:
            self._world_size = 1
            self._rank = 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_main(self) -> bool:
        return self._rank == 0

    @property
    def enabled(self) -> bool:
        return self._world_size > 1

    @staticmethod
    def _check_rank3(x: torch.Tensor) -> None:
        """Validate rank before any unpacking, so the error names the problem."""
        if x.dim() != 3:
            raise ValueError(
                f"expected a 3-D [tokens, heads, head_dim] tensor, got shape "
                f"{tuple(x.shape)}"
            )

    def _check(self, seq_dim_name: str, seq_len: int, heads: int) -> None:
        w = self._world_size
        if seq_len % w:
            raise ValueError(
                f"{seq_dim_name} ({seq_len}) must be divisible by ulysses world "
                f"size ({w}); pad the packed sequence before the all-to-all"
            )
        if heads % w:
            raise ValueError(
                f"head count ({heads}) must be divisible by ulysses world size "
                f"({w})"
            )

    def scatter_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[S_local, H, D] -> [S_total, H_local, D] (pre-attention)."""
        w = self._world_size
        if w == 1:
            return x
        self._check_rank3(x)
        s_local, heads, dim = x.shape
        self._check("local sequence length", s_local * w, heads)
        h_local = heads // w

        # Group the head axis by destination rank, then make rank the leading
        # axis so all_to_all_single's equal-split contract lines up.
        send = x.view(s_local, w, h_local, dim).permute(1, 0, 2, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self._group)
        # recv[i] is rank i's tokens for our head slice -> concatenate on tokens.
        return recv.reshape(w * s_local, h_local, dim)

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[S_total, H_local, D] -> [S_local, H, D] (post-attention)."""
        w = self._world_size
        if w == 1:
            return x
        self._check_rank3(x)
        s_total, h_local, dim = x.shape
        self._check("total sequence length", s_total, h_local * w)
        s_local = s_total // w

        send = x.view(w, s_local, h_local, dim).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self._group)
        # recv[i] is head-group i for our token slice -> concatenate on heads.
        return recv.permute(1, 0, 2, 3).reshape(s_local, h_local * w, dim)

    def broadcast_object(self, obj, src: int = 0):
        """Share a Python object from ``src`` to the group.

        Used by MAIN_RANK_BROADCAST stages, where the work is serial but every
        rank needs the result.
        """
        if self._world_size == 1:
            return obj
        holder = [obj if self._rank == src else None]
        dist.broadcast_object_list(holder, src=src, group=self._group)
        return holder[0]

    def barrier(self) -> None:
        if self._world_size > 1:
            dist.barrier(group=self._group)
