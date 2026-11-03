# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed allowed-token sets, independent of the rule that produced them."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TokenConstraint:
    """A hard token-ID rule, not a probability preference.

    Small static ID lists can compile once to an allowed-token bitmask. Dynamic
    grammars fill that representation directly; GPU channel state machines apply
    the same allow/deny semantics without transferring lists to the host.
    """

    mode: Literal["allow", "deny"]
    token_ids: tuple[int, ...]

    @classmethod
    def allow(cls, token_ids):
        return cls("allow", tuple(token_ids))

    @classmethod
    def deny(cls, token_ids):
        return cls("deny", tuple(token_ids))

    @classmethod
    def force(cls, token_id):
        return cls.allow((token_id,))

    def to_bitmask(self, vocab_size):
        """Compile on the CPU outside the per-token sampling hot path."""
        if self.mode not in ("allow", "deny") or vocab_size <= 0:
            raise ValueError("Invalid token constraint mode or vocabulary size")
        if any(
            type(token) is not int or not 0 <= token < vocab_size
            for token in self.token_ids
        ):
            raise ValueError("Constraint token ID is outside the vocabulary")
        if (self.mode == "allow" and not self.token_ids) or (
            self.mode == "deny" and len(set(self.token_ids)) == vocab_size
        ):
            raise ValueError("Token constraint must leave at least one allowed token")
        mask = torch.full(
            (1, (vocab_size + 31) // 32),
            -1 if self.mode == "deny" else 0,
            dtype=torch.int32,
        )
        for token in self.token_ids:
            value = int(mask[0, token // 32])
            bit = 1 << (token % 32)
            value = (value | bit if self.mode == "allow" else value & ~bit) & 0xFFFFFFFF
            mask[0, token // 32] = value if value < 2**31 else value - 2**32
        return mask


class AllowedTokenMask:
    def __init__(self, vocab_size):
        self.words = (vocab_size + 31) // 32
        self.capacity = 0
        self.device = None
        self.copy_done = None
        self.use_done = None

    def prepare(self, count, device):
        # CPU writes must not race an earlier asynchronous H2D transfer.
        if self.copy_done is not None and not self.copy_done.query():
            self.copy_done.synchronize()
        if count > self.capacity or device != self.device:
            # Growth/device changes are rare; keep the old device storage alive
            # until its last kernel finishes, including on a different stream.
            if self.use_done is not None:
                self.use_done.synchronize()
            self.capacity = max(count, self.capacity * 2)
            self.device = device
            gpu = device.type == "cuda"
            self.host_masks = torch.empty(
                (self.capacity, self.words),
                dtype=torch.int32,
                device="cpu",
                pin_memory=gpu,
            )
            self.host_rows = torch.empty(
                self.capacity, dtype=torch.int32, device="cpu", pin_memory=gpu
            )
            self.row_array = self.host_rows.numpy()
            if gpu:
                self.device_masks = torch.empty_like(self.host_masks, device=device)
                self.device_rows = torch.empty_like(self.host_rows, device=device)
                self.copy_done = torch.cuda.Event()
                self.use_done = torch.cuda.Event()
            else:
                self.copy_done = self.use_done = None
        return self.host_masks[:count]

    def apply(self, logits, rows):
        count = len(rows)
        if not count:
            return
        if logits.device.type != "cuda":
            apply_bitmask_reference(logits, self.host_masks, rows)
            return

        from atom.model_ops.triton_token_constraints import apply_packed_bitmask

        stream = torch.cuda.current_stream(logits.device)
        # A caller may change streams between batches. Order device-buffer reuse
        # after the prior kernel without making the host wait for that kernel.
        stream.wait_event(self.use_done)
        self.row_array[:count] = rows
        masks = self.device_masks[:count]
        indices = self.device_rows[:count]
        masks.copy_(self.host_masks[:count], non_blocking=True)
        indices.copy_(self.host_rows[:count], non_blocking=True)
        self.copy_done.record(stream)
        apply_packed_bitmask(logits, masks, indices)
        self.use_done.record(stream)


def apply_bitmask_reference(logits, masks, rows):
    """Pure CPU oracle for the packed-mask GPU kernel."""
    tokens = torch.arange(logits.shape[1])
    for i, row in enumerate(rows):
        allowed = ((masks[i, tokens // 32] >> (tokens % 32)) & 1).bool()
        logits[row].masked_fill_(~allowed, float("-inf"))
