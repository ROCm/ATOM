# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed allowed-token sets, independent of the rule that produced them."""

import torch

from atom.sampling_params import TokenConstraint

__all__ = [
    "AllowedTokenMask",
    "TokenConstraint",
    "TokenRuleBuffer",
    "apply_bitmask_reference",
    "apply_token_rules_reference",
]


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


class TokenRuleBuffer:
    """Small reusable rule metadata; deny rules never upload a vocabulary mask."""

    def __init__(self):
        self.capacity = 0
        self.copy_done = None
        self.use_done = None

    def prepare(self, rules, vocab_size, device):
        data = []
        ids = []
        for row, prompt_len, rule in rules:
            if any(token >= vocab_size for token in rule.token_ids):
                raise ValueError("Constraint token ID is outside the vocabulary")
            data.extend(
                (
                    row,
                    -1 if rule.until is None else prompt_len + rule.until - 1,
                    int(rule.mode == "allow"),
                    len(ids),
                    len(rule.token_ids),
                )
            )
            ids.extend(rule.token_ids)
        data.extend(ids)
        if self.copy_done is not None and not self.copy_done.query():
            self.copy_done.synchronize()
        if len(data) > self.capacity:
            if self.use_done is not None:
                self.use_done.synchronize()
            self.capacity = max(len(data), 2 * self.capacity)
            self.host = torch.empty(self.capacity, dtype=torch.int64, pin_memory=True)
            self.device = torch.empty_like(self.host, device=device)
            self.copy_done = torch.cuda.Event()
            self.use_done = torch.cuda.Event()
        self.host.numpy()[: len(data)] = data
        stream = torch.cuda.current_stream(device)
        stream.wait_event(self.use_done)
        self.device[: len(data)].copy_(self.host[: len(data)], non_blocking=True)
        self.copy_done.record(stream)
        return self.device, len(rules)

    def apply(self, rules, logits, target, positions, cu_seqlens, metadata):
        if logits.device.type != "cuda":
            apply_token_rules_reference(
                rules, logits, target, positions, cu_seqlens, metadata
            )
            return
        from atom.model_ops.triton_token_constraints import apply_token_rules

        packed, count = self.prepare(rules, logits.shape[-1], logits.device)
        apply_token_rules(
            logits, target, packed, count, positions, cu_seqlens, metadata
        )
        self.use_done.record(torch.cuda.current_stream(logits.device))


def apply_token_rules_reference(rules, logits, target, positions, cu_seqlens, metadata):
    """CPU oracle: every speculative row has its own generated-token position."""
    for row, prompt_len, rule in rules:
        cutoff = None if rule.until is None else prompt_len + rule.until - 1
        if metadata is None:
            rows = [(logits, row, int(cu_seqlens[row + 1]) - 1)]
        else:
            start = int(metadata.cu_num_draft_tokens[row - 1]) if row else 0
            end = int(metadata.cu_num_draft_tokens[row])
            rows = [
                (target, j, int(metadata.target_logits_indices[j]))
                for j in range(start, end)
            ]
            rows.append((logits, row, int(metadata.bonus_logits_indices[row])))
        for tensor, index, pos in rows:
            if cutoff is None or int(positions[pos]) < cutoff:
                if rule.mode == "deny":
                    tensor[index, list(rule.token_ids)] = -float("inf")
                else:
                    keep = torch.zeros(tensor.shape[-1], dtype=torch.bool)
                    keep[list(rule.token_ids)] = True
                    tensor[index].masked_fill_(~keep, -float("inf"))
