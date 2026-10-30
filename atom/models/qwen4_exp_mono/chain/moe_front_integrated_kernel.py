"""C1 q4 router logits -> compact M16 tasks and output clear.

The next launch consumes these tasks with Native A4W4 G1/G2. No quantization
is performed here: Native G1 already quantizes BF16 mixed inline. All task
publication and independent output-clear CTAs finish before G1 on the stream.
The current scope admits four real rows on TP1, without padding or EP.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROWS, HIDDEN, EXPERTS, TOPK, BM = 4, 2560, 513, 11, 16


@triton.jit
def _prepare(
    logits,
    ids_in,
    weights_in,
    ids,
    weights,
    sorted_ids,
    sorted_weights,
    experts,
    valid,
    m_indices,
    reverse,
    out,
    LOGIT_STRIDE: tl.constexpr,
    ID_STRIDE: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    FROM_LOGITS: tl.constexpr,
    SELECT_SORT: tl.constexpr,
    CLEAR_TILE: tl.constexpr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
):
    program = tl.program_id(0)
    if program == 0:
        row = tl.arange(0, ROWS)
        slot = tl.arange(0, 16)
        if FROM_LOGITS:
            col = tl.arange(0, 512)
            x = tl.load(logits + row[:, None] * LOGIT_STRIDE + col[None, :]).to(
                tl.float32
            )
            row_max = tl.max(x, 1)
            # BF16 logits permit an int32 (value, -expert) key. Fold signed zero.
            bits = x.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
            ordered = bits ^ ((bits >> 31) & 0x7FFF)
            ordered = tl.where(x == 0.0, 0, ordered)
            key = (ordered << 9) | (511 - col[None, :])
            selected = tl.zeros((ROWS, 16), tl.int32)
            probability = tl.zeros((ROWS, 16), tl.float32)
            total = tl.zeros((ROWS,), tl.float32)
            if SELECT_SORT:
                ranked = tl.topk(key, 16, dim=1, descending=True)
                selected = 511 - (ranked & 511)
                values = tl.load(logits + row[:, None] * LOGIT_STRIDE + selected).to(
                    tl.float32
                )
                exp_values = libdevice.exp(values - row_max[:, None])
                # Preserve Native's sequential top-10 accumulation order.
                for k in tl.static_range(10):
                    total += tl.sum(tl.where(slot[None, :] == k, exp_values, 0.0), 1)
                probability = exp_values * (1.0 / total[:, None])
            else:
                for k in tl.static_range(10):
                    best = tl.max(key, 1)
                    index = 511 - (best & 511)
                    value = tl.load(logits + row * LOGIT_STRIDE + index).to(tl.float32)
                    exponent = libdevice.exp(value - row_max)
                    selected = tl.where(slot[None, :] == k, index[:, None], selected)
                    probability = tl.where(
                        slot[None, :] == k, exponent[:, None], probability
                    )
                    total += exponent
                    key = tl.where(col[None, :] == index[:, None], -2147483647, key)
                probability *= 1.0 / total[:, None]
            shared = tl.load(logits + row * LOGIT_STRIDE + 512).to(tl.float32)
            selected = tl.where(slot[None, :] == 10, 512, selected)
            probability = tl.where(
                slot[None, :] == 10,
                1.0 / (1.0 + libdevice.exp(-shared[:, None])),
                probability,
            )
            tl.store(
                ids + row[:, None] * TOPK + slot[None, :],
                selected,
                slot[None, :] < TOPK,
            )
            tl.store(
                weights + row[:, None] * TOPK + slot[None, :],
                probability,
                slot[None, :] < TOPK,
            )
        else:
            selected = tl.load(
                ids_in + row[:, None] * ID_STRIDE + slot[None, :],
                slot[None, :] < TOPK,
                1023,
            )
            probability = tl.load(
                weights_in + row[:, None] * WEIGHT_STRIDE + slot[None, :],
                slot[None, :] < TOPK,
                0.0,
            )
            tl.store(
                ids + row[:, None] * TOPK + slot[None, :],
                selected,
                slot[None, :] < TOPK,
            )
            tl.store(
                weights + row[:, None] * TOPK + slot[None, :],
                probability,
                slot[None, :] < TOPK,
            )
        # Stable expert-major ordering; each expert has at most four picks.
        # Every nonempty expert therefore owns exactly one M16 tile.
        pick = tl.arange(0, ROWS * 16)
        eid = tl.reshape(tl.where(slot[None, :] < TOPK, selected, 1023), (ROWS * 16,))
        prob = tl.reshape(probability, (ROWS * 16,))
        order = tl.sort(eid * (ROWS * 16) + pick, descending=False)
        sorted_pick = order % (ROWS * 16)
        sorted_expert = order // (ROWS * 16)
        live = sorted_expert < EXPERTS
        previous = tl.gather(sorted_expert, tl.maximum(pick - 1, 0), 0)
        first = live & ((pick == 0) | (sorted_expert != previous))
        tile = tl.cumsum(first.to(tl.int32), 0) - 1
        first_pos = tl.associative_scan(tl.where(first, pick, 0), 0, _maximum)
        local = pick - first_pos
        tile_count = tl.sum(first.to(tl.int32), 0)
        # Each tile head initializes its own padding; no other head writes it.
        padded = tile[:, None] * BM + slot[None, :]
        tl.store(sorted_ids + padded, ROWS, first[:, None])
        tl.store(sorted_weights + padded, 0.0, first[:, None])
        tl.store(m_indices + padded, ROWS, first[:, None])
        tl.store(experts + tile, sorted_expert, first)
        tl.debug_barrier()
        dst = tile * BM + local
        token = sorted_pick // 16
        route = sorted_pick % 16
        tl.store(sorted_ids + dst, (route << 24) | token, live)
        tl.store(sorted_weights + dst, tl.gather(prob, sorted_pick, 0), live)
        tl.store(m_indices + dst, token, live)
        tl.store(reverse + token * TOPK + route, dst, live)
        tl.store(valid, tile_count * BM)
        tl.store(valid + 1, ROWS)
    else:
        offset = (program - 1) * CLEAR_TILE + tl.arange(0, CLEAR_TILE)
        tl.store(out + offset, 0.0, offset < ROWS * HIDDEN)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@dataclass
class Workspace:
    ids: torch.Tensor
    weights: torch.Tensor
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    experts: torch.Tensor
    valid: torch.Tensor
    m_indices: torch.Tensor
    reverse: torch.Tensor
    out: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def buf(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)

        capacity = ROWS * TOPK * BM
        return cls(
            buf((ROWS, TOPK), torch.int32),
            buf((ROWS, TOPK), torch.float32),
            buf(capacity, torch.int32),
            buf(capacity, torch.float32),
            buf(capacity // BM, torch.int32),
            buf(2, torch.int32),
            buf(capacity, torch.int32),
            buf(ROWS * TOPK, torch.int32),
            buf((ROWS, HIDDEN), torch.bfloat16),
        )

    def prepare(self, logits, *, ids=None, weights=None, select="loop"):
        assert select in ("loop", "sort")
        from_logits = ids is None
        if from_logits:
            assert logits.shape == (ROWS, EXPERTS) and logits.dtype == torch.bfloat16
            assert logits.stride(1) == 1
            ids, weights = self.ids, self.weights
        else:
            assert ids.shape == weights.shape == (ROWS, TOPK)
            assert ids.dtype == torch.int32 and weights.dtype == torch.float32
            assert ids.stride(1) == weights.stride(1) == 1
        return _prepare[(1 + triton.cdiv(ROWS * HIDDEN, 1024),)](
            logits,
            ids,
            weights,
            self.ids,
            self.weights,
            self.sorted_ids,
            self.sorted_weights,
            self.experts,
            self.valid,
            self.m_indices,
            self.reverse,
            self.out,
            logits.stride(0),
            ids.stride(0),
            weights.stride(0),
            FROM_LOGITS=from_logits,
            SELECT_SORT=select == "sort",
            CLEAR_TILE=1024,
            ROWS=ROWS,
            HIDDEN=HIDDEN,
            EXPERTS=EXPERTS,
            TOPK=TOPK,
            BM=BM,
            num_warps=4,
            enable_fp_fusion=False,
        )
