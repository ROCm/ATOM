# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Engram conditional-memory module for DeepSeek-V4.1-Flash.

Port of /data/DeepSeek-V4.1-Flash/inference/engram.py and model.py:Engram.
The hash layout must match the reference exactly: every multiplier is derived
from the compressed vocab size, and the per-(layer, ngram, head) moduli are
consecutive distinct primes drawn in a fixed order.
"""

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_CACHE_DIR = os.environ.get("ATOM_CACHE_DIR", "/root/.cache/atom")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Collapse token ids that normalize to the same string."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def cached_compressed_token_map(tokenizer, model_path: str) -> tuple[list[int], int]:
    key = hashlib.sha256(
        f"{model_path}:{len(tokenizer)}:engram_token_map_v1".encode()
    ).hexdigest()[:16]
    path = os.path.join(_CACHE_DIR, f"engram_token_map_{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
        return blob["lookup"], blob["size"]
    lookup, size = build_compressed_token_map(tokenizer)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"lookup": lookup, "size": size}, f)
    os.replace(tmp, path)
    return lookup, size


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(args.engram_max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(args.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=args.engram_max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=args.engram_n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Hash ids for the n-grams ending at each token of a flat batch.

    Unlike the reference, which keeps a dense [batch, max_seq_len] id cache,
    this keeps only the last `max_ngram_size - 1` compressed ids per request
    slot, which is all a lookback of that depth can reach.
    """

    DEAD = -1

    def __init__(
        self,
        layout: EngramLayout,
        token_map: list[int],
        compressed_vocab_size: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.layout = layout
        self.lookback = layout.max_ngram_size - 1
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        tm = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = int(tm[pad_token_id].item())
        self.register_buffer("token_map", tm, persistent=False)
        self.register_buffer(
            "primes", torch.tensor(layout.primes, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(
                layout.layer_ids, layout.max_ngram_size, compressed_vocab_size
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        history: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids/positions: [N] flat. history: [N, lookback] compressed ids of
        positions p-1..p-lookback, DEAD where unavailable. Returns [N, n_layers, n_hash_cols].
        """
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)

        tokens = [compressed]
        blocked = compressed == self.DEAD
        for shift in range(1, self.layout.max_ngram_size):
            source = history[:, shift - 1]
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        stacked = torch.stack(tokens, dim=-1)

        products = stacked.unsqueeze(1) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def build_history(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_state: torch.Tensor,
    ) -> torch.Tensor:
        """Lookback ids for a flat batch laid out as contiguous per-sequence runs.

        slot_state: [num_seqs, lookback] carried across steps, newest first.
        Returns [N, lookback]; also updates slot_state in place for the next step.
        """
        compressed = self.token_map[input_ids]
        n, lb = input_ids.numel(), self.lookback
        num_seqs = cu_seqlens.numel() - 1
        seq_id = torch.repeat_interleave(
            torch.arange(num_seqs, device=input_ids.device),
            cu_seqlens[1:] - cu_seqlens[:-1],
        )
        offset_in_seq = torch.arange(n, device=input_ids.device) - cu_seqlens[seq_id]

        hist = torch.empty((n, lb), dtype=torch.int64, device=input_ids.device)
        for k in range(1, lb + 1):
            from_batch = offset_in_seq >= k
            src = torch.where(
                from_batch,
                torch.arange(n, device=input_ids.device) - k,
                torch.zeros(1, dtype=torch.int64, device=input_ids.device),
            )
            in_batch = compressed[src]
            carried = slot_state[
                seq_id,
                torch.clamp(
                    torch.full_like(offset_in_seq, k - 1) - offset_in_seq, min=0
                ),
            ]
            hist[:, k - 1] = torch.where(from_batch, in_batch, carried)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        last = cu_seqlens[1:] - 1
        col = torch.arange(lb, device=input_ids.device)
        from_chunk = col.unsqueeze(0) < lengths.unsqueeze(1)
        chunk_val = compressed[torch.clamp(last.unsqueeze(1) - col.unsqueeze(0), min=0)]
        old_val = slot_state.gather(
            1, torch.clamp(col.unsqueeze(0) - lengths.unsqueeze(1), 0, lb - 1)
        )
        slot_state.copy_(torch.where(from_chunk, chunk_val, old_val))
        return hist
