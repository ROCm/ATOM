# SPDX-License-Identifier: MIT
"""Engram configuration, tokenizer compression and n-gram hashing.

Derived from ROCm/ATOM PR #2185. Table residency and asynchronous staging live
in `tables` and `host`, respectively; every kernel is under `device/`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atom.utils import envs

logger = logging.getLogger(__name__)

# Matches the reference: layer seeds are spaced by this prime so two layers
# never draw the same multiplier sequence.
_LAYER_SEED_STRIDE = 10007


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    """First prime strictly greater than `start` that is not already in `seen`.

    `seen` is shared across every (layer, ngram, head) so each hash head lands in
    its own slice of the table. That sharing is why the per-layer row counts
    differ (a later layer's primes are found after an earlier layer's).
    """
    candidate = start + 1
    while True:
        if candidate not in seen and _is_prime(candidate):
            return candidate
        candidate += 1


@dataclass(frozen=True)
class EngramConfig:
    """The engram_* block of a DeepSeek text config."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    vocab_size: int
    n_heads: int
    head_dim: int
    pad_token_id: int
    compressed_vocab_size: int
    seed: int = 0

    @classmethod
    def from_hf(cls, text_config: dict) -> EngramConfig | None:
        """Build from a HF `text_config`; None when the model has no engram."""
        if "engram_layer_ids" not in text_config:
            return None
        return cls(
            layer_ids=tuple(text_config["engram_layer_ids"]),
            num_embeddings=tuple(text_config["engram_num_embeddings"]),
            max_ngram_size=int(text_config["engram_max_ngram_size"]),
            vocab_size=int(text_config["engram_vocab_size"]),
            n_heads=int(text_config["engram_n_heads"]),
            head_dim=int(text_config["engram_head_dim"]),
            pad_token_id=int(text_config["engram_pad_token_id"]),
            compressed_vocab_size=int(text_config["engram_compressed_vocab_size"]),
            seed=int(text_config.get("engram_seed", 0)),
        )

    @property
    def ngram_orders(self) -> tuple[int, ...]:
        """The n of each n-gram order, 2..max_ngram_size inclusive."""
        return tuple(range(2, self.max_ngram_size + 1))

    @property
    def num_hash_heads(self) -> int:
        """Hash heads per engram layer: one per (ngram order, head)."""
        return len(self.ngram_orders) * self.n_heads


class CompressedTokenizer:
    """Maps token ids onto a smaller vocabulary of normalized surface forms.

    Two tokens that normalize to the same string (case, accents, whitespace)
    share a compressed id, so an n-gram hash keys on what the text says rather
    than on which of several encodings produced it.

    Building the table decodes every token in the vocabulary, which costs tens of
    seconds, so the result is cached on disk. The reference does not cache; at
    129,280 tokens that cost lands on every single server start.
    """

    _CACHE_VERSION = 1

    def __init__(
        self, tokenizer, expected_size: int | None = None, cache_dir: str | None = None
    ):
        self._tokenizer = tokenizer
        self.lookup_table, self.num_new_token = self._load_or_build(cache_dir)
        if expected_size is not None and self.num_new_token != expected_size:
            raise ValueError(
                f"compressed vocab is {self.num_new_token}, config says "
                f"{expected_size}. The tokenizer does not match the checkpoint, "
                f"and every engram hash would index the wrong rows."
            )

    def __len__(self) -> int:
        return self.num_new_token

    def _cache_key(self) -> str:
        vocab = self._tokenizer.get_vocab()
        h = hashlib.sha256()
        h.update(str(self._CACHE_VERSION).encode())
        h.update(str(len(vocab)).encode())
        for tok in sorted(vocab)[:1024]:
            h.update(tok.encode("utf-8", "replace"))
        return h.hexdigest()[:16]

    def _load_or_build(self, cache_dir: str | None) -> tuple[np.ndarray, int]:
        path = (
            Path(cache_dir or envs.ATOM_ENGRAM_CACHE_DIR)
            / f"compressed_vocab_{self._cache_key()}.npz"
        )
        if path.is_file():
            try:
                blob = np.load(path)
                logger.info("engram: loaded compressed-vocab table from %s", path)
                return blob["lookup"], int(blob["num_new_token"])
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                # A truncated or stale cache must not take the server down; the
                # table is reproducible, so fall through and rebuild it. These
                # are what a damaged .npz raises -- anything else is a real bug
                # and should propagate.
                logger.warning("engram: unreadable cache %s, rebuilding", path)

        lookup, num_new_token = self._build()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, lookup=lookup, num_new_token=num_new_token)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("engram: could not cache compressed vocab: %s", exc)
        return lookup, num_new_token

    def _build(self) -> tuple[np.ndarray, int]:
        from tokenizers import Regex, normalizers

        # U+E000: a Private Use Area sentinel that cannot occur in real token
        # text, used to shield a lone-space token from Strip() (restored below).
        sentinel = chr(0xE000)
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

        vocab_size = len(self._tokenizer)
        key_to_new: dict[str, int] = {}
        lookup = np.empty(vocab_size, dtype=np.int64)
        next_id = 0
        for tid in range(vocab_size):
            text = self._tokenizer.decode([tid], skip_special_tokens=False)
            if chr(0xFFFD) in text:
                # U+FFFD (REPLACEMENT CHARACTER) is what decode() emits for a
                # byte-fallback piece -- a raw byte or a fragment of a multi-byte
                # character that is not valid UTF-8 on its own. Such a piece does
                # not survive the round trip, so key it on the raw token instead.
                key = self._tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = normalizer.normalize_str(text)
                key = norm if norm else text
            nid = key_to_new.get(key)
            if nid is None:
                nid = next_id
                key_to_new[key] = nid
                next_id += 1
            lookup[tid] = nid
        return lookup, next_id

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(input_ids, dtype=np.int64)
        out = arr.copy()
        valid = arr >= 0
        out[valid] = self.lookup_table[arr[valid]]
        return out


class NgramHashMapping:
    """Per-layer n-gram hashes, bit-identical to the reference implementation."""

    def __init__(self, config: EngramConfig, compressed_tokenizer: CompressedTokenizer):
        self.config = config
        self.tokenizer = compressed_tokenizer
        self.tokenizer_vocab_size = len(compressed_tokenizer)
        self.pad_id = int(compressed_tokenizer.lookup_table[config.pad_token_id])

        half_bound = max(
            1, int(np.iinfo(np.int64).max // self.tokenizer_vocab_size) // 2
        )
        self.layer_multipliers: dict[int, np.ndarray] = {}
        for layer_id in config.layer_ids:
            rng = np.random.default_rng(
                int(config.seed + _LAYER_SEED_STRIDE * int(layer_id))
            )
            r = rng.integers(
                low=0, high=half_bound, size=(config.max_ngram_size,), dtype=np.int64
            )
            # Odd multipliers keep the low bit of the mix informative.
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.head_vocab_sizes = self._derive_head_vocab_sizes()
        self.head_offsets = {
            layer_id: np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(np.int64)
            for layer_id, sizes in self.head_vocab_sizes.items()
        }

    def _derive_head_vocab_sizes(self) -> dict[int, np.ndarray]:
        """One distinct prime per (layer, ngram order, head), in reference order.

        The per-layer sums are checked against `engram_num_embeddings` from the
        checkpoint config: they match only if the prime search ran in exactly the
        same order, which makes this a real check that the row layout agrees with
        the trained tables rather than a plausible-looking guess.
        """
        cfg = self.config
        seen: set[int] = set()
        sizes: dict[int, np.ndarray] = {}
        for layer_id in cfg.layer_ids:
            heads: list[int] = []
            for _ in cfg.ngram_orders:
                start = cfg.vocab_size - 1
                for _ in range(cfg.n_heads):
                    prime = _next_prime(start, seen)
                    seen.add(prime)
                    heads.append(prime)
                    start = prime
            sizes[layer_id] = np.asarray(heads, dtype=np.int64)

        for layer_id, expected in zip(cfg.layer_ids, cfg.num_embeddings):
            got = int(sizes[layer_id].sum())
            if got != expected:
                raise ValueError(
                    f"engram layer {layer_id}: derived {got} table rows but the "
                    f"checkpoint declares {expected}. The hash-head layout does "
                    f"not match the trained tables."
                )
        return sizes

    def compress_tokens(self, input_ids, token_mask=None):
        tokens = self.tokenizer(input_ids)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2:
            raise ValueError("Engram token IDs must have shape [batch, tokens]")
        if token_mask is not None:
            mask = np.asarray(token_mask, dtype=bool)
            if mask.shape != tokens.shape:
                raise ValueError("Engram token mask must match token IDs")
            tokens = np.where(mask, tokens, -1)
        if np.any(tokens < -1):
            raise ValueError("Only DEAD=-1 is a valid negative compressed token")
        return tokens

    def hash_layer(
        self, input_ids, layer_id, compress=True, *, history=None, token_mask=None
    ):
        """Hash each query with up to three prior compressed IDs/DEAD markers.

        History is compressed already. DEAD stops the entire lookback, including
        for an image query itself; every blocked position contributes pad_id.
        This computes the requested layer only and never mutates request state.
        """
        x = (
            self.compress_tokens(input_ids, token_mask)
            if compress
            else np.asarray(input_ids, dtype=np.int64)
        )
        if x.ndim == 1:
            x = x[None, :]
        if not compress and token_mask is not None:
            mask = np.asarray(token_mask, dtype=bool)
            if mask.shape != x.shape:
                raise ValueError("Engram token mask must match token IDs")
            x = np.where(mask, x, -1)
        if x.ndim != 2 or np.any(x < -1):
            raise ValueError("Expected compressed [batch, tokens] IDs or DEAD=-1")
        width = self.config.max_ngram_size - 1
        if history is None:
            history = np.full((x.shape[0], width), -1, dtype=np.int64)
        history = np.asarray(history, dtype=np.int64)
        if history.shape != (x.shape[0], width) or np.any(history < -1):
            raise ValueError(f"Engram history must have shape [batch, {width}]")
        combined = np.concatenate((history, x), axis=1)
        positions = width + np.arange(x.shape[1])
        blocked = np.zeros_like(x, dtype=bool)
        rolling = np.zeros_like(x)
        pieces = []
        multipliers = self.layer_multipliers[layer_id]
        for shift in range(self.config.max_ngram_size):
            source = combined[:, positions - shift]
            blocked |= source == -1
            token = np.where(blocked, self.pad_id, source)
            rolling ^= token * multipliers[shift]
            if shift:
                begin = (shift - 1) * self.config.n_heads
                sizes = self.head_vocab_sizes[layer_id][
                    begin : begin + self.config.n_heads
                ]
                pieces.append(rolling[..., None] % sizes)
        return np.concatenate(pieces, axis=-1)

    def hash_all_layers(self, input_ids, *, history=None, token_mask=None):
        compressed = self.compress_tokens(input_ids, token_mask)
        return {
            layer_id: self.hash_layer(
                compressed, layer_id, compress=False, history=history
            )
            for layer_id in self.config.layer_ids
        }

    def advance_history(self, history, compressed_tokens, accepted_lengths=None):
        """Return the last compressed IDs after each accepted prefix only.

        Used by lifecycle code; tentative/rejected tokens never mutate history.
        No full-sequence token cache is required by the hash algorithm.
        """
        tokens = np.asarray(compressed_tokens, dtype=np.int64)
        if tokens.ndim != 2:
            raise ValueError("Expected [batch, tokens] compressed IDs")
        batch, count = tokens.shape
        width = self.config.max_ngram_size - 1
        if history is None:
            history = np.full((batch, width), -1, dtype=np.int64)
        history = np.asarray(history, dtype=np.int64)
        if history.shape != (batch, width):
            raise ValueError(f"Engram history must have shape [batch, {width}]")
        lengths = (
            np.full(batch, count)
            if accepted_lengths is None
            else np.asarray(accepted_lengths)
        )
        if (
            lengths.shape != (batch,)
            or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths < 0)
            or np.any(lengths > count)
        ):
            raise ValueError(
                "Accepted lengths must describe prefixes of this token batch"
            )
        combined = np.concatenate((history, tokens), axis=1)
        return (
            np.stack(
                [
                    combined[row, length : length + width]
                    for row, length in enumerate(lengths)
                ]
            )
            if batch
            else history.copy()
        )

    def to_row_indices(self, hash_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """Fold per-head hashes into absolute row indices of the layer's table."""
        return hash_ids + self.head_offsets[layer_id][None, None, :]
