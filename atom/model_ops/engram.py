"""Engram: n-gram hash -> multi-head embedding lookup, computed on the host.

DeepSeek's Engram (https://github.com/deepseek-ai/Engram) augments a few decoder
layers with a lookup into very large n-gram embedding tables. The tables cannot
live in HBM -- DeepSeek-V4.1-Flash carries two of them, 384,006,168 and
384,016,682 rows of 256 fp8 values, ~101.5 GB each -- so the lookup runs on the
host and its result is staged into a pinned buffer and copied to the device on a
side stream, overlapped with the previous step's GPU work.

The hashing here reproduces the reference implementation (engram_demo_v1.py)
exactly; anything else silently indexes the wrong rows of a trained table.
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    kernel_size: int = 4

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
            kernel_size=int(text_config.get("engram_kernel_size", 4)),
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
        cache_dir = cache_dir or os.environ.get(
            "ATOM_ENGRAM_CACHE_DIR", str(Path.home() / ".cache" / "atom" / "engram")
        )
        path = Path(cache_dir) / f"compressed_vocab_{self._cache_key()}.npz"
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

        vocab_size = len(self._tokenizer)
        key_to_new: dict[str, int] = {}
        lookup = np.empty(vocab_size, dtype=np.int64)
        next_id = 0
        for tid in range(vocab_size):
            text = self._tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                # Byte-fallback pieces do not survive a round trip through
                # decode; key them on the raw token instead.
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

    def hash_layer(
        self, input_ids: np.ndarray, layer_id: int, compress: bool = True
    ) -> np.ndarray:
        """Hash ids for one layer, shape [B, T, num_hash_heads].

        Only the requested layer is computed. The reference hashes every layer
        and discards the rest, which doubles the work for a two-layer model.
        """
        x = (
            self.tokenizer(input_ids)
            if compress
            else np.asarray(input_ids, dtype=np.int64)
        )
        if x.ndim == 1:
            x = x[None, :]
        _, seq_len = x.shape

        multipliers = self.layer_multipliers[layer_id]
        head_sizes = self.head_vocab_sizes[layer_id]

        # shifted[k] is the token k positions back, left-padded with pad_id.
        shifted = [x]
        for k in range(1, self.config.max_ngram_size):
            shifted.append(
                np.pad(
                    x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
                )[:, :seq_len]
            )

        out = np.empty(
            (x.shape[0], seq_len, self.config.num_hash_heads), dtype=np.int64
        )
        head = 0
        for order_idx, n in enumerate(self.config.ngram_orders):
            mix = shifted[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifted[k] * multipliers[k])
            base = order_idx * self.config.n_heads
            for j in range(self.config.n_heads):
                out[:, :, head] = mix % int(head_sizes[base + j])
                head += 1
        return out

    def hash_all_layers(self, input_ids: np.ndarray) -> dict[int, np.ndarray]:
        compressed = self.tokenizer(input_ids)
        return {
            layer_id: self.hash_layer(compressed, layer_id, compress=False)
            for layer_id in self.config.layer_ids
        }

    def to_row_indices(self, hash_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """Fold per-head hashes into absolute row indices of the layer's table."""
        return hash_ids + self.head_offsets[layer_id][None, None, :]
