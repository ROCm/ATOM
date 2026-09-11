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
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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


# ---------------------------------------------------------------------------
# Host-resident tables and the prefetch that hides them (was engram_host.py).
# ---------------------------------------------------------------------------


class HostEmbeddingTable:
    """One engram layer's table, memory-mapped and gathered row-wise.

    The reference keeps the table as a float32 numpy array, which for this model
    would be 393 GB per layer -- 786 GB of host RAM for the pair, before any
    staging buffer. Rows are kept in their stored dtype and converted only after
    the gather, so the resident cost is the page cache the OS chooses to keep.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        num_rows: int,
        head_dim: int,
        scale: torch.Tensor | None = None,
    ):
        if tensor.shape[0] != num_rows:
            raise ValueError(f"table has {tensor.shape[0]} rows, expected {num_rows}")
        if tensor.shape[1] != head_dim:
            raise ValueError(
                f"table row is {tensor.shape[1]} wide, expected {head_dim}"
            )
        self._tensor = tensor
        self.num_rows = num_rows
        self.head_dim = head_dim
        self._scale = scale
        self.block_size = 0
        if scale is not None:
            # gather() does `scale.to(float32)`; that decodes 2**(code-127) only
            # for a float8 E8M0 dtype. A raw uint8 exponent-code table would be
            # read as plain magnitudes (~127x off), so fail loud instead.
            if not scale.is_floating_point():
                raise ValueError(
                    f"engram block scale must be a float8 (E8M0) dtype, got "
                    f"{scale.dtype}"
                )
            if scale.shape[0] != num_rows:
                raise ValueError(
                    f"scale has {scale.shape[0]} rows, expected {num_rows}"
                )
            if head_dim % scale.shape[1]:
                raise ValueError(
                    f"head_dim {head_dim} is not divisible by {scale.shape[1]} "
                    f"scale blocks"
                )
            self.block_size = head_dim // scale.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def gather(
        self, row_indices: np.ndarray, out_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Gather rows named by `row_indices` ([...] ints) -> [..., head_dim].

        Out-of-range indices are a bug in the hash layout rather than something
        to clamp away quietly: a clamp turns a wrong table into plausible
        numbers, which is far harder to notice than an exception.
        """
        flat = np.ascontiguousarray(row_indices.reshape(-1))
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_rows):
            raise IndexError(
                f"engram row index out of range: [{flat.min()}, {flat.max()}] "
                f"not within [0, {self.num_rows})"
            )
        index = torch.from_numpy(flat)
        # One fancy-index over the whole batch, not a row at a time: measured on
        # the real table that is ~0.9 us/row against ~8 us/row for a Python loop.
        rows = self._tensor[index].to(out_dtype)
        if self._scale is not None:
            # Block-quantized: each scale covers `block_size` consecutive values
            # of a row. Skipping this does not fail, it returns values two orders
            # of magnitude off, so it is not optional.
            scale = self._scale[index].to(out_dtype)
            rows = (
                rows.reshape(-1, scale.shape[1], self.block_size) * scale.unsqueeze(-1)
            ).reshape(-1, self.head_dim)
        return rows.reshape(*row_indices.shape, self.head_dim)


class EngramPrefetchCache:
    """Bounded per-request store of prefetched embeddings.

    The reference uses an unbounded module-level dict keyed by sequence id and
    never removes anything, so a long-running server accumulates one entry per
    request served. This is an LRU with an explicit `drop` for finished
    requests, and it is the only shared state between the worker and the runner.
    """

    def __init__(self, capacity: int = 4096):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    def put(self, seq_id: int, layer_id: int, value: torch.Tensor) -> None:
        with self._lock:
            key = (seq_id, layer_id)
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def take(self, seq_id: int, layer_id: int) -> torch.Tensor | None:
        with self._lock:
            return self._store.pop((seq_id, layer_id), None)

    def contains(self, seq_id: int, layer_id: int) -> bool:
        """Non-destructive probe. `take` consumes, so miss detection needs this."""
        with self._lock:
            return (seq_id, layer_id) in self._store

    def drop(self, seq_id: int) -> None:
        """Forget everything for a finished request."""
        with self._lock:
            for key in [k for k in self._store if k[0] == seq_id]:
                del self._store[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class EngramPrefetcher:
    """Runs hash + host gather for the next step while the GPU works on this one.

    One worker thread, not a thread per step: the work is numpy and torch gather
    which release the GIL, and an unbounded thread-per-step (as in the reference)
    both races with its own consumer and contends with the runner for the GIL.

    `submit` returns a Future so the consumer can wait for a specific step rather
    than hoping the daemon finished. When the result is not ready in time the
    caller computes it inline -- correctness never depends on the race.
    """

    def __init__(
        self,
        hash_mapping,
        tables: dict[int, HostEmbeddingTable],
        cache_capacity: int = 4096,
    ):
        self._hash_mapping = hash_mapping
        self._tables = tables
        self.cache = EngramPrefetchCache(cache_capacity)
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engram-prefetch"
        )
        self._inflight: Future | None = None

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._hash_mapping.config.layer_ids

    def compute(
        self, seq_ids: list[int], token_ids: np.ndarray
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Hash `token_ids` ([B, T]) and gather, for every engram layer."""
        results: dict[tuple[int, int], torch.Tensor] = {}
        compressed = self._hash_mapping.tokenizer(token_ids)
        for layer_id in self.layer_ids:
            hashes = self._hash_mapping.hash_layer(compressed, layer_id, compress=False)
            rows = self._hash_mapping.to_row_indices(hashes, layer_id)
            gathered = self._tables[layer_id].gather(rows)
            for i, seq_id in enumerate(seq_ids):
                results[(seq_id, layer_id)] = gathered[i]
        return results

    def submit_compute(
        self,
        seq_ids: list[int],
        token_ids: np.ndarray | None = None,
        *,
        token_source: Callable[[], np.ndarray] | None = None,
    ) -> Future:
        """Queue a prefetch. Cheap and non-blocking; the caller keeps going.

        `token_source`, when given, is called ON THE WORKER to produce the ids --
        it waits on an async device->host copy off the caller's thread, so the
        token D2H never blocks the compute thread. Otherwise `token_ids` is used
        directly.
        """

        def _run() -> None:
            toks = token_source() if token_source is not None else token_ids
            for key, value in self.compute(seq_ids, toks).items():
                self.cache.put(key[0], key[1], value)

        self._inflight = self._pool.submit(_run)
        return self._inflight

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queued prefetch lands. True if it did."""
        if self._inflight is None:
            return True
        try:
            self._inflight.result(timeout=timeout)
            return True
        except TimeoutError:
            return False

    def drop_requests(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.cache.drop(seq_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class EngramHost:
    """Ties the host prefetch to the device step.

    Ordering per step, with nothing added to the critical path:

      1. the previous step's sampled ids land on the host (the runner already
         copies them asynchronously and synchronizes before use), and
         `prefetch_next` queues hash + gather for them on the worker;
      2. `stage_embeddings` copies whatever the worker produced into a pinned buffer and
         issues the H2D on a side stream, recording an event;
      3. `wait_for_embeddings` makes the compute stream wait on that event, so the
         engram layers read staged rows rather than racing the copy.

    A sequence whose prefetch has not landed is computed inline in `stage_embeddings`. The
    result is identical either way -- the prefetch only decides whether the work
    was already done, never what the answer is.
    """

    def __init__(
        self,
        prefetcher: EngramPrefetcher,
        max_num_tokens: int,
        num_hash_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        from atom.utils import CpuGpuBuffer

        self.prefetcher = prefetcher
        self.max_num_tokens = max_num_tokens
        self.embed_width = num_hash_heads * head_dim
        self.device = device
        # pin_memory only on device: a CPU-only build (the unit tests) has no
        # pinned allocator and would raise on construction.
        self.buffers = {
            layer_id: CpuGpuBuffer(
                max_num_tokens,
                self.embed_width,
                dtype=dtype,
                device=device,
                pin_memory=device.type == "cuda",
                with_numpy=False,
            )
            for layer_id in prefetcher.layer_ids
        }
        # Streams and events exist only on device; the CPU path (unit tests) runs
        # everything synchronously and leaves them None. `copy_stream`/`copy_done`
        # carry the staging H2D; `_token_*` carry the async device->host of the
        # just-sampled token -- the main thread only launches that copy on the
        # side stream and records the event, and the worker waits on it before
        # reading, so the per-step token D2H never blocks the compute thread. The
        # pinned buffer and event are reused: `stage_embeddings` waits on the
        # worker before the next step can overwrite them.
        if device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device)
            self.copy_done = torch.cuda.Event()
            self._token_d2h_stream = torch.cuda.Stream(device)
            self._token_event = torch.cuda.Event()
            self._token_host = torch.empty(
                max_num_tokens, dtype=torch.int64
            ).pin_memory()
        else:
            self.copy_stream = self.copy_done = None
            self._token_d2h_stream = self._token_event = self._token_host = None
        self._staged_rows = 0

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.prefetcher.layer_ids

    def stage_embeddings(
        self,
        seq_ids: list[int],
        token_ids: np.ndarray | None = None,
    ) -> int:
        """Fill the staging buffers for `seq_ids`; returns the row count staged.

        `token_ids` is only consulted for sequences the prefetch missed -- the
        caller passes the cheap host-side source (the scheduler's committed
        anchor), correct for the just-admitted rows that are the only misses.
        """
        num_rows = len(seq_ids)
        if num_rows > self.max_num_tokens:
            raise ValueError(
                f"{num_rows} rows exceeds staging capacity {self.max_num_tokens}"
            )
        self.prefetcher.wait(timeout=None)

        # Probe without consuming: a hit still has to be readable by the fill
        # loop below, which is what makes this `contains` and not `take`.
        # A seq is a miss unless EVERY layer is cached: entries evict per
        # (seq, layer), so probing only layer_ids[0] would call a half-evicted
        # seq a hit and then fail in the take loop below.
        missing = [
            i
            for i, seq_id in enumerate(seq_ids)
            if any(
                not self.prefetcher.cache.contains(seq_id, lid)
                for lid in self.layer_ids
            )
        ]
        computed: dict[tuple[int, int], torch.Tensor] = {}
        if missing:
            if token_ids is None:
                raise RuntimeError(
                    f"engram prefetch missed {len(missing)} sequences and no token "
                    f"ids were supplied to recompute them"
                )
            miss_ids = [seq_ids[i] for i in missing]
            computed = self.prefetcher.compute(miss_ids, token_ids[missing])

        for layer_id in self.layer_ids:
            cpu = self.buffers[layer_id].cpu
            for row, seq_id in enumerate(seq_ids):
                value = computed.get((seq_id, layer_id))
                if value is None:
                    value = self.prefetcher.cache.take(seq_id, layer_id)
                if value is None:
                    raise RuntimeError(
                        f"no engram embedding for seq {seq_id} layer {layer_id}"
                    )
                cpu[row].copy_(value.reshape(-1)[: self.embed_width])

        if self.copy_stream is not None:
            # Capture the compute stream BEFORE entering the copy_stream context:
            # inside it `current_stream()` would return copy_stream, making the
            # wait a no-op self-wait that fails to order the H2D after the
            # previous forward's reads of these buffers.
            compute_stream = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(compute_stream)
                for buffer in self.buffers.values():
                    buffer.copy_to_gpu(num_rows)
                self.copy_done.record(self.copy_stream)
        else:
            for buffer in self.buffers.values():
                buffer.copy_to_gpu(num_rows)
        self._staged_rows = num_rows
        return num_rows

    def stage_dummy(self, num_rows: int) -> int:
        """Stage zeros for a warmup or capture pass.

        A dummy forward has no real sequences to look up, but it still runs the
        engram layers, so the buffers have to be the right size and the H2D has
        to happen -- warmup exists to touch exactly this path. Zeros keep the
        shapes honest without inventing token ids.
        """
        num_rows = min(int(num_rows), self.max_num_tokens)
        for buffer in self.buffers.values():
            buffer.cpu[:num_rows].zero_()
        if self.copy_stream is not None:
            # Capture the compute stream BEFORE entering the copy_stream context:
            # inside it `current_stream()` would return copy_stream, making the
            # wait a no-op self-wait that fails to order the H2D after the
            # previous forward's reads of these buffers.
            compute_stream = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(compute_stream)
                for buffer in self.buffers.values():
                    buffer.copy_to_gpu(num_rows)
                self.copy_done.record(self.copy_stream)
        else:
            for buffer in self.buffers.values():
                buffer.copy_to_gpu(num_rows)
        self._staged_rows = num_rows
        return num_rows

    def wait_for_embeddings(self) -> None:
        """Order the compute stream behind the staging H2D."""
        if self.copy_done is not None:
            torch.cuda.current_stream(self.device).wait_event(self.copy_done)

    def embeddings(self, layer_id: int) -> torch.Tensor:
        """Staged rows for one layer, [staged_rows, num_hash_heads * head_dim]."""
        return self.buffers[layer_id].gpu[: self._staged_rows]

    def prefetch_next(self, seq_ids: list[int], tokens) -> None:
        """Queue the next step's hash + gather for the just-sampled tokens.

        `tokens` is this step's sampled ids ([N] or [N, T]) -- the next step's
        model input. On device the last id per row is copied to the host
        ASYNCHRONOUSLY on a side stream, and the worker waits on a CUDA event
        before reading it, so the main thread never blocks on the D2H. A host
        array (tests) is submitted directly.
        """
        n = len(seq_ids)
        if not n:
            return
        if torch.is_tensor(tokens) and tokens.is_cuda:
            col = tokens.detach().reshape(n, -1)[:, -1].to(torch.int64)
            self._token_d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._token_d2h_stream):
                self._token_host[:n].copy_(col, non_blocking=True)
            # Keep `col` from being recycled by the allocator until the side
            # stream's copy has consumed it.
            col.record_stream(self._token_d2h_stream)
            self._token_event.record(self._token_d2h_stream)
            event, host = self._token_event, self._token_host

            def _read() -> np.ndarray:
                event.synchronize()
                return np.array(host[:n]).reshape(n, 1)

            self.prefetcher.submit_compute(seq_ids, token_source=_read)
        else:
            token_ids = np.asarray(tokens).reshape(n, -1)[:, -1:].astype(np.int64)
            self.prefetcher.submit_compute(seq_ids, token_ids)

    def drop_requests(self, seq_ids: list[int]) -> None:
        self.prefetcher.drop_requests(seq_ids)

    def shutdown(self) -> None:
        self.prefetcher.shutdown()
