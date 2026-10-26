# SPDX-License-Identifier: MIT
"""Prompt tokenization that reuses earlier requests' work.

An agentic conversation resends its whole history every turn, so the API
server would re-tokenize the whole context on the TTFT path, at a cost that
grows with the context rather than with the new messages. Two levels, tried
in order:

1. The conversation's earlier turn. Each prompt is kept, with its ids up to
   its last added token and where every added token in it ends. A later
   prompt (bucketed by its first characters) reuses those ids up to the last
   added token both share -- a turn's generation prompt is rewritten once it
   is answered, so the shared part usually ends a little before the stored
   one, and the later prompt then replaces it.
2. Everything else, split at added tokens (`segment_split`): each text
   segment's ids are cached by content, and missing ones are encoded in one
   parallel batch. This also serves edits in the middle, shared system
   prompts and tool schemas, and forks.

Both cut only at added tokens, which keeps them exact; the conditions are in
`segment_split`. The first prompts are also compared with the plain encoder,
and any difference turns the cache off for good.

The result is still a list over the whole prompt, so a request costs time
linear in its length (splitting the uncovered part, copying ids); what goes
away is re-encoding text already encoded.

Locking: one lock guards the LRU, the prefix index and the counters; hashing,
splitting, encoding, comparing and assembling ids all run outside it.
"""

import hashlib
import logging
import sys
import threading
import time
from typing import NamedTuple

import numpy as np

from atom.model_engine.byte_lru import ByteLRU
from atom.model_engine.segment_split import AddedTokenSplitter

logger = logging.getLogger("atom")

# Prompts checked against the plain encoder before the cache is trusted.
_VERIFY_FIRST = 4
# Prompts shorter than this (characters) are encoded whole: splitting, hashing
# and joining cost more than they save there.
_MIN_CACHED_CHARS = 100_000
# Leading characters that bucket a conversation's stored prompt: long enough
# to take in its first messages; a full comparison settles each candidate.
_PREFIX_KEY_CHARS = 16_384
# Stored prompts per bucket. Agentic conversations share a long system prompt
# and tool list, so a bucket holds every concurrent conversation of a
# deployment. A lookup compares the prompt with every entry in its bucket, so
# its cost grows linearly with the number of concurrent conversations, each
# costing about a comparison of the shared start (on the API server thread):
# with a ~100K-character shared start, ~0.8 ms at 32 entries and ~6 ms at
# 256, more for non-ASCII text. Re-encoding such a prompt costs tens to
# hundreds of ms.
_PREFIX_BUCKET = 256
# Characters compared per step when looking for where two prompts diverge.
_COMPARE_CHUNK = 1 << 16


class _Prefix(NamedTuple):
    """A prompt and the ids of its first `len_chars` characters, which end
    right after an added token; `cut_chars` / `cut_ids` mark where each added
    token in that span ends (characters, ids)."""

    text: str
    len_chars: int
    ids: np.ndarray  # int32, without the BOS/EOS `encode` wraps a prompt in
    cut_chars: np.ndarray  # int64, ascending; the last is `len_chars`
    cut_ids: np.ndarray  # int64

    @property
    def payload_bytes(self) -> int:
        # The arrays own their data, so getsizeof counts header and data; the
        # tuple itself and its `len_chars` int are retained too.
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.len_chars)
            + sys.getsizeof(self.text)
            + sys.getsizeof(self.ids)
            + sys.getsizeof(self.cut_chars)
            + sys.getsizeof(self.cut_ids)
        )


def _shared_chars(a: str, b: str, limit: int) -> int:
    """Length of the common prefix of `a` and `b`, up to `limit`, comparing
    chunk by chunk so the copying stays linear in what they share."""
    i = 0
    while i < limit:
        j = min(i + _COMPARE_CHUNK, limit)
        if a[i:j] == b[i:j]:
            i = j
            continue
        lo, hi = i, j  # the first difference is in [lo, hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if a[lo:mid] == b[lo:mid]:
                lo = mid
            else:
                hi = mid
        return lo
    return limit


class SegmentTokenizer:
    """``encode(prompt)`` equal to ``tokenizer.encode(prompt)``, reusing
    earlier prompts' ids (module docstring)."""

    def __init__(self, tokenizer, budget_bytes: int):
        self._tokenizer = tokenizer
        self._split = AddedTokenSplitter(tokenizer)
        self.enabled = self._split.segments_exact()
        if not self.enabled:
            logger.info("segment tokenizer: off (added tokens are not plain splits)")
        self._prefix_ok = self._split.prefix_cuts_stable()
        self._lock = threading.Lock()
        # Segments (key: content hash) and stored prompts (key: ("p", n))
        # share one byte budget.
        self._lru = ByteLRU(budget_bytes, on_evict=self._evicted)
        self._buckets: dict[int, list[int]] = {}
        self._next_prefix = 0
        self._verified = 0
        # [ids reused, ids encoded, seconds, prompts, prefix hits]
        self._stats = [0, 0, 0.0, 0, 0]

    @classmethod
    def maybe_create(cls, tokenizer, budget_bytes: int):
        """A cache for `tokenizer`, or None: turned off, not a fast (Rust)
        tokenizer, or splitting would not be exact."""
        if budget_bytes <= 0:
            return None
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return None
        if not isinstance(getattr(tokenizer, "_tokenizer", None), Tokenizer):
            return None
        try:
            seg = cls(tokenizer, budget_bytes)
        except Exception:
            logger.warning("segment tokenizer: off", exc_info=True)
            return None
        return seg if seg.enabled else None

    # -- public ------------------------------------------------------------

    def encode(self, prompt: str) -> list[int]:
        if not self.enabled or len(prompt) < _MIN_CACHED_CHARS:
            return self._tokenizer.encode(prompt)
        # Decided before encoding: a request either checks its own result or
        # was admitted once enough others had.
        with self._lock:
            verify = self._verified < _VERIFY_FIRST
        ids = self._encode_cached(prompt)
        if verify:
            ref = self._tokenizer.encode(prompt)
            with self._lock:
                if ids != ref:
                    if self.enabled:
                        logger.warning(
                            "segment tokenizer: result differs from the plain "
                            "encoder; turning the cache off"
                        )
                    self.enabled = False
                    # Nothing reads the cache any more; release it. Requests
                    # in flight keep their own references.
                    self._lru.clear()
                    self._buckets.clear()
                elif self.enabled:
                    self._verified += 1
            return ref
        with self._lock:
            enabled = self.enabled
        # Turned off while this request encoded: its ids are unchecked.
        return ids if enabled else self._tokenizer.encode(prompt)

    # -- the two levels ----------------------------------------------------

    def _encode_cached(self, prompt: str) -> list[int]:
        t0 = time.perf_counter()
        split = self._split
        base, k = self._find_prefix(prompt)
        start = 0 if base is None else int(base.cut_chars[k])

        parts = split.parts(prompt[start:])
        keys = [
            None if extracted else hashlib.blake2b(p.encode(), digest_size=16).digest()
            for p, extracted in parts
        ]
        # This request's own references to cached ids: another request may
        # evict them while this one encodes the missing segments.
        found: dict[bytes, np.ndarray] = {}
        missing: dict[bytes, str] = {}
        with self._lock:
            for (piece, _), key in zip(parts, keys):
                if key is None or key in found or key in missing:
                    continue
                ids = self._lru.get(key)
                if ids is None:
                    missing[key] = piece
                else:
                    found[key] = ids
        fresh: dict[bytes, np.ndarray] = {}
        if missing:
            encodings = split.backend.encode_batch(
                list(missing.values()), add_special_tokens=False
            )
            fresh = {
                k: np.asarray(e.ids, dtype=np.int32) for k, e in zip(missing, encodings)
            }
            with self._lock:
                # Turned off (and cleared) while this request encoded: keep
                # nothing, or it would be held with no reader.
                for key, ids in fresh.items() if self.enabled else ():
                    # An empty encoding is not worth an entry.
                    if len(ids):
                        self._lru.put(key, ids, ids.nbytes)

        # Assemble outside the lock, as one array.
        chunks = [split.head]
        n_base = 0 if base is None else int(base.cut_ids[k])
        if n_base:
            chunks.append(base.ids[:n_base])
        new_cut_chars, new_cut_ids = [], []
        chars, n_ids = start, n_base
        for (piece, extracted), key in zip(parts, keys):
            chars += len(piece)
            if extracted:
                chunks.append(split.id_array(piece))
                n_ids += 1
                new_cut_chars.append(chars)
                new_cut_ids.append(n_ids)
                continue
            ids = found.get(key)
            if ids is None:
                ids = fresh[key]
            chunks.append(ids)
            n_ids += len(ids)
        chunks.append(split.tail)
        body = np.concatenate(chunks)
        out = body.tolist()

        # With no added token past the reused cut, the new entry would be a
        # part of `base`, which already serves it.
        if new_cut_chars:
            self._store_prefix(prompt, body, base, k, new_cut_chars, new_cut_ids)
        fresh_ids = sum(len(v) for v in fresh.values())
        self._count(len(out) - fresh_ids, fresh_ids, time.perf_counter() - t0, base)
        return out

    def _find_prefix(self, prompt: str):
        """`(entry, k)`: the stored prompt sharing the most with `prompt` and
        its last cut `k` within what they share; `(None, -1)` if none."""
        if not self._prefix_ok or len(prompt) <= _PREFIX_KEY_CHARS:
            return None, -1
        bucket = hash(prompt[:_PREFIX_KEY_CHARS])
        # Peek: only the entry used is made recent, not every candidate.
        with self._lock:
            entries = [
                (pid, e)
                for pid, e in (
                    (pid, self._lru.peek(("p", pid)))
                    for pid in self._buckets.get(bucket, ())
                )
                if e is not None
            ]
        best, best_pid, best_k, best_chars = None, None, -1, 0
        for pid, entry in entries:
            shared = _shared_chars(
                prompt, entry.text, min(len(prompt), entry.len_chars)
            )
            k = int(np.searchsorted(entry.cut_chars, shared, side="right")) - 1
            if k >= 0 and entry.cut_chars[k] > best_chars:
                best, best_pid, best_k = entry, pid, k
                best_chars = int(entry.cut_chars[k])
        if best is not None:
            with self._lock:
                self._lru.get(("p", best_pid))
        return best, best_k

    def _store_prefix(self, prompt, body, base, k, new_cut_chars, new_cut_ids):
        """Keep `prompt` for later turns; it replaces `base` if it extends it
        whole, and sits beside it otherwise (another conversation with the
        same start, a fork)."""
        if not self._prefix_ok:
            return
        head = len(self._split.head)
        if base is None:
            cut_chars = np.asarray(new_cut_chars, dtype=np.int64)
            cut_ids = np.asarray(new_cut_ids, dtype=np.int64)
        else:
            cut_chars = np.concatenate(
                [base.cut_chars[: k + 1], np.asarray(new_cut_chars, dtype=np.int64)]
            )
            cut_ids = np.concatenate(
                [base.cut_ids[: k + 1], np.asarray(new_cut_ids, dtype=np.int64)]
            )
        if not len(cut_chars) or cut_chars[-1] <= _PREFIX_KEY_CHARS:
            return
        entry = _Prefix(
            prompt,
            int(cut_chars[-1]),
            body[head : head + int(cut_ids[-1])].copy(),
            cut_chars,
            cut_ids,
        )
        # The new prompt supersedes `base` when what it does not share of
        # `base` is added tokens only, no text: the previous turn's generation
        # prompt, which the template rewrites once the turn is answered
        # (DeepSeek-V4.1 thinking mode turns `<think>` into `</think>`).
        # Text there is another conversation or a fork, kept beside it.
        replaces = base is not None and int(base.cut_ids[-1] - base.cut_ids[k]) == (
            len(base.cut_ids) - 1 - k
        )
        bucket = hash(prompt[:_PREFIX_KEY_CHARS])
        with self._lock:
            if not self.enabled:  # as for segments above
                return
            pids = self._buckets.get(bucket, [])
            if replaces:
                for pid in list(pids):
                    if self._lru.peek(("p", pid)) is base:
                        self._lru.pop(("p", pid))
                        pids.remove(pid)
            pid = self._next_prefix
            self._next_prefix += 1
            pids.append(pid)
            self._buckets[bucket] = pids
            if len(pids) > _PREFIX_BUCKET:
                self._lru.pop(("p", pids.pop(0)))
            self._lru.put(("p", pid), entry, entry.payload_bytes)
            if ("p", pid) not in self._lru:  # larger than the whole budget
                pids.remove(pid)
            if not pids:
                self._buckets.pop(bucket, None)

    def _evicted(self, key, value) -> None:
        """LRU eviction callback (lock held): unindex a stored prompt."""
        if isinstance(key, tuple):
            bucket = hash(value.text[:_PREFIX_KEY_CHARS])
            pids = self._buckets.get(bucket)
            if pids is not None and key[1] in pids:
                pids.remove(key[1])
                if not pids:
                    del self._buckets[bucket]

    def _count(self, reused, encoded, seconds, base) -> None:
        with self._lock:
            st = self._stats
            st[0] += reused
            st[1] += encoded
            st[2] += seconds
            st[3] += 1
            st[4] += base is not None
            snapshot = list(st) if st[3] % 500 == 0 else None
            held = self._lru.bytes
        if snapshot is not None:
            logger.info(
                "segment tokenizer: %d prompts, %.1f%% of ids reused, %.1f ms per "
                "prompt, %d prefix hits, %.0f MB held",
                snapshot[3],
                100 * snapshot[0] / max(1, snapshot[0] + snapshot[1]),
                1e3 * snapshot[2] / snapshot[3],
                snapshot[4],
                held / 2**20,
            )
