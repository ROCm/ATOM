# SPDX-License-Identifier: MIT
"""Prompt tokenization that reuses the previous turn's work.

An agentic conversation resends its whole history every turn, so each request
re-tokenizes hundreds of thousands of tokens it tokenized a moment ago; for
DeepSeek-V4.1 that is ~0.24 s at 115K tokens and ~1.6 s at 656K, on the TTFT
path. A chat template separates messages with added (special) tokens, and a
fast tokenizer splits on added tokens before anything else, so the text
between two of them tokenizes the same alone as in the whole prompt. This
encodes those segments independently -- the new ones in one parallel batch --
and keeps each segment's ids in an LRU keyed by its content, so a turn costs
roughly its new messages (DeepSeek Recipe's "special-token segmentation").

Exactness is checked twice: statically (no added token strips neighbouring
whitespace or matches single words, either of which would make a segment
depend on its neighbours) and against the plain encoder on the first prompts;
any mismatch turns the cache off for good and falls back.
"""

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict

import numpy as np

logger = logging.getLogger("atom")

# Prompts checked against the plain encoder before the cache is trusted.
_VERIFY_FIRST = 4


class SegmentTokenizer:
    """``encode(prompt)`` equal to ``tokenizer.encode(prompt)``, cached by
    added-token-delimited segment."""

    def __init__(self, tokenizer, max_cached_tokens: int):
        self._tokenizer = tokenizer
        self._backend = tokenizer._tokenizer
        self._max_tokens = max_cached_tokens
        self._lock = threading.Lock()
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._cached_tokens = 0
        self._verified = 0
        # Tokens served from the cache vs freshly encoded, and time spent.
        self._stats = [0, 0, 0.0, 0]
        self._added = dict(tokenizer.get_added_vocab())
        self._split = re.compile(
            "("
            + "|".join(re.escape(t) for t in sorted(self._added, key=len, reverse=True))
            + ")"
        )
        # What `encode` wraps a prompt in (BOS/EOS), found on a probe.
        probe = "a"
        full = tokenizer.encode(probe)
        bare = tokenizer.encode(probe, add_special_tokens=False)
        start = next(
            (
                i
                for i in range(len(full) - len(bare) + 1)
                if full[i : i + len(bare)] == bare
            ),
            None,
        )
        self.enabled = start is not None and self._segments_are_independent()
        if start is not None:
            self._head, self._tail = full[:start], full[start + len(bare) :]
        if not self.enabled:
            logger.info("segment tokenizer: off (added tokens are not plain splits)")

    @classmethod
    def maybe_create(cls, tokenizer, max_cached_tokens: int):
        """A cache for `tokenizer`, or None: turned off, not a fast (Rust)
        tokenizer, no added tokens to split on, or splitting would not be
        exact."""
        if max_cached_tokens <= 0:
            return None
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return None
        if not isinstance(getattr(tokenizer, "_tokenizer", None), Tokenizer):
            return None
        try:
            seg = cls(tokenizer, max_cached_tokens)
        except Exception:
            logger.warning("segment tokenizer: off", exc_info=True)
            return None
        return seg if seg.enabled and seg._added else None

    def _segments_are_independent(self) -> bool:
        decoder = getattr(self._tokenizer, "added_tokens_decoder", {}) or {}
        return all(
            not (t.lstrip or t.rstrip or t.single_word) for t in decoder.values()
        )

    def encode(self, prompt: str) -> list[int]:
        if not self.enabled:
            return self._tokenizer.encode(prompt)
        ids = self._encode_segments(prompt)
        if self._verified < _VERIFY_FIRST:
            ref = self._tokenizer.encode(prompt)
            if ids != ref:
                self.enabled = False
                logger.warning(
                    "segment tokenizer: result differs from the plain encoder; "
                    "turning the cache off"
                )
                return ref
            self._verified += 1
        return ids

    def _encode_segments(self, prompt: str) -> list[int]:
        t0 = time.perf_counter()
        parts = [p for p in self._split.split(prompt) if p]
        keys: list[bytes | None] = []
        missing: dict[bytes, str] = {}
        with self._lock:
            for part in parts:
                if part in self._added:
                    keys.append(None)
                    continue
                key = hashlib.blake2b(part.encode(), digest_size=16).digest()
                keys.append(key)
                if key not in self._cache:
                    missing[key] = part
        # Encoded outside the lock: one batch, parallel inside the tokenizer.
        fresh = {}
        if missing:
            encodings = self._backend.encode_batch(
                list(missing.values()), add_special_tokens=False
            )
            fresh = {
                k: np.asarray(e.ids, dtype=np.int32) for k, e in zip(missing, encodings)
            }
        out: list[int] = list(self._head)
        with self._lock:
            for key, ids in fresh.items():
                if key not in self._cache:
                    self._cache[key] = ids
                    self._cached_tokens += len(ids)
            for part, key in zip(parts, keys):
                if key is None:
                    out.append(self._added[part])
                    continue
                ids = fresh.get(key)
                if ids is None:
                    ids = self._cache[key]
                if key in self._cache:
                    self._cache.move_to_end(key)
                out.extend(ids.tolist())
            while self._cached_tokens > self._max_tokens and self._cache:
                _, old = self._cache.popitem(last=False)
                self._cached_tokens -= len(old)
        out.extend(self._tail)
        fresh_tokens = sum(len(v) for v in fresh.values())
        with self._lock:
            st = self._stats
            st[0] += len(out) - fresh_tokens
            st[1] += fresh_tokens
            st[2] += time.perf_counter() - t0
            st[3] += 1
            if st[3] % 500 == 0:
                logger.info(
                    "segment tokenizer: %d prompts, %.1f%% of tokens reused, "
                    "%.1f ms per prompt, %d tokens cached",
                    st[3],
                    100 * st[0] / max(1, st[0] + st[1]),
                    1e3 * st[2] / st[3],
                    self._cached_tokens,
                )
        return out
