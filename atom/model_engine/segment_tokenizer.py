# SPDX-License-Identifier: MIT
"""Prompt tokenization that reuses the previous turn's work.

An agentic conversation resends its whole history every turn, so each request
re-tokenizes the whole context it tokenized a moment ago, on the TTFT path, at
a cost that grows with the context rather than with the new messages. A chat
template separates messages with added (special) tokens, and a
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
import json
import logging
import re
import threading
import time
from collections import OrderedDict

import numpy as np

logger = logging.getLogger("atom")

# Prompts checked against the plain encoder before the cache is trusted.
_VERIFY_FIRST = 4
# Prompts shorter than this (characters) are encoded whole: splitting, hashing
# and joining cost more than they save there.
_MIN_CACHED_CHARS = 100_000


def _alternation(tokens):
    """A capturing leftmost-longest pattern for `tokens`, or None."""
    tokens = sorted(tokens, key=len, reverse=True)
    if not tokens:
        return None
    return re.compile("(" + "|".join(re.escape(t) for t in tokens) + ")")


def _offset_sensitive(pre_tokenizer) -> bool:
    """Whether a pre-tokenizer treats the start of the input specially, so a
    segment encoded alone differs from the same text inside the prompt."""
    if not isinstance(pre_tokenizer, dict):
        return False
    kind = pre_tokenizer.get("type")
    if kind == "Sequence":
        return any(_offset_sensitive(p) for p in pre_tokenizer.get("pretokenizers", []))
    if kind == "Metaspace":
        return pre_tokenizer.get("prepend_scheme", "always") != "never"
    if kind == "ByteLevel":
        return bool(pre_tokenizer.get("add_prefix_space"))
    return False


def _normalizer_is_identity(backend) -> bool:
    """True when the fast tokenizer has no normalizer, or an empty sequence."""
    try:
        normalizer = json.loads(backend.to_str()).get("normalizer")
    except (ValueError, AttributeError):
        return False
    return normalizer is None or (
        normalizer.get("type") == "Sequence" and not normalizer.get("normalizers")
    )


class SegmentTokenizer:
    """``encode(prompt)`` equal to ``tokenizer.encode(prompt)``, cached by
    added-token-delimited segment."""

    def __init__(self, tokenizer, max_cached_tokens: int):
        self._tokenizer = tokenizer
        self._backend = tokenizer._tokenizer
        self._max_tokens = max_cached_tokens
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._cached_tokens = 0
        self._verified = 0
        # Tokens served from the cache vs freshly encoded, and time spent.
        self._stats = [0, 0, 0.0, 0]
        self._added = dict(tokenizer.get_added_vocab())
        # The backend extracts non-normalized added tokens from the raw text
        # first, then normalized ones from what is left; each pass takes the
        # leftmost-longest match. Two patterns reproduce that order.
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        normalized = {t.content for t in decoder.values() if t.normalized}
        self._split_raw = _alternation(t for t in self._added if t not in normalized)
        self._split_normalized = _alternation(t for t in self._added if t in normalized)
        self._lock = threading.Lock()
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
        if start is not None:
            self._head, self._tail = full[:start], full[start + len(bare) :]
        self.enabled = (
            start is not None
            and self._segments_are_independent()
            and self._probes_match()
        )
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
        """Whether splitting the raw prompt at added-token spellings is what
        the fast tokenizer does itself.

        An added token that strips whitespace or matches single words depends
        on its neighbours. A normalized added token is matched after the
        normalizer runs, so its raw spelling may not be where the tokenizer
        splits (NFC folds `a` + U+0301 into one character); those are only
        safe when there is no normalizer.
        """
        decoder = getattr(self._tokenizer, "added_tokens_decoder", {}) or {}
        if any(t.lstrip or t.rstrip or t.single_word for t in decoder.values()):
            return False
        # A normalizer runs per segment too and may treat its start specially
        # (e.g. `Prepend`), so only the identity is accepted.
        if not _normalizer_is_identity(self._backend):
            return False
        try:
            pre = json.loads(self._backend.to_str()).get("pre_tokenizer")
        except (ValueError, AttributeError):
            return False
        return not _offset_sensitive(pre)

    def _probes_match(self) -> bool:
        """Encode a few added-token placements both ways once, at startup."""
        added = list(self._added)
        picks = {added[0], added[-1], min(added, key=len), max(added, key=len)}
        for tok in picks:
            for probe in (f"a{tok}a b", f"{tok} x\n{tok}", f" y {tok}\t1 2"):
                if self._join(probe) != self._tokenizer.encode(probe):
                    return False
        return True

    def encode(self, prompt: str) -> list[int]:
        if not self.enabled or len(prompt) < _MIN_CACHED_CHARS:
            return self._tokenizer.encode(prompt)
        # Decided before encoding: a request either checks its own result or
        # was admitted once enough others had.
        with self._lock:
            verify = self._verified < _VERIFY_FIRST
        ids = self._encode_segments(prompt)
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
                elif self.enabled:
                    self._verified += 1
            return ref
        with self._lock:
            enabled = self.enabled
        # Turned off while this request encoded: its ids are unchecked.
        return ids if enabled else self._tokenizer.encode(prompt)

    def _parts(self, prompt: str) -> list[str]:
        parts = [prompt]
        for pattern in (self._split_raw, self._split_normalized):
            if pattern is None:
                continue
            nxt = []
            for part in parts:
                if part in self._added:
                    nxt.append(part)
                else:
                    nxt.extend(p for p in pattern.split(part) if p)
            parts = nxt
        return parts

    def _join(self, prompt: str) -> list[int]:
        """Segment-wise encoding without the cache (startup probe)."""
        out = list(self._head)
        for part in self._parts(prompt):
            if part in self._added:
                out.append(self._added[part])
            else:
                out.extend(self._backend.encode(part, add_special_tokens=False).ids)
        out.extend(self._tail)
        return out

    def _encode_segments(self, prompt: str) -> list[int]:
        t0 = time.perf_counter()
        parts = self._parts(prompt)
        keys: list[bytes | None] = []
        missing: dict[bytes, str] = {}
        # This request's own references to the ids it found cached: another
        # request may evict them while this one encodes its missing segments.
        found: dict[bytes, np.ndarray] = {}
        with self._lock:
            for part in parts:
                if part in self._added:
                    keys.append(None)
                    continue
                key = hashlib.blake2b(part.encode(), digest_size=16).digest()
                keys.append(key)
                ids = self._cache.get(key)
                if ids is None:
                    missing[key] = part
                else:
                    found[key] = ids
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
                # An empty encoding would cost a cache entry but no budget,
                # so eviction would never bound how many accumulate.
                if len(ids) and key not in self._cache:
                    self._cache[key] = ids
                    self._cached_tokens += len(ids)
            for part, key in zip(parts, keys):
                if key is None:
                    out.append(self._added[part])
                    continue
                ids = found.get(key)
                if ids is None:
                    ids = fresh[key]
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
