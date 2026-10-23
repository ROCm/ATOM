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

On top of that, each prompt is kept cut after its last added token, with its
ids and where each added token in it ends. The next turn of the conversation
(found by its first characters) reuses those ids up to the last added token
both prompts share -- the generation prompt of a turn is rewritten once it is
answered, so the shared part usually ends a little before the stored one --
and only the rest goes through the segment cache. Cutting at an added token
keeps this exact as well; it is off when one added token contains another.

Exactness is checked twice: statically (no added token strips neighbouring
whitespace or matches single words, either of which would make a segment
depend on its neighbours) and against the plain encoder on the first prompts;
any mismatch turns the cache off for good and falls back.
"""

import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

logger = logging.getLogger("atom")

# Prompts checked against the plain encoder before the cache is trusted.
_VERIFY_FIRST = 4
# Prompts shorter than this (characters) are encoded whole: splitting, hashing
# and joining cost more than they save there.
_MIN_CACHED_CHARS = 100_000
# Leading characters that index a conversation's stored prefix: long enough
# to take in its first messages, so conversations sharing only a system prompt
# land in the same bucket rarely, and the check below settles it anyway.
_PREFIX_KEY_CHARS = 16_384
# Stored prefixes per bucket (forks of one conversation, look-alike starts).
_PREFIX_BUCKET = 8


class _Prefix(NamedTuple):
    """A prompt cut after its last added token, the ids of exactly that, and
    where each added token in it ends -- the points a later prompt that
    diverges from it can still reuse it up to."""

    text: str
    ids: np.ndarray  # int32, without the BOS/EOS `encode` wraps a prompt in
    cut_chars: np.ndarray  # int64, ascending; the last is len(text)
    cut_ids: np.ndarray  # int64, ids before each cut

    @property
    def cost(self) -> int:
        """Budget units of 4 bytes, one per int32 token id: everything kept."""
        return (
            self.ids.nbytes
            + sys.getsizeof(self.text)
            + self.cut_chars.nbytes
            + self.cut_ids.nbytes
        ) // 4


def _cuts_are_stable(added, normalized) -> bool:
    """Whether a prompt cut right after an added token `t` extracts the same
    up to the cut whatever follows it -- what reusing a stored prefix needs.

    Within one extraction pass that fails only when another added token
    contains `t` (it could start at or before `t` and run past the cut):
    leftmost-longest matching takes `t` first otherwise. Across passes it also
    fails when a non-normalized token -- extracted from the raw text first --
    could start inside a normalized `t` and run past the cut (normalized `ab`,
    non-normalized `bc`, then `c` appended).
    """
    tokens = list(added)
    if any(a != b and a in b for a in tokens for b in tokens):
        return False
    raw = [u for u in tokens if u not in normalized]
    return not any(
        t[-k:] == u[:k]
        for t in normalized
        for u in raw
        for k in range(1, min(len(t), len(u)))
    )


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
        self._prefix_ok = _cuts_are_stable(self._added, normalized)
        self._prefixes: OrderedDict[int, _Prefix] = OrderedDict()
        self._prefix_buckets: dict[int, list[int]] = {}
        self._prefix_tokens = 0
        self._next_prefix = 0
        self._prefix_hits = 0

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

    def _find_prefix(self, prompt: str):
        """`(entry, k)`: the stored prefix sharing the most with `prompt`, and
        its last cut `k` that `prompt` agrees with up to; or None."""
        if not self._prefix_ok or len(prompt) <= _PREFIX_KEY_CHARS:
            return None
        bucket = hash(prompt[:_PREFIX_KEY_CHARS])
        with self._lock:
            entries = [
                self._prefixes[pid]
                for pid in self._prefix_buckets.get(bucket, ())
                if pid in self._prefixes
            ]
        best = None
        for entry in entries:
            cuts = entry.cut_chars
            if prompt.startswith(entry.text):
                k = len(cuts) - 1
            else:
                # Agreement up to a cut is monotone in the cut: bisect for the
                # last cut both prompts share.
                lo, hi = -1, len(cuts) - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    c = int(cuts[mid])
                    if c <= len(prompt) and prompt[:c] == entry.text[:c]:
                        lo = mid
                    else:
                        hi = mid - 1
                k = lo
            if k >= 0 and (best is None or cuts[k] > best[0].cut_chars[best[1]]):
                best = (entry, k)
        return best

    def _store_prefix(self, prompt: str, ids, cut_chars, cut_ids, base):
        """Remember `prompt` up to its last cut. `base`, when given, is the
        stored prefix this prompt extends whole, which it replaces; a prompt
        that only shares part of one (another conversation with the same
        start, a fork) is kept beside it."""
        if not self._prefix_ok or not cut_chars or cut_chars[-1] <= _PREFIX_KEY_CHARS:
            return
        cut = cut_chars[-1]
        bucket = hash(prompt[:_PREFIX_KEY_CHARS])
        with self._lock:
            if base is not None:
                for pid in list(self._prefix_buckets.get(bucket, ())):
                    if self._prefixes.get(pid) is base:
                        self._drop_prefix(pid, bucket)
            # After the drop, which removes an emptied bucket.
            pids = self._prefix_buckets.setdefault(bucket, [])
            pid = self._next_prefix
            self._next_prefix += 1
            entry = _Prefix(
                prompt[:cut],
                ids,
                np.asarray(cut_chars, dtype=np.int64),
                np.asarray(cut_ids, dtype=np.int64),
            )
            self._prefixes[pid] = entry
            self._prefix_tokens += entry.cost
            pids.append(pid)
            if len(pids) > _PREFIX_BUCKET:
                self._drop_prefix(pids[0], bucket)
            while self._prefix_tokens > self._max_tokens and self._prefixes:
                old_pid = next(iter(self._prefixes))
                old = self._prefixes[old_pid]
                self._drop_prefix(old_pid, hash(old.text[:_PREFIX_KEY_CHARS]))

    def _drop_prefix(self, pid: int, bucket: int) -> None:
        entry = self._prefixes.pop(pid, None)
        if entry is not None:
            self._prefix_tokens -= entry.cost
        pids = self._prefix_buckets.get(bucket)
        if pids is not None and pid in pids:
            pids.remove(pid)
            if not pids:
                del self._prefix_buckets[bucket]

    def _encode_segments(self, prompt: str) -> list[int]:
        t0 = time.perf_counter()
        # A conversation's earlier turn, cut after its last added token: its
        # ids are reused whole, and only the rest -- including the text after
        # that cut -- is split and looked up below.
        found_prefix = self._find_prefix(prompt)
        base = None if found_prefix is None else found_prefix[0]
        if base is None:
            start, cut_chars, cut_ids = 0, [], []
        else:
            k = found_prefix[1]
            # Extending it whole retires it; sharing only part leaves it be.
            extends = k == len(base.cut_chars) - 1
            start = int(base.cut_chars[k])
            cut_chars = base.cut_chars[: k + 1].tolist()
            cut_ids = base.cut_ids[: k + 1].tolist()
        parts = self._parts(prompt[start:])
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
        if base is not None:
            out.extend(base.ids[: cut_ids[-1]].tolist())
        with self._lock:
            for key, ids in fresh.items():
                # An empty encoding would cost a cache entry but no budget,
                # so eviction would never bound how many accumulate.
                if len(ids) and key not in self._cache:
                    self._cache[key] = ids
                    self._cached_tokens += len(ids)
            chars = start
            for part, key in zip(parts, keys):
                chars += len(part)
                if key is None:
                    out.append(self._added[part])
                    cut_chars.append(chars)
                    cut_ids.append(len(out) - len(self._head))
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
        if cut_chars:
            self._store_prefix(
                prompt,
                np.asarray(
                    out[len(self._head) : len(self._head) + cut_ids[-1]],
                    dtype=np.int32,
                ),
                cut_chars,
                cut_ids,
                base if base is not None and extends else None,
            )
        out.extend(self._tail)
        if base is not None:
            with self._lock:
                self._prefix_hits += 1
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
                    "%.1f ms per prompt, %d tokens cached, %d prefix hits",
                    st[3],
                    100 * st[0] / max(1, st[0] + st[1]),
                    1e3 * st[2] / st[3],
                    self._cached_tokens,
                    self._prefix_hits,
                )
        return out
