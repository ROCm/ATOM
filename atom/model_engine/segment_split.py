# SPDX-License-Identifier: MIT
"""Splitting a prompt at added (special) tokens the way a fast tokenizer does,
and deciding whether encoding the pieces separately is exact.

A fast tokenizer extracts added tokens before anything else: non-normalized
ones from the raw text, then normalized ones from what is left, each pass
leftmost-longest. The text between two extracted tokens is then normalized,
pre-tokenized and encoded on its own, so encoding it alone gives the same ids
-- unless a token strips its neighbours' whitespace, the normalizer or
pre-tokenizer treats the start of its input specially, the model samples
(BPE dropout), or the two passes can
disagree about where a token is. `AddedTokenSplitter.segments_exact` and
`prefix_cuts_stable` decide those cases from the tokenizer's own description,
and a few placements are encoded both ways as a backstop.
"""

import json
import re

import numpy as np

_END = ""  # marks a token's end in the trie; never a character


def _alternation(tokens):
    """A capturing leftmost-longest pattern for `tokens`, or None.

    Built from a trie of the tokens rather than one alternative per token:
    a flat alternation of a thousand literals sharing a first character (a
    tokenizer's `<|...|>` family) retries every literal at each such
    character of the text, which made splitting slower than encoding."""
    trie: dict = {}
    for t in set(tokens):
        node = trie
        for ch in t:
            node = node.setdefault(ch, {})
        node[_END] = None
    if not trie:
        return None
    return re.compile("(" + _trie_pattern(trie) + ")")


def _trie_pattern(node: dict) -> str:
    """The continuations below `node`. The children start with different
    characters, so at most one can match, and a token ending at `node` is
    taken only when no longer one matches: leftmost-longest, as before."""
    alts = []
    for ch, child in node.items():
        if ch == _END:
            continue
        run = [ch]
        # A chain of single children is one literal (also keeps the
        # recursion shallow for very long tokens).
        while _END not in child and len(child) == 1:
            ((ch, child),) = child.items()
            run.append(ch)
        alts.append(re.escape("".join(run)) + _trie_pattern(child))
    if not alts:
        return ""
    if len(alts) == 1:
        return f"(?:{alts[0]})?" if _END in node else alts[0]
    body = "(?:" + "|".join(alts) + ")"
    return body + "?" if _END in node else body


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


def _deterministic(model) -> bool:
    """Whether the model encodes a given text the same way every time. BPE
    with dropout samples merges per call, so a segment's ids must not be
    shared between occurrences or requests."""
    return not (isinstance(model, dict) and model.get("dropout"))


def _normalizer_is_identity(description: dict) -> bool:
    normalizer = description.get("normalizer")
    return normalizer is None or (
        normalizer.get("type") == "Sequence" and not normalizer.get("normalizers")
    )


def _cuts_are_stable(added, normalized) -> bool:
    """Whether a prompt cut right after an added token `t` extracts the same up
    to the cut whatever follows it -- what reusing a stored prefix needs.

    Within one pass that fails only when another added token contains `t` (it
    could start at or before `t` and run past the cut); leftmost-longest takes
    `t` first otherwise. Across passes it also fails when a non-normalized
    token, extracted from the raw text first, could start inside a normalized
    `t` and run past the cut (normalized `ab`, non-normalized `bc`, then `c`).
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


class AddedTokenSplitter:
    """A tokenizer's added-token extraction, reproduced on the host."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.backend = tokenizer._tokenizer
        self.added: dict[str, int] = dict(tokenizer.get_added_vocab())
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
        self._decoder = list(decoder.values())
        self._normalized = {t.content for t in self._decoder if t.normalized}
        self._raw_pattern = _alternation(
            t for t in self.added if t not in self._normalized
        )
        self._normalized_pattern = _alternation(
            t for t in self.added if t in self._normalized
        )
        # One-element arrays for added-token ids, shared by every prompt.
        self._id_arrays = {
            t: np.asarray([i], dtype=np.int32) for t, i in self.added.items()
        }
        # What `encode` wraps a prompt in (BOS/EOS), found on a probe.
        full = tokenizer.encode("a")
        bare = tokenizer.encode("a", add_special_tokens=False)
        start = next(
            (
                i
                for i in range(len(full) - len(bare) + 1)
                if full[i : i + len(bare)] == bare
            ),
            None,
        )
        self.wrapped = start is not None
        if self.wrapped:
            self.head = np.asarray(full[:start], dtype=np.int32)
            self.tail = np.asarray(full[start + len(bare) :], dtype=np.int32)

    def id_array(self, token: str) -> np.ndarray:
        return self._id_arrays[token]

    def segments_exact(self) -> bool:
        """Whether encoding the pieces of `parts` separately is exact."""
        if not self.added or not self.wrapped:
            return False
        # With special tokens split like text, `parts` would still extract
        # them as single ids; the probes need not hit a special one.
        if getattr(self.tokenizer, "split_special_tokens", False) or getattr(
            self.backend, "encode_special_tokens", False
        ):
            return False
        if any(t.lstrip or t.rstrip or t.single_word for t in self._decoder):
            return False
        try:
            description = json.loads(self.backend.to_str())
        except (ValueError, AttributeError):
            return False
        if not _deterministic(description.get("model")):
            return False
        # A normalizer runs per segment too, and a normalized added token is
        # matched after it runs; only the identity is accepted.
        if not _normalizer_is_identity(description):
            return False
        if _offset_sensitive(description.get("pre_tokenizer")):
            return False
        return self._probes_match()

    def prefix_cuts_stable(self) -> bool:
        return _cuts_are_stable(self.added, self._normalized)

    def parts(self, text: str) -> list[tuple[str, bool]]:
        """`text` as `(piece, is_added_token)`, in the backend's order. Whether
        a piece was extracted is tracked, not inferred from its spelling: text
        that happens to spell an added token is still split by the passes."""
        parts = [(text, False)]
        for pattern in (self._raw_pattern, self._normalized_pattern):
            if pattern is None:
                continue
            nxt = []
            for piece, extracted in parts:
                if extracted:
                    nxt.append((piece, True))
                    continue
                # A capturing split alternates text and matches.
                for i, sub in enumerate(pattern.split(piece)):
                    if sub:
                        nxt.append((sub, i % 2 == 1))
            parts = nxt
        return parts

    def encode_pieces(self, text: str) -> list[int]:
        """`tokenizer.encode(text)` by pieces, without any cache."""
        out = self.head.tolist()
        for piece, extracted in self.parts(text):
            if extracted:
                out.append(self.added[piece])
            else:
                out.extend(self.backend.encode(piece, add_special_tokens=False).ids)
        out.extend(self.tail.tolist())
        return out

    def _probes_match(self) -> bool:
        """A few added-token placements, encoded both ways once, at startup."""
        added = list(self.added)
        picks = {added[0], added[-1], min(added, key=len), max(added, key=len)}
        for tok in picks:
            for probe in (f"a{tok}a b", f"{tok} x\n{tok}", f" y {tok}\t1 2"):
                if self.encode_pieces(probe) != self.tokenizer.encode(probe):
                    return False
        return True
