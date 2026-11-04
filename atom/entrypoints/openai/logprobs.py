# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request-owned logprob processing for visible text.

TokenBytes maps tokenizer IDs to original bytes. LogprobText preserves those
positions through parsing. LogprobsProcessor decodes deltas and claims records
from request-local records; LogprobCollector connects it to coalesced engine output.
Chat wire-format assembly lives in serving_chat.py.
"""

import codecs
import json
import logging
from bisect import bisect_right


class LogprobError(RuntimeError):
    """An internal decoding/attribution failure, never a client validation error."""


class LogprobText(str):
    """Decoded text retaining byte spans for token-logprob attribution after parsing."""

    def __new__(cls, parts, owner):
        parts = tuple(p for p in parts if p[0])
        obj = super().__new__(cls, "".join(p[0] for p in parts))
        obj.parts = parts
        obj.owner = owner
        return obj

    @property
    def source_ranges(self):
        return [(lo, hi) for _, lo, hi in self.parts]

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return str.__getitem__(self, key)
        start, end, step = key.indices(len(self))
        if step != 1:
            raise ValueError("Logprob text requires contiguous slices")
        parts, at = [], 0
        for text, lo, hi in self.parts:
            left, right = max(0, start - at), min(len(text), end - at)
            if left < right:
                if hi - lo == len(text.encode("utf-8")):
                    first = lo + len(text[:left].encode("utf-8"))
                    last = lo + len(text[:right].encode("utf-8"))
                else:
                    # A replacement part is one character mapped to the bytes
                    # that caused it. Never infer its source length from U+FFFD.
                    assert len(text) == 1 and left == 0 and right == 1
                    first, last = lo, hi
                parts.append((text[left:right], first, last))
            at += len(text)
        return LogprobText(parts, self.owner)

    def __add__(self, other):
        return join_text((self, other))

    def __radd__(self, other):
        return join_text((other, self))


def join_text(parts):
    parts = list(parts)
    if not parts or type(parts[0]) is str and all(type(p) is str for p in parts):
        return "".join(parts)
    traced = next((p for p in parts if isinstance(p, LogprobText)), None)
    if traced is None:
        return "".join(parts)
    spans = []
    for part in parts:
        if not part:
            continue
        if not isinstance(part, LogprobText) or part.owner is not traced.owner:
            raise LogprobError("Parser lost source-byte provenance")
        spans.extend(part.parts)
    return LogprobText(spans, traced.owner)


class _LogprobStore:
    """Request-local records; retain unclaimed spans while parsers buffer text."""

    def __init__(self, top_k):
        self.top_k = top_k
        self.records = []
        self.ends = []
        self.cursor = 0

    def append(self, start, end, token, record):
        top = record.top[: self.top_k]
        values = (start, end, token, record.logprob, record.rank)
        self.records.append(values + tuple(value for pair in top for value in pair))
        self.ends.append(end)

    def within(self, start, end):
        first = bisect_right(self.ends, start, lo=self.cursor)
        for index in range(first, len(self.records)):
            record = self.records[index]
            if record is None:
                continue
            if record[0] >= end:
                break
            self.records[index] = None
            yield record
        while self.cursor < len(self.records) and self.records[self.cursor] is None:
            self.cursor += 1
        if self.cursor >= 1024:
            del self.records[: self.cursor]
            del self.ends[: self.cursor]
            self.cursor = 0

    def discard_before(self, offset):
        count = bisect_right(self.ends, offset)
        if count:
            del self.records[:count]
            del self.ends[:count]
            self.cursor = max(0, self.cursor - count)

    def close(self):
        self.records.clear()
        self.ends.clear()
        self.cursor = 0


class TokenBytes:
    def __init__(self, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        state = json.loads(decoder.__getstate__()) if decoder is not None else {}
        if state.get("type") != "ByteLevel":
            raise ValueError("Chat logprobs currently requires a ByteLevel tokenizer")
        self.tokenizer = tokenizer
        self.special = frozenset(tokenizer.all_special_ids) | {
            token_id
            for token_id, token in getattr(
                tokenizer, "added_tokens_decoder", {}
            ).items()
            if token.special
        }
        # Inverse of the reversible GPT-2/ByteLevel byte alphabet.
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = list(bs)
        extra = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + extra)
                extra += 1
        self.byte_decoder = dict(zip(map(chr, cs), bs))
        self.cache = {}

    def raw(self, token_id):
        if token_id in self.cache:
            return self.cache[token_id]
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        # Model vocabularies may be padded beyond the tokenizer vocabulary.
        # Such a candidate has no text bytes; its probability remains valid.
        if token is None:
            return b""
        try:
            raw = bytes(self.byte_decoder[c] for c in token)
        except KeyError:
            # Added non-byte-alphabet tokens (e.g. structural markers).
            raw = token.encode("utf-8")
        if len(self.cache) >= 8192:
            self.cache.clear()
        self.cache[token_id] = raw
        return raw

    def candidate(self, token_id, logprob):
        raw = self.raw(token_id)
        return {
            "token": raw.decode("utf-8", errors="replace"),
            "bytes": list(raw),
            "logprob": float(logprob),
        }


class LogprobsProcessor:
    def __init__(self, codec, top_k=0):
        self.codec = codec
        self.store = _LogprobStore(top_k)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.source_end = 0

    def feed(self, token_ids, records, *, finished=False):
        if token_ids and (records is None or len(token_ids) != len(records)):
            raise LogprobError("Missing or misaligned engine token logprobs")
        pending = self.decoder.getstate()[0]
        start = self.source_end - len(pending)
        pieces = []
        for token_id, record in zip(token_ids, records or ()):
            if int(token_id) in self.codec.special:
                continue
            raw = self.codec.raw(token_id)
            end = self.source_end + len(raw)
            self.store.append(self.source_end, end, token_id, record)
            self.source_end = end
            pieces.append(raw)
        raw = b"".join(pieces)
        text = self.decoder.decode(raw, final=finished)
        remaining = len(self.decoder.getstate()[0])
        data = pending + raw
        if remaining:
            data = data[:-remaining]
        parts = []
        # Usually one exact UTF-8 span. For malformed/truncated bytes, map each
        # replacement character to the original bytes, not its three-byte encoding.
        while data:
            try:
                decoded = data.decode("utf-8")
            except UnicodeDecodeError as error:
                if error.start:
                    parts.append(
                        (
                            data[: error.start].decode("utf-8"),
                            start,
                            start + error.start,
                        )
                    )
                parts.append(("\ufffd", start + error.start, start + error.end))
                start += error.end
                data = data[error.end :]
            else:
                parts.append((decoded, start, start + len(data)))
                break
        sourced = LogprobText(parts, self)
        if sourced != text:
            raise LogprobError("Token-byte decoder lost source alignment")
        return sourced

    def take(self, text):
        if not text:
            return []
        if not isinstance(text, LogprobText) or text.owner is not self:
            raise LogprobError("Visible text has no token-byte source spans")
        result = []
        for start, end in text.source_ranges:
            for record in self.store.within(start, end):
                _, _, token, logprob, rank, *top = record
                entry = self.codec.candidate(token, logprob if rank <= 20 else -9999.0)
                entry["top_logprobs"] = [
                    self.codec.candidate(t, p)
                    for t, p in zip(top[::2], top[1::2])
                    if t >= 0
                ]
                result.append(entry)
        return result

    def take_fields(self, fields):
        """Assign boundary tokens once, in source order for both JSON and SSE."""
        if any(
            value and (not isinstance(value, LogprobText) or value.owner is not self)
            for value in fields.values()
        ):
            raise LogprobError("Visible text has no token-byte source spans")
        values = {field: [] for field in fields}
        # A replayed Kimi answer has the same source as its earlier reasoning.
        # Preserve the first delivery's ownership when both spans are identical.
        spans = sorted(
            (start, end, 0 if field == "reasoning_content" else 1, field, text)
            for field, value in fields.items()
            if value
            for text, start, end in value.parts
        )
        for start, end, _, field, text in spans:
            values[field].extend(self.take(LogprobText([(text, start, end)], self)))
        return values

    def discard_resolved(self, held_text):
        # Called only after the tool parser and pending SSE deltas are drained.
        # Keep the UTF-8 prefix and reasoning marker prefix, including any token
        # straddling their boundaries. Everything older has been classified.
        offset = self.source_end - len(self.decoder.getstate()[0])
        if held_text:
            if not isinstance(held_text, LogprobText):
                raise LogprobError("Buffered text has no token-byte source spans")
            offset = min(offset, min(start for start, _ in held_text.source_ranges))
        self.store.discard_before(offset)

    def close(self):
        self.store.close()


class LogprobCollector:
    """Observe raw deltas without changing the existing text/SSE generators."""

    def __init__(self, source, codec, top_k, n=1):
        self.source = source
        self.trackers = [LogprobsProcessor(codec, top_k) for _ in range(n)]

    async def get(self):
        payload = await self.source.get()
        index, chunk = payload if isinstance(payload, tuple) else (0, payload)
        try:
            if chunk.get("finish_reason") == "logprob_error":
                raise LogprobError(
                    "Engine returned missing or misaligned token logprobs"
                )
            chunk["text"] = self.trackers[index].feed(
                chunk.get("token_ids", []),
                chunk.get("logprobs"),
                finished=bool(chunk.get("finished")),
            )
        except LogprobError as exc:
            logging.getLogger("atom").exception(
                "Logprob decoding failed for choice %s", index
            )
            chunk.update(
                text="",
                finished=True,
                finish_reason="logprob_error",
                logprob_error=str(exc),
            )
        return payload
