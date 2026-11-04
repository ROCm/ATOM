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
from bisect import bisect_left, bisect_right

from transformers.convert_slow_tokenizer import bytes_to_unicode


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
        first = bisect_left(self.ends, start, lo=self.cursor)
        for index in range(first, len(self.records)):
            record = self.records[index]
            if record is None:
                continue
            zero_width = record[0] == record[1]
            if record[0] > end or (record[0] == end and not zero_width):
                break
            if record[1] == start and not zero_width:
                continue
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
        if getattr(tokenizer, "clean_up_tokenization_spaces", False):
            raise ValueError(
                "Chat logprobs does not support tokenizer whitespace cleanup; "
                "the tokenizer's decoding policy is not changed by this request"
            )
        self.tokenizer = tokenizer
        self.special = frozenset(tokenizer.all_special_ids) | {
            token_id
            for token_id, token in getattr(
                tokenizer, "added_tokens_decoder", {}
            ).items()
            if token.special
        }
        inverse = {char: byte for byte, char in bytes_to_unicode().items()}
        self.byte_table = {}
        for token, token_id in tokenizer.get_vocab().items():
            try:
                raw = bytes(inverse[char] for char in token)
            except KeyError:
                raw = token.encode("utf-8")
            self.byte_table[token_id] = raw

    def raw(self, token_id):
        # Padded model-vocabulary IDs have no tokenizer representation.
        return self.byte_table.get(int(token_id), b"")

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
        # JSON has no event order; reasoning was delivered before replayed content.
        ordered = dict(
            sorted(fields.items(), key=lambda item: item[0] != "reasoning_content")
        )
        return claim_logprobs([ordered])[0]

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


def claim_logprobs(deltas):
    """Claim source spans once, breaking equal starts by delivery order."""
    values = [{} for _ in deltas]
    spans = []
    for i, delta in enumerate(deltas):
        for priority, (field, text) in enumerate(delta.items()):
            if field not in ("content", "reasoning_content"):
                continue
            values[i][field] = []
            if not text:
                continue
            if not isinstance(text, LogprobText):
                raise LogprobError("Visible text has no token-byte source spans")
            for value, start, end in text.parts:
                spans.append((start, i, priority, end, field, value, text.owner))
    for start, i, _, end, field, value, owner in sorted(spans):
        values[i][field].extend(owner.take(LogprobText([(value, start, end)], owner)))
    return values
