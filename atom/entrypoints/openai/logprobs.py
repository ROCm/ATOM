# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Request-owned logprob processing for visible text.

TokenBytes maps tokenizer IDs to original bytes. SourceText preserves those
positions through parsing. LogprobsProcessor decodes deltas and claims records
from its bounded spool; LogprobCollector connects it to coalesced engine output.
Chat wire-format assembly lives in serving_chat.py.
"""

import codecs
import json
import struct
from contextlib import ExitStack
from tempfile import SpooledTemporaryFile


class SourceText(str):
    """Parsed text retaining original byte spans, including UTF-8 replacements."""

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
            raise ValueError("Source text requires contiguous slices")
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
        return SourceText(parts, self.owner)

    def __add__(self, other):
        return join_text((self, other))

    def __radd__(self, other):
        return join_text((other, self))


def join_text(parts):
    parts = list(parts)
    if not parts or type(parts[0]) is str and all(type(p) is str for p in parts):
        return "".join(parts)
    traced = next((p for p in parts if isinstance(p, SourceText)), None)
    if traced is None:
        return "".join(parts)
    spans = []
    for part in parts:
        if not part:
            continue
        if not isinstance(part, SourceText) or part.owner is not traced.owner:
            raise ValueError("Parser lost source-byte provenance")
        spans.extend(part.parts)
    return SourceText(spans, traced.owner)


class _LogprobStore:
    """Fixed-width records spill after 64 KiB; ownership needs no Python index."""

    MEMORY_LIMIT = 64 * 1024

    def __init__(self, top_k):
        self.layout = struct.Struct("<QQqdi" + "qd" * top_k + "?")
        self.top_k = top_k
        self._files = ExitStack()
        self.file = self._files.enter_context(
            SpooledTemporaryFile(  # noqa: SIM115 - owned by the request ExitStack
                max_size=self.MEMORY_LIMIT, mode="w+b"
            )
        )
        self.count = 0

    def append(self, start, end, token, record):
        top = list(record.top[: self.top_k])
        top += [(-1, 0.0)] * (self.top_k - len(top))
        values = [start, end, token, record.logprob, record.rank]
        values.extend(value for pair in top for value in pair)
        self.file.seek(self.count * self.layout.size)
        self.file.write(self.layout.pack(*values, False))
        self.count += 1

    def read(self, index):
        self.file.seek(index * self.layout.size)
        return self.layout.unpack(self.file.read(self.layout.size))

    def within(self, start, end):
        left, right = 0, self.count
        while left < right:
            middle = (left + right) // 2
            if self.read(middle)[1] <= start:
                left = middle + 1
            else:
                right = middle
        for index in range(left, self.count):
            record = self.read(index)
            if record[0] >= end:
                break
            if not record[-1]:
                # Store ownership in the spool, not an ever-growing Python set.
                # A partially visible token still contributes its probability.
                self.file.seek((index + 1) * self.layout.size - 1)
                self.file.write(b"\x01")
                yield record[:-1]

    def close(self):
        self._files.close()


class TokenBytes:
    def __init__(self, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        state = json.loads(decoder.__getstate__()) if decoder is not None else {}
        if state.get("type") != "ByteLevel":
            raise ValueError("Chat logprobs currently requires a ByteLevel tokenizer")
        self.tokenizer = tokenizer
        self.special = frozenset(tokenizer.all_special_ids)
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
            raise RuntimeError("Missing or misaligned engine token logprobs")
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
        sourced = SourceText(parts, self)
        if sourced != text:
            raise RuntimeError("Token-byte decoder lost source alignment")
        return sourced

    def take(self, text):
        if not text:
            return []
        if not isinstance(text, SourceText) or text.owner is not self:
            raise RuntimeError("Visible text has no token-byte source spans")
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
        values = {field: [] for field in fields}
        spans = sorted(
            (start, end, field, text)
            for field, value in fields.items()
            if value
            for text, start, end in value.parts
        )
        for start, end, field, text in spans:
            values[field].extend(self.take(SourceText([(text, start, end)], self)))
        return values

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
        chunk["text"] = self.trackers[index].feed(
            chunk.get("token_ids", []),
            chunk.get("logprobs"),
            finished=bool(chunk.get("finished")),
        )
        return payload
