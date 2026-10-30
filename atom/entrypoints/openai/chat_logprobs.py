# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep generated token probabilities aligned with visible Chat byte spans."""

import json
from functools import lru_cache

from .sse import data_frame, iter_sse_data


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


@lru_cache(maxsize=4)
def token_bytes(tokenizer):
    return TokenBytes(tokenizer)


class LogprobTracker:
    def __init__(self, codec, top_k=0):
        self.codec = codec
        self.top_k = top_k
        self.raw = bytearray()
        self.spans = []
        self.cursor = 0
        self.emitted = 0

    def feed(self, token_ids, records):
        if records is None or len(token_ids) != len(records):
            raise RuntimeError("Missing or misaligned engine token logprobs")
        for token_id, record in zip(token_ids, records):
            if int(token_id) in self.codec.special:
                continue  # Mirrors skip_special_tokens in the text decoder.
            start = len(self.raw)
            self.raw.extend(self.codec.raw(token_id))
            self.spans.append((start, len(self.raw), token_id, record))

    def take(self, text):
        if not text:
            return []
        visible = text.encode("utf-8")
        start = self.raw.find(visible, self.cursor)
        end = start + len(visible)
        if start < 0 and "\ufffd" in text:
            # A token cap may leave incomplete UTF-8. Match the text decoder's
            # replacement behavior while retaining the original token bytes.
            prefix = text.split("\ufffd", 1)[0].encode("utf-8")
            starts = (
                [self.raw.find(prefix, self.cursor)]
                if prefix
                else [self.cursor] + [lo for lo, *_ in self.spans if lo >= self.cursor]
            )
            for lo in starts:
                if lo < 0:
                    continue
                for _, hi, *_ in self.spans:
                    if (
                        hi > lo
                        and self.raw[lo:hi].decode("utf-8", errors="replace") == text
                    ):
                        start, end = lo, hi
                        break
                if start >= 0:
                    break
        if start < 0:
            raise RuntimeError(
                "Visible text cannot be aligned to generated token bytes"
            )
        result = []
        while self.emitted < len(self.spans):
            _lo, hi, token_id, record = self.spans[self.emitted]
            if hi > end:
                break  # A parser may split a token across two text deltas.
            self.emitted += 1
            if hi <= start:
                continue  # Reasoning/tool delimiters removed by the parsers.
            entry = self.codec.candidate(
                token_id, record.logprob if record.rank <= 20 else -9999.0
            )
            entry["top_logprobs"] = [
                self.codec.candidate(t, p) for t, p in record.top[: self.top_k]
            ]
            result.append(entry)
        self.cursor = end
        # Streaming requests retain only unconsumed parser/UTF-8 fragments.
        # Top-K records for the entire response must not accumulate here.
        self.spans = self.spans[self.emitted :]
        self.emitted = 0
        cut = min(self.cursor, self.spans[0][0]) if self.spans else self.cursor
        if cut:
            del self.raw[:cut]
            self.cursor -= cut
            self.spans = [
                (lo - cut, hi - cut, token, record)
                for lo, hi, token, record in self.spans
            ]
        return result


class LogprobCollector:
    """Observe raw deltas without changing the existing text/SSE generators."""

    def __init__(self, source, codec, top_k, n=1):
        self.source = source
        self.trackers = [LogprobTracker(codec, top_k) for _ in range(n)]

    async def get(self):
        payload = await self.source.get()
        index, chunk = payload if isinstance(payload, tuple) else (0, payload)
        if chunk.get("token_ids"):
            self.trackers[index].feed(chunk["token_ids"], chunk.get("logprobs"))
        return payload


async def stream_logprobs(source, trackers):
    try:
        async for frame in source:
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    tracker = trackers[choice["index"]]
                    values = {}
                    for field in ("reasoning_content", "content"):
                        if text := choice.get("delta", {}).get(field):
                            values[field] = tracker.take(text)
                    if values:
                        choice["logprobs"] = values
                yield data_frame(chunk)
    finally:
        await source.aclose()


def response_logprobs(response, outputs, codec, top_k):
    for choice, output in zip(response.choices, outputs):
        tracker = LogprobTracker(codec, top_k)
        tracker.feed(output["token_ids"], output.get("logprobs"))
        values = {"content": []}
        for field in ("reasoning_content", "content"):
            if text := choice["message"].get(field):
                values[field] = tracker.take(text)
        choice["logprobs"] = values
    return response
