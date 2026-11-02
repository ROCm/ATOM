# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Keep generated token probabilities aligned with visible Chat byte spans."""

import codecs
import json
from functools import lru_cache

from .logprob_store import LogprobStore
from .source_text import SourceText


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
        self.store = LogprobStore(top_k)
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

    def close(self):
        self.store.close()


class LogprobCollector:
    """Observe raw deltas without changing the existing text/SSE generators."""

    def __init__(self, source, codec, top_k, n=1):
        self.source = source
        self.trackers = [LogprobTracker(codec, top_k) for _ in range(n)]

    async def get(self):
        payload = await self.source.get()
        index, chunk = payload if isinstance(payload, tuple) else (0, payload)
        chunk["text"] = self.trackers[index].feed(
            chunk.get("token_ids", []),
            chunk.get("logprobs"),
            finished=bool(chunk.get("finished")),
        )
        return payload


async def stream_logprobs(source, trackers):
    # create_chat_chunk attaches probabilities before serialization, while source
    # spans still exist. Never search rendered SSE text or re-encode its frames.
    try:
        async for frame in source:
            yield frame
    finally:
        try:
            await source.aclose()
        finally:
            for tracker in trackers:
                tracker.close()


def response_logprobs(
    response,
    outputs,
    codec,
    top_k,
    *,
    reasoning,
    tools=None,
    tool_choice=None,
    tool_parser_cls=None,
):
    from .serving_chat import _tool_parser_for_request
    from .tool_parser import parse_tool_calls
    from .tool_parser.registry import forbids_tool_calls

    for choice, output in zip(response.choices, outputs):
        tracker = LogprobTracker(codec, top_k)
        try:
            source = tracker.feed(
                output["token_ids"], output.get("logprobs"), finished=True
            )
            thought, answer = reasoning.split(source)
            content, _ = parse_tool_calls(
                answer,
                tools,
                parser_cls=_tool_parser_for_request(tool_parser_cls, tools),
                suppress_calls=forbids_tool_calls(tool_choice),
            )
            # Consume spans independently; reasoning and content can be interleaved
            # in source order, then joined into separate non-streaming fields.
            values = {"content": tracker.take(content)}
            choice["message"]["content"] = str(content)
            if thought is not None:
                choice["message"]["reasoning_content"] = str(thought)
                values["reasoning_content"] = tracker.take(thought)
            choice["logprobs"] = values
        finally:
            tracker.close()
    return response
