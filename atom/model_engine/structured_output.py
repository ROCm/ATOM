# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Target-logit constraints with incremental committed grammar state.

Only the previous forward's output prefix is retained to bridge deferred CPU
delivery. Older requests reconstruct from the scheduler's committed history.
Only the immediately preceding batch retains matchers. Prefill/replay or a
history mismatch reconstructs from the scheduler rather than reusing stale state.
"""

from functools import lru_cache


def structured_stop_token_ids(config, tokenizer):
    stops = list(getattr(config, "stop_token_ids", None) or [])
    if tokenizer.eos_token_id is not None:
        stops.append(tokenizer.eos_token_id)
    if not stops:
        raise ValueError("Structured output requires a configured EOS token")
    return list(dict.fromkeys(stops))


def tokenizer_fingerprint(tokenizer, stop_token_ids):
    """Identify the actual token mapping/decoder and effective termination IDs."""
    import hashlib
    import json

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Structured output requires a serializable fast tokenizer")
    payload = {
        "tokenizer": json.loads(backend.to_str()),
        "stops": sorted(stop_token_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_worker_grammar(runner, schema, tool_format, fingerprint, vocab_size):
    """Compile on the execution rank via RPC, before the request is enqueued.

    The RPC dispatcher catches failures without terminating the worker loop.
    No GPU operation or inter-rank collective is performed here.
    """
    structured = getattr(runner, "structured_output", None)
    if structured is None:
        from atom.model_engine.llm_engine import _load_tokenizer
        from atom.model_ops.embed_head import ParallelLMHead

        # Use the physical target projection, never a first-forward tensor or
        # an independently guessed HF vocab. Reject unsupported projections at
        # admission instead of discovering a mismatch during model execution.
        sizes = {
            head.num_embeddings
            for head in runner.model.modules()
            if isinstance(head, ParallelLMHead)
        }
        if sizes != {vocab_size}:
            raise ValueError(
                "Structured output vocabulary differs from the target LM head"
            )
        tokenizer = _load_tokenizer(
            runner.config.model, runner.config.trust_remote_code
        )
        stops = structured_stop_token_ids(runner.config, tokenizer)
        worker_fingerprint = tokenizer_fingerprint(tokenizer, stops)
        if worker_fingerprint != fingerprint:
            raise ValueError(
                "Structured output tokenizer/stop tokens differ from the API"
            )
        structured = StructuredOutput(tokenizer, vocab_size, stops)
        structured.fingerprint = worker_fingerprint
        runner.structured_output = structured
    if structured.vocab_size != vocab_size or structured.fingerprint != fingerprint:
        raise ValueError("Structured output deployment changed; restart the server")
    structured.compile(schema, tool_format)
    return {"fingerprint": fingerprint, "vocab_size": vocab_size}


class StructuredOutput:
    def __init__(self, tokenizer, vocab_size, stop_token_ids):
        from llguidance.hf import from_tokenizer

        from . import structured_grammar as grammar

        self.xgr = grammar
        self.stop_ids = list(stop_token_ids)
        info = from_tokenizer(tokenizer, n_vocab=vocab_size, eos_token=self.stop_ids)
        self.compiler = grammar.GrammarCompiler(info)
        self.compile = lru_cache(maxsize=128)(self.compiler.compile_json_schema)
        self.vocab_size = vocab_size
        self.pending = {}
        self.states = {}
        self.current = {}

    def clear(self):
        self.pending.clear()
        self.states.clear()
        self.current.clear()

    def advance(self, matcher, active, marker, history, token):
        if active and not matcher.accept_token(token):
            raise ValueError("Committed output violates the structured-output grammar")
        history.append(token)
        if not active and tuple(history[-len(marker) :]) == tuple(marker):
            active = True
        return active

    def matcher(self, schema, marker, history):
        compiled = (
            self.compile(*schema) if isinstance(schema, tuple) else self.compile(schema)
        )
        matcher = self.xgr.GrammarMatcher(compiled)
        body = history
        if marker:
            end = next(
                (
                    i + len(marker)
                    for i in range(len(history))
                    if tuple(history[i : i + len(marker)]) == tuple(marker)
                ),
                None,
            )
            if end is None:
                return matcher, False
            body = history[end:]
        for token in body:
            if not matcher.accept_token(token):
                raise ValueError(
                    "Committed output violates the structured-output grammar"
                )
            if token in self.stop_ids:
                break
        return matcher, True

    def apply(self, logits, batch, metadata):
        requests = getattr(batch, "structured_requests", {})
        previous, self.pending = self.pending, {}
        states, self.states = self.states, {}
        self.current = {}
        if not requests:
            return {}
        target = metadata.target_logits_indices.tolist() if metadata else []
        bonus = metadata.bonus_logits_indices.tolist() if metadata else []
        lengths = metadata.num_draft_tokens_np.tolist() if metadata else []
        input_ids = metadata.input_ids.tolist() if metadata else []
        if getattr(self, "mask_buffer", None) is None:
            from .structured_output_mask import StructuredOutputMask

            self.mask_buffer = StructuredOutputMask(self.vocab_size)
        mask_count = len(target) + len(bonus) if metadata else len(requests)
        masks = self.mask_buffer.prepare(mask_count, logits.device)
        prefixes = {}
        rows = []
        for i, (schema, marker, committed) in requests.items():
            if batch.is_final_chunk is not None and not batch.is_final_chunk[i]:
                continue
            req_id = batch.req_ids[i]
            history = list(committed)
            pending = previous.get(req_id)
            # A prefill/replay uses the scheduler's authoritative history.
            if (
                i >= batch.total_seqs_num_prefill
                and pending is not None
                and pending[: len(history)] == history
            ):
                history = list(pending)
            prefixes[i] = (schema, marker, history)
            state = states.get(req_id)
            if (
                i >= batch.total_seqs_num_prefill
                and state is not None
                and state[:3] == (schema, marker, history)
            ):
                matcher, active = state[3:]
            else:
                matcher, active = self.matcher(schema, marker, history)
            self.current[i] = (matcher, active)
            # The speculative branch may be rejected. Never advance the
            # committed matcher with draft tokens; clone it once per request.
            if metadata:
                matcher = matcher.copy()
            if metadata:
                start = sum(lengths[:i])
                branch_rows = target[start : start + lengths[i]] + [bonus[i]]
            else:
                branch_rows = [i]
            branch = list(history)
            unreachable = False
            for pos, row in enumerate(branch_rows):
                mask = masks[len(rows) : len(rows) + 1]
                if active and matcher.is_terminated():
                    # Deferred delivery may leave a final forward in flight
                    # after EOS, before the scheduler retires this request.
                    mask.zero_()
                    for stop in self.stop_ids:
                        value = (
                            int(mask[0, stop // 32]) | (1 << (stop % 32))
                        ) & 0xFFFFFFFF
                        mask[0, stop // 32] = value if value < 2**31 else value - 2**32
                    unreachable = True
                elif active and not unreachable:
                    matcher.fill_next_token_bitmask(mask)
                else:
                    mask.fill_(-1)
                    if not unreachable:
                        for stop in self.stop_ids:
                            value = (
                                int(mask[0, stop // 32]) & ~(1 << (stop % 32))
                            ) & 0xFFFFFFFF
                            mask[0, stop // 32] = (
                                value if value < 2**31 else value - 2**32
                            )
                rows.append(row)
                if metadata and pos < len(branch_rows) - 1:
                    draft = input_ids[row + 1]
                    if active and not unreachable and not matcher.accept_token(draft):
                        unreachable = True
                    if active and matcher.is_terminated():
                        unreachable = True
                    if not unreachable:
                        branch.append(draft)
                        if not active and tuple(branch[-len(marker) :]) == tuple(
                            marker
                        ):
                            active = True
        if rows:
            self.mask_buffer.apply(logits, rows)
        return prefixes

    def commit(self, sampled, batch, prefixes):
        if not prefixes:
            return
        ids = sampled.reshape(len(batch.req_ids), -1).tolist()
        for i, (schema, marker, history) in prefixes.items():
            emitted = [token for token in ids[i] if token >= 0]
            for j, token in enumerate(emitted):
                if token in self.stop_ids:
                    emitted = emitted[: j + 1]
                    break
            full = list(history)
            matcher, active = self.current[i]
            for token in emitted:
                # An already-finished request can have one forward in flight.
                if active and matcher.is_terminated() and token in self.stop_ids:
                    full.append(token)
                    break
                active = self.advance(matcher, active, marker, full, token)
            self.pending[batch.req_ids[i]] = full
            self.states[batch.req_ids[i]] = (schema, marker, full, matcher, active)
        self.current.clear()
