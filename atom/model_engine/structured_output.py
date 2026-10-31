# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Target-logit constraints with incremental committed grammar state.

Only the previous forward's output prefix is retained to bridge deferred CPU
delivery. Older requests reconstruct from the scheduler's committed history.
Only the immediately preceding batch retains matchers. Prefill/replay or a
history mismatch reconstructs from the scheduler rather than reusing stale state.
"""

from functools import lru_cache

import torch


def structured_stop_token_ids(config, tokenizer):
    stops = list(getattr(config, "stop_token_ids", None) or [])
    if tokenizer.eos_token_id is not None:
        stops.append(tokenizer.eos_token_id)
    if not stops:
        raise ValueError("Structured output requires a configured EOS token")
    return list(dict.fromkeys(stops))


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
        matcher = self.xgr.GrammarMatcher(self.compile(schema))
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
        prefixes = {}
        rows, masks = [], []
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
                mask = self.xgr.allocate_token_bitmask(1, self.vocab_size)
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
                masks.append(mask)
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
            bits = torch.cat(masks).to(logits.device)
            tokens = torch.arange(self.vocab_size, device=logits.device)
            allowed = ((bits[:, tokens // 32] >> (tokens % 32)) & 1).bool()
            indexes = torch.tensor(rows, device=logits.device)
            selected = logits[indexes]
            selected.masked_fill_(~allowed, float("-inf"))
            logits[indexes] = selected
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
