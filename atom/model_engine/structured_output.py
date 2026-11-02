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
        self.commit_records = None
        self.draft_plan = None

    def prepare(self, batch, metadata):
        """Publish compact draft snapshots before launching the target forward."""
        self.draft_plan = None
        requests = getattr(batch, "structured_requests", {})
        if not requests or metadata is None:
            return
        import torch

        from .structured_output_transfer import AsyncTokenCopy

        # Host mirrors are consumed here, while this step owns the staging slot.
        # Copy row numbers into Python, never retain a reusable NumPy view.
        target = metadata.target_logits_indices_np
        bonus = metadata.bonus_logits_indices_np
        if target is None or bonus is None:
            raise ValueError("Structured draft preparation requires host indices")
        lengths = metadata.num_draft_tokens_np
        starts = lengths.cumsum() - lengths
        plan, pieces, offset = {}, [], 0
        for i in requests:
            if batch.is_final_chunk is not None and not batch.is_final_chunk[i]:
                continue
            start, count = int(starts[i]), int(lengths[i])
            rows = target[start : start + count].tolist()
            plan[i] = (rows + [int(bonus[i])], offset, count)
            # Target rows are contiguous within a request's verification span.
            # cat owns the snapshot before an input buffer can be reused.
            if count:
                pieces.append(metadata.input_ids[rows[0] + 1 : rows[0] + 1 + count])
            offset += count
        self.draft_plan = plan
        if pieces:
            if getattr(self, "draft_copy", None) is None:
                self.draft_copy = AsyncTokenCopy(metadata.input_ids.device)
            self.draft_copy.start(torch.cat(pieces))

    def _finish_commit(self):
        records = getattr(self, "commit_records", None)
        if records is not None:
            # Called after this step's forward has been submitted. Only wait for
            # last step's token copy, never for the current forward or sampler.
            ids = self.commit_copy.wait().tolist()
            self.commit_records = None
            self._commit_ids(ids, records)

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
        if not requests:
            self.clear()
            return {}
        self._finish_commit()
        previous, self.pending = self.pending, {}
        states, self.states = self.states, {}
        self.current = {}
        plan = getattr(self, "draft_plan", None)
        drafts = []
        if metadata is not None:
            if plan is None:
                # CPU reference tests can execute without a runner. GPU callers
                # must prefetch: never silently reintroduce synchronous D2H.
                if metadata.input_ids.device.type != "cpu":
                    raise ValueError("Structured draft tokens were not prefetched")
                target = metadata.target_logits_indices.tolist()
                bonus = metadata.bonus_logits_indices.tolist()
                lengths = metadata.num_draft_tokens_np.tolist()
                input_ids = metadata.input_ids.tolist()
                plan = {}
                for i in requests:
                    start = sum(lengths[:i])
                    rows = target[start : start + lengths[i]]
                    plan[i] = (rows + [bonus[i]], len(drafts), len(rows))
                    drafts.extend(input_ids[row + 1] for row in rows)
            elif any(count for _, _, count in plan.values()):
                drafts = self.draft_copy.wait().tolist()
        self.draft_plan = None
        if getattr(self, "mask_buffer", None) is None:
            from .structured_output_mask import StructuredOutputMask

            self.mask_buffer = StructuredOutputMask(self.vocab_size)
        mask_count = (
            sum(len(rows) for rows, _, _ in plan.values())
            if metadata
            else len(requests)
        )
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
                branch_rows, draft_start, _ = plan[i]
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
                    draft = drafts[draft_start + pos]
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
        # Own request identities and matcher states across scheduler RPCs and
        # batch reorder/removal. Do not retain the mutable ScheduledBatch.
        records = [(batch.req_ids[i], prefixes[i], self.current[i]) for i in prefixes]
        sampled = sampled.reshape(len(batch.req_ids), -1)
        if sampled.device.type == "cuda":
            import torch

            from .structured_output_transfer import AsyncTokenCopy

            if getattr(self, "commit_copy", None) is None:
                self.commit_copy = AsyncTokenCopy(sampled.device)
            snapshot = torch.cat([sampled[i : i + 1] for i in prefixes])
            self.commit_copy.start(snapshot)
            self.commit_records = records
        else:
            self._commit_ids([sampled[i].tolist() for i in prefixes], records)
        self.current = {}

    def _commit_ids(self, ids, records):
        for tokens, (req_id, (schema, marker, history), state) in zip(ids, records):
            emitted = [token for token in tokens if token >= 0]
            for j, token in enumerate(emitted):
                if token in self.stop_ids:
                    emitted = emitted[: j + 1]
                    break
            full = list(history)
            matcher, active = state
            for token in emitted:
                # An already-finished request can have one forward in flight.
                if active and matcher.is_terminated() and token in self.stop_ids:
                    full.append(token)
                    break
                active = self.advance(matcher, active, marker, full, token)
            self.pending[req_id] = full
            self.states[req_id] = (schema, marker, full, matcher, active)
