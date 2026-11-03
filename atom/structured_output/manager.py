# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent per-request grammar state, delta acknowledgements and draft rollback."""

import hashlib
import json
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock

import torch

from atom.model_ops.token_constraints import AllowedTokenMask, TokenConstraint

from . import backend_guidance as grammar
from .utils import AsyncTokenCopy


def structured_stop_token_ids(config, tokenizer=None):
    stops = config.end_token_ids
    if not stops:
        raise ValueError("Structured output requires a configured EOS token")
    return list(stops)


def tokenizer_fingerprint(tokenizer, stop_token_ids):
    """Identify the actual token mapping/decoder and effective termination IDs."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    encoding = getattr(tokenizer, "model", None)
    if backend is not None:
        mapping = json.loads(backend.to_str())
    elif hasattr(encoding, "_mergeable_ranks"):
        # A slow tiktoken tokenizer: its encoding is the token mapping.
        ranks = hashlib.sha256()
        for token, rank in sorted(
            encoding._mergeable_ranks.items(), key=lambda x: x[1]
        ):
            ranks.update(rank.to_bytes(4, "little") + token)
        mapping = {
            "tiktoken": ranks.hexdigest(),
            "special": sorted(encoding._special_tokens.items()),
            "pattern": encoding._pat_str,
        }
    else:
        raise ValueError("Structured output requires a serializable tokenizer")
    payload = {"tokenizer": mapping, "stops": sorted(stop_token_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _compile_worker_grammar(
    runner, schema, tool_format, reasoning_end, fingerprint, vocab_size, request_ids=()
):
    """Compile on the execution rank via RPC, before the request is enqueued.

    The RPC dispatcher catches failures without terminating the worker loop.
    No GPU operation or inter-rank collective is performed here.
    """
    from atom.utils import envs

    if envs.ATOM_ENABLE_RELAXED_MTP:
        raise ValueError("Structured output cannot use relaxed MTP verification")
    structured = getattr(runner, "structured_output", None)
    if structured is None:
        from atom.model_engine.llm_engine import load_tokenizer
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
        tokenizer = load_tokenizer(runner.config.model, runner.config.trust_remote_code)
        stops = structured_stop_token_ids(runner.config, tokenizer)
        worker_fingerprint = tokenizer_fingerprint(tokenizer, stops)
        if worker_fingerprint != fingerprint:
            raise ValueError(
                "Structured output tokenizer/stop tokens differ from the API"
            )
        structured = StructuredOutput(tokenizer, vocab_size, stops, worker_fingerprint)
        runner.structured_output = structured
    if structured.vocab_size != vocab_size or structured.fingerprint != fingerprint:
        raise ValueError("Structured output deployment changed; restart the server")
    compiled = structured.compile(schema, tool_format, reasoning_end)
    # Matchers own their compiled grammar after admission, independently of the
    # bounded compilation cache. Build them before any request can be scheduled.
    result = {"fingerprint": fingerprint, "vocab_size": vocab_size}
    if request_ids:
        result["compiled"] = compiled
    return result


def _prune_compilations(runner):
    # Remove completed records even if an older compilation is still running.
    # Active futures remain owned until completion; cache hits must prune too.
    records = runner.structured_compilations
    for key, future in list(records.items()):
        if len(records) <= 128:
            break
        if future.done():
            records.pop(key)


def prepare_worker_grammar(
    runner, schema, tool_format, reasoning_end, fingerprint, vocab_size, request_ids=()
):
    """Poll background compilation without blocking the worker's forward loop."""
    _prune_compilations(runner)
    structured = getattr(runner, "structured_output", None)
    if (
        structured is not None
        and structured.fingerprint == fingerprint
        and structured.vocab_size == vocab_size
    ):
        compiled = structured.grammars.get((schema, tool_format, reasoning_end))
        if compiled is not None:
            states = {
                req_id: RequestState(structured.backend.GrammarMatcher(compiled))
                for req_id in request_ids
                if req_id not in structured.states
            }
            structured.states.update(states)
            return {"ready": True, "fingerprint": fingerprint, "vocab_size": vocab_size}
    key = (
        schema,
        tool_format,
        reasoning_end,
        fingerprint,
        vocab_size,
        tuple(request_ids),
    )
    if runner.structured_compile_executor is None:
        runner.structured_compile_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="grammar-compile"
        )
    future = runner.structured_compilations.get(key)
    if future is None:
        future = runner.structured_compile_executor.submit(
            _compile_worker_grammar, runner, *key
        )
        runner.structured_compilations[key] = future
    if not future.done():
        return {"ready": False}
    try:
        result = future.result()
    except Exception:
        runner.structured_compilations.pop(key, None)
        raise
    if request_ids:
        structured = runner.structured_output
        compiled = result["compiled"]
        states = {
            req_id: RequestState(structured.backend.GrammarMatcher(compiled))
            for req_id in request_ids
            if req_id not in structured.states
        }
        structured.states.update(states)
    runner.structured_compilations.move_to_end(key)
    _prune_compilations(runner)
    return {"ready": True, "fingerprint": fingerprint, "vocab_size": vocab_size}


@dataclass
class RequestState:
    matcher: grammar.GrammarMatcher
    acknowledged: int = 0
    pending: deque = field(default_factory=deque)

    def acknowledge(self, update, replay):
        if update.start != self.acknowledged:
            raise ValueError("Structured token delta is out of order")
        for token in update.tokens:
            if self.pending:
                if self.pending.popleft() != token:
                    raise ValueError("Structured output differs from committed tokens")
            elif not self.matcher.accept_token(token):
                raise ValueError("Committed output violates the requested grammar")
            self.acknowledged += 1
        if replay and self.pending:
            self.matcher.rollback(len(self.pending))
            self.pending.clear()


class StructuredOutput:
    def __init__(self, tokenizer, vocab_size, stop_token_ids, fingerprint=None):
        self.backend = grammar
        self.stop_ids = list(stop_token_ids)
        info = grammar.grammar_tokenizer(tokenizer, vocab_size, self.stop_ids)
        self.compiler = grammar.GrammarCompiler(info)
        self.fingerprint = fingerprint
        self.vocab_size = vocab_size
        self.grammars = OrderedDict()
        self.compile_lock = Lock()
        self.states = {}
        self.errors = {}
        self.current = {}
        self.draft_plan = None
        self.commit_records = None
        self.draft_copy = None
        self.commit_copy = None
        self.mask_buffer = AllowedTokenMask(vocab_size)
        self.stop_mask = TokenConstraint.allow(self.stop_ids).to_bitmask(vocab_size)

    def compile(self, schema, tool_format=None, reasoning_end=""):
        key = (schema, tool_format, reasoning_end)
        with self.compile_lock:
            compiled = self.grammars.get(key)
            if compiled is None:
                compiled = self.compiler.compile_json_schema(*key)
                self.grammars[key] = compiled
                if len(self.grammars) > 128:
                    self.grammars.popitem(last=False)
            else:
                self.grammars.move_to_end(key)
            return compiled

    def release(self, request_ids):
        for req_id in request_ids:
            self.states.pop(req_id, None)
            self.errors.pop(req_id, None)
            self.current.pop(req_id, None)
        # Keep even an empty record list until the outstanding D2H has finished;
        # its pinned buffer must not be reused while that copy is in flight.
        if self.commit_records is not None:
            removed = set(request_ids)
            self.commit_records = [
                (i, req, state)
                for i, req, state in self.commit_records
                if req not in removed
            ]

    def prepare(self, batch, metadata):
        """Snapshot only this step's constrained drafts before target forward."""
        self.draft_plan = None
        requests = batch.structured_requests
        if not requests or metadata is None:
            return
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
            if count:
                pieces.append(metadata.input_ids[rows[0] + 1 : rows[0] + 1 + count])
            offset += count
        self.draft_plan = plan
        if pieces:
            if self.draft_copy is None:
                self.draft_copy = AsyncTokenCopy(metadata.input_ids.device)
            self.draft_copy.start(torch.cat(pieces))

    def _finish_commit(self):
        if self.commit_records is not None:
            # The next target forward is already submitted. Wait only for the
            # previous step's compact token snapshot, never the current forward.
            ids = self.commit_copy.wait().tolist()
            records, self.commit_records = self.commit_records, None
            self._commit_ids(ids, records)

    def _state(self, req_id, update, replay):
        state = self.states.get(req_id)
        if state is None:
            if update.schema is None:
                raise ValueError("Missing initial structured request configuration")
            compiled = self.grammars.get(update.schema)
            if compiled is None:
                raise ValueError("Structured preparation expired before admission")
            state = RequestState(self.backend.GrammarMatcher(compiled))
            self.states[req_id] = state
        state.acknowledge(update, replay)
        return state

    def _fill_rows(self, matcher, masks, drafts):
        consumed = 0
        try:
            reachable = True
            for pos, mask in enumerate(masks):
                if matcher.is_terminated() or not reachable:
                    mask.copy_(self.stop_mask[0])
                else:
                    matcher.fill_next_token_bitmask(mask.view(1, -1))
                if pos < len(drafts) and reachable:
                    token = drafts[pos]
                    reachable = bool((int(mask[token // 32]) >> (token % 32)) & 1)
                    if reachable and not matcher.is_terminated():
                        if not matcher.accept_token(token):
                            raise ValueError("Draft token disagrees with grammar mask")
                        consumed += 1
        finally:
            # A speculative branch never owns committed state. Roll it back
            # even when a mask fill fails, then terminate only that request.
            if consumed:
                matcher.rollback(consumed)

    def apply(self, logits, batch, metadata):
        if not batch.structured_requests:
            return {}
        self._finish_commit()
        self.current = {}
        plan = self.draft_plan
        drafts = []
        if metadata is not None:
            if plan is None:
                raise ValueError("Structured draft tokens were not prefetched")
            if any(count for _, _, count in plan.values()):
                drafts = self.draft_copy.wait().tolist()
        self.draft_plan = None
        count = (
            sum(len(v[0]) for v in plan.values())
            if metadata
            else len(batch.structured_requests)
        )
        masks = self.mask_buffer.prepare(count, logits.device)
        rows = []
        for i, update in batch.structured_requests.items():
            req_id = batch.req_ids[i]
            final = batch.is_final_chunk is None or batch.is_final_chunk[i]
            branch_rows, start, size = (
                plan.get(i, ([], 0, 0)) if metadata else ([i], 0, 0)
            )
            request_masks = masks[len(rows) : len(rows) + len(branch_rows)]
            try:
                if req_id in self.errors:
                    raise ValueError(self.errors[req_id])
                state = self._state(req_id, update, i < batch.total_seqs_num_prefill)
                if not final:
                    continue
                self._fill_rows(
                    state.matcher, request_masks, drafts[start : start + size]
                )
                self.current[i] = state
            except (ValueError, RuntimeError) as exc:
                self.errors[req_id] = str(exc)
                request_masks.copy_(self.stop_mask.expand_as(request_masks))
            rows.extend(branch_rows)
        if rows:
            self.mask_buffer.apply(logits, rows)
        return dict(self.current)

    def commit(self, sampled, batch, prefixes):
        if not prefixes:
            return
        records = [
            (j, batch.req_ids[i], state)
            for j, (i, state) in enumerate(prefixes.items())
        ]
        sampled = sampled.reshape(len(batch.req_ids), -1)
        if sampled.device.type == "cuda":
            if self.commit_copy is None:
                self.commit_copy = AsyncTokenCopy(sampled.device)
            self.commit_copy.start(torch.cat([sampled[i : i + 1] for i in prefixes]))
            self.commit_records = records
        else:
            self._commit_ids([sampled[i].tolist() for i in prefixes], records)
        self.current = {}

    def _commit_ids(self, ids, records):
        for index, req_id, state in records:
            if req_id in self.errors:
                continue
            try:
                for token in ids[index]:
                    if token < 0:
                        continue
                    if state.matcher.is_terminated() and token in self.stop_ids:
                        break
                    if not state.matcher.accept_token(token):
                        raise ValueError(
                            "Committed output violates the requested grammar"
                        )
                    state.pending.append(token)
                    if token in self.stop_ids:
                        break
            except (ValueError, RuntimeError) as exc:
                self.errors[req_id] = str(exc)
