# SPDX-License-Identifier: MIT
"""Audit real scheduler state against raw tokens, without changing model math.

The oracle owns a raw-token ledger. It derives Engram history by slicing that
ledger, gathers embeddings without the prefetch/staging cache, and checks greedy
acceptance directly against target argmax. Scheduler outputs are checked only
once finalized, including terminal EOS/stop/cap truncation.
"""

from collections import Counter

import numpy as np
import torch

from atom.utils.forward_context import get_forward_context


class RuntimeStateAudit:
    def __init__(self, runner):
        self.runner = runner
        self.builder = runner.attn_metadata_builder
        self.preparer = self.builder.engram
        if self.preparer is None:
            raise ValueError("State audit requires checkpoint Engram tables")
        self._prepare = self.preparer.prepare
        self._run = runner.run_model
        self._commit = self.builder.commit_speculative_state
        self.preparer.prepare = self.prepare
        runner.run_model = self.run
        self.builder.commit_speculative_state = self.commit
        self.records = []
        self.sequences = {}
        self.pending = None

    def bind(self, sequences):
        if self.sequences or self.pending is not None:
            raise RuntimeError("Previous audit group was not finalized")
        self.sequences = {seq.id: seq for seq in sequences}
        self.ledger = {seq.id: [] for seq in sequences}
        self.samples = {seq.id: {} for seq in sequences}
        self.checked = {seq.id: seq.num_prompt_tokens for seq in sequences}
        self.stats = Counter()
        self.accepted = Counter()
        self.batch_sizes = Counter()

    def history(self, tokens):
        width = self.preparer.mapping.config.max_ngram_size - 1
        tail = self.preparer.mapping.compress_tokens(
            np.asarray([tokens[-width:]], dtype=np.int64)
        )[0].tolist()
        return [-1] * (width - len(tail)) + tail

    def prepare(self, spans, token_ids, histories, *, dummy=False, token_mask=None):
        if dummy:
            return self._prepare(
                spans, token_ids, histories, dummy=True, token_mask=token_mask
            )
        if self.pending is not None:
            raise RuntimeError("Uncommitted audit inputs")
        if token_mask is not None and not np.asarray(token_mask).all():
            raise ValueError("Raw-token state audit supports text requests only")
        ids = token_ids.detach().cpu().tolist()
        inputs = []
        for span, actual_history in zip(spans, histories):
            seq = self.sequences[span.request_id]
            prefix = self.ledger[span.request_id]
            # A cached prompt may enter after zero; its raw IDs are authoritative.
            if span.position > len(prefix):
                if span.position > seq.num_prompt_tokens:
                    raise AssertionError("State advanced past observed input tokens")
                prefix = list(seq.prompt_token_ids[: span.position])
            prefix = prefix[: span.position]
            expected = self.history(prefix)
            if list(actual_history) != expected:
                raise AssertionError(
                    f"Request {span.request_id} history at {span.position}: "
                    f"{list(actual_history)} != raw-prefix {expected}"
                )
            tokens = ids[span.token_slice]
            full = prefix + tokens
            common = min(len(full), seq.num_prompt_tokens)
            if full[:common] != list(seq.prompt_token_ids[:common]):
                raise AssertionError("Forwarded prompt IDs disagree with request")
            inputs.append((span, prefix, tokens))
            self.stats["incoming_histories"] += 1
        prepared = self._prepare(
            spans, token_ids, histories, dummy=False, token_mask=token_mask
        )
        mapping = self.preparer.mapping
        width = mapping.config.max_ngram_size - 1
        for i, (span, prefix, tokens) in enumerate(inputs):
            full = prefix + tokens
            if prepared.histories[i].tolist() != self.history(full):
                raise AssertionError("Prepared history disagrees with raw prefix")
            # Hash raw lookback+queries with no supplied history. This avoids
            # both the runtime history and cached immutable lookup snapshots.
            lookback = prefix[-width:]
            raw = np.asarray([lookback + tokens], dtype=np.int64)
            expected_compressed = mapping.compress_tokens(raw)[0, len(lookback) :]
            if not np.array_equal(prepared.compressed_rows[i], expected_compressed):
                raise AssertionError("Prepared compressed token rows disagree")
            for layer, values in prepared.embeddings.items():
                hashes = mapping.hash_layer(raw, layer)[:, len(lookback) :]
                rows = mapping.to_row_indices(hashes, layer)[0]
                expected = self.preparer.host.prefetcher._tables[layer].gather(rows)
                expected = expected.reshape(span.length, -1).to(values.dtype)
                actual = values[0, span.token_slice].detach().cpu()
                if not torch.equal(actual, expected):
                    raise AssertionError(
                        f"Request {span.request_id} staged Engram layer {layer} differs"
                    )
                self.stats["embedding_rows"] += span.length
            self.stats["prepared_histories"] += 1
        self.pending = inputs
        self.batch_sizes[len(spans)] += 1
        return prepared

    def check_commit(self, metadata, lengths):
        if self.pending is None:
            raise AssertionError("State commit has no audited inputs")
        for (span, prefix, tokens), length in zip(self.pending, lengths):
            full = prefix + tokens[:length]
            expected = [len(full)] + self.history(full)
            actual = metadata.cache.cursor[span.slot].cpu().tolist()
            if actual != expected:
                raise AssertionError(
                    f"Request {span.request_id} committed cursor/history "
                    f"{actual} != raw-prefix {expected}"
                )
            self.ledger[span.request_id] = full
            self.stats["committed_histories"] += 1
        self.pending = None

    def save_samples(self, span, values):
        seq = self.sequences[span.request_id]
        for position, token in values:
            if position >= seq.num_prompt_tokens:
                self.samples[span.request_id][position] = token

    def run(self, input_ids, batch):
        logits, hidden = self._run(input_ids, batch)
        metadata = get_forward_context().attn_metadata
        if metadata.dummy:
            return logits, hidden
        step = metadata.step
        if logits is None:
            if step.tentative:
                raise AssertionError("Verification forward returned no logits")
            self.check_commit(metadata, [span.length for span in step.requests])
            self.stats["middle_prefill_forwards"] += 1
            return logits, hidden
        self.top1 = logits.argmax(-1).cpu().tolist()
        if not step.tentative:
            prefill = get_forward_context().context.is_prefill
            for i, span in enumerate(step.requests):
                row = i if prefill else span.offset
                self.save_samples(span, [(span.end, self.top1[row])])
            self.check_commit(metadata, [span.length for span in step.requests])
            self.stats["ordinary_forwards"] += 1
        else:
            self.stats["verify_forwards"] += 1
        return logits, hidden

    def commit(self, metadata, last_token_indices):
        step = metadata.step
        if not step.tentative:
            return self._commit(metadata, last_token_indices)
        lengths = (last_token_indices - step.cu_seqlens_q[:-1] + 1).tolist()
        for (span, _, tokens), length in zip(self.pending, lengths):
            scores = self.top1[span.offset : span.offset + span.length]
            expected = next(
                (i + 1 for i in range(span.length - 1) if tokens[i + 1] != scores[i]),
                span.length,
            )
            if length != expected:
                raise AssertionError(
                    f"Request {span.request_id} accepted {length} inputs; "
                    f"target argmax requires {expected}"
                )
            outputs = tokens[1:length] + [scores[length - 1]]
            self.save_samples(
                span,
                [(span.position + i + 1, token) for i, token in enumerate(outputs)],
            )
            self.accepted[length - 1] += 1
            self.stats["verified_inputs"] += span.length
        self._commit(metadata, last_token_indices)
        self.check_commit(metadata, lengths)

    def observe(self):
        for request_id, seq in self.sequences.items():
            end = seq.num_finalized_tokens
            finalized = list(seq.token_ids[:end])
            prefix = self.ledger[request_id]
            common = min(len(prefix), end)
            if prefix[:common] != finalized[:common]:
                raise AssertionError("Committed input IDs differ from finalized output")
            for position in range(self.checked[request_id], end):
                expected = self.samples[request_id].get(position)
                if finalized[position] != expected:
                    raise AssertionError(
                        f"Request {request_id} finalized token {position}: "
                        f"{finalized[position]} != target sampler {expected}"
                    )
                self.stats["finalized_generated_tokens"] += 1
            self.checked[request_id] = end

    def finish(self):
        self.observe()
        if self.pending is not None or not all(
            seq.is_finished for seq in self.sequences.values()
        ):
            raise AssertionError("Audit group ended with unfinished request state")
        self.records.append(
            {
                "checks": dict(self.stats),
                "accepted_draft_counts": dict(self.accepted),
                "batch_sizes": dict(self.batch_sizes),
                "requests": [
                    {
                        "id": seq.id,
                        "prompt_tokens": seq.num_prompt_tokens,
                        "output_tokens": seq.num_finalized_tokens
                        - seq.num_prompt_tokens,
                        "leave_reason": str(seq.leave_reason),
                    }
                    for seq in self.sequences.values()
                ],
            }
        )
        self.sequences = {}

    def close(self):
        self.preparer.prepare = self._prepare
        self.runner.run_model = self._run
        self.builder.commit_speculative_state = self._commit
