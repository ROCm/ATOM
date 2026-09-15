# SPDX-License-Identifier: MIT
"""Event timing and accepted-prefix counts for isolated scheduler benchmarks."""

import statistics

import torch

from atom.utils.forward_context import get_forward_context


class RuntimeStats:
    def __init__(self, runner):
        self.records = []
        self.prefixes = []
        self._target = runner.run_model
        runner.run_model = self.target
        if getattr(runner, "drafter", None) is not None:
            self._draft = runner.drafter.propose
            runner.drafter.propose = self.draft
        self._commit = runner.attn_metadata_builder.commit_speculative_state
        runner.attn_metadata_builder.commit_speculative_state = self.commit

    def timed(self, kind, operation, args, kwargs):
        step = get_forward_context().attn_metadata.step
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        result = operation(*args, **kwargs)
        end.record()
        self.records.append(
            (kind, tuple(span.length for span in step.requests), start, end)
        )
        return result

    def target(self, *args, **kwargs):
        forward = get_forward_context()
        kind = "prefill" if forward.context.is_prefill else "verify"
        return self.timed(kind, self._target, args, kwargs)

    def draft(self, *args, **kwargs):
        return self.timed("draft", self._draft, args, kwargs)

    def commit(self, metadata, last_indices):
        if metadata.step.tentative:
            counts = last_indices - metadata.step.cu_seqlens_q[:-1]
            self.prefixes.append(
                (
                    counts.detach().clone(),
                    [span.length - 1 for span in metadata.step.requests],
                )
            )
        return self._commit(metadata, last_indices)

    def take(self):
        torch.cuda.synchronize()
        result = {}
        for kind in ("prefill", "verify", "draft"):
            rows = [
                (lengths, start.elapsed_time(end))
                for label, lengths, start, end in self.records
                if label == kind
            ]
            result[kind] = {
                "calls": len(rows),
                "total_ms": sum(row[1] for row in rows),
                "median_ms": statistics.median(row[1] for row in rows) if rows else 0,
                "measurements": [
                    {
                        "requests": len(lengths),
                        "target_tokens": sum(lengths),
                        "query_lengths": list(lengths),
                        "milliseconds": elapsed,
                    }
                    for lengths, elapsed in rows
                ],
            }
        counts = [
            int(value) for prefix, _ in self.prefixes for value in prefix.tolist()
        ]
        offered = sum(sum(widths) for _, widths in self.prefixes)
        result["accepted_drafts"] = sum(counts)
        result["verified_drafts"] = offered
        result["acceptance_rate"] = sum(counts) / offered if offered else None
        result["acceptance_histogram"] = {str(i): counts.count(i) for i in range(6)}
        result["count_scope"] = (
            "target-accepted drafts before EOS/stop/output-cap truncation"
        )
        self.records.clear()
        self.prefixes.clear()
        return result
