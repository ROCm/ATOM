# SPDX-License-Identifier: MIT
"""Replay each verify row from a pre-forward STATE/PAGE snapshot."""

import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.utils.forward_context import get_forward_context


class VerifyTrace:
    def __init__(
        self,
        runner,
        inspect_position=None,
        dump_directory=None,
        *,
        replay_batch=False,
        trace_layers=None,
        persistent_shadow=False,
    ):
        self.runner = runner
        self.persistent_shadow = persistent_shadow
        self.shadow = None
        self.replay_batch = replay_batch
        self.trace_layers = (
            trace_layers if trace_layers is not None else (2 if replay_batch else None)
        )
        self.records = []
        self._run = runner.run_model
        self._commit = runner.attn_metadata_builder.commit_speculative_state
        runner.run_model = self.run
        runner.attn_metadata_builder.commit_speculative_state = self.commit
        self.pending = None
        self.inspect_position = inspect_position
        self.layers = None
        if inspect_position is not None:
            from .dspark_layer_trace import LayerTrace

            self.layers = LayerTrace(
                runner.model,
                dump_directory=dump_directory,
                max_layers=self.trace_layers,
                request_projections=not replay_batch,
            )

    @torch.inference_mode()
    def run(self, input_ids, batch):
        forward = get_forward_context()
        metadata = forward.attn_metadata
        step, cache = metadata.step, metadata.cache
        if not step.tentative:
            if self.persistent_shadow and self.shadow is not None:
                raise ValueError("Persistent shadow requires one prefill-only start")
            return self._run(input_ids, batch)
        reused_shadow = self.persistent_shadow and self.shadow is not None
        shadow = (
            self.shadow
            if reused_shadow
            else PagedAttentionCache(
                cache.geometry, cache.num_pages, cache.num_slots, cache.pool.device
            )
        )
        if not reused_shadow:
            shadow.backing.copy_(cache.backing)
        if self.persistent_shadow:
            self.shadow = shadow
        inspect = None
        if self.layers is not None:
            inspect = next(
                (
                    i
                    for i, span in enumerate(step.requests)
                    if span.position <= self.inspect_position < span.end
                ),
                None,
            )
            if inspect is not None:
                self.layers.mode = "block"
                self.layers.row = (
                    step.requests[inspect].offset
                    + self.inspect_position
                    - step.requests[inspect].position
                )
        logits, hidden = self._run(input_ids, batch)
        if self.layers is not None:
            self.layers.mode = None
        record = {
            "requests": [],
            "persistent_shadow": self.persistent_shadow,
            "shadow_reused": reused_shadow,
            "traced_layers": self.trace_layers or len(self.runner.model.layers),
        }
        expected_states = []
        draft_flag = forward.context.is_draft
        forward.context.is_draft = True  # Oracle replay must not overwrite target aux.
        try:
            rows = [[] for _ in step.requests]
            states = [[] for _ in step.requests]
            if self.replay_batch:
                groups = [
                    [
                        (i, span, offset)
                        for i, span in enumerate(step.requests)
                        if offset < span.length
                    ]
                    for offset in range(step.max_length)
                ]
            else:
                groups = [
                    [(i, span, offset)]
                    for i, span in enumerate(step.requests)
                    for offset in range(span.length)
                ]
            for group in groups:
                token_rows = [span.offset + offset for _, span, offset in group]
                if self.layers is not None:
                    self.layers.mode = "serial" if inspect is not None else None
                    self.layers.rows = token_rows
                spans = [
                    RequestSpan(
                        span.request_id,
                        span.position + offset,
                        j,
                        1,
                        span.slot,
                        span.block_ids,
                    )
                    for j, (_, span, offset) in enumerate(group)
                ]
                local = shadow.begin_step(spans)
                shadow.prepare_state(local)
                gather = torch.tensor(
                    token_rows, device=input_ids.device, dtype=torch.long
                )
                embeddings = {
                    layer: values.index_select(1, gather)
                    for layer, values in metadata.engram_embeddings.items()
                }
                values = self.runner.model.forward_hidden(
                    input_ids[gather][None], shadow, local, embeddings
                )
                sequential_logits = self.runner.model.head(
                    self.runner.model.norm(values)
                )[0]
                shadow.finish_step(
                    local,
                    torch.stack(
                        [cache.pending.cursors[i, offset, 1:] for i, _, offset in group]
                    ),
                )
                for j, (i, span, offset) in enumerate(group):
                    full, serial = logits[token_rows[j]], sequential_logits[j]
                    ids = torch.unique(
                        torch.cat((full.topk(4).indices, serial.topk(4).indices))
                    )
                    log_p, log_q = serial.log_softmax(-1), full.log_softmax(-1)
                    rows[i].append(
                        {
                            "position": span.position + offset,
                            "input_id": int(input_ids[token_rows[j]]),
                            "block_top1": int(full.argmax()),
                            "serial_top1": int(serial.argmax()),
                            "max_logit_error": float((full - serial).abs().max()),
                            "kl": float((log_p.exp() * (log_p - log_q)).sum()),
                            "candidate_ids": ids.tolist(),
                            "block_scores": full[ids].tolist(),
                            "serial_scores": serial[ids].tolist(),
                        }
                    )
                    states[i].append(
                        {
                            name: shadow.state.view(name)[:, span.slot].clone()
                            for name in ("tail_values", "tail_scores", "cursor")
                        }
                    )
            record["replay_mode"] = (
                "batched_single_step" if self.replay_batch else "single_request"
            )
            record["replay_request_counts"] = [len(group) for group in groups]
            record["requests"] = [
                {"id": span.request_id, "start": span.position, "rows": rows[i]}
                for i, span in enumerate(step.requests)
            ]
            expected_states = states
        finally:
            forward.context.is_draft = draft_flag
            if self.layers is not None:
                self.layers.mode = None
                self.layers.rows = None
        if inspect is not None:
            record["layer_comparison"] = {
                "request_id": step.requests[inspect].request_id,
                "position": self.inspect_position,
                **self.layers.compare(
                    step.requests[inspect].offset
                    + self.inspect_position
                    - step.requests[inspect].position
                ),
            }
        self.pending = (record, expected_states)
        return logits, hidden

    def commit(self, metadata, last_token_indices):
        self._commit(metadata, last_token_indices)
        if self.pending is None:
            return
        record, states = self.pending
        step, cache = metadata.step, metadata.cache
        lengths = (last_token_indices - step.cu_seqlens_q[:-1] + 1).tolist()
        for i, (span, length) in enumerate(zip(step.requests, lengths)):
            row = record["requests"][i]
            row["accepted_inputs"] = length
            row["committed_state_max_error"] = {
                name: float(
                    (cache.state.view(name)[:, span.slot].float() - expected.float())
                    .abs()
                    .max()
                )
                for name, expected in states[i][length - 1].items()
            }
        if self.persistent_shadow:
            # Replay visited rejected queries too. Restore only the selected
            # prefix's small state; PAGE visibility is cursor-bounded and the
            # physical window slack preserves the entire accepted window.
            for i, (span, length) in enumerate(zip(step.requests, lengths)):
                for name, value in states[i][length - 1].items():
                    self.shadow.state.view(name)[:, span.slot].copy_(value)
            comparisons = [
                compare_visible_cache(
                    cache, self.shadow, span, length, len(self.runner.model.layers)
                )
                for span, length in zip(step.requests, lengths)
            ]
            ranks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ranks, comparisons)
            record["committed_cache_ranks"] = ranks
        self.records.append(record)
        self.pending = None


def compare_visible_cache(cache, shadow, span, length, target_layers):
    """Locate differing visible rows; ignore rejected and uninitialized storage."""
    end = span.position + length
    device = cache.pool.device
    geometry = cache.geometry
    positions = torch.arange(max(0, end - geometry.window_size), end, device=device)
    physical = positions % geometry.ring_slots
    actual = cache.state.view("window")[:target_layers, span.slot, physical]
    expected = shadow.state.view("window")[:target_layers, span.slot, physical]
    differences = (actual != expected).sum(-1).cpu()
    errors = (actual.float() - expected.float()).abs().amax(dim=(-1, -2)).tolist()
    positions = positions.tolist()
    windows = [
        {
            "layer": layer,
            "max_error": errors[layer],
            "positions": [
                position for position, count in zip(positions, counts) if count
            ],
            "unequal": int(counts.sum()),
        }
        for layer, counts in enumerate(differences)
        if counts.any()
    ]
    global_rows = []
    for owner, ratio in geometry.owners:
        ids = torch.arange(end // ratio, device=device)
        blocks = torch.tensor(span.block_ids, device=device)
        for kind in ("main", "index"):
            name = f"{kind}_{owner}"
            pages = cache.pages.view(name)[0]
            rows_per_page = pages.shape[1]
            page_ids, offsets = blocks[ids // rows_per_page], ids % rows_per_page
            actual = pages[page_ids, offsets]
            expected = shadow.pages.view(name)[0, page_ids, offsets]
            counts = (actual != expected).sum(-1)
            if counts.any():
                global_rows.append(
                    {
                        "field": name,
                        "rows": ids[counts != 0].tolist(),
                        "unequal": int(counts.sum()),
                        "max_error": float(
                            (actual.float() - expected.float()).abs().max()
                        ),
                    }
                )
    return {
        "request_id": span.request_id,
        "end": end,
        "window": windows,
        "global": global_rows,
    }
