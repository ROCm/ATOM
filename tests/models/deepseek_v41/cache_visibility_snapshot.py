# SPDX-License-Identifier: MIT
"""Hash what a reader at the committed cursor is allowed to see, per step.

Every cache comparison so far has been DSpark against DSpark -- block forward
against serial shadow -- which is blind to anything both arms read alike. This
snapshots the cache in a form that can be compared against a *baseline* run.

Two things make that comparison possible even though the arms do not share a
layout. `ring_slots` is `window_size + speculative_tokens`, so baseline rings are
128 rows and a width-5 DSpark ring is 133: the same logical position lands on a
different physical row. And the rows past the cursor differ by design, because
baseline never writes a draft there. So this gathers by *logical position* and
covers only the visible prefix -- the window positions a reader at `p` may
attend, and the compressed rows below `(p+1)//ratio`. A difference here is a
reader seeing something baseline never wrote, which is the thing in question.

Enable with ATOM_DSPARK_CACHE_SNAPSHOT=<path>; optionally restrict cost with
ATOM_DSPARK_SNAPSHOT_LAYERS=7,8 and ATOM_DSPARK_SNAPSHOT_STEPS=40. Off by
default and never imported by production code paths.
"""

import base64
import hashlib
import json
import os
import pathlib

import torch


def _digest(tensor):
    if tensor.numel() == 0:
        return "empty"
    raw = tensor.detach().contiguous().cpu()
    return hashlib.blake2b(
        raw.view(torch.uint8).numpy().tobytes(), digest_size=8
    ).hexdigest()


class IndexProbe:
    """What the indexer was offered and what it chose, per attention layer.

    `_update_global` sizes the candidate tile with the whole verify block --
    `(step.position + step.length) // ratio` -- so a width-5 DSpark step offers
    two more compressed rows than baseline does at the same committed position,
    and those two are written from draft tokens by `write_global`, which has no
    rollback. `index_topk` is far larger than the number of candidates this
    early, so `count = min(topk, width)` keeps everything it is offered and the
    only thing standing between a draft-derived row and the attention is the
    per-token bound passed as `visible_lengths`. This records both.

    Call order within a forward is layer order, so the ordinal names the layer.
    """

    def __init__(self, handle):
        self.handle, self.call, self.position, self.recording = handle, 0, None, False

    def begin(self, position):
        self.call, self.position, self.recording = 0, position, True

    def observe(self, keys, visible_lengths, selected):
        if not self.recording:
            return
        chosen = selected[0, 0]
        self.handle.write(
            json.dumps(
                {
                    "position": self.position,
                    "call": self.call,
                    "offered": int(keys.shape[1]),
                    "visible_lengths": visible_lengths.tolist(),
                    "kept": int((chosen >= 0).sum()),
                    "max_id": int(chosen.max()) if chosen.numel() else -1,
                    "ids": chosen[chosen >= 0].tolist(),
                }
            )
            + "\n"
        )
        self.call += 1


def install_index_probe(handle):
    """Wrap the indexer entry point in place; the model is left untouched."""
    from atom.models.deepseek_v41 import attention

    probe = IndexProbe(handle)
    original = attention.select_indices

    def traced(q, weights, keys, visible_lengths, **kwargs):
        result = original(q, weights, keys, visible_lengths, **kwargs)
        probe.observe(keys, visible_lengths, result[0])
        return result

    attention.select_indices = traced
    return probe


class CacheVisibilitySnapshot:
    """One JSONL row per (request, committed position). Requires an unpacked cache."""

    def __init__(self, path, layers=None, max_steps=None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.layers, self.max_steps, self.written = layers, max_steps, 0
        self.handle = self.path.open("w")
        self.index_handle = self.path.with_name(
            self.path.name.replace("visibility", "indexer")
        ).open("w")
        self.probe = install_index_probe(self.index_handle)

    @classmethod
    def from_env(cls, rank):
        root = os.environ.get("ATOM_DSPARK_CACHE_SNAPSHOT")
        if not root:
            return None
        chosen = os.environ.get("ATOM_DSPARK_SNAPSHOT_LAYERS")
        steps = os.environ.get("ATOM_DSPARK_SNAPSHOT_STEPS")
        return cls(
            pathlib.Path(root) / f"visibility_rank{rank}.jsonl",
            layers=[int(v) for v in chosen.split(",")] if chosen else None,
            max_steps=int(steps) if steps else None,
        )

    def _window_prefix(self, cache, layer, slot, position):
        """The rows a reader at `position` may attend, oldest first."""
        geometry = cache.geometry
        # Matches `_indices`: `first = max(0, pos - WINDOW + 1)` read up to,
        # but not including, the block's own first position.
        first = max(0, position - geometry.window_size + 1)
        if first >= position:
            return torch.empty(0)
        rows = torch.arange(first, position, device=cache.pool.device)
        ring = cache.state.view("window")[layer, slot]
        return ring[rows % geometry.ring_slots]

    def _global_prefix(self, cache, kind, owner, ratio, blocks, position):
        count = position // ratio
        if count <= 0:
            return torch.empty(0)
        pages = (
            cache.index_planes[owner]
            if kind == "index"
            else cache.pages.view(f"main_{owner}")[0]
        )
        per_page = pages.shape[1]
        ids = torch.arange(count, device=pages.device)
        return pages[blocks[ids // per_page].long(), ids % per_page]

    def capture(self, cache, step, block_tables):
        if cache.packed or (
            self.max_steps is not None and self.written >= self.max_steps
        ):
            self.probe.recording = False
            return
        geometry = cache.geometry
        # The forward that follows is the one the probe should attribute.
        self.probe.begin(step.requests[0].position if step.requests else None)
        layers = self.layers if self.layers is not None else range(geometry.layers)
        for i, span in enumerate(step.requests):
            position, slot = span.position, span.slot
            row = {
                "request_id": span.request_id,
                "position": position,
                "ring_slots": geometry.ring_slots,
                "cursor": cache.cursor[slot].tolist(),
                "window": {
                    str(layer): _digest(
                        self._window_prefix(cache, layer, slot, position)
                    )
                    for layer in layers
                },
                "main": {},
                "index": {},
                "tails": {},
            }
            for owner, ratio in geometry.owners:
                for kind in ("main", "index"):
                    row[kind][str(owner)] = _digest(
                        self._global_prefix(
                            cache, kind, owner, ratio, block_tables[i], position
                        )
                    )
            for name in ("tail_values", "tail_scores"):
                row["tails"][name] = _digest(cache.state.view(name)[:, slot])
            # The row this step's predecessor just wrote, verbatim. A digest
            # says the arms differ; only the values say whether that is one
            # bf16 ULP or something a reduction order cannot produce.
            if position:
                ring = cache.state.view("window")
                row["last_row"] = {
                    str(layer): base64.b64encode(
                        ring[layer, slot, (position - 1) % geometry.ring_slots]
                        .detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).decode()
                    for layer in layers
                }
            self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()
        self.written += 1

    def close(self):
        self.handle.close()
