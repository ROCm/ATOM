# SPDX-License-Identifier: MIT
"""Draft metadata publication from actual stage-private native pools.

No engine installation, graph publication, lease acquisition or context-write
completion is inferred here. The caller must retire context producers first.
"""

from dataclasses import fields
from itertools import pairwise

import torch

from atom.models.deepseek_v4.mono.binding import LayerStep, input_identity
from atom.models.deepseek_v4.mono.draft_plan import plan_draft_block
from atom.mono.runtime.consensus import MonoUnsupported


def _publication_binding(model, configs, steps):
    """Static owners/layout plus immutable RoPE contents, not live KV values."""

    def rope_binding(tensor):
        # Inference tensors have no version counter. Only an already-FP32
        # contiguous table can safely share that unversioned storage directly.
        try:
            version = tensor._version
        except RuntimeError:
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise MonoUnsupported(
                    "unversioned draft RoPE requires direct FP32 storage"
                )
            version = None
        return id(tensor), input_identity(tensor), version

    return (
        tuple(
            tuple((f.name, getattr(cfg, f.name)) for f in fields(cfg))
            for cfg in configs
        ),
        tuple(
            (
                id(stage),
                id(stage.attn),
                stage.window_size,
                stage.attn.swa_window,
                input_identity((stage.attn.swa_plane, stage.attn.swa_plane_rope)),
                rope_binding(stage.attn.rotary_emb.cos_cache),
                rope_binding(stage.attn.rotary_emb.sin_cache),
            )
            for stage in model.mtp
        ),
        input_identity(tuple(vars(step) for step in steps)),
        tuple((rope_binding(step.cos), rope_binding(step.sin)) for step in steps),
    )


class PreparedDraftPublication:
    """Session-owned static publication binding; not a replay credential.

    Preparation retains converted RoPE tables once. Publishing still validates
    every live address and the full write/alias set before copying integers.
    The session must fence old readers, check its frontier and retire on error.
    """

    def __init__(self, model, configs, steps):
        self.model, self.configs, self.steps = model, tuple(configs), tuple(steps)
        for stage, step in zip(model.mtp, self.steps):
            for name in ("cos", "sin"):
                source = getattr(stage.attn.rotary_emb, f"{name}_cache")
                try:
                    _ = source._version
                except RuntimeError:
                    table = source.reshape(source.shape[0], -1)
                    if input_identity(table) != input_identity(getattr(step, name)):
                        raise MonoUnsupported(
                            "unversioned draft RoPE cannot retain a converted copy"
                        )
        self.binding = _publication_binding(model, self.configs, self.steps)

    def publish(self, anchors, slots):
        if _publication_binding(self.model, self.configs, self.steps) != self.binding:
            raise MonoUnsupported("prepared draft publication binding changed")
        return native_draft_steps(
            self.model, self.configs, anchors, slots, out=self.steps, _prepared=True
        )


def draft_write_views(steps, configs):
    """Bounded native FP8 draft write set for startup save/restore.

    BF16 draft KV stays mailbox-only. FP8 writes each destination into both
    native planes; never snapshot the entire shared multi-layer allocation.
    Call only after full-stack preflight, before the first write.
    """
    steps, configs = tuple(steps), tuple(configs)
    if len(steps) != len(configs):
        raise MonoUnsupported("one draft write-set config per stage required")
    views = []
    for step, cfg in zip(steps, configs):
        if not cfg.kv_fp8:
            continue
        rows = sorted({int(row) for row in step.dest_rows[0].tolist()})
        planes = step.kv_cache
        if not isinstance(planes, tuple) or len(planes) != 2:
            raise MonoUnsupported("draft FP8 write set requires two planes")
        for plane in planes:
            if any(row < 0 or row >= plane.shape[0] for row in rows):
                raise MonoUnsupported("draft write set exceeds native allocation")
            for row in rows:
                views.append(plane[row : row + 1])
    return tuple(views)


def native_draft_steps(model, configs, anchors, slots, *, out=None, _prepared=False):
    """Build a whole stack before the caller can run the first cache write.

    Host anchors/slots are explicit snapshots of the current frontier, never
    recovered from a stale speculative ell. Only live, unpadded requests are
    represented. Stages share row numbers but must not share writable bytes.
    Optional ``out`` retains exact-shape metadata addresses after checking the
    entire stack. Publish outside capture, after retiring previous readers;
    this is not graph replay authorization or a padding/lifecycle contract.
    """
    configs, stages = tuple(configs), tuple(model.mtp)
    out = None if out is None else tuple(out)
    if out is not None and len(out) != len(stages):
        raise MonoUnsupported("one static draft metadata destination per stage")
    if _prepared and out is None:
        raise MonoUnsupported("prepared draft publication requires fixed storage")
    anchors, slots = tuple(anchors), tuple(slots)
    if not stages or len(configs) != len(stages):
        raise MonoUnsupported("one draft cache config per stage required")
    steps, plans, byte_ranges = [], [], []
    staged_metadata = None
    call_plans = {}
    for ordinal, (stage, cfg) in enumerate(zip(stages, configs)):
        if cfg.compress_ratio or cfg.draft_block_width <= 0:
            raise MonoUnsupported(
                "native draft cache requires uncompressed block config"
            )
        attn = stage.attn
        if stage.window_size != cfg.window:
            raise MonoUnsupported("native draft history differs from compiled window")
        nope, rope = (
            getattr(attn, "swa_plane", None),
            getattr(attn, "swa_plane_rope", None),
        )
        if not isinstance(nope, torch.Tensor):
            raise MonoUnsupported("native draft cache plane not bound")
        # Native pools use a one-byte FP8 dtype for packed data and inline
        # scales. The mono kernel consumes those same bytes, not an FP8 cast.
        if cfg.kv_fp8 and nope.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            nope = nope.view(torch.uint8)
        if (
            nope.ndim != 2
            or nope.shape[1] != (512 if cfg.kv_fp8 else cfg.head_dim)
            or nope.dtype != (torch.uint8 if cfg.kv_fp8 else torch.bfloat16)
            or not nope.is_contiguous()
        ):
            raise MonoUnsupported("native draft cache plane layout mismatch")
        if cfg.kv_fp8:
            if (
                not isinstance(rope, torch.Tensor)
                or rope.shape != (nope.shape[0], cfg.rope_dim)
                or rope.dtype != torch.bfloat16
                or not rope.is_contiguous()
                or nope.device != rope.device
            ):
                raise MonoUnsupported("native draft two-buffer cache layout mismatch")
            cache, planes = (nope, rope), (nope, rope)
        else:
            if rope is not None:
                raise MonoUnsupported("native BF16 draft must use a single cache plane")
            if not cfg.bf16_kv_roundtrip or cfg.quant_query_fp8:
                raise MonoUnsupported(
                    "native BF16 draft requires KV QAT and unquantized Q"
                )
            cache, planes = nope, (nope,)
        plan_key = (
            attn.swa_window,
            cfg.draft_block_width,
            stage.window_size,
            nope.shape[0],
        )
        plan = call_plans.get(plan_key)
        if plan is None:
            plan = plan_draft_block(
                anchors,
                slots,
                attn.swa_window,
                width=cfg.draft_block_width,
                history=stage.window_size,
                pool_rows=nope.shape[0],
            )
            # Only equal address formulas and capacities share this call's
            # integer plan. Stage byte ranges still use their own plane bases.
            call_plans[plan_key] = plan
        if plans and plan != plans[0]:
            raise MonoUnsupported("draft stage window address contracts differ")
        plans.append(plan)
        # Native per-layer views deliberately extend to the same plane end.
        # Compare addressed bytes (reads as well as writes), not storage/view
        # overlap: interleaved layer windows must remain independent even when
        # a draft write would alias another stage's history rather than output.
        rows = sorted({row for keys in plan.indices for row in keys if row >= 0})
        for plane in planes:
            row_bytes = plane.shape[1] * plane.element_size()
            base = plane.data_ptr()
            byte_ranges.extend(
                (base + row * row_bytes, base + (row + 1) * row_bytes) for row in rows
            )
        device = nope.device
        # Host staging is also the dynamic shape reference for fixed outputs.
        # No temporary GPU metadata allocation is needed before publication.
        if out is not None and staged_metadata is not None:
            # Stage plans were compared above in this invocation. Share only
            # immutable CPU copy sources; each stage keeps private GPU outputs
            # and still passes its own full layout/address/alias checks.
            pos, dest, indices = staged_metadata
        else:
            pos = torch.tensor(plan.positions, dtype=torch.int32)
            dest = torch.tensor(
                (plan.dest_rows, (0,) * len(plan.positions)), dtype=torch.int32
            )
            indices = torch.tensor(plan.indices, dtype=torch.int32)
            if out is not None:
                staged_metadata = pos, dest, indices
        if indices.shape[1] != cfg.n_keys:
            raise MonoUnsupported(
                "native draft key capacity differs from compiled config"
            )
        tables = []
        for name, source in zip(
            ("cos", "sin"), (attn.rotary_emb.cos_cache, attn.rotary_emb.sin_cache)
        ):
            if (
                source.device != device
                or source.ndim < 2
                or source.shape[-1] != cfg.rope_dim // 2
            ):
                raise MonoUnsupported("native draft RoPE layout mismatch")
            table = (
                getattr(out[ordinal], name)
                if _prepared
                else source.reshape(source.shape[0], -1).float().contiguous()
            )
            if (
                table.shape[1] != cfg.rope_dim // 2
                or max(plan.positions) >= table.shape[0]
            ):
                raise MonoUnsupported("native draft position exceeds RoPE table")
            tables.append(table)
        if out is None:
            pos, dest, indices = (tensor.to(device) for tensor in (pos, dest, indices))
        steps.append(LayerStep(pos, cache, dest, indices, *tables, state={}))
    ordered = sorted(byte_ranges)
    if any(left[1] > right[0] for left, right in pairwise(ordered)):
        raise MonoUnsupported("draft stages cannot share addressed cache bytes")
    if out is not None:
        destinations = []
        for target, source in zip(out, steps):
            old_cache = target.kv_cache
            new_cache = source.kv_cache
            old_planes = old_cache if isinstance(old_cache, tuple) else (old_cache,)
            new_planes = new_cache if isinstance(new_cache, tuple) else (new_cache,)
            if input_identity(old_planes) != input_identity(new_planes):
                raise MonoUnsupported("static draft cache binding changed")
            if target.block_tables is not None or target.state:
                raise MonoUnsupported("static draft requires native window metadata")
            for name in ("positions", "dest_rows", "indices", "cos", "sin"):
                old, new = getattr(target, name), getattr(source, name)
                if (
                    old.shape != new.shape
                    or old.dtype != new.dtype
                    or old.device != source.cos.device
                    or not old.is_contiguous()
                ):
                    raise MonoUnsupported("static draft metadata layout changed")
                destinations.append((old, new))
        writable = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            for tensor, _ in destinations
        )
        if any(left[1] > right[0] for left, right in pairwise(writable)):
            raise MonoUnsupported("static draft metadata destinations alias")
        cache_ranges = [
            (plane.data_ptr(), plane.data_ptr() + plane.numel() * plane.element_size())
            for step in steps
            for plane in (
                step.kv_cache if isinstance(step.kv_cache, tuple) else (step.kv_cache,)
            )
        ]
        if any(a < d and c < b for a, b in writable for c, d in cache_ranges):
            raise MonoUnsupported("static draft metadata aliases native cache")
        # No partial publication for a rejected later stage. A device copy
        # failure still requires retiring the owning session, not retrying.
        for target, source in destinations:
            if target is not source:
                target.copy_(source)
        return out
    return tuple(steps)
