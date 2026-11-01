# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Compiled V4 variants and shared scratch/peer arenas; no production cache bindings."""

from __future__ import annotations

from typing import ClassVar

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.deepseek_v4.mono.build import _build_launches, kernel_kwargs
from atom.models.deepseek_v4.mono.config import (
    LAUNCH_PHASES,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.kernels.plan import layout, stage_tasks
from atom.mono.runtime.consensus import bind_agreed

__all__ = ["Dsv4Variant"]


class Dsv4Variant:
    """The compiled kernel, scratch and symmetric buffer shared by all layers of one attention variant.

    Scratch cannot be shared across variants (different mailbox layouts). Within one, fresh epoch
    tags make sharing safe, so ``step`` lives here: one counter per scratch, advanced once per step.
    """

    _cache: ClassVar[dict] = {}
    _adv_cache: ClassVar[dict] = {}

    def __init__(
        self,
        cfg,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
        poll_timeout_us: int | None = None,
        tokens_per_seq: int = 1,
        launch_shape: str = "one",
        qkv_batch_limit: int | None = None,
        qkv_samples_per_pass: int | None = None,
        qkv_norm_handoff: bool = False,
        attention_mode: str = "causal",
    ):
        moe_mode = as_moe_mode(moe_mode)
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if tokens_per_seq <= 0 or samples % tokens_per_seq:
            raise ValueError(
                f"{samples} samples are not whole runs of {tokens_per_seq} tokens per sequence"
            )
        self.poll_timeout_us = poll_timeout_us
        self.tokens_per_seq = tokens_per_seq
        if launch_shape not in LAUNCH_PHASES:
            raise ValueError(f"unknown launch_shape {launch_shape!r}")
        if attention_mode not in ("causal", "draft_block"):
            raise ValueError(f"unknown attention_mode {attention_mode!r}")
        if attention_mode == "draft_block":
            if cfg.compress_ratio or cfg.indexed:
                raise ValueError("draft block variant requires uncompressed stages")
            if launch_shape != "three":
                raise ValueError("draft block variant requires the pre/core/moe chain")
        if samples == 14 and (tokens_per_seq != 7 or launch_shape != "three"):
            raise ValueError("candidate S14 requires q7 and the three-phase contract")
        if launch_shape == "one" and (
            qkv_batch_limit is not None
            or qkv_samples_per_pass is not None
            or qkv_norm_handoff
        ):
            raise ValueError("the legacy ONE launch does not take split-QKV tuning")
        # Refuse invalid launch contracts before importing GPU/IPC backends.
        # Host-only admission checks must not require a visible device.
        from atom.models.deepseek_v4.mono.kernels.layer import (
            build_advance_step,
        )
        from atom.models.deepseek_v4.mono.runtime import EpochPeerBuffer

        self.launch_shape = launch_shape
        self.qkv_batch_limit = qkv_batch_limit
        self.qkv_samples_per_pass = qkv_samples_per_pass
        self.qkv_norm_handoff = qkv_norm_handoff
        self.attention_mode = attention_mode
        self.key = _variant_key(
            cfg,
            samples,
            npes,
            moe_mode,
            timeline,
            poll_timeout_us,
            tokens_per_seq,
            launch_shape,
            qkv_batch_limit,
            qkv_samples_per_pass,
            qkv_norm_handoff,
            attention_mode,
        )
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(
            samples,
            cfg.heads,
            npes,
            cfg.window,
            moe_mode,
            qkv_norm_handoff=qkv_norm_handoff,
            **dims,
        )
        refresh_modes = (True, False) if cfg.indexed else (True,)
        self.launch_sets = {}
        for refresh in refresh_modes:
            cache_key = (self.key, refresh)
            built = Dsv4Variant._cache.get(cache_key)
            if built is None:
                kwargs = kernel_kwargs(
                    cfg,
                    tokens_per_seq=tokens_per_seq,
                    timeline=timeline,
                    moe_mode=moe_mode,
                    index_topk_refresh=refresh,
                    qkv_batch_limit=qkv_batch_limit,
                    qkv_samples_per_pass=qkv_samples_per_pass,
                    qkv_norm_handoff=qkv_norm_handoff,
                    attention_mode=attention_mode,
                )
                built = _build_launches(
                    cfg,
                    samples,
                    npes,
                    launch_shape=launch_shape,
                    poll_timeout_us=poll_timeout_us,
                    **kwargs,
                )
                Dsv4Variant._cache[cache_key] = built
            elif not isinstance(built, tuple):
                built = (built,)
            self.launch_sets[refresh] = built
        self.launches = self.launch_sets[
            cfg.index_topk_refresh if cfg.indexed else True
        ]
        # the step advance clears mailbox pairs; the plain-f32 "xqd" region sits last and is skipped
        self.scr_pairs = self.scr_layout["xqd"] // 8
        self.sym_pairs = self.sym_layout["_bytes"] // 8
        adv_key = (self.scr_pairs, self.sym_pairs)
        if adv_key not in Dsv4Variant._adv_cache:
            Dsv4Variant._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Dsv4Variant._adv_cache[adv_key]
        self.stages = stage_tasks(
            samples,
            cfg.heads,
            window=cfg.window,
            qkv_norm_handoff=qkv_norm_handoff,
            index_topk_refresh=cfg.index_topk_refresh,
            **dims,
        )
        import torch.distributed as dist

        group = dist.group.WORLD if npes > 1 and group is None else group

        def allocate():
            self.scratch = torch.zeros(
                self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
            )
            self.step = torch.zeros(1, dtype=torch.int32, device=dev)
            self.hang = torch.zeros(1, dtype=torch.int32, device=dev)

        bind_agreed(allocate, group)
        self.peer_buffer = EpochPeerBuffer(
            self.sym_layout["_bytes"], rank=rank, npes=npes, group=group
        )
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs so tags can wrap; graph-capturable."""
        _run_compiled(
            self._advance,
            self.step.data_ptr(),
            self.scratch.data_ptr(),
            self.sym,
            torch.cuda.current_stream(),
        )

    def close(self):
        self.peer_buffer.close()
