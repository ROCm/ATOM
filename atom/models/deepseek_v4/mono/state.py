# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Compiled V4 variants and shared scratch/peer arenas; no production cache bindings."""

from __future__ import annotations

from typing import ClassVar

import torch

from atom.models.deepseek_v4.mono.config import (
    BLOCK_TOKENS,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.kernels.plan import layout, stage_tasks
from atom.mono.runtime.consensus import bind_agreed

__all__ = ["Dsv4Variant", "fp4_pool_shapes"]


def fp4_pool_shapes(cfg, samples: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Codes/scales of the gfx950 paged FP4 index pool, in runs of blocks."""
    k_pb = BLOCK_TOKENS // cfg.compress_ratio
    nb = -(-cfg.n_compressed // k_pb)
    ng = cfg.index_head_dim // 32
    return (samples, nb, ng * k_pb * 16), (samples, nb, ng * k_pb)


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
        finite_observer_layer: int = -1,
    ):
        from atom.models.deepseek_v4.mono.kernels.layer import (
            build_advance_step,
            build_dsv4_kernel,
        )
        from atom.mono.runtime.peer_memory import SymmetricPeerBuffer

        moe_mode = as_moe_mode(moe_mode)
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if tokens_per_seq <= 0 or samples % tokens_per_seq:
            raise ValueError(
                f"{samples} samples are not whole runs of {tokens_per_seq} tokens per sequence"
            )
        self.poll_timeout_us = poll_timeout_us
        self.tokens_per_seq = tokens_per_seq
        if launch_shape not in ("one", "two"):
            raise ValueError(f"launch_shape must be one/two, got {launch_shape!r}")
        if launch_shape == "one" and (
            qkv_batch_limit is not None
            or qkv_samples_per_pass is not None
            or qkv_norm_handoff
        ):
            raise ValueError("the legacy ONE launch does not take split-QKV tuning")
        self.launch_shape = launch_shape
        self.qkv_batch_limit = qkv_batch_limit
        self.qkv_samples_per_pass = qkv_samples_per_pass
        self.qkv_norm_handoff = qkv_norm_handoff
        self.finite_observer_layer = finite_observer_layer
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
            finite_observer_layer,
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
        phases = ("one",) if launch_shape == "one" else ("attention", "moe")
        refresh_modes = (True, False) if cfg.indexed else (True,)
        self.launch_sets = {}
        for refresh in refresh_modes:
            cache_key = (self.key, refresh)
            built = Dsv4Variant._cache.get(cache_key)
            if built is None:
                built = tuple(
                    build_dsv4_kernel(
                        samples,
                        cfg.heads,
                        npes,
                        cfg.window,
                        scale=cfg.softmax_scale,
                        timeline=timeline,
                        moe_mode=moe_mode,
                        poll_timeout_us=poll_timeout_us,
                        tokens_per_seq=tokens_per_seq,
                        swiglu_limit=cfg.swiglu_limit,
                        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                        hc_eps=cfg.hc_eps,
                        bf16_kv_roundtrip=cfg.bf16_kv_roundtrip,
                        window_rows=cfg.cache_rows,
                        index_topk_refresh=refresh,
                        phase=phase,
                        qkv_batch_limit=qkv_batch_limit,
                        qkv_samples_per_pass=qkv_samples_per_pass,
                        qkv_norm_handoff=qkv_norm_handoff,
                        finite_observer_layer=finite_observer_layer,
                        **dims,
                    )
                    for phase in phases
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
            # Slot 0 is the bounded-poll timeout. Diagnostic builds add NaN
            # boundary bits without inserting kernels between full layers.
            self.hang = torch.zeros(
                9 if finite_observer_layer >= 0 else 1,
                dtype=torch.int32,
                device=dev,
            )

        bind_agreed(allocate, group)
        self.peer_buffer = SymmetricPeerBuffer(
            self.sym_layout["_bytes"], rank=rank, npes=npes, group=group
        )
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses

    def hang_detected(self) -> bool:
        """Whether a launch timed out a poll (needs ``poll_timeout_us``). Synchronizes."""
        return bool(self.hang[0].item())

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs so tags can wrap; graph-capturable."""
        self._advance(
            self.step.data_ptr(),
            self.scratch.data_ptr(),
            self.sym,
            stream=torch.cuda.current_stream(),
        )

    def close(self):
        self.peer_buffer.close()
