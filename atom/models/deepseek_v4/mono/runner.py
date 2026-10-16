# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Whole-stack DeepSeek-V4 Mono execution; serving lifecycle lives in session."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from atom.models.deepseek_v4.mono.cache import LayerStep
from atom.models.deepseek_v4.mono.config import (
    MAX_LAYERS_PER_STEP,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.state import Dsv4Variant
from atom.models.deepseek_v4.mono.weights import pack_layer_weights
from atom.mono.runtime.compile import (
    compile_only,
    launcher_resources,
    require_resident_grid,
)
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed, tp_agree

logger = logging.getLogger("atom")

if TYPE_CHECKING:
    from atom.models.deepseek_v4.mono.layer import Dsv4MonoKernel


__all__ = [
    "Dsv4Variant",
    "MoeMode",
    "V4MonoDecodeRunner",
    "shape_dims",
]


@dataclass(frozen=True)
class RowsBuild:
    """One exact row width: two attention variants, independent layer owners."""

    layers: tuple[Dsv4MonoKernel, ...]
    variants: tuple[Dsv4Variant, ...]
    hidden: tuple[torch.Tensor, torch.Tensor]


class V4MonoDecodeRunner:
    """A target layer stack on the PR ONE builder, with V4.1-style prepare/forward.

    No drafter is required. Exact row width and ``tokens_per_seq`` describe the
    device geometry; speculative acceptance/rollback stays with the caller.
    The optional serving adapter binds ATOM's real cache after allocation.
    All layers use external rolling state, so changing width does not reset or
    duplicate a request's state. Target and draft must not share this runner.
    """

    def __init__(
        self,
        weights,
        *,
        rank=0,
        npes=1,
        group=None,
        packed=None,
        experts=None,
        moe_mode=MoeMode.A8W4,
        timeline=False,
        finite_observer_layer=-1,
    ):
        import torch.distributed as dist

        self.group = dist.group.WORLD if npes > 1 and group is None else group
        self.rank, self.npes = rank, npes
        self.weights = tuple(weights)
        self.moe_mode = as_moe_mode(moe_mode)
        self.timeline = timeline
        self.finite_observer_layer = int(finite_observer_layer)
        self._lane_builds = {}
        self._refused = set()
        self._closed = False
        self.packed = []

        def bind():
            if not self.weights or len(self.weights) > MAX_LAYERS_PER_STEP:
                raise MonoUnsupported("invalid target layer count")
            geometry = {(w.cfg.hidden, w.cfg.hc_mult) for w in self.weights}
            if len(geometry) != 1:
                raise MonoUnsupported("target layers must share the residual geometry")
            if packed is not None and len(packed) != len(self.weights):
                raise MonoUnsupported("one packed weight owner is required per layer")
            if experts is not None and len(experts) != len(self.weights):
                raise MonoUnsupported("one external expert bank is required per layer")
            for i, w in enumerate(self.weights):
                w.cfg.validate()
                self.packed.append(
                    packed[i]
                    if packed is not None
                    else pack_layer_weights(
                        w.t, self.moe_mode, None if experts is None else experts[i]
                    )
                )

        bind_agreed(bind, self.group)

    @classmethod
    def from_model(cls, model, max_model_len: int, *, rank=0, npes=8, group=None):
        """Prepare-time binding of the real 61-layer target, without installing it."""
        import torch.distributed as dist

        from atom.models.deepseek_v4.mono.config import V4Config
        from atom.models.deepseek_v4.mono.weights import bind_layer

        group = dist.group.WORLD if npes > 1 and group is None else group
        bindings = []

        def bind():
            from atom.models.deepseek_v4 import _V4_USE_REF_QUANT

            if _V4_USE_REF_QUANT:
                raise MonoUnsupported(
                    "the real BF16 profile requires ordinary reference quantization off"
                )
            if len(model.layers) != 61:
                raise MonoUnsupported(
                    "the first real target profile is the 61-layer V4-Pro"
                )
            index_topk_source_layer = -1
            for block in model.layers:
                indexed = block.attn.compress_ratio == 4
                index_topk_refresh = indexed and not block.attn.skip_topk
                if index_topk_refresh:
                    index_topk_source_layer = int(block.layer_id)
                elif indexed and index_topk_source_layer < 0:
                    raise MonoUnsupported(
                        "a CSA index reuse layer precedes its refresh producer"
                    )
                cfg = V4Config(
                    compress_ratio=block.attn.compress_ratio,
                    max_seq=max_model_len,
                    kv_fp8=block.attn.kv_fp8,
                    bf16_kv_roundtrip=False,
                    shared_bf16_block128=True,
                    routed_down_bf16=True,
                    compressor_projection_bf16=True,
                    attention_fp8_bf16=True,
                    indexer_hadamard=False,
                    index_topk_refresh=index_topk_refresh if indexed else True,
                    index_topk_source_layer=index_topk_source_layer if indexed else -1,
                )
                bindings.append(bind_layer(block, cfg))

        bind_agreed(bind, group)
        runner = cls(
            [b.weights for b in bindings],
            packed=[b.packed for b in bindings],
            rank=rank,
            npes=npes,
            group=group,
        )
        runner.bindings = tuple(bindings)
        runner.model = model
        return runner

    @staticmethod
    def _compile(launch):
        """Compile before allocating peers; all ranks vote before any launch."""
        n = sum(
            name != "stream"
            for name in inspect.signature(launch._original_func).parameters
        )
        with compile_only():
            launch(*([0] * n))
        return launcher_resources(launch)

    @staticmethod
    def _launch_plan(rows: int) -> dict:
        """Resource-proven launch shape for one exact live-row micro-bucket."""
        if rows == 1:
            return {
                "launch_shape": "one",
                "qkv_batch_limit": None,
                "qkv_samples_per_pass": None,
                "qkv_norm_handoff": False,
            }
        if rows in (2, 4, 8):
            return {
                "launch_shape": "two",
                "qkv_batch_limit": 1,
                "qkv_samples_per_pass": 1,
                "qkv_norm_handoff": True,
            }
        raise MonoUnsupported(
            "the production decode plan supports exact 1/2/4/8-row micro-buckets"
        )

    def prepare(self, rows: int, *, tokens_per_seq: int = 1, lanes: int = 1) -> bool:
        """Prepare an exact width outside capture; failures agree before IPC.

        S1 uses the legacy whole-layer ONE launch. S2/S4/S8 use attention -> MoE
        under one owner/arena/tag with one advance after both launches. SGPR
        spill into registers/LDS is not itself a refusal. Repeated micro-buckets
        in one serving step receive separate stable lanes, so a faster TP rank
        cannot overwrite a peer mailbox that a slower rank is still consuming.
        """
        from atom.models.deepseek_v4.mono.kernels.layer import build_dsv4_kernel
        from atom.models.deepseek_v4.mono.layer import Dsv4MonoKernel

        if self._closed:
            raise RuntimeError("target runner is closed")
        if lanes <= 0:
            raise ValueError("lanes must be positive")
        key = (rows, tokens_per_seq)
        if key in self._lane_builds:
            if lanes <= len(self._lane_builds[key]):
                return True
            raise MonoUnsupported(
                "an exact width cannot grow its prepared lane count after binding"
            )
        if key in self._refused:
            return False
        cfgs, builds = {}, {}
        failure = None
        try:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("prepare the width before graph capture")
            for binding in getattr(self, "bindings", ()):
                binding.assert_current()
            if tokens_per_seq <= 0 or rows % tokens_per_seq:
                raise MonoUnsupported("rows must form complete consecutive token runs")
            plan = self._launch_plan(rows)
            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            if device.multi_processor_count != 256:
                raise MonoUnsupported("the PR grid requires 256 gfx950 CUs")
            for w in self.weights:
                cfg = w.cfg
                validate_shard(
                    rows,
                    cfg.heads,
                    self.rank,
                    self.npes,
                    cfg.window,
                    cfg.compress_ratio,
                )
                vk = _variant_key(
                    cfg,
                    rows,
                    self.npes,
                    self.moe_mode,
                    self.timeline,
                    None,
                    tokens_per_seq,
                    finite_observer_layer=self.finite_observer_layer,
                    **plan,
                )
                cfgs.setdefault(vk, cfg)
            for vk, cfg in cfgs.items():
                refresh_modes = (True, False) if cfg.indexed else (True,)
                for refresh in refresh_modes:
                    cache_key = (vk, refresh)
                    launches = Dsv4Variant._cache.get(cache_key)
                    if launches is None:
                        phases = (
                            ("one",)
                            if plan["launch_shape"] == "one"
                            else ("attention", "moe")
                        )
                        launches = tuple(
                            build_dsv4_kernel(
                                rows,
                                cfg.heads,
                                self.npes,
                                cfg.window,
                                scale=cfg.softmax_scale,
                                timeline=self.timeline,
                                moe_mode=self.moe_mode,
                                tokens_per_seq=tokens_per_seq,
                                swiglu_limit=cfg.swiglu_limit,
                                hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                                hc_eps=cfg.hc_eps,
                                bf16_kv_roundtrip=cfg.bf16_kv_roundtrip,
                                window_rows=cfg.cache_rows,
                                phase=phase,
                                index_topk_refresh=refresh,
                                qkv_batch_limit=plan["qkv_batch_limit"],
                                qkv_samples_per_pass=plan["qkv_samples_per_pass"],
                                qkv_norm_handoff=plan["qkv_norm_handoff"],
                                finite_observer_layer=self.finite_observer_layer,
                                **shape_dims(cfg),
                            )
                            for phase in phases
                        )
                    elif not isinstance(launches, tuple):
                        launches = (launches,)
                    for launch in launches:
                        for resource in self._compile(launch):
                            if resource["scratch"] or resource["lds"] > 163840:
                                raise MonoUnsupported(
                                    f"unsafe {rows}-row {plan['launch_shape']} resource: {resource}"
                                )
                        require_resident_grid(launch, 256, 512)
                    builds[cache_key] = launches
        except Exception as error:
            failure = error
            logger.exception(
                "V4 mono: preparing rows=%d tokens_per_seq=%d failed",
                rows,
                tokens_per_seq,
            )
        if not tp_agree(failure is None, self.group):
            self._refused.add(key)
            return False
        Dsv4Variant._cache.update(builds)
        lane_builds = []
        try:
            for _lane in range(lanes):
                variants, layers = {}, []
                for vk, cfg in cfgs.items():
                    variants[vk] = Dsv4Variant(
                        cfg,
                        rows,
                        self.rank,
                        self.npes,
                        self.group,
                        timeline=self.timeline,
                        moe_mode=self.moe_mode,
                        tokens_per_seq=tokens_per_seq,
                        finite_observer_layer=self.finite_observer_layer,
                        **plan,
                    )
                    bind_agreed(
                        lambda v=variants[vk]: self._compile(v._advance), self.group
                    )
                # Layer constructor failures are local; agree before proceeding to
                # another collective and before letting any rank launch this stack.
                hidden = None
                failure = None
                try:
                    for w, packed_layer in zip(self.weights, self.packed):
                        vk = _variant_key(
                            w.cfg,
                            rows,
                            self.npes,
                            self.moe_mode,
                            self.timeline,
                            None,
                            tokens_per_seq,
                            finite_observer_layer=self.finite_observer_layer,
                            **plan,
                        )
                        layers.append(
                            Dsv4MonoKernel(
                                w,
                                rows,
                                self.rank,
                                self.npes,
                                variant=variants[vk],
                                packed=packed_layer,
                                moe_mode=self.moe_mode,
                                tokens_per_seq=tokens_per_seq,
                                own_state=False,
                            )
                        )
                    cfg = self.weights[0].cfg
                    shape = (
                        (rows, cfg.hidden)
                        if cfg.hc_mult == 1
                        else (rows, cfg.hc_mult, cfg.hidden)
                    )
                    hidden = tuple(
                        torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                        for _ in range(2)
                    )
                # Allocation/compile failures from any backend must participate
                # in the TP-wide refusal instead of escaping on one rank.
                except Exception as error:  # noqa: BLE001
                    failure = error
                if not tp_agree(failure is None, self.group):
                    raise MonoUnsupported(
                        "a rank failed to allocate the target width"
                    ) from failure
                lane_builds.append(
                    RowsBuild(tuple(layers), tuple(variants.values()), hidden)
                )
            self._lane_builds[key] = tuple(lane_builds)
            return True
        except Exception:
            for build in lane_builds:
                for variant in build.variants:
                    variant.close()
            for variant in locals().get("variants", {}).values():
                if not any(
                    variant is v for build in lane_builds for v in build.variants
                ):
                    variant.close()
            self._refused.add(key)
            raise

    def _build_for(self, key: tuple[int, int], lane: int = 0) -> RowsBuild:
        builds = self._lane_builds.get(key)
        if builds is None:
            raise MonoUnsupported("target width has not been prepared")
        if not 0 <= lane < len(builds):
            raise MonoUnsupported(f"target width has no prepared lane {lane}")
        return builds[lane]

    def index_policy(self) -> dict:
        """Describe the compiled CSA refresh/reuse schedule.

        A reuse hit means one CSA layer invocation consumes its producer's
        tagged ``i_sel`` without launching the index compressor tail, query,
        score, or top-k stages.  Layer counts are independent of exact-row
        micro-bucketing; the serving session accounts for those executions.
        """
        bindings = getattr(self, "bindings", ())
        refresh, reuse = [], []
        for ordinal, weights in enumerate(self.weights):
            cfg = weights.cfg
            if not cfg.indexed:
                continue
            layer_id = (
                bindings[ordinal].layer_id if ordinal < len(bindings) else ordinal
            )
            (refresh if cfg.index_topk_refresh else reuse).append(int(layer_id))
        return {
            "refresh_csa_layers": len(refresh),
            "reuse_csa_layers": len(reuse),
            "refresh_layer_ids": refresh,
            "reuse_layer_ids": reuse,
            "suppressed_stages_per_reuse_hit": [
                "index_compressor_tail",
                "index_query_b",
                "index_query",
                "index_weight_projection",
                "index_score",
                "index_topk",
            ],
        }

    def forward(
        self,
        h,
        steps: tuple[LayerStep, ...],
        *,
        tokens=None,
        tokens_per_seq=1,
        x_out=None,
        lane=0,
    ):
        """Whole target stack; caller supplies real RoPE/cache/state per layer.

        This does not allocate outputs, prepare widths, advance per layer, or
        roll back caches. It returns the settled V4 mHC residual (LM head stays
        ordinary). Eligibility/TP-uniform metadata must be agreed upstream.
        """
        if self._closed:
            raise RuntimeError("target runner is closed")
        key = (h.shape[0], tokens_per_seq)
        build = self._build_for(key, lane)
        if len(steps) != len(build.layers):
            raise ValueError("one LayerStep is required per target layer")
        self.preflight(
            h,
            steps,
            tokens=tokens,
            tokens_per_seq=tokens_per_seq,
            x_out=x_out,
            lane=lane,
        )
        # Alias-free ping-pong: the external input can be a previous returned view.
        buf0 = build.hidden[0]
        first = int(h.data_ptr() == buf0.data_ptr())
        for i, (layer, step) in enumerate(zip(build.layers, steps)):
            out = build.hidden[(i + first) % 2]
            if i == len(build.layers) - 1 and x_out is not None:
                if x_out.data_ptr() == h.data_ptr():
                    raise ValueError(
                        "target output must not alias its last layer input"
                    )
                out = x_out
            h = layer.forward(
                h,
                step.positions,
                step.kv_cache,
                step.dest_rows,
                step.indices,
                step.cos,
                step.sin,
                x_out=out,
                layer=i,
                advance=False,
                tokens=tokens,
                block_tables=step.block_tables,
                env_rows=step.env_rows,
                state=step.state,
            )
        for variant in build.variants:
            variant.advance_step()
        return h

    def preflight(self, h, steps, *, tokens=None, tokens_per_seq=1, x_out=None, lane=0):
        """All layers are checked before the first cache write, not mid-stack."""
        build = self._build_for((h.shape[0], tokens_per_seq), lane)
        if len(steps) != len(build.layers):
            raise MonoUnsupported("one real step per layer is required")
        for binding in getattr(self, "bindings", ()):
            binding.assert_current()
        current = h
        first = int(h.data_ptr() == build.hidden[0].data_ptr())
        for i, (layer, step) in enumerate(zip(build.layers, steps)):
            out = (
                x_out
                if i == len(build.layers) - 1 and x_out is not None
                else build.hidden[(i + first) % 2]
            )
            if out.data_ptr() == current.data_ptr():
                raise MonoUnsupported("layer output must not alias its input")
            layer.validate_inputs(
                current,
                step.positions,
                step.kv_cache,
                step.dest_rows,
                step.indices,
                step.cos,
                step.sin,
                x_out=out,
                layer=i,
                advance=False,
                tokens=tokens,
                block_tables=step.block_tables,
                env_rows=step.env_rows,
                state=step.state,
            )
            current = out

    def forward_tokens(
        self,
        input_ids,
        positions,
        steps,
        *,
        tokens_per_seq=1,
        x_out=None,
        lane=0,
    ):
        """Token→settled residual. The ordinary model retains head and sampling."""
        h = self.model.mhc_input(input_ids)
        return self.forward(
            h,
            steps,
            tokens=input_ids,
            tokens_per_seq=tokens_per_seq,
            x_out=x_out,
            lane=lane,
        )

    def close(self):
        """Collective, once every replay has retired; never mutate rolling state."""
        if self._closed:
            return
        for builds in self._lane_builds.values():
            for build in builds:
                for variant in build.variants:
                    variant.close()
        self._lane_builds.clear()
        self._closed = True
