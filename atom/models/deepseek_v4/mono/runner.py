# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""Whole-stack DeepSeek-V4 Mono runner and serving lifecycle."""

from __future__ import annotations

import inspect
import logging
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from atom.models.deepseek_v4.mono.config import (
    MAX_LAYERS_PER_STEP,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.contract import (
    BatchContinuation,
    ContinuationRow,
    continues_batch,
)
from atom.models.deepseek_v4.mono.profile import ORDINARY_PROFILE
from atom.models.deepseek_v4.mono.state import Dsv4Variant, LayerStep, V4CacheAdapter
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

_session_lease_lock = threading.Lock()
_session_lease_owner: weakref.ReferenceType | None = None
_session_generation = 0


def _live_session():
    return None if _session_lease_owner is None else _session_lease_owner()


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


class V4MonoSession:
    """Serving lifecycle/dispatch on the same runner; no second model wrapper.

    Publication happens before graph replay, because ATOM rotates H2D buffers.
    A graph captured as Mono is never replayed for a refused/forked step.
    This bring-up gate votes outside capture; its cost remains in step timing.
    """

    profile = ORDINARY_PROFILE

    def __init__(self, model, config, builder, profile=ORDINARY_PROFILE):
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        self.model, self.config, self.builder = model, config, builder
        self.profile = profile
        self.policy = getattr(config, "mono_decode", "auto")
        self.rank, self.group = (
            get_tensor_model_parallel_rank(),
            get_tp_group().cpu_group,
        )
        self.runner = self.adapter = None
        self.armed = False
        self.disabled = False
        self.refusal = None
        self.graph_keys = set()
        self.graph_generations: dict[tuple[int, int], int] = {}
        self.selected_steps = 0
        self.graph_replays = 0
        self.eager_forwards = 0
        self.decode_attempts = 0
        self.fallback_steps = 0
        self.fallback_reasons: dict[str, int] = {}
        self.microbucket_stacks = 0
        self.index_refresh_executions = 0
        self.index_reuse_executions = 0
        self.index_suppressed_rows = 0
        self.live_started = False
        self.weights_ready = True
        self.requires_cache_rebind = False
        self.entry_sources = ()
        self.output = None
        self.token_out = None
        self.owner_generation = 0
        self.graph_generation = 0
        self.publication_generation = 0
        self.publication_slot_generations: tuple[tuple[int, int], ...] = ()
        self.capture_publication = False
        self.slot_generations: dict[int, int] = {}
        self.batch_continuation: BatchContinuation | None = None
        self.full_publications = 0
        self.fast_publications_by_width: dict[int, int] = {}

    @property
    def force(self) -> bool:
        return getattr(self, "policy", "auto") == "force"

    @property
    def uses_greedy_terminal(self) -> bool:
        return self.profile.greedy_terminal

    def enforce_selection(self, context, selected: bool) -> None:
        """Make ``force`` a qualification contract, without affecting prefill.

        Auto mode deliberately falls back before the first cache write. Force
        mode uses the same admission decision but turns a refused *live decode*
        into an error, so a benchmark cannot silently measure the native path.
        Capture-time dummy rows and prefill remain native by design.
        """
        if selected or not self.force:
            return
        fc = context.context
        if fc is None or fc.is_prefill or getattr(fc, "is_dummy_run", False):
            return
        raise MonoUnsupported(
            "V4 Mono force policy refused a live decode step: "
            f"{self.refusal or 'unsupported step contract'}"
        )

    def _acquire_lease(self):
        """Own the one live Mono arena/cache binding in this rank process."""
        global _session_lease_owner, _session_generation
        with _session_lease_lock:
            holder = _live_session()
            if holder is self:
                raise MonoUnsupported("this V4 Mono session is already prepared")
            if holder is not None:
                raise MonoUnsupported(
                    "another V4 Mono session still owns the live arena/cache binding"
                )
            _session_generation += 1
            self.owner_generation = _session_generation
            _session_lease_owner = weakref.ref(self)

    def _release_lease(self):
        global _session_lease_owner
        with _session_lease_lock:
            if _live_session() is self:
                _session_lease_owner = None
        self.owner_generation = 0

    def owns_live_session(self) -> bool:
        with _session_lease_lock:
            return self.owner_generation != 0 and _live_session() is self

    def slots_reset(self, slots) -> None:
        """Invalidate any publication that predates request-state slot reuse."""
        changed = {int(slot) for slot in slots}
        for slot in changed:
            self.slot_generations[slot] = self.slot_generations.get(slot, 0) + 1
        if self.armed and any(
            slot in changed for slot, _ in self.publication_slot_generations
        ):
            self.armed = False
            self.publication_generation = 0
            self.publication_slot_generations = ()
            if self.adapter is not None:
                self.adapter.clear_publication()
            self.refusal = (
                "request-state slot reset invalidated the published Mono step"
            )
        if self.batch_continuation is not None and any(
            row.slot in changed for row in self.batch_continuation.rows
        ):
            self.batch_continuation = None

    def close(self):
        # A graph generation is a capture lifetime, not merely a session-owner
        # lifetime.  Advancing it here makes any external/stale continuation
        # token unusable even if this Python session is prepared again.
        self.graph_generation = getattr(self, "graph_generation", 0) + 1
        self.armed = False
        self.graph_keys.clear()
        self.graph_generations.clear()
        self.publication_generation = 0
        self.publication_slot_generations = ()
        self.capture_publication = False
        self.batch_continuation = None
        if self.adapter is not None:
            self.adapter.clear_publication()
        if self.runner is not None:
            self.runner.close()
        self.runner = self.adapter = None
        self.entry_sources = ()
        self.output = None
        self.token_out = None
        self._release_lease()

    def graph_is_current(self, key: tuple[int, int]) -> bool:
        """Whether ``key`` was captured by the live owner generation."""
        return (
            key in self.graph_keys
            and self.owner_generation != 0
            and self.graph_generations.get(key) is not None
            and self.owns_live_session()
        )

    def remember_graph(self, key: tuple[int, int], selected: bool) -> None:
        """Bind or remove one graph under a new capture generation."""
        self.graph_generation += 1
        if (
            self.batch_continuation is not None
            and self.batch_continuation.graph_key == key
        ):
            self.batch_continuation = None
        if selected:
            self.graph_keys.add(key)
            self.graph_generations[key] = self.graph_generation
        else:
            self.graph_keys.discard(key)
            self.graph_generations.pop(key, None)

    def prepare_weights(self) -> bool:
        """Pack and retain Mono weights before the KV pool is sized.

        The production weight layout needs a one-time, lossless repack for the
        imported kernels.  Keeping that allocation behind ``allocate_kv_cache``
        made it invisible to ``get_num_blocks``: the ordinary cache consumed
        the complete memory budget first and Mono then sat on top of it.  This
        phase has no cache dependency, so materialize it early and let the
        existing pool planner observe the actual resident bytes.

        Cache adapters, scratch arenas and terminal buffers remain in
        ``prepare`` because they require the final pool geometry.  Repeated
        calls are idempotent; a completed cache binding also owns a valid
        weight binding.
        """
        if self.disabled or not self.weights_ready:
            return False
        if self.runner is not None:
            return True
        try:
            bind_agreed(self._acquire_lease, self.group)
            self.runner = V4MonoDecodeRunner.from_model(
                self.model.model,
                self.config.max_model_len,
                rank=self.rank,
                npes=8,
                group=self.group,
            )
            logger.info(
                "V4 Mono weights prepared before KV sizing on TP rank %d",
                self.rank,
            )
            return True
        except Exception as error:
            self.refusal = str(error)
            logger.exception("V4 Mono weight prepare refused before KV sizing")
            self.close()
            self.disabled = True
            if self.force:
                raise
            return False

    def prepare(self, *, cache_rebound=False):
        if self.disabled or not self.weights_ready:
            return False
        if self.requires_cache_rebind and not cache_rebound:
            self.refusal = (
                "updated weights require an explicit cache/state rebind before "
                "Mono prepare"
            )
            return False
        # A complete old cache binding must retire before rebinding.  A runner
        # with no adapter is the weight-only phase installed by
        # ``prepare_weights`` and must survive the initial KV allocation.
        if self.adapter is not None:
            torch.cuda.synchronize()
            engine = self.builder.model_runner
            engine.graphs.clear()
            engine.graph_pool = None
            if hasattr(engine, "graph_logits"):
                engine.graph_logits.clear()
            self.close()
        if not self.prepare_weights():
            return False
        try:
            adapters = []
            declared_capture_sizes = self.config.capture_sizes
            max_declared_requests = max(declared_capture_sizes, default=1)
            policy_request_limit = (
                self.profile.max_requests
                if self.force
                else max(self.profile.auto_request_widths)
            )
            max_requests = (
                max(1, min(int(max_declared_requests), policy_request_limit))
                if self.config.compilation_config.cudagraph_mode.name == "FULL"
                else policy_request_limit
            )
            max_rows = max_requests * self.profile.tokens_per_seq
            bind_agreed(
                lambda: adapters.append(
                    V4CacheAdapter(
                        self.model.model,
                        self.builder,
                        [w.cfg for w in self.runner.weights],
                        max_rows=max_rows,
                        tokens_per_seq=self.profile.tokens_per_seq,
                        max_requests=max_requests,
                    )
                ),
                self.group,
            )
            self.adapter = adapters[0]
            from atom.models.deepseek_v4.mono.weights import _signature

            inner = self.model.model

            def bind_entry():
                self.entry_sources = tuple(
                    (
                        owner,
                        name,
                        getattr(owner, name),
                        _signature(getattr(owner, name)),
                    )
                    for owner, name in (
                        (inner.embed, "weight"),
                        (inner.norm, "weight"),
                        (inner.head, "weight"),
                        (inner, "hc_head_fn"),
                        (inner, "hc_head_scale"),
                        (inner, "hc_head_base"),
                    )
                )

            bind_agreed(bind_entry, self.group)
            for rows in self.adapter.prepared_widths:
                if not self.runner.prepare(
                    rows,
                    tokens_per_seq=self.profile.tokens_per_seq,
                    lanes=self.adapter.bucket_count(rows),
                ):
                    raise MonoUnsupported(f"S{rows} binary admission failed")
                for ordinal in range(self.adapter.bucket_count(rows)):
                    bucket = self.adapter.bucket(rows, ordinal)
                    build = self.runner._build_for(
                        (rows, self.profile.tokens_per_seq), ordinal
                    )
                    bind_agreed(
                        lambda b=build, x=bucket, lane=ordinal: self.runner.preflight(
                            b.hidden[0],
                            x.layers,
                            tokens=x.tokens,
                            tokens_per_seq=self.profile.tokens_per_seq,
                            lane=lane,
                        ),
                        self.group,
                    )
            cfg = self.runner.weights[0].cfg
            shape = (
                (self.adapter.max_rows, cfg.hidden)
                if cfg.hc_mult == 1
                else (self.adapter.max_rows, cfg.hc_mult, cfg.hidden)
            )

            def allocate_terminal():
                self.output = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                self.token_out = torch.empty(
                    self.adapter.max_rows, dtype=torch.int32, device="cuda"
                )

            bind_agreed(allocate_terminal, self.group)
            logger.info(
                "V4 Mono profile %s exact-row widths %s real cache prepared on TP rank %d",
                self.profile.name,
                self.adapter.prepared_widths,
                self.rank,
            )
            self.refusal = None
            if cache_rebound:
                self.live_started = False
                self.requires_cache_rebind = False
            return True
        except Exception as error:
            self.refusal = str(error)
            logger.exception("V4 Mono exact-row prepare refused before launch")
            self.close()
            self.disabled = True
            if self.force:
                raise
            return False

    def publish(
        self,
        input_ids,
        positions,
        context,
        *,
        capture=False,
        allow_auto=True,
    ):
        self.armed = False
        self.publication_generation = 0
        self.publication_slot_generations = ()
        self.capture_publication = False
        if self.adapter is not None:
            self.adapter.clear_publication()
        if self.runner is None:
            self.batch_continuation = None
            return False
        # Mono serves decode only.  A real prefill or an eager dummy warmup is
        # already a TP-uniform scheduler mode and cannot launch the Mono graph;
        # walking 61 weight owners and entering Gloo consensus before refusing
        # it only adds TTFT.  Capture-time dummy decode is intentionally kept:
        # it is how the selected Mono graph is constructed.
        if not capture and (
            context.context.is_prefill
            or getattr(context.context, "is_dummy_run", False)
        ):
            self.batch_continuation = None
            return False
        is_live_decode = bool(
            not capture
            and not context.context.is_prefill
            and not getattr(context.context, "is_dummy_run", False)
        )
        if is_live_decode:
            self.decode_attempts += 1
        if is_live_decode and not self.force and not allow_auto:
            self.refusal = (
                "sampling/logprob request is outside the V4 Mono auto profile"
            )
            self.batch_continuation = None
            self.fallback_steps += 1
            self.fallback_reasons[self.refusal] = (
                self.fallback_reasons.get(self.refusal, 0) + 1
            )
            return False
        graph_key = (
            int(context.context.running_bs),
            int(context.attn_metadata.max_seqlen_q),
        )
        if not self.profile.accepts_graph(*graph_key, force=self.force):
            self.refusal = (
                f"graph shape {graph_key} is not enabled for V4 Mono "
                f"{self.profile.name} profile"
            )
            self.batch_continuation = None
            if is_live_decode:
                self.fallback_steps += 1
                self.fallback_reasons[self.refusal] = (
                    self.fallback_reasons.get(self.refusal, 0) + 1
                )
            return False
        batch_contract = None
        if (
            self.profile.continuation
            and is_live_decode
            and self.batch_continuation is not None
            and self.batch_continuation.owner_generation == self.owner_generation
            and self.graph_is_current(self.batch_continuation.graph_key)
            and self.graph_generations.get(self.batch_continuation.graph_key)
            == self.batch_continuation.graph_generation
            and all(
                self.slot_generations.get(row.slot, 0) == row.slot_generation
                for row in self.batch_continuation.rows
            )
            and self.weights_ready
        ):
            batch_contract = self.adapter.continuation_contract(context)
        if continues_batch(self.batch_continuation, graph_key, batch_contract):
            # The anchor step already agreed owner/cache/layer ABI and the
            # scheduler broadcasts the same position/slot/block-table frontier
            # to every TP worker. Ordinary greedy output and MTP target/draft
            # sampling already obey the engine's TP-uniform device contract, so
            # the next rows need no host object collective or D2H validation.
            # Do not turn a rank-local enqueue failure into a slow-path
            # collective: peers have made the same fast decision and would
            # otherwise wait in a different phase.
            self.adapter.write(
                input_ids,
                positions,
                context,
                validate_static=False,
                validate_device_values=False,
                supported=True,
            )
            rows = tuple(
                ContinuationRow(
                    slot,
                    self.slot_generations.get(slot, 0),
                    position,
                    blocks,
                )
                for position, slot, blocks in batch_contract
            )
            self.batch_continuation = BatchContinuation(
                self.owner_generation,
                graph_key,
                self.graph_generations[graph_key],
                rows,
            )
            width = graph_key[0] * graph_key[1]
            self.fast_publications_by_width[width] = (
                self.fast_publications_by_width.get(width, 0) + 1
            )
            return self._arm_publication(capture=False)
        # Host consensus is outside the captured graph. In particular, a
        # rank-local owner failure cannot leave its peers launching alone.
        try:
            supported = (
                self.owns_live_session()
                and self.adapter.supported(context)
                and input_ids.ndim == 1
                and positions.ndim == 1
                and input_ids.numel() == positions.numel()
                and input_ids.numel() <= self.adapter.max_rows
                and self.weights_ready
            )
            if (
                supported
                and getattr(context.context, "is_dummy_run", False)
                and not capture
            ):
                supported = False
            self.adapter.assert_current()
            for binding in self.runner.bindings:
                binding.assert_current()
            from atom.models.deepseek_v4.mono.weights import _signature

            if any(
                getattr(owner, name) is not value or _signature(value) != signature
                for owner, name, value, signature in self.entry_sources
            ):
                raise MonoUnsupported(
                    "embedding/head owner changed; retire graphs and reprepare Mono"
                )
        # Owner and adapter checks raise backend-specific errors; convert every
        # one into the same TP-agreed native fallback decision.
        except Exception as error:  # noqa: BLE001
            self.refusal = str(error)
            supported = False
        if not tp_agree(supported, self.group):
            self.batch_continuation = None
            if is_live_decode:
                reason = self.refusal or (
                    "tp_peer_refused" if supported else "unsupported_contract"
                )
                self.fallback_steps += 1
                self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            return False
        try:

            def publish_and_preflight():
                self.adapter.write(input_ids, positions, context)
                for _, bucket in self.adapter.published_buckets:
                    build = self.runner._build_for(
                        (bucket.width, self.profile.tokens_per_seq), bucket.ordinal
                    )
                    self.runner.preflight(
                        build.hidden[0],
                        bucket.layers,
                        tokens=bucket.tokens,
                        tokens_per_seq=self.profile.tokens_per_seq,
                        lane=bucket.ordinal,
                    )

            bind_agreed(publish_and_preflight, self.group)
            import torch.distributed as dist

            local_contract = self.adapter.publication_consensus_tensor()
            contracts = [
                torch.empty_like(local_contract)
                for _ in range(dist.get_world_size(self.group))
            ]
            dist.all_gather(contracts, local_contract, group=self.group)
            if any(
                not torch.equal(contract, contracts[0]) for contract in contracts[1:]
            ):
                raise MonoUnsupported(
                    "TP ranks published different token/position/slot/block tables"
                )
        # Publication may fail in torch.distributed, cache adapters, or the
        # compiled preflight; all must refuse before any rank launches Mono.
        except Exception as error:  # noqa: BLE001
            self.refusal = str(error)
            self.adapter.clear_publication()
            if is_live_decode:
                reason = self.refusal or "metadata_publication_refused"
                self.fallback_steps += 1
                self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            logger.warning(
                "V4 Mono metadata publication refused before launch: %s", error
            )
            self.batch_continuation = None
            return False
        if is_live_decode:
            self.full_publications += 1
        published_rows = len(self.adapter.published_contract)
        continuation_contract = (
            self.adapter.continuation_contract(context)
            if self.profile.continuation and is_live_decode
            else None
        )
        if (
            self.profile.continuation
            and is_live_decode
            and published_rows in self.adapter.prepared_widths
            and continuation_contract is not None
            and self.graph_is_current(graph_key)
        ):
            rows = tuple(
                ContinuationRow(
                    slot,
                    self.slot_generations.get(slot, 0),
                    position,
                    blocks,
                )
                for position, slot, blocks in continuation_contract
            )
            self.batch_continuation = BatchContinuation(
                self.owner_generation,
                graph_key,
                self.graph_generations[graph_key],
                rows,
            )
        else:
            self.batch_continuation = None
        return self._arm_publication(capture=capture)

    def _arm_publication(self, *, capture: bool) -> bool:
        slots = tuple(
            sorted({contract[2] for contract in self.adapter.published_contract})
        )
        self.publication_generation = self.owner_generation
        self.publication_slot_generations = tuple(
            (slot, self.slot_generations.get(slot, 0)) for slot in slots
        )
        self.armed = True
        self.capture_publication = capture
        self.refusal = None
        if not capture:
            self.selected_steps += (
                1  # successful publications, not a device-launch counter
            )
            policy = self.runner.index_policy()
            buckets = tuple(self.adapter.published_buckets)
            self.microbucket_stacks += len(buckets)
            self.index_refresh_executions += len(buckets) * policy["refresh_csa_layers"]
            self.index_reuse_executions += len(buckets) * policy["reuse_csa_layers"]
            self.index_suppressed_rows += (
                sum(bucket.width for _, bucket in buckets) * policy["reuse_csa_layers"]
            )
        return True

    def serving_status(self) -> dict:
        """Host counters for a completed serving run; no device reads or syncs."""
        selected = self.selected_steps
        return {
            "policy": self.policy,
            "profile": self.profile.name,
            "decode_attempts": self.decode_attempts,
            "selected_steps": selected,
            "graph_replays": self.graph_replays,
            "eager_forwards": self.eager_forwards,
            "fallback_steps": self.fallback_steps,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "mono_hit_rate": (
                selected / self.decode_attempts if self.decode_attempts else 0.0
            ),
            "graph_replay_rate": self.graph_replays / selected if selected else 0.0,
            "last_refusal": self.refusal,
            "publication": {
                "full_consensus": self.full_publications,
                "batch_fast": sum(self.fast_publications_by_width.values()),
                "c1_fast": self.fast_publications_by_width.get(1, 0),
                "fast_by_width": {
                    str(width): count
                    for width, count in sorted(self.fast_publications_by_width.items())
                },
            },
            "owner_generation": self.owner_generation,
            "graph_generations": sorted(
                [key[0], key[1], generation]
                for key, generation in self.graph_generations.items()
            ),
            "index_reuse": self.index_reuse_status(),
        }

    def index_reuse_status(self) -> dict:
        """Static policy plus serving-generation refresh/reuse hit counters."""
        policy = (
            self.runner.index_policy()
            if self.runner is not None
            else {
                "refresh_csa_layers": 0,
                "reuse_csa_layers": 0,
                "refresh_layer_ids": [],
                "reuse_layer_ids": [],
                "suppressed_stages_per_reuse_hit": [],
            }
        )
        return {
            **policy,
            "microbucket_stacks": self.microbucket_stacks,
            "refresh_layer_executions": self.index_refresh_executions,
            "reuse_layer_executions": self.index_reuse_executions,
            "index_task_suppression_hits": self.index_reuse_executions,
            "index_suppressed_live_rows": self.index_suppressed_rows,
        }

    def finish_capture_publication(self) -> None:
        """Retire the host authorization retained across warmup and capture."""
        self.armed = False
        self.capture_publication = False
        self.publication_generation = 0
        self.publication_slot_generations = ()
        if self.adapter is not None:
            self.adapter.clear_publication()

    def greedy_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Finish the token boundary without materializing full-vocab logits.

        The layer stack returns the pre-head mHC residual. This applies the same
        terminal mHC reduction and RMSNorm as ``compute_logits``, then performs
        a TP-sharded argmax that exchanges only one ``(value, token)`` pair per
        rank and row. ``token_out`` has a stable address, so this terminal can be
        recorded in the Mono CUDA graph.
        """
        if self.token_out is None or hidden_states.ndim != 3:
            raise MonoUnsupported("Mono greedy terminal is not prepared")
        rows = hidden_states.shape[0]
        if not 1 <= rows <= self.token_out.numel():
            raise MonoUnsupported("Mono greedy terminal rows exceed the prepared width")
        inner = self.model.model
        x = inner.head.hc_head(
            hidden_states,
            inner.hc_head_fn,
            inner.hc_head_scale,
            inner.hc_head_base,
        )
        x = inner.norm(x)
        return inner.head.compute_argmax_token(x, out=self.token_out[:rows])

    def forward(self, input_ids, positions):
        if not self.armed:
            raise MonoUnsupported("Mono input metadata has not been published")
        try:
            if (
                not self.owns_live_session()
                or self.publication_generation != self.owner_generation
                or any(
                    self.slot_generations.get(slot, 0) != generation
                    for slot, generation in self.publication_slot_generations
                )
            ):
                raise MonoUnsupported(
                    "published Mono metadata belongs to a stale owner/slot generation"
                )
            rows = self.adapter.published_rows
            if (
                not rows
                or input_ids.ndim != 1
                or positions.ndim != 1
                or input_ids.numel() != positions.numel()
                or not rows <= input_ids.numel() <= self.adapter.max_rows
                or self.output is None
            ):
                raise MonoUnsupported(
                    "published exact-row metadata does not match model input"
                )
            out = self.output[: input_ids.numel()]
            if input_ids.numel() > rows:
                out[rows:].zero_()
            for offset, bucket in self.adapter.published_buckets:
                end = offset + bucket.width
                self.runner.forward_tokens(
                    bucket.tokens,
                    bucket.positions,
                    bucket.layers,
                    tokens_per_seq=self.profile.tokens_per_seq,
                    x_out=out[offset:end],
                    lane=bucket.ordinal,
                )
            return out
        finally:
            self.armed = False  # authorization is one forward, never a stale model hook
            if not self.capture_publication:
                self.publication_generation = 0
                self.publication_slot_generations = ()
                if self.adapter is not None:
                    self.adapter.clear_publication()

    def invalidate_weights(self):
        """Called before weight mutation, after readers retire; no captured retry."""
        torch.cuda.synchronize()
        engine = self.builder.model_runner
        engine.graphs.clear()
        engine.graph_pool = None
        if hasattr(engine, "graph_logits"):
            engine.graph_logits.clear()
        self.close()
        self.weights_ready = False
        self.requires_cache_rebind = True
        self.refusal = (
            "weight update in progress; Mono must be prepared after the final bucket"
        )

    def weights_updated(self, *, cache_reset):
        """Keep the updated model on native until an actual cache rebind.

        ``clear_kv_cache`` can reset bytes without replacing the V4 cache and
        request-state owners.  Treating that notification as a complete rebind
        used to recapture against an ambiguous lifecycle boundary.  The next
        real ``allocate_kv_cache`` calls ``prepare(cache_rebound=True)`` and is
        the only path that may install Mono again.
        """
        self.weights_ready = True
        self.requires_cache_rebind = True
        self.refusal = (
            "updated weights require an explicit cache/state rebind before "
            "Mono prepare"
        )
