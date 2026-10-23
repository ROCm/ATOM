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

from atom.models.deepseek_v4.mono.binding import (
    LayerStep,
    input_identity,
    tensor_owners,
)
from atom.models.deepseek_v4.mono.build import kernel_kwargs, launch_plan
from atom.models.deepseek_v4.mono.config import (
    LAUNCH_PHASES,
    MAX_LAYERS_PER_STEP,
    MoeMode,
    _variant_key,
    as_moe_mode,
    shape_dims,
    validate_shard,
)
from atom.models.deepseek_v4.mono.runtime import require_resident_grid
from atom.models.deepseek_v4.mono.state import Dsv4Variant
from atom.models.deepseek_v4.mono.weights import pack_layer_weights
from atom.mono.runtime.compile import compile_only, launcher_resources
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


@dataclass(frozen=True)
class LaneGraph:
    """Backbone-only graph with lane-local outputs, never packed-row offsets."""

    graph: object
    build: RowsBuild
    inputs: tuple
    output: torch.Tensor
    aux: dict[int, torch.Tensor]
    buffers: tuple
    owners: tuple
    runtime: tuple


def _lane_runtime_storage(build):
    """Addresses consumed by captured epoch advance and peer-wait kernels."""
    return tuple(
        (
            getattr(variant, "scratch", None),
            getattr(variant, "step", None),
            getattr(variant, "hang", None),
            getattr(variant, "peers", None),
            getattr(variant, "sym", None),
            getattr(getattr(variant, "peer_buffer", None), "storage", None),
            getattr(getattr(variant, "peer_buffer", None), "local_address", None),
        )
        for variant in build.variants
    )


class V4MonoDecodeRunner:
    """A target layer stack with prepared one/two/three-phase launches.

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
        attention_mode="causal",
    ):
        import torch.distributed as dist

        self.group = dist.group.WORLD if npes > 1 and group is None else group
        self.rank, self.npes = rank, npes
        self.weights = tuple(weights)
        self.moe_mode = as_moe_mode(moe_mode)
        self.timeline = timeline
        if attention_mode not in ("causal", "draft_block"):
            raise MonoUnsupported(f"unknown attention mode {attention_mode!r}")
        self.attention_mode = attention_mode
        self._lane_builds = {}
        self._lane_graphs = {}
        self._refused = set()
        self._closed = False
        self._aux_outputs = {}
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
                if self.attention_mode == "draft_block" and (
                    w.cfg.compress_ratio or w.cfg.indexed
                ):
                    raise MonoUnsupported("draft runner requires uncompressed stages")
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
    def from_model(
        cls,
        model,
        max_model_len: int,
        *,
        rank=0,
        npes=8,
        group=None,
        compressor_ring_extra: int | None = None,
    ):
        """Prepare-time binding of the real 61-layer target, without installing it."""
        import torch.distributed as dist

        from atom.models.deepseek_v4.mono.config import V4Config
        from atom.models.deepseek_v4.mono.weights import bind_layer

        group = dist.group.WORLD if npes > 1 and group is None else group
        bindings = []
        if compressor_ring_extra is not None and (
            not isinstance(compressor_ring_extra, int)
            or isinstance(compressor_ring_extra, bool)
            or compressor_ring_extra < 0
        ):
            raise MonoUnsupported(
                "physical compressor ring slack must be a nonnegative integer"
            )

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
                    compressor_ring_rows=(
                        None
                        if compressor_ring_extra is None
                        else (8 if indexed else block.attn.compress_ratio)
                        + compressor_ring_extra
                    ),
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

    def prepare(self, rows: int, *, tokens_per_seq: int = 1, lanes: int = 1) -> bool:
        """Prepare an exact width outside capture; failures agree before IPC.

        S1 uses the legacy whole-layer ONE launch. S2/S4/S8 use attention -> MoE
        under one owner/arena/tag with one advance after both launches. The
        complete S7/q7 candidate splits attention into pre/core, retaining one
        advance after all three phases. SGPR
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
            if rows == 14 and tokens_per_seq != 7:
                raise MonoUnsupported("candidate S14 requires two complete q7 requests")
            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            if device.multi_processor_count != 256:
                raise MonoUnsupported("the PR grid requires 256 gfx950 CUs")
            plan = launch_plan(
                rows,
                tokens_per_seq=tokens_per_seq,
                attention_mode=self.attention_mode,
            )
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
                    attention_mode=self.attention_mode,
                    **plan,
                )
                cfgs.setdefault(vk, cfg)
            for vk, cfg in cfgs.items():
                refresh_modes = (True, False) if cfg.indexed else (True,)
                for refresh in refresh_modes:
                    cache_key = (vk, refresh)
                    launches = Dsv4Variant._cache.get(cache_key)
                    if launches is None:
                        phases = LAUNCH_PHASES[plan["launch_shape"]]
                        kwargs = kernel_kwargs(
                            cfg,
                            tokens_per_seq=tokens_per_seq,
                            timeline=self.timeline,
                            moe_mode=self.moe_mode,
                            index_topk_refresh=refresh,
                            attention_mode=self.attention_mode,
                            **{k: v for k, v in plan.items() if k != "launch_shape"},
                        )
                        launches = tuple(
                            build_dsv4_kernel(
                                rows,
                                cfg.heads,
                                self.npes,
                                cfg.window,
                                phase=phase,
                                **kwargs,
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
                        attention_mode=self.attention_mode,
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
                            attention_mode=self.attention_mode,
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
                                timeline=self.timeline,
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
        aux_row_offset=0,
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
        validated = self.preflight(
            h,
            steps,
            tokens=tokens,
            tokens_per_seq=tokens_per_seq,
            x_out=x_out,
            lane=lane,
        )
        aux = self._aux_views(h, build, steps, x_out, aux_row_offset)
        return self._forward_resolved(h, steps, build, validated, aux, tokens)

    def forward_buckets(self, h, buckets, *, x_out):
        """Resolve an entire packed verify step before any target/cache write.

        Contracts remain local to this invocation. No public bypass accepts a
        caller's previous validation, and no resolved contract survives a step.
        Each bucket owns its exact rows/q arena lane; request boundaries and
        live token metadata are supplied by the cache adapter's publication.
        """
        if self._closed:
            raise RuntimeError("target runner is closed")
        buckets = tuple(buckets)
        if (
            x_out.shape != h.shape
            or x_out.dtype != h.dtype
            or x_out.device != h.device
            or not x_out.is_contiguous()
            or torch._C._overlaps(h, x_out)
        ):
            raise MonoUnsupported("packed target output geometry/dtype/device mismatch")
        at = 0
        resolved = []
        aux_inputs = []
        graph_storage = []
        lanes = set()
        # All lanes consume the same weight owners. Check them once for this
        # complete step, before resolving any lane and before the first write.
        # No binding result is retained across calls.
        self._assert_bindings_current()
        for offset, bucket in buckets:
            end = offset + bucket.width
            q = bucket.tokens_per_seq
            if offset != at or bucket.width <= 0 or end > h.shape[0]:
                raise MonoUnsupported("verify buckets must exactly partition live rows")
            key = (bucket.width, q, bucket.ordinal)
            if key in lanes:
                raise MonoUnsupported("verify buckets cannot reuse an arena lane")
            lanes.add(key)
            build = self._build_for((bucket.width, q), bucket.ordinal)
            current, out = h[offset:end], x_out[offset:end]
            validated = self._preflight_layers(
                current,
                bucket.layers,
                tokens=bucket.tokens,
                tokens_per_seq=q,
                x_out=out,
                lane=bucket.ordinal,
            )
            aux_inputs.append((current, build, bucket.layers, out, offset))
            graph = getattr(self, "_lane_graphs", {}).get(key)
            if graph is not None:
                self._require_lane_graph(graph, build, bucket.layers, bucket.tokens)
                graph_storage.append((graph.output, graph.aux))
            resolved.append((offset, current, bucket, build, validated, graph))
            at = end
        if at != h.shape[0] or not buckets:
            raise MonoUnsupported("verify buckets must exactly partition live rows")
        aux_views = self._aux_views_many(
            aux_inputs, protected_inputs=(h, x_out, graph_storage)
        )
        # An engine FULL capture records kernels directly. Do not capture a
        # child graph launch: its backend support and retirement semantics are
        # distinct from ordinary lane replay. Keep the same validated buffers
        # and operation topology, with no runtime graph creation.
        outer_capture = (
            h.device.type == "cuda"
            and bool(graph_storage)
            and (torch.cuda.is_current_stream_capturing())
        )
        # Every lane's layer bindings and aux destinations were checked before
        # this loop. Execution consumes them once, without another preflight.
        for (offset, current, bucket, build, validated, graph), aux in zip(
            resolved, aux_views
        ):
            if graph is None or outer_capture:
                self._forward_resolved(
                    current, bucket.layers, build, validated, aux, bucket.tokens
                )
            else:
                # The graph retains only lane-local addresses. Packed offsets
                # and aux destinations belong to this validated publication.
                build.hidden[0].copy_(current)
                graph.graph.replay()
                x_out[offset : offset + bucket.width].copy_(graph.output)
                for layer_id, destination in aux.items():
                    destination.copy_(graph.aux[layer_id])
        return x_out

    @staticmethod
    def _graph_inputs(steps, tokens):
        return input_identity((tokens, tuple(vars(step) for step in steps)))

    def _require_lane_graph(self, graph, build, steps, tokens):
        if graph.build is not build or graph.inputs != self._graph_inputs(
            steps, tokens
        ):
            raise MonoUnsupported(
                "lane graph metadata/cache binding changed; retire and recapture"
            )
        if graph.aux.keys() != self._aux_outputs.keys():
            raise MonoUnsupported("lane graph aux contract changed")
        if graph.buffers != input_identity((build.hidden, graph.output, graph.aux)):
            raise MonoUnsupported("lane graph captured storage changed")
        if graph.runtime != input_identity(_lane_runtime_storage(build)):
            raise MonoUnsupported("lane graph epoch/peer storage changed")

    def capture_lane(self, steps, *, tokens, rows, tokens_per_seq=1, lane=0):
        """Capture a prepared, already-warmed backbone on valid published state.

        This executes the stack once, including cache writes and epoch advance.
        The lifecycle owner must snapshot/restore request state if this is a
        warmup, and coordinate all ranks before calling. Never call during live
        layout selection or inside an engine graph capture. Embedding, terminal
        and draft graphs remain the session/engine's responsibility.
        """
        key = (rows, tokens_per_seq, lane)
        graphs = getattr(self, "_lane_graphs", {})
        prepared = []

        def prepare_capture():
            if (
                self._closed
                or key in graphs
                or torch.cuda.is_current_stream_capturing()
            ):
                raise MonoUnsupported(
                    "lane capture requires an open uncaptured prepared lane"
                )
            build = self._build_for((rows, tokens_per_seq), lane)
            h = build.hidden[0]
            if h.device.type != "cuda":
                raise MonoUnsupported("lane capture requires CUDA storage")
            output = torch.empty_like(h)
            validated = self.preflight(
                h,
                steps,
                tokens=tokens,
                tokens_per_seq=tokens_per_seq,
                x_out=output,
                lane=lane,
            )
            self._aux_views(h, build, steps, output, 0)
            aux = {
                layer_id: torch.empty(
                    (rows, h.shape[-1]), device=h.device, dtype=h.dtype
                )
                for layer_id in self._aux_outputs
            }
            prepared.append(
                (build, output, validated, aux, self._graph_inputs(steps, tokens))
            )

        bind_agreed(prepare_capture, self.group)
        build, output, validated, aux, inputs = prepared[0]
        captured = []

        def capture():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._forward_resolved(
                    build.hidden[0], steps, build, validated, aux, tokens
                )
            captured.append(graph)

        # Never publish a graph on ranks whose peers failed capture. A failure
        # can still have advanced request state; the lifecycle owner restores
        # its warmup snapshot before retrying or retiring the runner.
        bind_agreed(capture, self.group)
        graphs[key] = LaneGraph(
            captured[0],
            build,
            inputs,
            output,
            aux,
            input_identity((build.hidden, output, aux)),
            # CUDA graphs retain device pointers, not Python tensor owners.
            # Keep captured metadata/cache/state alive until graph retirement;
            # otherwise an allocator could recycle a stale pointer before the
            # publication identity check observes a replacement.
            tensor_owners(
                (
                    tokens,
                    tuple(vars(step) for step in steps),
                    validated,
                    _lane_runtime_storage(build),
                )
            ),
            input_identity(_lane_runtime_storage(build)),
        )
        self._lane_graphs = graphs

    def _forward_resolved(self, h, steps, build, validated, aux, tokens):
        """Execute only the call-local resolution made by a public forward."""
        # Preflight resolved the alias-free ping-pong outputs for this call.
        for i, (layer, step) in enumerate(zip(build.layers, steps)):
            h = layer._forward_validated(
                h,
                step.positions,
                step.dest_rows,
                step.indices,
                step.cos,
                step.sin,
                validated=validated[i],
                layer=i,
                tokens=tokens,
                env_rows=step.env_rows,
            )
            if i in aux:
                # Native DSpark captures the settled post-layer mHC residual
                # averaged over connections. Write before ping-pong reuse;
                # out= retains drafter ownership and allocates no result tensor.
                torch.mean(h, dim=1, out=aux[i])
        for variant in build.variants:
            variant.advance_step()
        return h

    def bind_aux_outputs(self, layer_ids, buffers):
        """Bind drafter-owned post-layer captures before any graph is built.

        Embedding/input taps use a different semantic contract and are not
        accepted here. Rebinding a captured runner would leave graph writes
        pointing at retired storage, so callers must create a fresh runner.
        """
        if self._closed or self._lane_builds:
            raise MonoUnsupported("aux outputs must bind before width preparation")
        layer_ids, buffers = tuple(layer_ids), tuple(buffers)
        if len(layer_ids) != len(buffers):
            raise MonoUnsupported("one aux buffer per target layer is required")
        if any(type(i) is not int or not 0 <= i < len(self.weights) for i in layer_ids):
            raise MonoUnsupported("aux capture requires real target output layer IDs")
        if tuple(sorted(set(layer_ids))) != layer_ids:
            raise MonoUnsupported("aux capture layer IDs must be strictly increasing")
        self._aux_outputs = dict(zip(layer_ids, buffers))

    def _aux_views(self, h, build, steps, x_out, offset):
        """Check every destination before the first layer/cache write."""
        return self._aux_views_many(((h, build, steps, x_out, offset),))[0]

    def _aux_views_many(self, entries, *, protected_inputs=()):
        """Resolve aux for all lanes under one call-local alias check.

        Check full destination buffers against every lane, including later
        layers/lanes, before execution. Only identical tensor objects are
        deduplicated; views and mutations are checked again on every call.
        """
        entries = tuple(entries)
        outputs = getattr(self, "_aux_outputs", {})
        if not outputs:
            return tuple({} for _ in entries)
        protected = []
        seen_tensors = set()

        def tensors(value):
            if isinstance(value, torch.Tensor):
                # Shared metadata/pool tensors recur across layers. Deduplicate
                # only the identical live object within this call: distinct
                # views (including offset/empty aliases) still need checking.
                identity = id(value)
                if identity not in seen_tensors:
                    seen_tensors.add(identity)
                    protected.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    tensors(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    tensors(item)

        tensors(protected_inputs)
        for h, build, steps, x_out, offset in entries:
            if type(offset) is not int or offset < 0:
                raise MonoUnsupported("aux row offset must be a nonnegative integer")
            tensors((h, *build.hidden, x_out))
            for step in steps:
                for value in vars(step).values():
                    tensors(value)
        for buffer in outputs.values():
            if any(
                (
                    not isinstance(buffer, torch.Tensor)
                    or buffer.ndim != 2
                    or buffer.shape[1] != h.shape[2]
                    or buffer.shape[0] < offset + h.shape[0]
                    or buffer.device != h.device
                    or buffer.dtype != h.dtype
                    or not buffer.is_contiguous()
                    or buffer.requires_grad
                )
                for h, _, _, _, offset in entries
            ):
                raise MonoUnsupported(
                    "aux buffer geometry/dtype/device/capacity mismatch"
                )
            if any(
                value is not None and torch._C._overlaps(buffer, value)
                for value in protected
            ):
                raise MonoUnsupported(
                    "aux buffer aliases target input/output/cache storage"
                )
            tensors(buffer)
        return tuple(
            {
                layer_id: buffer[offset : offset + h.shape[0]]
                for layer_id, buffer in outputs.items()
            }
            for h, _, _, _, offset in entries
        )

    def preflight(self, h, steps, *, tokens=None, tokens_per_seq=1, x_out=None, lane=0):
        """Resolve all layers before the first write; caller-local, never cached."""
        self._assert_bindings_current()
        return self._preflight_layers(
            h,
            steps,
            tokens=tokens,
            tokens_per_seq=tokens_per_seq,
            x_out=x_out,
            lane=lane,
        )

    def lane_execution_consensus_tensor(self, buckets, *, capacity):
        """Owned CPU ABI for the exact eager/graph sequence on every TP rank.

        Metadata equality alone does not prove execution equality: one rank
        may have a captured lane while another would run its eager stack.
        Encode graph membership after publication preflight, before launch.
        Capacity is deployment-fixed so different bucket counts can be agreed
        safely with the same tensor collective shape.
        """
        buckets = tuple(buckets)
        if type(capacity) is not int or capacity < 1 or len(buckets) > capacity:
            raise MonoUnsupported("lane execution exceeds its fixed consensus ABI")
        payload = torch.full((1 + capacity * 5,), -1, dtype=torch.int64)
        payload[0] = len(buckets)
        graphs = getattr(self, "_lane_graphs", {})
        for index, (offset, bucket) in enumerate(buckets):
            key = (bucket.width, bucket.tokens_per_seq, bucket.ordinal)
            base = 1 + index * 5
            payload[base : base + 5] = torch.tensor(
                (offset, *key, int(key in graphs)), dtype=torch.int64
            )
        return payload

    def preflight_publication(self, buckets, *, x_out):
        """Validate all published lanes under one call-local weight check.

        No resolved contract escapes this call. Forward still validates its
        actual embedding/output views after publication consensus; this only
        deduplicates shared owners within the publication, not across steps.
        """
        self._assert_bindings_current()
        self._preflight_publication_layers(buckets, x_out=x_out)

    def _preflight_publication_layers(self, buckets, *, x_out):
        """Resolve publication after the caller's same-transaction weight check.

        Private serving split, not a cached binding credential. Every layer,
        graph pointer, output and aux alias is still checked in this call.
        The session must check weights before its TP admission verdict; no
        user code or weight replacement may intervene before this resolution.
        """
        aux_inputs = []
        graph_storage = []
        for offset, bucket in buckets:
            q = bucket.tokens_per_seq
            build = self._build_for((bucket.width, q), bucket.ordinal)
            self._preflight_layers(
                build.hidden[0],
                bucket.layers,
                tokens=bucket.tokens,
                tokens_per_seq=q,
                lane=bucket.ordinal,
            )
            graph = getattr(self, "_lane_graphs", {}).get(
                (bucket.width, q, bucket.ordinal)
            )
            if graph is not None:
                # Publication runs under session TP consensus, before any
                # rank enters a captured peer-wait kernel. Checking only in
                # forward could let a late local refusal strand other ranks.
                self._require_lane_graph(graph, build, bucket.layers, bucket.tokens)
                graph_storage.append((graph.output, graph.aux))
            aux_inputs.append((build.hidden[0], build, bucket.layers, x_out, offset))
        self._aux_views_many(aux_inputs, protected_inputs=(x_out, graph_storage))

    def _assert_bindings_current(self):
        for binding in getattr(self, "bindings", ()):
            binding.assert_current()

    def _preflight_layers(
        self, h, steps, *, tokens=None, tokens_per_seq=1, x_out=None, lane=0
    ):
        """Private lane resolution after this call's shared-owner check."""
        build = self._build_for((h.shape[0], tokens_per_seq), lane)
        if len(steps) != len(build.layers):
            raise MonoUnsupported("one real step per layer is required")
        current = h
        validated = []
        first = int(h.data_ptr() == build.hidden[0].data_ptr())
        for i, (layer, step) in enumerate(zip(build.layers, steps)):
            out = (
                x_out
                if i == len(build.layers) - 1 and x_out is not None
                else build.hidden[(i + first) % 2]
            )
            if out.data_ptr() == current.data_ptr():
                raise MonoUnsupported("layer output must not alias its input")
            contract = layer.validate_inputs(
                current,
                step.positions,
                step.kv_cache,
                step.dest_rows,
                step.indices,
                step.cos,
                step.sin,
                x_out=out,
                layer=i,
                tokens=tokens,
                block_tables=step.block_tables,
                state=step.state,
            )
            validated.append(contract)
            current = out
        return tuple(validated)

    def forward_tokens(
        self,
        input_ids,
        positions,
        steps,
        *,
        tokens_per_seq=1,
        x_out=None,
        lane=0,
        aux_row_offset=0,
    ):
        """Token→settled residual. The ordinary model retains head and sampling."""
        from atom.models.deepseek_v4.mono.embedding import mhc_input

        h = mhc_input(self.model, input_ids)
        return self.forward(
            h,
            steps,
            tokens=input_ids,
            tokens_per_seq=tokens_per_seq,
            x_out=x_out,
            lane=lane,
            aux_row_offset=aux_row_offset,
        )

    def close(self):
        """Collective, once every replay has retired; never mutate rolling state."""
        if self._closed:
            return
        # The owner retires device work before close. Drop graph references
        # before their captured scratch/peer storage is released.
        getattr(self, "_lane_graphs", {}).clear()
        for builds in self._lane_builds.values():
            for build in builds:
                for variant in build.variants:
                    variant.close()
        self._lane_builds.clear()
        self._aux_outputs = {}
        self._closed = True
