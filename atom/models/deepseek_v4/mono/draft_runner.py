# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent V4 draft backbone execution; admission/cache publication is external.

This runner never calls native block_backbone as a fallback. The caller must
bind actual draft weights and publish stage-private LayerSteps after target
context writes. Markov/confidence sampling remains with the original model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import pairwise

import torch

from atom.models.deepseek_v4.mono.binding import input_identity, tensor_owners
from atom.models.deepseek_v4.mono.config import V4Config
from atom.models.deepseek_v4.mono.runner import (
    V4MonoDecodeRunner,
    _lane_runtime_storage,
)
from atom.models.deepseek_v4.mono.weights import BoundLayer, _signature, bind_layer
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed


@dataclass(frozen=True)
class _ResolvedDraftRequest:
    """Invocation-owned lane bindings; not persistent replay authorization."""

    start: int
    end: int
    hidden: torch.Tensor
    steps: tuple
    build: object
    validated: tuple
    aux: dict
    tokens: torch.Tensor


@dataclass(frozen=True)
class _DraftBlockCapture:
    """Owned captured storage, not authorization to replay a live request."""

    graph: object
    inputs: tuple
    outputs: tuple
    resolved: tuple
    identity: tuple
    owners: tuple
    source_steps: tuple = ()
    source_identity: tuple = ()
    sampled: bool = False
    epilogue: object = None


def _draft_config_storage(cfg):
    """Freeze the explicit V4 numerical/geometry contract, not tensor contents."""
    if not isinstance(cfg, V4Config):
        raise MonoUnsupported("draft capture requires a V4 layer config")
    return tuple((field.name, getattr(cfg, field.name)) for field in fields(cfg))


def _draft_execution_storage(request):
    """Actual launch bindings in addition to the published LayerStep owners."""
    layers = getattr(request.build, "layers", ())
    if len(layers) != len(request.validated):
        raise MonoUnsupported("draft capture layer contracts changed")
    contracts = []
    for layer, validated in zip(layers, request.validated):
        if len(validated) != 8:
            raise MonoUnsupported("draft capture layer contract is incomplete")
        cfg, *bindings = validated
        contracts.append(
            (
                id(layer),
                _draft_config_storage(cfg),
                tuple(bindings),
                _draft_config_storage(layer.W.cfg),
                layer.packed,
                layer.hc_sb,
                layer.scratch,
                layer.sym,
                layer.peers,
                layer.timeline,
                layer.step,
                id(layer.variant),
                layer.variant.hang,
                layer.rank,
                tuple(id(launch) for launch in layer.launches),
            )
        )
    return tuple(contracts)


def _draft_capture_storage(inputs, outputs, resolved):
    return (
        inputs,
        outputs,
        tuple(
            (
                request.start,
                request.end,
                request.hidden,
                request.tokens,
                tuple(vars(step) for step in request.steps),
                request.build.hidden,
                request.aux,
                _lane_runtime_storage(request.build),
                _draft_execution_storage(request),
            )
            for request in resolved
        ),
    )


def _draft_scalar_contract(model, stages):
    """Runtime metadata baked into the draft embedding/compiled stack."""

    def scalars(owner, names):
        return tuple(
            (name, type(value), value)
            for name in names
            for value in (getattr(owner, name, None),)
        )

    return (
        (type(model.noise_token_id), model.noise_token_id),
        (type(model.hc_mult), model.hc_mult),
        tuple(
            tuple(
                (type(value), value)
                for value in (
                    stage.stage_id,
                    stage.num_stages,
                    stage.window_size,
                    stage.attn.compress_ratio,
                    stage.attn.kv_fp8,
                )
            )
            for stage in stages
        ),
        tuple(
            (
                scalars(
                    stage,
                    (
                        "hc_mult",
                        "hc_sinkhorn_iters",
                        "hc_eps",
                        "norm_eps",
                        "HC_POST_MULT",
                    ),
                ),
                scalars(
                    stage.attn,
                    (
                        "eps",
                        "softmax_scale",
                        "head_dim",
                        "n_local_heads",
                        "n_local_groups",
                    ),
                ),
                scalars(
                    getattr(stage, "ffn", None),
                    ("routed_scaling_factor", "swiglu_limit"),
                ),
            )
            for stage in stages
        ),
        scalars(stages[-1].norm, ("eps", "dim")),
        scalars(model.head, ("norm_eps", "hc_eps")),
        scalars(
            getattr(stages[-1], "markov_head", None),
            ("vocab_size", "rank", "fused_sample"),
        ),
    )


def _draft_output_module_tree(model, stages):
    """Retain child owners as well as root identities across post-load reloads."""
    return tuple(
        tuple(module.named_modules())
        for module in (
            model.embed,
            model.head,
            stages[-1].norm,
            getattr(stages[-1], "markov_head", None),
            getattr(stages[-1], "confidence_head", None),
        )
        if module is not None
    )


def _same_module_tree(bound, current):
    """Compare module names and identities, never module equality or weights."""
    return len(bound) == len(current) and all(
        len(old) == len(new)
        and all(
            old_name == new_name and old_owner is new_owner
            for (old_name, old_owner), (new_name, new_owner) in zip(old, new)
        )
        for old, new in zip(bound, current)
    )


class V4MonoDraftRunner(V4MonoDecodeRunner):
    """A separately owned block-bidirectional stack, not a target session.

    Preparation, full-stack preflight, peer ownership and epoch advancement
    reuse the existing implementation. Cache lifecycle and serving admission
    are not inferred from successfully constructing this runner.
    """

    def __init__(self, weights, *, draft_model, **kwargs):
        if "attention_mode" in kwargs:
            raise MonoUnsupported("draft runner owns its block attention semantics")
        weights = tuple(weights)
        stages = tuple(draft_model.mtp)
        if not weights or len(weights) != len(stages):
            raise MonoUnsupported("one weight binding per draft stage is required")
        if draft_model.embed is None or draft_model.head is None:
            raise MonoUnsupported("draft embed/head must be shared before binding")
        last = stages[-1]
        if any(
            not hasattr(last, name)
            for name in (
                "hc_head_fn",
                "hc_head_scale",
                "hc_head_base",
                "norm",
            )
        ):
            raise MonoUnsupported("draft final reduction weights are missing")
        super().__init__(weights, attention_mode="draft_block", **kwargs)
        self.draft_model = draft_model
        self._draft_width = None
        self._clear_request_storage()

    def _clear_request_storage(self):
        """Drop fixed request buffers only at construction or after retirement."""
        self._request_destinations = None
        self._request_completion = None
        self._request_output = None
        self._request_output_completion = None
        self._request_tokens = None
        self._request_hidden = None

    @classmethod
    def from_model(cls, model, max_model_len, *, num_draft, rank=0, npes=8, group=None):
        """Bind loaded draft stages without installing an engine dispatch.

        ``model`` is the undecorated wrapper's inner owner, not the target.
        Actual layer packing uses the same strict post-load binder; draft
        geometry/quantization mismatches are not silently normalized away.
        """
        import torch.distributed as dist

        group = dist.group.WORLD if npes > 1 and group is None else group
        if (
            not model.mtp
            or type(num_draft) is not int
            or not 1 <= num_draft <= model.mtp[0].window_size
        ):
            raise MonoUnsupported("draft width must be within the native window")
        bindings, output_bindings, scalar_bindings, module_trees = [], [], [], []
        stages = tuple(model.mtp)

        def bind():
            from atom.models.deepseek_v4 import _V4_USE_REF_QUANT

            if _V4_USE_REF_QUANT:
                raise MonoUnsupported(
                    "draft requires ordinary reference quantization off"
                )
            if not stages or model.embed is None or model.head is None:
                raise MonoUnsupported(
                    "loaded draft stages and shared embed/head required"
                )
            scalar_contract = _draft_scalar_contract(model, stages)
            output_tree = _draft_output_module_tree(model, stages)
            if type(model.noise_token_id) is not int or model.noise_token_id < 0:
                raise MonoUnsupported("draft noise token must be a nonnegative integer")
            for ordinal, block in enumerate(stages):
                if block.stage_id != ordinal or block.num_stages != len(stages):
                    raise MonoUnsupported("draft stage order/count mismatch")
                if block.attn.compress_ratio != 0:
                    raise MonoUnsupported("compressed draft stage is not yet qualified")
                cfg = V4Config(
                    compress_ratio=0,
                    max_seq=max_model_len,
                    kv_fp8=block.attn.kv_fp8,
                    # Native BF16 draft fake-quantizes NoPE KV after QK/RoPE;
                    # unlike ordinary BF16 target, it retains the QAT seam.
                    # Exact BF16 rounding still needs device qualification.
                    bf16_kv_roundtrip=not block.attn.kv_fp8,
                    # The native FP8 two-buffer path always group-quantizes
                    # Q, irrespective of qkn's BF16-path quant_q argument.
                    # Keep BF16-cache drafts unquantized; KV_FP8 gates this
                    # option inside the attention kernel.
                    quant_query_fp8=block.attn.kv_fp8,
                    draft_block_width=num_draft,
                    shared_bf16_block128=True,
                    routed_down_bf16=True,
                    compressor_projection_bf16=True,
                    attention_fp8_bf16=True,
                    indexer_hadamard=False,
                )
                bindings.append(bind_layer(block, cfg))
            if _draft_scalar_contract(model, stages) != scalar_contract:
                raise MonoUnsupported("draft scalars changed during binding")
            scalar_bindings.append(scalar_contract)
            # Keep final reduction/norm and shared embedding/head owners alive;
            # replacement or in-place reload invalidates before a cache write.
            last = stages[-1]
            sources = []
            for name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
                tensor = getattr(last, name)
                if not isinstance(tensor, torch.Tensor):
                    raise MonoUnsupported(f"draft final {name} must be a tensor")
                sources.append((last, name, tensor, _signature(tensor)))
            for tree in output_tree:
                for _, child in tree:
                    tensors = (
                        *child.named_parameters(recurse=False),
                        *child.named_buffers(recurse=False),
                    )
                    for name, tensor in tensors:
                        sources.append((child, name, tensor, _signature(tensor)))
            current_tree = _draft_output_module_tree(model, stages)
            if not _same_module_tree(output_tree, current_tree):
                raise MonoUnsupported("draft output module tree changed during binding")
            module_trees.append(output_tree)
            output_bindings.append(
                BoundLayer(
                    int(last.layer_id),
                    bindings[-1].weights,
                    {},
                    model,
                    tuple(sources),
                )
            )

        bind_agreed(bind, group)
        runner = cls(
            [binding.weights for binding in bindings],
            draft_model=model,
            packed=[binding.packed for binding in bindings],
            rank=rank,
            npes=npes,
            group=group,
        )
        runner.bindings = tuple(bindings)
        runner._draft_output_bindings = tuple(output_bindings)
        runner._bound_draft_stages = stages
        runner._bound_draft_modules = (model.embed, model.head, stages[-1].norm)
        runner._bound_draft_output_tree = module_trees[0]
        runner._bound_draft_scalars = scalar_bindings[0]
        return runner

    def _assert_bindings_current(self):
        super()._assert_bindings_current()
        if hasattr(self, "_bound_draft_stages"):
            model = self.draft_model
            stages = tuple(model.mtp)
            if len(stages) != len(self._bound_draft_stages) or any(
                old is not new for old, new in zip(self._bound_draft_stages, stages)
            ):
                raise MonoUnsupported("draft stage replacement invalidated binding")
            if _draft_scalar_contract(model, stages) != self._bound_draft_scalars:
                raise MonoUnsupported(
                    "draft runtime scalar contract invalidated binding"
                )
            modules = (model.embed, model.head, stages[-1].norm)
            if any(
                old is not new for old, new in zip(self._bound_draft_modules, modules)
            ):
                raise MonoUnsupported(
                    "draft output module replacement invalidated binding"
                )
            current_tree = _draft_output_module_tree(model, stages)
            if not _same_module_tree(self._bound_draft_output_tree, current_tree):
                raise MonoUnsupported(
                    "draft output child module replacement invalidated binding"
                )
            for binding in self._draft_output_bindings:
                binding.assert_current()

    def prepare(self, rows, *, tokens_per_seq=1, lanes=1):
        self._assert_bindings_current()
        if self._draft_width is not None and tokens_per_seq != self._draft_width:
            raise MonoUnsupported("draft width cannot change after preparation")
        if any(w.cfg.draft_block_width != tokens_per_seq for w in self.weights):
            raise MonoUnsupported("draft key capacity must match the prepared width")
        ready = super().prepare(rows, tokens_per_seq=tokens_per_seq, lanes=lanes)
        if ready:
            self._draft_width = tokens_per_seq
        return ready

    def prepare_requests(self, max_requests):
        """Prepare one complete request per independent peer arena lane."""
        if type(max_requests) is not int or max_requests <= 0:
            raise MonoUnsupported("positive draft request capacity required")
        width = self.weights[0].cfg.draft_block_width
        ready = self.prepare(width, tokens_per_seq=width, lanes=max_requests)
        if ready:
            prepared = []

            def allocate():
                previous = getattr(self, "_request_completion", None)
                if previous is not None:
                    previous.synchronize()
                output_reader = getattr(self, "_request_output_completion", None)
                if output_reader is not None:
                    output_reader.synchronize()
                prototype = self._build_for((width, width), 0).hidden[0]
                device = prototype.device
                prepared.append(
                    (
                        tuple(
                            tuple(
                                torch.empty(
                                    (2, width), dtype=torch.int32, device=device
                                )
                                for _ in self.weights
                            )
                            for _ in range(max_requests)
                        ),
                        torch.empty(
                            (max_requests * width, *prototype.shape[1:]),
                            dtype=prototype.dtype,
                            device=device,
                        ),
                        {
                            dtype: torch.empty(
                                (max_requests, width), dtype=dtype, device=device
                            )
                            for dtype in (torch.int32, torch.int64)
                        },
                        torch.empty(
                            (max_requests * width, *prototype.shape[1:]),
                            dtype=prototype.dtype,
                            device=device,
                        ),
                    )
                )

            bind_agreed(allocate, getattr(self, "group", None))
            (
                self._request_destinations,
                self._request_output,
                self._request_tokens,
                self._request_hidden,
            ) = prepared[0]
            self._request_completion = None
            self._request_output_completion = None
        return ready

    def close(self):
        previous = getattr(self, "_request_output_completion", None)
        if previous is not None:
            previous.synchronize()
        super().close()
        self._clear_request_storage()

    @staticmethod
    def _request_steps(steps, start, end, *, out=None):
        """Slice rows, optionally reusing caller-owned two-plane destinations.

        The caller must retire previous readers before reuse. This helper
        does not authorize capture or own stream/session lifecycle.
        """
        steps = tuple(steps)
        if any(step.block_tables is not None or step.state for step in steps):
            raise MonoUnsupported("draft request lanes require native window metadata")
        if out is not None:
            out = tuple(out)
            if len(out) != len(steps):
                raise MonoUnsupported("one fixed lane destination per draft stage")
            for step, target in zip(steps, out):
                source = step.dest_rows[:, start:end]
                if (
                    target.shape != source.shape
                    or target.dtype != source.dtype
                    or target.device != source.device
                    or not target.is_contiguous()
                    or target.stride(0) != end - start
                ):
                    raise MonoUnsupported(
                        "fixed draft lane destination layout mismatch"
                    )
            ranges = sorted(
                (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in out
            )
            if any(a[1] > b[0] for a, b in pairwise(ranges)):
                raise MonoUnsupported("fixed draft lane destinations alias")
            sources = [
                tensor
                for step in steps
                for tensor in (
                    step.positions,
                    step.dest_rows,
                    step.indices,
                    step.cos,
                    step.sin,
                    *(
                        step.kv_cache
                        if isinstance(step.kv_cache, tuple)
                        else (step.kv_cache,)
                    ),
                )
            ]
            if any(
                torch._C._overlaps(target, source)
                for target in out
                for source in sources
            ):
                raise MonoUnsupported(
                    "fixed draft lane destination aliases source/cache"
                )
            for step, target in zip(steps, out):
                target.copy_(step.dest_rows[:, start:end])
        return tuple(
            replace(
                step,
                positions=step.positions[start:end],
                # Kernel plane offsets use the exact lane width, not the
                # aggregate batch stride of a two-plane slice.
                dest_rows=(
                    step.dest_rows[:, start:end].contiguous()
                    if out is None
                    else out[ordinal]
                ),
                indices=step.indices[start:end],
            )
            for ordinal, step in enumerate(steps)
        )

    def _forward_draft_requests(self, hidden, steps, *, tokens, width, lane, out=None):
        """Resolve every whole-request lane before the first cache write.

        Do not admit aggregate B*K as a kernel width or reuse a peer arena
        within a step. Results and resolved contracts are invocation-local.
        Original graph/static publication remains separately guarded.
        An optional output belongs to the caller, which must retire all later
        epilogue/head readers before reuse. The backbone event is insufficient.
        """
        resolved, out = self._resolve_draft_requests(
            hidden, steps, tokens=tokens, width=width, lane=lane, out=out
        )
        self._execute_draft_requests(resolved, out)
        if getattr(self, "_request_destinations", None) is not None:
            completions = []

            def retire():
                completion = torch.cuda.Event()
                completion.record(torch.cuda.current_stream())
                completions.append(completion)

            bind_agreed(retire, getattr(self, "group", None))
            self._request_completion = completions[0]
        return out

    def _resolve_draft_requests(self, hidden, steps, *, tokens, width, lane, out=None):
        """TP-agreed live lane contracts, without any backbone submission.

        Capture may retain the storage but must revalidate live ownership and
        frontier separately before replay; returned contracts are call-local.
        Metadata copies are ordered after the previous reader event.
        """
        # Resident kernels handshake across ranks. Different request counts
        # must fail collectively before entering different lane sequences.
        resolved, outputs = [], []

        def resolve():
            if self._closed:
                raise RuntimeError("draft runner is closed")
            if hidden.shape[0] % width or tokens.numel() != hidden.shape[0]:
                raise MonoUnsupported("draft lanes must partition complete requests")
            fixed = getattr(self, "_request_destinations", None)
            if fixed is not None:
                if lane < 0 or lane + hidden.shape[0] // width > len(fixed):
                    raise MonoUnsupported("draft request exceeds fixed lane capacity")
                previous = getattr(self, "_request_completion", None)
                if previous is not None:
                    torch.cuda.current_stream().wait_event(previous)
            if out is not None:
                if (
                    not isinstance(out, torch.Tensor)
                    or out.shape != hidden.shape
                    or out.dtype != hidden.dtype
                    or out.device != hidden.device
                    or not out.is_contiguous()
                ):
                    raise MonoUnsupported("fixed draft output layout mismatch")
                sources = tensor_owners(
                    (hidden, tokens, tuple(vars(step) for step in steps), fixed)
                )
                if any(torch._C._overlaps(out, source) for source in sources):
                    raise MonoUnsupported("fixed draft output aliases input/cache")
            outputs.append(torch.empty_like(hidden) if out is None else out)
            for request, start in enumerate(range(0, hidden.shape[0], width)):
                end, ordinal = start + width, lane + request
                current, ids = hidden[start:end], tokens[start:end]
                published = self._request_steps(
                    steps, start, end, out=None if fixed is None else fixed[ordinal]
                )
                build = self._build_for((width, width), ordinal)
                if out is not None:
                    owners = tensor_owners(
                        (
                            getattr(build, "hidden", ()),
                            tuple(vars(layer) for layer in build.layers),
                            tuple(vars(v) for v in getattr(build, "variants", ())),
                        )
                    )
                    if any(torch._C._overlaps(out, owner) for owner in owners):
                        raise MonoUnsupported("fixed draft output aliases lane storage")
                if len(published) != len(build.layers):
                    raise MonoUnsupported("one LayerStep is required per draft stage")
                validated = self.preflight(
                    current,
                    published,
                    tokens=ids,
                    tokens_per_seq=width,
                    lane=ordinal,
                )
                aux = self._aux_views(current, build, published, None, 0)
                if out is not None and any(
                    torch._C._overlaps(out, source) for source in tensor_owners(aux)
                ):
                    raise MonoUnsupported(
                        "fixed draft output aliases auxiliary storage"
                    )
                resolved.append(
                    _ResolvedDraftRequest(
                        start, end, current, published, build, validated, aux, ids
                    )
                )

        bind_agreed(resolve, getattr(self, "group", None))
        return tuple(resolved), outputs[0]

    def _execute_draft_requests(self, resolved, out):
        """Submit an already agreed lane sequence to its allocated output.

        No metadata construction, host reads, preflight, allocation or TP vote
        belongs here. The caller retains every binding and retires all readers.
        This private execution split alone does not authorize graph replay:
        captured storage/epochs, padding and session credentials remain required.
        """
        for request in resolved:
            out[request.start : request.end].copy_(
                self._forward_resolved(
                    request.hidden,
                    request.steps,
                    request.build,
                    request.validated,
                    request.aux,
                    request.tokens,
                )
            )

    def _agree_request_layout(self, rows, width, lane):
        import torch.distributed as dist

        group = getattr(self, "group", None)
        if group is None:
            return
        wire = torch.tensor((rows, width, lane), dtype=torch.int64)
        # MIN(x, -x) carries both extrema in one collective. Keep the
        # complete layout verdict before any lane can enter a peer kernel.
        extrema = torch.cat((wire, -wire))
        dist.all_reduce(extrema, op=dist.ReduceOp.MIN, group=group)
        if not torch.equal(extrema[:3], -extrema[3:]):
            raise MonoUnsupported("TP ranks disagree on draft request lane layout")

    def _forward_draft_backbone(
        self, hidden, steps, *, tokens, tokens_per_seq, lane, out=None
    ):
        """Resolve the whole local stack and agree before any peer/cache write.

        Resolution is call-local; do not run public ``forward`` afterwards,
        which would repeat preflight outside the agreed transaction.
        """
        self._agree_request_layout(hidden.shape[0], tokens_per_seq, lane)
        if hidden.shape[0] > tokens_per_seq:
            return self._forward_draft_requests(
                hidden, steps, tokens=tokens, width=tokens_per_seq, lane=lane, out=out
            )
        resolved = []

        def resolve():
            if self._closed:
                raise RuntimeError("draft runner is closed")
            build = self._build_for((hidden.shape[0], tokens_per_seq), lane)
            if len(steps) != len(build.layers):
                raise MonoUnsupported("one LayerStep is required per draft stage")
            validated = self.preflight(
                hidden,
                steps,
                tokens=tokens,
                tokens_per_seq=tokens_per_seq,
                lane=lane,
            )
            aux = self._aux_views(hidden, build, steps, None, 0)
            resolved.append((build, validated, aux))

        bind_agreed(resolve, getattr(self, "group", None))
        build, validated, aux = resolved[0]
        return self._forward_resolved(hidden, steps, build, validated, aux, tokens)

    @torch.inference_mode()
    def forward_block(self, anchor_ids, anchor_positions, steps, *, num_draft, lane=0):
        """Return native-shaped (normed, pre-norm hidden), without sampling.

        Published steps contain expanded anchor+1..anchor+T positions and
        ordered [valid target window, whole draft block] keys. The adapter must
        independently qualify that contract before engine installation.
        """
        steps = tuple(steps)

        def validate_entry():
            if self._closed:
                raise RuntimeError("draft runner is closed")
            self._assert_bindings_current()
            if type(num_draft) is not int or num_draft != self._draft_width:
                raise MonoUnsupported("prepare the exact draft width before forward")
            if anchor_ids.ndim != 1 or anchor_positions.shape != anchor_ids.shape:
                raise MonoUnsupported(
                    "one anchor token and position per request required"
                )
            if anchor_ids.device != anchor_positions.device:
                raise MonoUnsupported("anchor token/position devices differ")
            if not anchor_ids.shape[0]:
                raise MonoUnsupported("empty draft blocks are not executable")
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("draft graph publication is not yet qualified")
            expected_positions = (
                anchor_positions[:, None]
                + torch.arange(1, num_draft + 1, device=anchor_positions.device)
            ).reshape(-1)
            if len(steps) != len(self.weights) or any(
                step.positions.shape != expected_positions.shape
                or step.positions.device != expected_positions.device
                or not torch.equal(step.positions, expected_positions)
                for step in steps
            ):
                raise MonoUnsupported("draft positions must match the current anchors")

        # Embedding may communicate, and the resident backbone polls peers.
        # A local binding/position failure must retire this call on every rank
        # before either operation, not leave healthy peers entering a kernel.
        bind_agreed(validate_entry, getattr(self, "group", None))
        batch = anchor_ids.shape[0]
        # Shared/vocab-sharded embedding may itself communicate. Agree before
        # it, not only before the later resident backbone handshake.
        self._agree_request_layout(batch * num_draft, num_draft, lane)
        inputs = []
        outputs = []
        token_outputs = []
        hidden_outputs = []

        def resolve_output():
            storage = getattr(self, "_request_output", None)
            tokens = getattr(self, "_request_tokens", None)
            hidden_storage = getattr(self, "_request_hidden", None)
            if (
                hidden_storage is not None
                and batch * num_draft > hidden_storage.shape[0]
            ):
                raise MonoUnsupported("draft hidden exceeds prepared request capacity")
            hidden_outputs.append(
                None if hidden_storage is None else hidden_storage[: batch * num_draft]
            )
            token_out = None if tokens is None else tokens.get(anchor_ids.dtype)
            if tokens is not None and (
                token_out is None
                or batch > token_out.shape[0]
                or token_out.shape[1] != num_draft
                or token_out.device != anchor_ids.device
                or not token_out.is_contiguous()
                or torch._C._overlaps(token_out, anchor_ids)
            ):
                raise MonoUnsupported("prepared draft token layout mismatch")
            token_outputs.append(None if token_out is None else token_out[:batch])
            if storage is not None and batch * num_draft > storage.shape[0]:
                raise MonoUnsupported("draft output exceeds prepared request capacity")
            previous = getattr(self, "_request_output_completion", None)
            if previous is not None and (
                token_out is not None
                or hidden_storage is not None
                or (storage is not None and batch > 1)
            ):
                torch.cuda.current_stream().wait_event(previous)
            outputs.append(
                None if storage is None or batch == 1 else storage[: batch * num_draft]
            )

        bind_agreed(resolve_output, getattr(self, "group", None))

        def embed_block():
            inputs.append(
                self._embed_draft_block(
                    anchor_ids,
                    num_draft,
                    ids_out=token_outputs[0],
                    hidden_out=hidden_outputs[0],
                )
            )

        # A failed eager allocation/embedding must not let healthy ranks enter
        # backbone preflight/launch. This is not recovery from a failed device
        # collective: such failures still require retiring the serving owner.
        bind_agreed(embed_block, getattr(self, "group", None))
        flat_ids, hidden = inputs[0]
        residual = self._forward_draft_backbone(
            hidden,
            steps,
            tokens=flat_ids,
            tokens_per_seq=num_draft,
            lane=lane,
            out=outputs[0],
        )
        result = self._reduce_draft_block(residual, batch, num_draft)
        if (
            outputs[0] is not None
            or token_outputs[0] is not None
            or hidden_outputs[0] is not None
        ):
            completions = []

            def retire_output():
                completion = torch.cuda.Event()
                completion.record(torch.cuda.current_stream())
                completions.append(completion)

            bind_agreed(retire_output, getattr(self, "group", None))
            self._request_output_completion = completions[0]
        return result

    def _embed_draft_block(self, anchor_ids, width, *, ids_out=None, hidden_out=None):
        """Device input construction after entry/layout agreement.

        Keep the original embedding callable and operation order. The caller
        validates and retires ids_out/hidden_out before reuse. The original
        embedding still allocates; graph capture remains separately guarded.
        """
        model = self.draft_model
        batch = anchor_ids.shape[0]
        ids = (
            anchor_ids.new_full((batch, width), model.noise_token_id)
            if ids_out is None
            else ids_out
        )
        flat_ids, hidden = self._execute_draft_embedding(anchor_ids, ids)
        if hidden_out is not None:
            self._validate_draft_hidden_output(hidden, hidden_out, flat_ids, anchor_ids)
        return flat_ids, self._expand_draft_hidden(hidden, hidden_out)

    def _execute_draft_embedding(self, anchor_ids, ids):
        """Device entry on caller-validated fixed token storage.

        Capture retains the original embedding allocation in its graph pool;
        runtime ownership/layout checks belong before capture or replay.
        This helper alone grants no graph or serving authorization.
        """
        ids.fill_(self.draft_model.noise_token_id)
        ids[:, 0].copy_(anchor_ids)
        flat_ids = ids.reshape(-1)
        hidden = self.draft_model.embed(flat_ids).reshape(flat_ids.numel(), -1)
        return flat_ids, hidden

    def _execute_resolved_draft_block(
        self, anchor_ids, ids, hidden, residual, resolved
    ):
        """Capture body on fixed, preflighted storage, excluding sampling.

        The capture owner must resolve every request against slices of hidden
        and ids, retain all storage and bindings, and authorize each replay
        against the live frontier. These invocation records alone are not
        replay credentials. No host validation or event recording belongs here.
        """
        _, embedded = self._execute_draft_embedding(anchor_ids, ids)
        self._expand_draft_hidden(embedded, hidden)
        self._execute_draft_requests(resolved, residual)
        return self._reduce_draft_block(residual, ids.shape[0], ids.shape[1])

    def _capture_resolved_draft_block(
        self, inputs, resolved, *, source_steps=(), epilogue=None
    ):
        """Capture previously validated, warmed startup bindings collectively.

        Executes the KV write set and advances epochs. Caller must retire all
        streams and restore its startup snapshot on success AND failure. No
        serving graph is published here, and live replay stays unauthorized.
        """
        captured = []

        def validate():
            if self._closed or torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("draft capture requires an open startup runner")
            self._assert_bindings_current()
            self._validate_draft_capture_inputs(inputs, resolved)
            if source_steps:
                self._validate_draft_capture_sources(source_steps, resolved)

        bind_agreed(validate, getattr(self, "group", None))

        def capture():
            graph = torch.cuda.CUDAGraph()
            # The distributed watchdog queries events on another thread.
            # Match DraftGraph's capture scope so it cannot invalidate ours.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                # Lane destinations have a width-local two-plane stride.
                # Record their publication into the graph rather than running
                # _request_steps and full lane preflight on every replay.
                for request in resolved if source_steps else ():
                    for source, target in zip(source_steps, request.steps):
                        target.dest_rows.copy_(
                            source.dest_rows[:, request.start : request.end]
                        )
                outputs = self._execute_resolved_draft_block(*inputs, resolved)
                if epilogue is not None:
                    outputs = epilogue(outputs, inputs[0], inputs[1].shape[1])
            captured.append((graph, outputs))

        bind_agreed(capture, getattr(self, "group", None))
        retained = []

        def retain():
            owner = self._retain_draft_capture(
                captured[0][0], inputs, captured[0][1], resolved
            )
            if epilogue is not None:
                owner = replace(owner, sampled=True, epilogue=epilogue)
            if source_steps:
                owner = replace(
                    owner,
                    source_steps=tuple(source_steps),
                    source_identity=input_identity(
                        tuple(vars(step) for step in source_steps)
                    ),
                    owners=(
                        *owner.owners,
                        *tensor_owners(tuple(vars(step) for step in source_steps)),
                    ),
                )
            retained.append(owner)

        bind_agreed(retain, getattr(self, "group", None))
        return retained[0]

    @staticmethod
    def _validate_draft_capture_sources(steps, resolved):
        """Prove captured lanes read the fixed native publication buffers.

        Full lane preflight/alias checks precede this startup-only check.
        Position/key views remain shared; only the width-strided destination
        planes are copied in the graph. Live address validity is still the
        native publisher's responsibility before any replay.
        """
        steps = tuple(steps)
        for request in resolved:
            if len(steps) != len(request.steps):
                raise MonoUnsupported("draft capture source stack changed")
            for source, target in zip(steps, request.steps):
                expected = replace(
                    source,
                    positions=source.positions[request.start : request.end],
                    indices=source.indices[request.start : request.end],
                    dest_rows=target.dest_rows,
                )
                if input_identity(vars(expected)) != input_identity(vars(target)):
                    raise MonoUnsupported("draft capture lane publication changed")

    def _validate_draft_capture_inputs(self, inputs, resolved):
        """Reject malformed storage before embedding or a captured KV write."""
        if len(inputs) != 4 or not resolved:
            raise MonoUnsupported("draft capture requires four inputs and live lanes")
        anchors, ids, hidden, residual = inputs
        width = self._draft_width
        if (
            any(not isinstance(t, torch.Tensor) for t in inputs)
            or anchors.ndim != 1
            or not anchors.numel()
            or ids.shape != (anchors.numel(), width)
            or anchors.dtype not in (torch.int32, torch.int64)
            or ids.dtype != anchors.dtype
            or hidden.ndim != 3
            or hidden.shape[0] != ids.numel()
            or hidden.shape[1] != self.draft_model.hc_mult
            or residual.shape != hidden.shape
            or residual.dtype != hidden.dtype
            or any(t.device != hidden.device or not t.is_contiguous() for t in inputs)
            or any(torch._C._overlaps(a, b) for a, b in pairwise(inputs))
            or torch._C._overlaps(anchors, hidden)
            or torch._C._overlaps(anchors, residual)
            or torch._C._overlaps(ids, residual)
            or len(resolved) != anchors.numel()
        ):
            raise MonoUnsupported("draft capture input layout/alias mismatch")
        for lane, request in enumerate(resolved):
            start, end = lane * width, (lane + 1) * width
            if (
                (request.start, request.end) != (start, end)
                or input_identity(request.hidden) != input_identity(hidden[start:end])
                or input_identity(request.tokens)
                != input_identity(ids.reshape(-1)[start:end])
                or request.build is not self._build_for((width, width), lane)
            ):
                raise MonoUnsupported("draft capture lane input binding mismatch")

    def _retain_draft_capture(self, graph, inputs, outputs, resolved):
        """Retain all pointers after successful TP-agreed capture.

        Capture execution and warmup write-set restore are the caller's job.
        This does not publish a graph or grant a live replay credential.
        """
        inputs, outputs, resolved = tuple(inputs), tuple(outputs), tuple(resolved)
        storage = _draft_capture_storage(inputs, outputs, resolved)
        return _DraftBlockCapture(
            graph,
            inputs,
            outputs,
            resolved,
            input_identity(storage),
            tensor_owners(storage),
        )

    def _require_draft_capture_storage(self, capture, resolved):
        """Necessary storage check; live frontier/epoch authorization is external."""
        if self._closed:
            raise MonoUnsupported("draft capture runner is closed")
        self._assert_bindings_current()
        if len(resolved) != len(capture.resolved) or any(
            old.build is not new.build for old, new in zip(capture.resolved, resolved)
        ):
            raise MonoUnsupported("draft capture lane owner changed")
        storage = _draft_capture_storage(capture.inputs, capture.outputs, resolved)
        if input_identity(storage) != capture.identity:
            raise MonoUnsupported("draft capture storage changed")
        if (
            capture.source_steps
            and input_identity(tuple(vars(step) for step in capture.source_steps))
            != capture.source_identity
        ):
            raise MonoUnsupported("draft capture source storage changed")

    def _require_draft_capture_publication(self, capture, steps):
        """Match a live, reader-fenced native publication to captured sources.

        The session/native publisher already checked the live frontier and
        addressed rows. Matching the complete source pointer/layout contract
        permits reuse of startup-preflighted lanes, not arbitrary metadata.
        """
        if (
            not capture.source_steps
            or input_identity(tuple(vars(step) for step in steps))
            != capture.source_identity
        ):
            raise MonoUnsupported("draft graph publication storage changed")

    def _validate_draft_hidden_output(self, hidden, out, *sources):
        """Host-only allocation contract; never part of captured execution."""
        if (
            out.shape != (hidden.shape[0], self.draft_model.hc_mult, hidden.shape[-1])
            or out.dtype != hidden.dtype
            or out.device != hidden.device
            or not out.is_contiguous()
            or any(torch._C._overlaps(out, source) for source in (hidden, *sources))
        ):
            raise MonoUnsupported("prepared draft hidden layout/alias mismatch")

    def _expand_draft_hidden(self, hidden, out=None):
        """Device-only expansion of an already validated invocation.

        This does not authorize replay or validate storage/owner lifetime.
        Copying the broadcast view preserves the original repeat values.
        """
        hc_mult = self.draft_model.hc_mult
        if out is None:
            return hidden.unsqueeze(1).repeat(1, hc_mult, 1)
        out.copy_(hidden.unsqueeze(1).expand(-1, hc_mult, -1))
        return out

    def _reduce_draft_block(self, residual, batch, width):
        """Original device epilogue; no sampler or host authorization.

        Markov consumes normed hidden, confidence consumes the pre-norm mHC
        reduction. Do not collapse the two outputs or replace the native norm.
        """
        model = self.draft_model
        last = model.mtp[-1]
        pre_norm = model.head.hc_head(
            residual,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
        )
        return last.norm(pre_norm), pre_norm.reshape(batch, width, -1)
