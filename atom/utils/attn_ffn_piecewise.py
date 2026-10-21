# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture numerical regions between compiled pieces through one decorator.

The common runner owns recording, replay and stable boundary outputs. Tensor
arguments are captured directly by default; ``copy_per_step`` requests staging
when their producer changes addresses. Backends prepare metadata and cross-layer
results through their normal lifecycle, with explicit in-place GPU producers.

V4 retains its decode startup-capture interface. Live prefill regions opt in with
``capture_prefill=True`` and use the current forward context's backend layout.
See ``docs/piecewise_core.md`` for the metadata and buffer lifetime contract.
"""

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any, get_args

__all__ = [
    "configure_piecewise_core",
    "decode_bucket_key",
    "piecewise_capture_shapes",
    "piecewise_core",
]


def piecewise_capture_shapes(model, capture_sizes, q_buckets):
    """Add declared stage token multiples within the startup builder's bounds.

    Stages declare only ``piecewise_capture_multiple``. Budgeting and capture
    use this same plan, including deduplication against ordinary BS * Q buckets.
    """
    if not capture_sizes or not q_buckets:
        return []
    multiples = {
        m.piecewise_capture_multiple
        for m in model.modules()
        if getattr(m, "piecewise_capture_multiple", 0)
    }
    if any(not isinstance(n, int) or n <= 0 for n in multiples):
        raise ValueError("piecewise_capture_multiple must be a positive integer")
    max_bs, max_q = max(capture_sizes), max(q_buckets)
    existing = {bs * q for bs in capture_sizes for q in q_buckets}
    tokens = set().union(*(range(n, max_bs * max_q + 1, n) for n in multiples))
    shapes = []
    for n in sorted(tokens - existing):
        q = next(
            (q for q in range(1, max_q + 1) if n % q == 0 and n // q <= max_bs), None
        )
        if q is not None:
            shapes.append((n // q, q))
    return shapes


def configure_piecewise_core(layer, *, outputs=None, copy_per_step=None):
    """Configure a layer's region boundaries once during runner initialization.

    Optional: the decorator otherwise owns its output store and uses its declared
    input policy. Runtime layouts with standalone eager producers can request
    staging here without adding graph arguments to the model's numerical calls.
    """
    if isinstance(copy_per_step, str):
        raise TypeError("copy_per_step must be a tuple of tensor argument names")
    layer.__dict__["_piecewise_core_options"] = (outputs, copy_per_step)


def decode_bucket_key(forward_context) -> tuple:
    """``(bucket_bs, q_eff)`` for a core captured per decode bucket.

    Nothing here is model-specific -- both come off the forward context -- so
    every decode core keys the same way. ``bucket_bs`` is the step's
    ``running_bs`` (a capture Context has no forward_mode, so batch_size is
    already the bucket).
    """
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    context = getattr(forward_context, "context", None)
    q_eff = int(getattr(attn_metadata, "max_seqlen_q", 1) or 1)
    forward_mode = getattr(context, "forward_mode", None)
    if forward_mode is not None and getattr(forward_mode, "running_bs", 0):
        bucket_bs = int(forward_mode.running_bs)
    else:
        bucket_bs = int(getattr(context, "scheduled_bs", 0) or 0)
    return (bucket_bs, q_eff)


def _is_decode(forward_context) -> bool:
    """Whether this step is a decode.

    Checked by NAME, like ``_annotation_names_tensor`` above and for the same
    reason: this module stays free of a torch import, and ``AttnState`` lives
    behind one. Any ``PREFILL_*`` reads as not-decode, which is the only
    distinction that matters here.
    """
    state = getattr(getattr(forward_context, "attn_metadata", None), "state", None)
    return getattr(state, "name", "") == "DECODE"


def _annotation_names_tensor(annotation: Any) -> bool:
    """Whether a parameter's annotation names a tensor.

    This is the whole rule that tells a graph INPUT from a pass-through config
    arg: ``torch.Tensor`` (and ``torch.Tensor | None``) is an input; an
    explicitly non-tensor annotation (``bool``, ``int``, an enum, ...) is config.
    Checked by NAME so this module stays free of a torch import, and handles the
    ``from __future__ import annotations`` case where the annotation is a string.
    An UNANNOTATED parameter counts as an input, so a core written before this
    split -- everything a tensor -- is unchanged.
    """
    if annotation is inspect.Parameter.empty:
        return True
    if isinstance(annotation, str):
        return "Tensor" in annotation
    # Bare ``torch.Tensor`` has no args; a Union/Optional yields its members.
    parts = get_args(annotation) or (annotation,)
    return any(getattr(p, "__name__", "") == "Tensor" for p in parts)


def _partition_inputs(fn: Callable) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The core's parameters after the leading layer, split into the graph's
    tensor INPUTS (in order) and the pass-through CONFIG args.

    One source rather than two: the split comes off the signature itself, so
    there is no names list beside the function to drift from it. A parameter is a
    graph input unless its annotation names no tensor (see
    ``_annotation_names_tensor``); a config arg is forwarded to the body
    untouched and BAKED into the graph at capture -- never cloned, never captured
    on an address -- so it must be invariant for a given graph key (or fold it
    into ``key``). Order matters (the runner expands inputs by it), so
    ``*args``/``**kwargs`` are rejected here rather than mis-expanded later. The
    leading parameter is the layer that owns the core, not one of its inputs.
    """
    inputs: list[str] = []
    passthrough: list[str] = []
    for i, (name, p) in enumerate(inspect.signature(fn).parameters.items()):
        if i == 0:
            continue
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            raise TypeError(
                f"{fn.__name__} must name its inputs explicitly; "
                f"'{'**' if p.kind is p.VAR_KEYWORD else '*'}{name}' has no "
                "declared order for the runner to expand by."
            )
        (inputs if _annotation_names_tensor(p.annotation) else passthrough).append(name)
    return tuple(inputs), tuple(passthrough)


def _resolve_zero_copy(names: tuple[str, ...], copied: frozenset) -> frozenset:
    """Which inputs the graph captures on directly.

    ``ATOM_ATTN_FFN_ZC`` overrides it with a comma-separated whitelist (empty
    string = capture on nothing), which is how a suspected input gets taken off
    the zero-copy path for one run without touching code.
    """
    override = os.environ.get("ATOM_ATTN_FFN_ZC")
    if override is not None:
        wanted = {p.strip() for p in override.split(",") if p.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(
                f"ATOM_ATTN_FFN_ZC names {sorted(unknown)} are not inputs; "
                f"known names are {list(names)}"
            )
        return frozenset(wanted)
    return frozenset(names) - copied


def piecewise_core(
    *,
    key: Callable[[Any], tuple] = lambda _fc: (),
    max_tokens: int | None = None,
    copy_per_step: tuple[str, ...] = (),
    capture_prefill: bool = False,
    token_dim: int = 0,
):
    """Give the decorated function its own cudagraph, keyed per layer and shape.

    ``key`` adds whatever else changes the graph's shape beyond the layer and the
    token count -- for a decode core that is the bucket, `decode_bucket_key`.
    By default only DECODE steps are captured. ``capture_prefill=True`` also
    permits live prefill capture over the current backend's ``cudagraph_key``.
    The backend prepares request inputs and persistent result workspaces in its
    normal build lifecycle. GPU producers write these buffers in place; Python
    side effects are not replayed. No metadata schema or transaction is inferred.
    Omitting ``piecewise`` uses the current forward context. The runner and its
    boundary output store are created lazily on the owning layer.
    The explicit decode API remains available for startup capture callers.
    ``max_tokens`` is an OPTIONAL extra cap on the rows
    a captured graph covers -- a memory lever, off by default. It used to be 512
    and to be the only gate, which silently excluded every decode bucket above
    ~85 sequences at DSpark q=6.

    Every TENSOR input is captured on directly -- the graph reads the producer's
    own tensor and nothing copies it -- EXCEPT the names in ``copy_per_step``,
    which the runner clones and refreshes each step. Name one when capturing on it
    is wrong, whatever the reason; the entry's own comment should say which.

    A parameter annotated as something other than a tensor (``bool``, ``int``, an
    enum, ...) is NOT a graph input: it is forwarded to the body untouched and
    baked into the graph at capture, so a core can keep its config flags in its
    own signature without the model having to strip them. Being baked, such an arg
    must be invariant for a given graph ``key`` (or folded into it).

    The wrapped function is called as ``fn(layer, **inputs)`` and returns one
    tensor. It routes three ways off two flags the caller passes each step:

      * ``piecewise=False`` -- FULL / eager / a PIECEWISE forward running under a
        FULL-runtime graph. Simply called: nothing is captured, and no downstream
        piece is holding an address to deliver to.
      * ``piecewise=True, capture=False`` -- plain PIECEWISE. The core stays
        eager, but the dense piece downstream was captured reading a fixed
        address, so the result is delivered into the persistent output slot.
      * ``piecewise=True, capture=True`` -- the core gets its own cudagraph
        (AF_PIECEWISE): captured in the capture pass, replayed after.

    So ``capture`` is the mode gate -- with it off this collapses to exactly the
    eager-attention-plus-stable-buffer path plain PIECEWISE always ran, and the
    caller has no branch of its own to keep.
    """

    if isinstance(copy_per_step, str):
        # `("positions")` is a string, not a 1-tuple, and `frozenset` of it is a
        # set of CHARACTERS -- which subtracts nothing from the input names, so
        # the entry silently means "copy nothing" while reading as its opposite.
        # That shipped once. A one-name tuple needs its trailing comma.
        raise TypeError(
            f"copy_per_step must be a tuple of names, got the string "
            f"{copy_per_step!r}. A single name needs a trailing comma: "
            f'("{copy_per_step}",). Without it this silently copies nothing.'
        )
    copied = frozenset(copy_per_step)

    def decorate(fn: Callable) -> Callable:
        input_names, passthrough_names = _partition_inputs(fn)
        signature = inspect.signature(fn)
        unknown = copied - set(input_names)
        if unknown:
            raise ValueError(
                f"copy_per_step names unknown tensor inputs: {sorted(unknown)}"
            )
        zero_copy = _resolve_zero_copy(input_names, copied)

        @functools.wraps(fn)
        def wrapper(
            layer,
            *args,
            piecewise=None,
            capture=None,
            runner=None,
            outputs=None,
            forward_context=None,
            capture_key=None,
            prepare=None,
            zero_copy_inputs=None,
            **inputs,
        ):
            if piecewise is None:
                if not capture_prefill:
                    piecewise = False
                else:
                    from atom.utils.forward_context import get_forward_context

                    if forward_context is None:
                        forward_context = get_forward_context()
                    mode = getattr(forward_context, "cudagraph_runtime_mode", None)
                    piecewise = getattr(mode, "name", None) == "PIECEWISE"
            metadata = getattr(forward_context, "attn_metadata", None)
            backend_key = getattr(metadata, "cudagraph_key", None)
            if capture is None:
                capture = backend_key is not None
            # FULL / eager: the body runs directly, so a caller on that path only
            # states `piecewise=False` and none of the capture collaborators.
            if not piecewise:
                return fn(layer, *args, **inputs)

            if args or capture_prefill:
                bound = signature.bind(layer, *args, **inputs)
                bound.apply_defaults()
                inputs = dict(list(bound.arguments.items())[1:])

            if capture_prefill:
                from atom.utils.cuda_graph import CudagraphCaptureRunner

                configured_outputs, configured_copies = layer.__dict__.get(
                    "_piecewise_core_options", (None, None)
                )
                if outputs is None:
                    outputs = configured_outputs
                if zero_copy_inputs is None and configured_copies is not None:
                    unknown = set(configured_copies) - set(input_names)
                    if unknown:
                        raise ValueError(
                            f"copy_per_step names unknown tensor inputs: {sorted(unknown)}"
                        )
                    zero_copy_inputs = _resolve_zero_copy(
                        input_names, frozenset(configured_copies)
                    )

                if runner is None:
                    stores = layer.__dict__.setdefault("_piecewise_core_runners", {})
                    if fn not in stores:
                        stores[fn] = CudagraphCaptureRunner(
                            outputs=outputs, token_dim=token_dim
                        )
                    runner = stores[fn]
                if outputs is None:
                    outputs = runner.outputs

            layer_name = getattr(layer, "layer_name", id(layer))
            # Row count off the first input the caller actually passed. A core
            # whose signature covers several call shapes leaves the inputs of the
            # shapes it is not in as None -- declaration order still decides
            # which one anchors, but a None one is skipped rather than crashing.
            # The first present tensor supplies the configured token axis.
            anchor = next(
                (inputs[n] for n in input_names if inputs.get(n) is not None), None
            )
            if anchor is None:
                raise ValueError(
                    f"{fn.__name__} was called with every tensor input None; "
                    "the runner has no way to size the graph. Inputs are "
                    f"{list(input_names)}."
                )
            num_tokens = int(anchor.shape[token_dim])
            # Fixed capacity shares one layer slot across token prefixes;
            # otherwise each token count gets its own stable output address.
            out_key = (
                layer_name
                if getattr(outputs, "capacity", 0)
                else (layer_name, num_tokens)
            )
            if capture_prefill:
                out_key = (fn, out_key)
            # Config args are not graph inputs: bind them into the compute so they
            # are baked at capture, and hand the runner only the tensor inputs.
            # EVERY declared input, None included. A core whose signature spans
            # several call shapes leaves the inputs of the shapes it is not in as
            # None, and those parameters still have to be bound or the call is a
            # missing-argument TypeError. The runner already leaves a None alone
            # (`input_buffers` / `replay` both check), so it takes the same dict.
            tensor_inputs = {n: inputs.get(n) for n in input_names}
            core = functools.partial(
                fn, layer, **{n: inputs[n] for n in passthrough_names if n in inputs}
            )
            if prepare is not None:
                body = core

                def core(**values):
                    with prepare() as overrides:
                        return body(**values, **overrides)

            context = getattr(forward_context, "context", None)
            # `capture` off (plain PIECEWISE) short-circuits to the deliver tail
            # below: the core runs eager and only its result is stabilised, with
            # no graph of its own ever recorded.
            # Live prefill uses the metadata interface or the explicitly keyed
            # low-level API. The default decode policy is unchanged.
            capturing = getattr(forward_context, "in_hipgraph", False)
            runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
            eligible = (
                capture
                and not getattr(context, "is_dummy_run", False)
                and not getattr(
                    getattr(forward_context, "attn_metadata", None), "dummy", False
                )
                and (
                    not capturing
                    and (runtime_mode is None or runtime_mode.name == "PIECEWISE")
                    if capture_prefill
                    else getattr(forward_context, "attn_metadata", None) is not None
                    and _is_decode(forward_context)
                )
                and (max_tokens is None or num_tokens <= max_tokens)
            )
            direct = zero_copy
            if eligible:
                direct = (
                    zero_copy
                    if zero_copy_inputs is None
                    else frozenset(zero_copy_inputs)
                )
                if isinstance(zero_copy_inputs, str) or direct - set(input_names):
                    raise ValueError(
                        "zero_copy_inputs must name declared tensor inputs"
                    )
            if capture_prefill and capture_key is None:
                eligible = eligible and backend_key is not None
                if eligible:
                    handled, result = runner.run(
                        (
                            fn,
                            layer_name,
                            num_tokens,
                            backend_key,
                            tuple(key(forward_context)),
                            tuple(
                                (name, type(inputs[name]), inputs[name])
                                for name in passthrough_names
                                if isinstance(
                                    inputs.get(name),
                                    (str, int, float, bool, type(None)),
                                )
                            ),
                        ),
                        tensor_inputs,
                        core,
                        functools.partial(outputs.slot, out_key),
                        zero_copy=direct,
                    )
                    if handled:
                        return result
                eligible = False
            # num_tokens is a KEY DIM, not an incidental one: the graph is
            # captured at exactly this flat row count.
            graph_key = (
                (
                    capture_key
                    if capture_key is not None
                    else (layer_name, num_tokens) + tuple(key(forward_context))
                )
                if eligible
                else None
            )
            on_demand = capture_prefill and not capturing
            if eligible and on_demand and capture_key is None:
                raise ValueError("Live prefill capture requires a complete capture_key")
            if (
                on_demand
                and graph_key is not None
                and not runner.can_capture(graph_key)
            ):
                graph_key = None

            if graph_key is not None:
                if (capturing or on_demand) and not runner.has_graph(graph_key):
                    read_from, refresh = runner.input_buffers(
                        tensor_inputs, input_names, direct
                    )
                    if on_demand:
                        return runner.capture_on_demand(
                            graph_key,
                            read_from,
                            refresh,
                            core,
                            functools.partial(outputs.slot, out_key),
                        )
                    warm = core(**read_from)
                    out_slot = None if warm is None else outputs.slot(out_key, warm)
                    runner.capture(graph_key, read_from, refresh, core, out_slot)
                elif runner.has_graph(graph_key) and not capturing:
                    return runner.replay(graph_key, tensor_inputs)

            result = core(**tensor_inputs)
            if result is None:
                return None
            return outputs.deliver(out_key, result)

        wrapper.input_names = input_names
        wrapper.passthrough_names = passthrough_names
        wrapper.zero_copy_names = zero_copy
        wrapper.max_tokens = max_tokens
        return wrapper

    return decorate
