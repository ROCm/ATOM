# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture one op's body into its OWN cudagraph, between the dense pieces.

Under ``--cudagraph-mode AF_PIECEWISE`` the attention core gets a cudagraph of
its own, so small-batch decode replays through it instead of running eager
between the graphed dense pieces. A cudagraph bakes the ADDRESS of everything it
reads, so both sides of that boundary need one that does not move:

  * inputs -- the runner clones each one at capture and refreshes the clone at
    replay, so the producer upstream can allocate however it likes;
  * the output -- the dense piece downstream was captured reading one address
    per (layer, rows), so an eagerly computed result is delivered into it.

Both are the runner's business, not the model's. A model opts in by decorating
the method that IS its core -- no separate wrapper, the decorated function is the
core itself::

    @piecewise_core(key=decode_bucket_key, copy_per_step=("positions",))
    def _attn_core(self, *, x, q, kv_pre, ..., positions):
        ...the paged attention body...

and writes nothing else -- no staging buffers, no per-input policy, no dummy
forward to measure shapes. The inputs and their order come off the function's
own signature (the leading ``self``/layer is skipped). A caller on the FULL /
eager path just passes ``piecewise=False`` and the body runs directly.

What this deliberately does NOT do is let the producer write the graph's input
buffer directly. That saves the clone's copy, but only by making every producer
in the chain aware of the capture -- which is what the previous design did, and
is what made this feature reach into the model layer. Two small copies per layer
(`q` and `kv_pre`; the other inputs were already copied) buy that back.
"""

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any, get_args

__all__ = [
    "decode_bucket_key",
    "piecewise_core",
]


def decode_bucket_key(forward_context) -> tuple:
    """``(bucket_bs, q_eff)`` for a core captured per decode bucket.

    Nothing here is model-specific -- both come off the forward context -- so
    every decode core keys the same way. ``bucket_bs`` is the ceil-to-captured
    graph_bs (``forward_mode.effective_bs`` at replay; a capture Context has no
    forward_mode, so batch_size == graph_bs == bucket).
    """
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    context = getattr(forward_context, "context", None)
    q_eff = int(getattr(attn_metadata, "max_seqlen_q", 1) or 1)
    forward_mode = getattr(context, "forward_mode", None)
    if forward_mode is not None and getattr(forward_mode, "effective_bs", 0):
        bucket_bs = int(forward_mode.effective_bs)
    else:
        bucket_bs = int(getattr(context, "batch_size", 0) or 0)
    return (bucket_bs, q_eff)


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
    max_tokens: int = 512,
    copy_per_step: tuple[str, ...] = (),
):
    """Give the decorated function its own cudagraph, keyed per layer and shape.

    ``key`` adds whatever else changes the graph's shape beyond the layer and the
    token count -- for a decode core that is the bucket, `decode_bucket_key`.
    ``max_tokens`` bounds the rows a captured graph covers; a longer step (a
    prefill) runs eager.

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

    copied = frozenset(copy_per_step)

    def decorate(fn: Callable) -> Callable:
        input_names, passthrough_names = _partition_inputs(fn)
        zero_copy = _resolve_zero_copy(input_names, copied)

        @functools.wraps(fn)
        def wrapper(
            layer,
            *,
            piecewise,
            capture=False,
            runner=None,
            outputs=None,
            forward_context=None,
            **inputs,
        ):
            # FULL / eager: the body runs directly, so a caller on that path only
            # states `piecewise=False` and none of the capture collaborators.
            if not piecewise:
                return fn(layer, **inputs)

            layer_name = getattr(layer, "layer_name", id(layer))
            num_tokens = int(inputs[input_names[0]].shape[0])
            # One output slot per (layer, flat row count): the address the dense
            # piece downstream was captured reading.
            out_key = (layer_name, num_tokens)
            # Config args are not graph inputs: bind them into the compute so they
            # are baked at capture, and hand the runner only the tensor inputs.
            tensor_inputs = {n: inputs[n] for n in input_names if n in inputs}
            core = functools.partial(
                fn, layer, **{n: inputs[n] for n in passthrough_names if n in inputs}
            )

            context = getattr(forward_context, "context", None)
            # `capture` off (plain PIECEWISE) short-circuits to the deliver tail
            # below: the core runs eager and only its result is stabilised, with
            # no graph of its own ever recorded.
            eligible = (
                capture
                and not getattr(context, "is_dummy_run", False)
                and getattr(forward_context, "attn_metadata", None) is not None
                and num_tokens <= max_tokens
            )
            # num_tokens is a KEY DIM, not an incidental one: the graph is
            # captured at exactly this flat row count.
            graph_key = (
                (layer_name, num_tokens) + tuple(key(forward_context))
                if eligible
                else None
            )
            # True only inside the capture pass: WHICH KIND of forward this is,
            # building the graphs vs serving a real step.
            capturing = getattr(forward_context, "in_hipgraph", False)

            if graph_key is not None:
                if capturing and not runner.has_graph(graph_key):
                    # First time the capture pass reaches this key. Warm up on
                    # the buffers the graph will read, size the output slot from
                    # that result, then record.
                    read_from, refresh = runner.input_buffers(
                        tensor_inputs, input_names, zero_copy
                    )
                    out_slot = outputs.slot(out_key, core(**read_from))
                    runner.capture(graph_key, read_from, refresh, core, out_slot)
                elif runner.has_graph(graph_key) and not capturing:
                    return runner.replay(graph_key, tensor_inputs)

            # Everything else: not eligible, the capture pass revisiting a key,
            # or a step whose key was never captured. Also the tail of the
            # capture branch above -- it fed on clones, so its result is not this
            # forward's answer.
            return outputs.deliver(out_key, core(**tensor_inputs))

        wrapper.input_names = input_names
        wrapper.passthrough_names = passthrough_names
        wrapper.zero_copy_names = zero_copy
        wrapper.max_tokens = max_tokens
        return wrapper

    return decorate
