# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-agnostic half of the attention/FFN-wise piecewise cudagraph path.

Under ``--cudagraph-mode AF_PIECEWISE`` the attention core is captured into its
OWN cudagraph, separate from the dense pieces around it. Two things have to hold
for that graph to be replayable:

  * its inputs must live at FIXED addresses, so the producers upstream stage
    each one into a persistent buffer (``ZeroCopyBuffers``) instead of handing
    over a fresh tensor every step; and
  * the graph must be keyed on everything that changes its shape.

Neither is model-specific, but the first implementation grew inside DeepSeek-V4
and encoded V4 in both: the buffer set was V4's exact tensors, and the zero-copy
contract was a set of POSITIONAL INDICES into V4's nine-argument core. This
module keeps the mechanism and drops the model: a layer declares its inputs BY
NAME on an ``AttnFfnPiecewise`` it owns, and everything downstream addresses
them that way.

``CudagraphCaptureRunner`` (atom/utils/cuda_graph.py) still owns capture/replay
itself; this module owns the contract around it.
"""

import inspect
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "AttnFfnPiecewise",
    "BufferShape",
    "DecodeAttnFfnPiecewise",
    "ZeroCopyBuffers",
    "decode_bucket_key",
    "resolve_zero_copy_names",
]


class ZeroCopyBuffers:
    """Fixed-address staging buffers for one layer's captured-core inputs.

    Keyed by name, so nothing here knows what any particular model calls its
    tensors. Every buffer is allocated for ``max_tokens`` rows and handed out as
    ``buf[:n]``; the token axis is ALWAYS dim 0.

    Allocation happens once, post-load, from sample tensors: shapes and dtypes
    depend on the quantisation config, so they are learned from a dummy forward
    rather than declared.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = int(max_tokens)
        self._bufs: dict[str, torch.Tensor | None] = {}

    def alloc_like(self, name: str, sample: torch.Tensor | None) -> None:
        """Allocate ``name`` with the sample's shape and dtype.

        A None sample registers None: the input exists in the signature but not
        in this configuration (an absent indexer, say), and ``stage`` passes it
        through untouched.
        """
        self._bufs[name] = torch.empty_like(sample) if sample is not None else None

    def alloc(self, name: str, width: int, dtype: torch.dtype, device: Any) -> None:
        """Allocate ``name`` as ``[max_tokens, width]``.

        For an input whose width is not readable off a sample -- the sample may
        be a view into a larger tensor, or the buffer may need to be contiguous
        where the sample is not.
        """
        self._bufs[name] = torch.empty(
            self.max_tokens, width, dtype=dtype, device=device
        )

    def get(self, name: str) -> torch.Tensor | None:
        return self._bufs.get(name)

    def fits(self, n: int) -> bool:
        return n <= self.max_tokens

    def stage(self, name: str, tensor: torch.Tensor | None):
        """Copy ``tensor`` (n tokens on dim 0) into ``name``'s buffer and return
        the fixed-address slice. None-safe both ways: an unregistered name or a
        None tensor passes the tensor through unchanged."""
        buf = self._bufs.get(name)
        if buf is None or tensor is None:
            return tensor
        n = tensor.shape[0]
        buf[:n].copy_(tensor)
        return buf[:n]

    def stage_inplace(
        self,
        name: str,
        n: int,
        produce_out: Callable[[torch.Tensor], Any],
        produce_plain: Callable[[], torch.Tensor],
        *,
        inplace: bool,
    ) -> torch.Tensor:
        """Fill ``name``'s buffer from a single-tensor producer.

        ``inplace`` says the producer can write the destination directly (a GEMM
        with ``out=``), which saves the copy; otherwise it produces normally and
        the result is copied in. Either way the return is ``buf[:n]``, at the
        address the graph was captured reading.
        """
        buf = self._bufs[name]
        if inplace:
            produce_out(buf[:n])
        else:
            buf[:n].copy_(produce_plain())
        return buf[:n]


@dataclass(frozen=True)
class BufferShape:
    """Allocate this staging buffer as ``[max_tokens, width]`` rather than from a
    sample. For an input whose sample is a view into a larger tensor, or whose
    buffer must be contiguous where the sample is not."""

    width: int
    dtype: torch.dtype


class AttnFfnPiecewise:
    """One attention layer's captured-core concern, owned BY the layer.

    Composed rather than mixed in: an attention layer is already large, and the
    capture concern has its own state (buffers, the input contract, the graph
    key) with no reason to share the layer's namespace or fight nn.Module for
    the MRO. The layer holds one of these; a model subclasses THIS.

    Subclass supplies ``describe_staging_buffers`` and ``core``; the inputs and
    their order come from ``core``'s signature. Decode cores should subclass
    ``DecodeAttnFfnPiecewise``, which fills in the graph key too.

    The core's own parameter list is the single source of truth. That is the
    point of the interface: the original design expressed the zero-copy set as
    indices into one model's argument tuple, so it could not be read without
    counting positions and could not survive a signature change. Every declared
    input is zero-copy unless the subclass names an exception.
    """

    # Inputs a subclass wants COPIED per step instead of captured on directly.
    # Empty is the norm: staging exists to give the graph fixed addresses, so
    # everything it declares is zero-copy unless something is known not to be.
    zero_copy_exclude: tuple[str, ...] = ()
    # Rows the staging buffers cover; a longer step falls back to the copy path.
    max_tokens: int = 512

    def __init__(self, layer, *, runner, enabled: bool):
        self.layer = layer
        self.runner = runner
        # Whether this layer's core is captured at all (the cudagraph mode).
        self.enabled = bool(enabled)
        self.buffers: ZeroCopyBuffers | None = None
        self.input_names = self.core_input_names()
        self.zero_copy_names = resolve_zero_copy_names(
            self.input_names,
            os.environ.get("ATOM_ATTN_FFN_ZC"),
            default_excluded=type(self).zero_copy_exclude,
        )

    @classmethod
    def core_input_names(cls) -> tuple[str, ...]:
        """The core's inputs, in order, read off ``core``'s own signature.

        One source rather than two: an ``input_names`` attribute beside the
        method is a second place to state the same thing, and the two drift.
        Order matters -- the runner expands arguments by it -- so the core must
        name its inputs explicitly; ``**kwargs`` carries no order and is
        rejected here rather than mis-expanded later.
        """
        names = []
        for name, p in inspect.signature(cls.core).parameters.items():
            if name == "self":
                continue
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                raise TypeError(
                    f"{cls.__name__}.core must name its inputs explicitly; "
                    f"'{'**' if p.kind is p.VAR_KEYWORD else '*'}{name}' has no "
                    "declared order for the runner to expand by."
                )
            names.append(name)
        return tuple(names)

    # ---- subclass implements -------------------------------------------------

    def describe_staging_buffers(self) -> dict[str, Any]:
        """Describe the staging buffer for every input. Allocates nothing --
        ``alloc`` does that; this only says what to allocate.

        Returns ``{name: spec}``, where a spec is one of:
          * a sample tensor -- allocate a buffer of the same shape and dtype;
          * a ``BufferShape`` -- allocate ``[max_tokens, width]`` explicitly,
            for an input whose sample is a view or must be made contiguous;
          * ``None`` -- this configuration does not produce that input (an
            absent indexer, say). It keeps its place in the signature and
            passes through unstaged.

        A subclass typically gets its samples by running a dummy forward
        through its own projection chain: shapes and dtypes vary with the quant
        config, so they are measured rather than declared, and that chain is
        the one part of this that cannot be generic.
        """
        raise NotImplementedError

    def core(self, **named: Any) -> torch.Tensor:
        """The compute captured into the graph, taking the declared inputs."""
        raise NotImplementedError

    def graph_key(self, forward_context) -> tuple:
        """Key components beyond the layer name and the token count.

        Default empty: the token count ``run`` always includes is the only shape
        most cores have. A core reading shape-carrying metadata (a per-request
        query length, say) must add it, or two differently-shaped steps share
        one graph.
        """
        del forward_context
        return ()

    # ---- provided ------------------------------------------------------------

    @property
    def device(self):
        return next(self.layer.parameters()).device

    def alloc(self) -> None:
        """Turn ``describe_staging_buffers`` into the staging pool. Call post-load,
        pre-capture. No-op when this layer is not captured."""
        if not self.enabled:
            return
        samples = self.describe_staging_buffers()
        device = self.device
        bufs = ZeroCopyBuffers(self.max_tokens)
        for name, spec in samples.items():
            if isinstance(spec, BufferShape):
                bufs.alloc(name, spec.width, spec.dtype, device)
            else:
                bufs.alloc_like(name, spec)
        self.buffers = bufs

    def staging(self, n: int) -> ZeroCopyBuffers | None:
        """The pool for a step of ``n`` tokens, or None when this step must not
        stage: the layer's core is not captured, or ``n`` overflows the buffers.

        None rather than a pass-through pool on purpose. The caller is a traced
        function, and which of the two modes it is in should be readable at the
        call site instead of hidden behind an object that quietly does nothing.
        """
        bufs = self.buffers
        return bufs if bufs is not None and bufs.fits(n) else None

    def run(
        self,
        *,
        forward_context,
        layer_name: str,
        named_args: dict[str, Any],
        out_key,
        piecewise: bool,
    ) -> torch.Tensor:
        """Capture / replay / eager for one invocation. Every caller goes through
        here, so WHETHER to capture, and under what key, is stated once."""
        num_tokens = int(named_args[self.input_names[0]].shape[0])
        context = getattr(forward_context, "context", None)
        capture_it = (
            self.enabled
            and not getattr(context, "is_dummy_run", False)
            and getattr(forward_context, "attn_metadata", None) is not None
            and num_tokens <= self.max_tokens
        )
        if not capture_it:
            return self.runner.stabilize(out_key, self.core(**named_args), piecewise)

        # num_tokens is a KEY DIM, not an incidental one: the core graph is
        # captured at exactly this flat row count so a zero-copy input's read
        # length equals what the producer upstream wrote (no padded tail).
        key = (layer_name, num_tokens) + tuple(self.graph_key(forward_context))
        return self.runner.run(
            key=key,
            out_key=out_key,
            named_args=named_args,
            input_names=self.input_names,
            zc=self.zero_copy_names,
            compute_fn=self.core,
            piecewise=piecewise,
            in_hipgraph=getattr(forward_context, "in_hipgraph", False),
        )


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


class DecodeAttnFfnPiecewise(AttnFfnPiecewise):
    """A core captured per decode bucket -- subclass this and implement
    ``core`` plus ``describe_staging_buffers``.

    Split from the base rather than folded into it because the key is exactly
    what a prefill core does differently: it has no ``max_seqlen_q`` bucket to
    key on. Keeping the decode notion in a subclass leaves the base's default
    free for that case instead of forcing it to override decode semantics.
    """

    def graph_key(self, forward_context) -> tuple:
        return decode_bucket_key(forward_context)


def resolve_zero_copy_names(
    all_names: Iterable[str],
    override: str | None,
    default_excluded: Iterable[str] = (),
) -> frozenset[str]:
    """Zero-copy set from names, with an env override for bisecting.

    ``override`` is a comma-separated name list (empty string = copy
    everything), which is how a suspected input gets taken out of the zero-copy
    path without a code change. Unset means every name except
    ``default_excluded``.
    """
    names = tuple(all_names)
    if override is not None:
        wanted = {p.strip() for p in override.split(",") if p.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(
                f"zero-copy override names {sorted(unknown)} are not inputs; "
                f"known names are {list(names)}"
            )
        return frozenset(wanted)
    return frozenset(names) - set(default_excluded)
