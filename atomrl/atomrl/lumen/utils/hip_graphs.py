# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""HIP/CUDA graph capture utilities for Lumen.

Provides graph-capture wrappers that record a training step (or sub-step)
into a replayable HIP graph, reducing kernel launch overhead. Handles
FP8 scaling manager state updates that must remain graph-safe.

Key classes:

- ``LumenGraphedCallable`` — wraps a single callable in a forward-only graph.
- ``LumenGraphedModule``   — wraps an ``nn.Module`` via ``LumenGraphedCallable``.
- ``LumenGraphedLayer``    — per-layer forward+backward graph capture with a
  custom ``autograd.Function`` that replays separate forward and backward
  CUDA graphs.  Modelled after TE's ``Graphed`` autograd function.
- ``capture_lumen_graphs`` — captures all transformer layers of a Megatron
  GPTModel and replaces their ``forward`` with graphed wrappers.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-only graph capture (unchanged from original)
# ---------------------------------------------------------------------------


class LumenGraphedCallable:
    """Wraps a callable (forward pass) in a CUDA/HIP graph.

    1. Warm up the callable with sample inputs
    2. Record a graph capturing the computation
    3. Replay the graph on subsequent calls

    The graph is re-recorded if input shapes change.

    Args:
        callable_fn: The function to graph-capture.
        sample_args: Sample arguments for warmup and capture.
        sample_kwargs: Sample keyword arguments.
        num_warmup: Number of warmup iterations before capture.
        pool: Optional CUDA memory pool for graph allocation.
    """

    def __init__(
        self,
        callable_fn: Callable,
        sample_args: tuple[torch.Tensor, ...],
        sample_kwargs: dict | None = None,
        num_warmup: int = 3,
        pool: Any | None = None,
    ):
        self._fn = callable_fn
        self._num_warmup = num_warmup
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: tuple[torch.Tensor, ...] | None = None
        self._static_output: torch.Tensor | None = None
        self._input_shapes: tuple[torch.Size, ...] | None = None
        self._pool = pool

        self._capture(sample_args, sample_kwargs or {})

    def _capture(self, args: tuple, kwargs: dict):
        """Warm up and capture the graph."""
        device = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_cuda:
                device = a.device
                break

        if device is None:
            logger.warning("No CUDA tensors in args, skipping graph capture")
            self._graph = None
            return

        self._static_inputs = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in args
        )
        self._input_shapes = tuple(
            a.shape if isinstance(a, torch.Tensor) else None for a in args
        )

        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(self._num_warmup):
                self._fn(*self._static_inputs, **kwargs)
        torch.cuda.current_stream(device).wait_stream(s)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=self._pool, stream=s):
            self._static_output = self._fn(*self._static_inputs, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self._graph is None:
            return self._fn(*args, **kwargs)

        for static, new in zip(self._static_inputs, args):
            if isinstance(static, torch.Tensor) and isinstance(new, torch.Tensor):
                if static.shape != new.shape:
                    logger.info("Input shape changed, re-capturing graph")
                    self._capture(args, kwargs)
                    return self._static_output
                static.copy_(new)

        self._graph.replay()
        return self._static_output

    def reset(self):
        """Release the captured graph."""
        self._graph = None
        self._static_inputs = None
        self._static_output = None


class LumenGraphedModule(nn.Module):
    """Wrapper that graph-captures a module's forward pass.

    Args:
        module: The module to wrap.
        sample_input: Sample input for graph capture.
        num_warmup: Warmup iterations.
        enabled: Whether graph capture is active.
    """

    def __init__(
        self,
        module: nn.Module,
        sample_input: torch.Tensor | None = None,
        num_warmup: int = 3,
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.enabled = enabled
        self._graphed: LumenGraphedCallable | None = None
        self._num_warmup = num_warmup

        if enabled and sample_input is not None:
            self._graphed = LumenGraphedCallable(
                module, (sample_input,), num_warmup=num_warmup
            )

    def forward(self, *args, **kwargs):
        if self._graphed is not None and self.enabled:
            return self._graphed(*args, **kwargs)
        return self.module(*args, **kwargs)

    def capture(self, sample_input: torch.Tensor):
        """Manually trigger graph capture with the given sample input."""
        self._graphed = LumenGraphedCallable(
            self.module, (sample_input,), num_warmup=self._num_warmup
        )

    def release_graph(self):
        """Release the captured graph."""
        if self._graphed is not None:
            self._graphed.reset()
            self._graphed = None


def lumen_make_graphed_callables(
    callables: list[Callable],
    sample_args: list[tuple[torch.Tensor, ...]],
    num_warmup: int = 3,
) -> list[LumenGraphedCallable]:
    """Graph-capture multiple callables sharing a memory pool.

    Args:
        callables: List of functions to capture.
        sample_args: Corresponding sample arguments.
        num_warmup: Warmup iterations per callable.

    Returns:
        List of LumenGraphedCallable instances.
    """
    pool = (
        getattr(torch.cuda, "graph_pool_handle", lambda: None)()
        if torch.cuda.is_available()
        else None
    )
    graphed = []
    for fn, args in zip(callables, sample_args):
        graphed_callable = LumenGraphedCallable(
            fn, args, num_warmup=num_warmup, pool=pool
        )
        graphed.append(graphed_callable)
    return graphed


# ---------------------------------------------------------------------------
# Per-layer forward+backward graph capture
# ---------------------------------------------------------------------------


class _FwdGraphedLayerFn(torch.autograd.Function):
    """Forward-only graph replay with eager backward.

    Forward: replays the captured CUDA graph (fast, reduced kernel-launch
    overhead).  Backward: re-runs the layer forward eagerly to build a fresh
    autograd tape, then calls ``torch.autograd.backward`` on that tape.  This
    avoids capturing the backward graph (which requires a full extra
    forward+backward warmup at 98 %+ memory utilization) while still getting
    the forward kernel-launch savings.
    """

    @staticmethod
    def forward(
        ctx,
        real_input,
        static_fwd_input,
        static_fwd_output,
        fwd_graph,
        layer_fwd_fn,
        static_kwargs,
    ):
        ctx.layer_fwd_fn = layer_fwd_fn
        ctx.static_kwargs = static_kwargs
        ctx.save_for_backward(real_input)

        static_fwd_input.copy_(real_input)
        fwd_graph.replay()

        return static_fwd_output.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        (real_input,) = ctx.saved_tensors
        inp = real_input.detach().requires_grad_(True)

        with torch.enable_grad():
            out = ctx.layer_fwd_fn(inp, **ctx.static_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            torch.autograd.backward(out, grad_output)

        return (inp.grad, None, None, None, None, None)


class LumenGraphedLayer:
    """Per-layer forward+backward graph capture with lazy initialization.

    Graph capture is deferred until the first real training call so that
    warmup happens on the *actual* data path (priming Triton/AITER JIT
    caches) and capture occurs only after the memory pool is established.

    Uses ``torch.autograd.grad(only_inputs=True)`` for warmup and backward
    capture to avoid "Cannot set grad twice" errors with
    ``gradient_accumulation_fusion`` and FP8 parameter storage.
    Saves and restores ``main_grad`` across warmup.

    Modelled after Megatron-LM's ``_CudaGraphRunner``.

    Args:
        layer: The ``nn.Module`` transformer layer.
        num_warmup: Eager forward+backward passes before graph capture.
    """

    _shared_pool = None

    @classmethod
    def get_shared_pool(cls):
        """Return a shared graph pool handle for all graphed layers.

        Sharing a pool allows ROCm to reuse memory between layers since
        they execute sequentially — one layer's intermediates can overlap
        with another's in the pool.
        """
        if cls._shared_pool is None and torch.cuda.is_available():
            cls._shared_pool = getattr(torch.cuda, "graph_pool_handle", lambda: None)()
        return cls._shared_pool

    def __init__(self, layer: nn.Module, num_warmup: int = 3):
        self.layer = layer
        self._original_forward = layer.forward
        self._num_warmup = num_warmup
        self._call_count = 0
        self._captured = False

        self._fwd_graph: torch.cuda.CUDAGraph | None = None
        self._static_input: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None
        self._static_kwargs: dict[str, Any] | None = None
        self._extra_outputs: tuple = ()

        self._pool = self.get_shared_pool()

    def _do_capture(
        self,
        hidden_states: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> Any:
        """Capture forward-only graph. No warmup (layer already warmed by
        num_warmup real training steps). Backward runs eagerly via recompute."""
        device = hidden_states.device
        fwd = self._original_forward

        self._static_input = hidden_states.clone().detach()
        self._static_kwargs = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        from atomrl.lumen.ops.quantize.linear import set_graph_capture_mode

        set_graph_capture_mode(True)

        try:
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))

            self._fwd_graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            with (
                torch.no_grad(),
                torch.cuda.graph(
                    self._fwd_graph,
                    pool=self._pool,
                    stream=s,
                    capture_error_mode="thread_local",
                ),
            ):
                fwd_out = fwd(self._static_input, **self._static_kwargs)
            torch.cuda.current_stream(device).wait_stream(s)
        finally:
            set_graph_capture_mode(False)

        if isinstance(fwd_out, tuple):
            self._static_output = fwd_out[0]
            self._extra_outputs = fwd_out[1:]
        else:
            self._static_output = fwd_out
            self._extra_outputs = ()

        self._captured = True
        return self._replay(hidden_states, kwargs)

    def _replay(
        self,
        hidden_states: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> Any:
        """Replay captured forward graph; backward is eager via recompute."""
        for k, v in kwargs.items():
            if k in self._static_kwargs and isinstance(v, torch.Tensor):
                self._static_kwargs[k].copy_(v)

        out = _FwdGraphedLayerFn.apply(
            hidden_states,
            self._static_input,
            self._static_output,
            self._fwd_graph,
            self._original_forward,
            self._static_kwargs,
        )
        if self._extra_outputs:
            return (out,) + self._extra_outputs
        return out

    def __call__(self, hidden_states, **kwargs):
        if self._captured:
            return self._replay(hidden_states, kwargs)

        if not torch.is_grad_enabled():
            return self._original_forward(hidden_states, **kwargs)

        self._call_count += 1

        if self._call_count <= self._num_warmup:
            return self._original_forward(hidden_states, **kwargs)

        from atomrl.lumen.ops.dispatch import _backend_cache

        uncached = [
            k
            for k in _backend_cache
            if k.endswith(":prev") and k[:-5] not in _backend_cache
        ]
        if uncached:
            logger.debug("Deferring capture: %d ops not yet cached", len(uncached))
            return self._original_forward(hidden_states, **kwargs)

        try:
            result = self._do_capture(hidden_states, kwargs)
            logger.info(
                "Graph capture succeeded for layer after %d calls",
                self._call_count,
            )
            return result
        except RuntimeError as e:
            logger.warning("Graph capture failed: %s — permanent eager fallback", e)
            self._captured = False
            if self._fwd_graph is not None:
                del self._fwd_graph
                self._fwd_graph = None
            self._static_input = None
            self._static_output = None
            self._static_kwargs = None
            torch.cuda.empty_cache()
            self._num_warmup = float("inf")
            return self._original_forward(hidden_states, **kwargs)

    def disable(self):
        self._captured = False
        self._fwd_graph = None
        self._call_count = 0

    def enable(self):
        pass


def install_lazy_graph_capture(
    model: nn.Module,
    num_warmup: int = 3,
    skip_recomputed_layers: int = 0,
    max_graphed_layers: int = 0,
) -> int:
    """Replace non-checkpointed transformer layers' forward with a lazy
    graph-capture wrapper.

    Layers using activation checkpointing (recompute) are **skipped**
    because the checkpoint backward re-runs the forward, which is
    incompatible with graph replay's inplace static-buffer updates.

    Args:
        model: Megatron GPTModel with ``.decoder.layers``.
        num_warmup: Number of eager forward passes before capture.
        skip_recomputed_layers: Number of leading layers to skip
            (matches ``recompute_num_layers`` from Megatron config).
        max_graphed_layers: Maximum number of layers to graph (0 = all eligible).
            Limits memory usage in the graph pool at high memory utilization.

    Returns:
        Number of layers wrapped.
    """
    if not hasattr(model, "decoder") or model.decoder is None:
        logger.warning("Model has no decoder attribute, skipping graph wrappers")
        return 0
    if not hasattr(model.decoder, "layers"):
        logger.warning("Model decoder has no layers, skipping graph wrappers")
        return 0

    layers = model.decoder.layers
    wrapped = 0
    for l_no, layer in enumerate(layers):
        if l_no < skip_recomputed_layers:
            continue
        if max_graphed_layers > 0 and wrapped >= max_graphed_layers:
            break
        graphed = LumenGraphedLayer(layer, num_warmup=num_warmup)
        layer.forward = graphed
        wrapped += 1

    logger.info(
        f"Installed lazy graph capture on {wrapped}/{len(layers)} transformer layers "
        f"(skipped {skip_recomputed_layers} recomputed, "
        f"max {max_graphed_layers if max_graphed_layers > 0 else 'unlimited'}, "
        f"capture after {num_warmup} warmup steps)"
    )
    return wrapped
