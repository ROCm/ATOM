# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
from aiter import logger

from atom.config import Config, CUDAGraphMode
from atom.utils import compilation_counter, weak_ref_tensors
from atom.utils.forward_context import get_forward_context

# from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
# from vllm.config import CUDAGraphMode, VllmConfig
# from vllm.forward_context import BatchDescriptor, get_forward_context
# from vllm.logger import init_logger
# from vllm.platforms import current_platform

# logger = init_logger(__name__)


class BatchDescriptor(NamedTuple):
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of
    items as minimal as possible to properly and uniquely describe the padded
    batch for cudagraph.
    """

    num_tokens: int
    uniform_decode: bool = False
    """
    False can also be used for an uniform decode batch to dispatch to the 
    cudagraph supporting non-uniform batches.
    """

    @property
    def non_uniform(self) -> "BatchDescriptor":
        """
        Return a non-uniform version of current batch descriptor.
        """
        return BatchDescriptor(self.num_tokens, uniform_decode=False)


@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


# One cudagraph pool per num_tokens bucket (default). A bucket's graphs only
# ever reuse memory among themselves, so nothing one bucket's graph wrote can be
# handed to another bucket's.
#
# Sharing one pool across buckets corrupts DeepSeek-V4 decode. Measured on
# V4-Pro-DSpark tp8, GSM8K 3-shot 250q, --cudagraph-mode PIECEWISE:
#
#   pool shared, 11 buckets   0.664      pool shared, 2 buckets [32,64]  0.956
#   pool shared, 1 bucket     0.976      per-bucket, 11 buckets          0.960
#
# Generations stay coherent for a few decode tokens and then collapse, prefill
# is unaffected, and it is intermittent. Removing cross-bucket sharing by either
# route (one bucket, or one pool each) fixes it; the corrupted memory is inside
# the compiled pieces, not on the graph->eager boundary — pinning every boundary
# tensor of the one V4 split op changes nothing (0.636).
_graph_pools: dict = {}

# ATOM_PER_BUCKET_POOL=0 restores the single shared pool. It buys little: on the
# config above the two modes differ by 1.11GB reserved per rank (202.83 ->
# 203.94GB) and the capture-time allocated delta is identical at 14.71GB, so the
# overlay was not reducing live footprint at all. The wider claim it was added
# for (~10GB vs ~35GB unshared on DSV4 TP8) does not reproduce there; a config
# with far more or larger buckets, or DP, may still see a real gap.
_shared_graph_pool: Any | None = None


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic.
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: Config,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = True

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams

        # self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        # Rebound on the first capture when the shared pool is opted into;
        # `_graph_pools` is only mutated in place, so it needs no declaration.
        global _shared_graph_pool

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.info(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            # ================= TODO lirong ========================
            # validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                # Default: one pool per num_tokens bucket, so no bucket's graph
                # can be handed memory another bucket's graph wrote. See the
                # module header for the accuracy data behind this default and
                # for what ATOM_PER_BUCKET_POOL=0 costs.
                _per_bucket = os.environ.get("ATOM_PER_BUCKET_POOL", "1") == "1"
                _bkey = batch_descriptor.num_tokens if batch_descriptor else 0
                if _per_bucket:
                    _pool = _graph_pools.get(_bkey)
                else:
                    # Match vLLM (platforms/interface.py:get_global_graph_pool):
                    # use a DEDICATED shareable pool handle created ONCE up front
                    # for EVERY graph.
                    if _shared_graph_pool is None:
                        _shared_graph_pool = torch.cuda.graph_pool_handle()
                    _pool = _shared_graph_pool
                # thread_local, not the "global" default: global mode invalidates
                # the capture when ANY thread makes an unsafe HIP call, and under
                # DP attention the NCCL watchdog thread polls hipEventQuery on
                # outstanding works every ~100ms. Piecewise captures one small
                # graph per layer with eager (collective-issuing) sections in
                # between, so that poll reliably lands inside a capture and kills
                # the process with hipErrorStreamCaptureUnsupported. thread_local
                # restricts the check to the capturing thread, which is the only
                # one touching this stream.
                # Capture on graph_capture()'s stream (vLLM parity), not torch's
                # own side stream, so pieces + collectives stay coherent.
                with torch.cuda.graph(
                    cudagraph,
                    pool=_pool,
                    stream=torch.cuda.current_stream(),
                    capture_error_mode="thread_local",
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            # first graph of a per-bucket pool -> remember its pool. (shared pool
            # is created up front via graph_pool_handle() above, nothing to do.)
            if _per_bucket and _bkey not in _graph_pools:
                _graph_pools[_bkey] = cudagraph.pool()

            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
