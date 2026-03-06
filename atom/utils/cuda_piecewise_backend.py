# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import dataclasses
from typing import Any, Callable

from atom.utils import Range
import torch.fx as fx

# import vllm.envs as envs
from atom.utils.backends import VllmBackend
from atom.utils.decorators import end_monitoring_torch_compile
from atom.config import Config
from pickle import Pickler
import contextvars
from contextlib import contextmanager
from collections.abc import Callable, Generator
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._logging._internal import trace_structured
import pickle
import torch
import logging
logger = logging.getLogger("atom")
@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore


@dataclasses.dataclass
class RangeEntry:
    compile_range: Range
    compiled: bool = False
    runnable: Callable[..., Any] = None  # type: ignore


_on_compilation_complete_callback: contextvars.ContextVar[Callable[[], None] | None] = (
    contextvars.ContextVar("on_compilation_complete_callback", default=None)
)

@contextmanager
def set_on_compilation_complete(
    callback: Callable[[], None],
) -> Generator[None, None, None]:
    token = _on_compilation_complete_callback.set(callback)
    try:
        yield
    finally:
        _on_compilation_complete_callback.reset(token)


class PiecewiseBackend:

    def __init__(
        self,
        graph: fx.GraphModule,
        vllm_config: Config,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        # compiled_graph_for_general_shape: Callable,
        vllm_backend: VllmBackend,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable[..., Any]] | None = None,
        submod_name: str = "",
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1

        self.compile_sizes: set[int] = set(self.compilation_config.compile_sizes)
        # self.compile_sizes = [1, 2, 4, 16384]

        self.first_run_finished = False

        # self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.returns_tuple = returns_tuple
        self.compiled_runnables = compiled_runnables
        self.submod_name = submod_name

        # self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to compile
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        self.on_compilation_complete = _on_compilation_complete_callback.get()
        # We only keep compilation management inside this class directly.
        self.range_entries: dict[Range, RangeEntry] = {}

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        self.to_be_compiled_ranges: set[Range] = set(self.compile_ranges)

        if self.compile_sizes is not None:
            for size in self.compile_sizes:
                if isinstance(size, str):
                    assert size == "cudagraph_capture_sizes"
                    raise NotImplementedError(
                        "cudagraph_capture_sizes not supported in compile_sizes."
                        "This should be handled in `post_init_cudagraph_sizes`."
                    )
                else:
                    assert isinstance(size, int)
                    range = Range(start=size, end=size)
                    if range not in self.compile_ranges:
                        self.range_entries[range] = RangeEntry(
                            compile_range=range,
                        )
                        self.to_be_compiled_ranges.add(range)

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

        # Track whether we've logged the graph for this subgraph (only log once)
        self._graph_logged = False


    def get_compiled_graph_wrapper(
        self, compiled_graph: Callable[..., Any]
    ) -> Callable[..., Any]:
        def compiled_graph_wrapper(*args: Any) -> Any:
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    def check_for_ending_compilation(self) -> None:
        if self.is_last_graph:
            end_monitoring_torch_compile(self.vllm_config)
            if self.on_compilation_complete is not None:
                self.on_compilation_complete()

    def to_bytes(self) -> dict[str, bytes]:
        class StandaloneCompiledArtifactsPickler(Pickler):
            def reducer_override(self, obj: object) -> Any:
                if isinstance(obj, CachingAutotuner):
                    obj.prepare_for_pickle()
                    return pickle.loads, (
                        pickle.dumps(
                            obj,
                        ),
                    )
                return NotImplemented

        def serialize(fn: Callable[..., Any]) -> bytes:
            assert hasattr(fn, "serialize"), "fn must have serialize method"
            with torch._functorch.config.patch("bundled_autograd_cache", True):
                entry = fn.serialize()

                f = io.BytesIO()
                StandaloneCompiledArtifactsPickler(f).dump(entry)
                result = f.getvalue()
            return result

        out = {}

        for range_key, entry in self.range_entries.items():
            if not entry.compiled:
                logger.debug(
                    "entry with range %s not compiled, so cannot get its bytes",
                    range_key,
                )
                continue
            if hasattr(entry.runnable, "serialize"):
                out[str(range_key)] = serialize(entry.runnable)

        return out

    def _fakify_args(self, args: tuple[Any, ...]) -> list[Any]:
        # We need to pass fake example_inputs, otherwise torch.compile
        # will fakify the example_inputs potentially causing some non dynamic
        # dimension to be be duck shaped to other existing shapes that have hints
        # matching their values.
        # This is problem because it can lead to unintended specializations!
        # if the new wrongly dynamic dim is specialized
        # it will force specializing the whole shape
        # torch.compile probably should not accept
        # non fake tensors as example inputs!
        # See issue https://github.com/vllm-project/vllm/issues/27899
        fake_example_inputs = []
        assert self.graph is not None
        for node in self.graph.graph.nodes:
            # All place holders come first
            if node.op == "placeholder":
                fake_example_inputs.append(node.meta["example_value"])
            else:
                break
        assert len(fake_example_inputs) == len(args)
        return fake_example_inputs

    def _log_compile_start(self, compile_range: Range):
        """Log compilation event for TORCH_TRACE/tlparse."""
        is_cudagraph_size = (
            self.compile_sizes is not None and compile_range.start in self.compile_sizes
        )
        subgraph_index = self.piecewise_compile_index
        submod_name = self.submod_name
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "vllm_piecewise_compile_start",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {
                    "piecewise_index": subgraph_index,
                    "submod_name": submod_name,
                    "total_piecewise_compiles": self.total_piecewise_compiles,
                    "compile_range_start": compile_range.start,
                    "compile_range_end": compile_range.end,
                    "is_single_size": compile_range.is_single_size(),
                    "is_cudagraph_capture_size": is_cudagraph_size,
                }
            ),
        )

        # Log the subgraph graph dump only once per subgraph (not per size)
        # to reduce log file size. The graph code is the same for all sizes.
        if not self._graph_logged:
            self._graph_logged = True
            assert self.graph is not None
            trace_structured(
                "graph_dump",
                metadata_fn=lambda: {
                    "name": f"vllm_{submod_name}",
                },
                payload_fn=lambda: self.graph.print_readable(print_output=False),
            )

    def _maybe_compile_for_range_entry(
        self, range_entry: RangeEntry, args: tuple[Any, ...]
    ) -> Any:
        if not range_entry.compiled:
            if self.compiled_runnables is not None:
                range_entry.runnable = self.get_compiled_graph_wrapper(
                    self.compiled_runnables[str(range_entry.compile_range)]
                )
            else:
                self._log_compile_start(range_entry.compile_range)

                # args are real arguments
                # fakify for range, real args for concrete size.
                # For concrete size, we clear the shape env in
                # compiler_manager.compile() so no need to fakify.
                args_list = (
                    self._fakify_args(args)
                    if not range_entry.compile_range.is_single_size()
                    else list(args)
                )

                with (
                    torch._functorch.config.patch("bundled_autograd_cache", True),
                ):
                    range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                        self.graph,
                        args_list,
                        self.vllm_backend.inductor_config,
                        self.compilation_config,
                        compile_range=range_entry.compile_range,
                        graph_index=self.piecewise_compile_index,
                        num_graphs=self.total_piecewise_compiles,
                    )

            range_entry.compiled = True
            self.to_be_compiled_ranges.remove(range_entry.compile_range)

            self.check_for_ending_compilation()

    def _find_range_for_shape(self, runtime_shape: int) -> RangeEntry | None:
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if self.compile_sizes is None:
            self.compile_sizes = []

        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    def __call__(self, *args: Any) -> Any:
        runtime_shape = int(args[self.sym_shape_indices[0]])
        range_entry = self._find_range_for_shape(runtime_shape)

        assert range_entry is not None, (
            f"Shape: {runtime_shape} out of considered ranges: {self.compile_ranges}"
        )

        self._maybe_compile_for_range_entry(range_entry, args)
        return range_entry.runnable(*args)
