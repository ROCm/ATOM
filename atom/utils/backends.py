# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import dataclasses
import logging
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional
import hashlib

from atom.utils.decorators import hash_factors
import torch
import torch.fx as fx
from atom.config import CompilationConfig, Config, CUDAGraphMode, DynamicShapesType
from atom.utils import (
    compilation_counter,
    envs,
    is_torch_equal_or_newer,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._logging._internal import trace_structured

from .compiler_inferface import (
    CompilerInterface,
    InductorAdaptor,
    InductorStandaloneAdaptor,
)

logger = logging.getLogger("atom")
from pydantic.dataclasses import dataclass

def get_static_graph_wrapper_cls(cls) -> str:
    return "atom.utils.cuda_graph.CUDAGraphWrapper"

@dataclass
class Range:
    """
    A range of numbers.
    Inclusive of start, inclusive of end.
    """

    start: int
    end: int

    def is_single_size(self) -> bool:
        return self.start == self.end

    def __contains__(self, size: int) -> bool:
        # Inclusive of start, inclusive of end
        return self.start <= size <= self.end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __str__(self) -> str:
        return f"({self.start}, {self.end})"

    def __repr__(self) -> str:
        return self.__str__()

import copy
class EagerAdaptor(CompilerInterface):
    name = "eager"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:

        compilation_counter.num_eager_compiles += 1
        # return graph, None
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        from torch._inductor import standalone_compile
        dynamic_shapes = "from_tracing_context"
        compiled_graph = standalone_compile(
            graph,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={"config_patches": {}},
        )

        return compiled_graph, None


def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:
    # return InductorAdaptor()
    if compilation_config.use_inductor:
    # if False:
        if is_torch_equal_or_newer("2.8.0.dev"):
            logger.debug("Using InductorStandaloneAdaptor")
            return InductorStandaloneAdaptor(compilation_config.compile_cache_save_format)
        else:
            logger.debug("Using InductorAdaptor")
            return InductorAdaptor()
    else:
        logger.info("Using EagerAdaptor")
        return EagerAdaptor()


def make_copy_and_call(
    sym_tensor_indices: list[int],
    input_buffers: list[torch.Tensor | None],
    callable_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a wrapper that copies inputs to static buffers before calling.

    This is used for cudagraph input copying where we need to copy dynamic
    tensors to static buffers before invoking the compiled graph.

    Args:
        sym_tensor_indices: Indices of tensors with symbolic shapes
        input_buffers: List of static buffers (can contain None for lazy init)
        callable_fn: The compiled function to call

    Returns:
        A wrapper function that copies inputs and calls the compiled function
    """

    def copy_and_call(*args: Any) -> Any:
        list_args = list(args)
        for i, index in enumerate(sym_tensor_indices):
            runtime_tensor = list_args[index]
            runtime_shape = runtime_tensor.shape[0]

            # lazy initialization of buffer on first call
            if input_buffers[i] is None:
                input_buffers[i] = runtime_tensor.clone()

            static_tensor = input_buffers[i][:runtime_shape]  # type: ignore[index]
            static_tensor.copy_(runtime_tensor)
            list_args[index] = static_tensor
        return callable_fn(*list_args)

    return copy_and_call

class StopCompiling(BaseException):
    pass


class CompilerManager:
    """
    A manager to manage the compilation process, including
    caching the compiled graph, loading the compiled graph,
    and compiling the graph.

    The cache is a dict mapping
    `(runtime_shape, graph_index, backend_name)`
    to `any_data` returned from the compiler.

    When serializing the cache, we save it to a Python file
    for readability. We don't use json here because json doesn't
    support int as key.
    """

    def __init__(self, compilation_config: CompilationConfig):
        self.cache: dict[tuple[Optional[int], int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)
        self.loaded_artifacts: dict[str, Any] = {}

    def compute_hash(self, vllm_config: Config) -> str:
        return self.compiler.compute_hash(vllm_config)

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        """
        Initialize the cache directory for the compiler.

        The organization of the cache directory is as follows:
        cache_dir=/path/to/hash_str/rank_i_j/prefix/
        inside cache_dir, there will be:
        - vllm_compile_cache.py
        - computation_graph.py
        - transformed_code.py

        for multiple prefixes, they can share the same
        base cache dir of /path/to/hash_str/rank_i_j/ ,
        to store some common compilation artifacts.
        """

        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "vllm_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            # load the cache from the file
            with open(self.cache_file_path) as f:
                # we use ast.literal_eval to parse the data
                # because it is a safe way to parse Python literals.
                # do not use eval(), it is unsafe.
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Optional[Callable]:
        if (runtime_shape, graph_index, self.compiler.name) not in self.cache:
            return None
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via " "handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph


    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        additional_inductor_config,
        compilation_config: CompilationConfig,
        compile_range: Range,
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
    ) -> Any:
        # print("runtime_shape", runtime_shape)
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        compiled_graph = self.load(graph, example_inputs, graph_index, runtime_shape)
        if compiled_graph is not None:
            if graph_index == num_graphs - 1:
                # after loading the last graph for this shape, record the time.
                # there can be multiple graphs due to piecewise compilation.
                now = time.time()
                elapsed = now - compilation_start_time
                if runtime_shape is None:
                    logger.info(
                        "Directly load the compiled graph(s) for dynamic shape "
                        "from the cache, took %.3f s",
                        elapsed,
                    )
                else:
                    logger.info(
                        "Directly load the compiled graph(s) for shape %s "
                        "from the cache, took %.3f s",
                        str(runtime_shape),
                        elapsed,
                    )
            return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            # Let compile_fx generate a key for us
            maybe_key = None
        else:
            maybe_key = "artifact_compile_range_"
            maybe_key += f"{compile_range.start}_{compile_range.end}"
            maybe_key += f"_subgraph_{graph_index}"
        # with self.compile_context(compile_range):
            # There is a compilation time optimization here.
            #
            # If the (input metdata, graph, compiler config) are the same, then
            # we want to avoid compiling the same artifact again. If we didn't
            # do this optimization, the backend compilation (InductorAdaptor or
            # InductorStandaloneAdaptor)
            # is able to cache hit and produce an artifact faster if it was
            # already created, but it is still a duplicate artifact that
            # requires unnecessary things e.g. disk IO.
            #
            # The optimization is: If the backend compilation cache hits,
            # then do an early return from the backend compilation and look up
            # which of the previous in-memory artifacts we created to reuse.
            #
            # We implemented this by monkey-patching torch (torch does not
            # easily expose the cache_key function), but in the future torch
            # should expose the cache_key function that we can just call
            # directly before invoking backend compilation.
        cache_key = None
        orig = torch._functorch._aot_autograd.autograd_cache.autograd_cache_key

        def autograd_cache_key(*args, **kwargs):
            result = orig(*args, **kwargs)
            if result is None:
                return None
            nonlocal cache_key
            cache_key = result[0]
            if cache_key in self.loaded_artifacts:
                raise StopCompiling()
            return result

        from unittest.mock import patch

        with (
            # Graphs that are isometric (different node names but same
            # structure) should be treated as the same.
            # torch._functorch.config.patch(autograd_cache_normalize_inputs=True),
            patch(
                "torch._functorch._aot_autograd.autograd_cache.autograd_cache_key",
                autograd_cache_key,
            ),
        ):
            try:
                compiled_graph, handle = self.compiler.compile(
                    graph,
                    example_inputs,
                    additional_inductor_config,
                    compile_range,
                    key=maybe_key,
                )
            except StopCompiling:
                assert cache_key is not None
                return self.loaded_artifacts[cache_key]
        if cache_key is not None and compiled_graph is not None:
            self.loaded_artifacts[cache_key] = compiled_graph

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        is_compile_cache_enabled = True
        if is_compile_cache_enabled and handle is not None:
            self.cache[(compile_range, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                logger.info_once(
                    "Cache the graph of compile range %s for later use",
                    str(compile_range),
                )
            logger.debug(
                "Store the %s-th graph for compile range%s from %s via handle %s",
                graph_index,
                str(compile_range),
                self.compiler.name,
                handle,
            )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            compilation_config.compilation_time += elapsed
            logger.info(
                "Compiling a graph for compile range %s takes %.2f s",
                str(compile_range),
                elapsed,
            )

        return compiled_graph

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


# used to judge whether the node should be split or not
def _split_judge_func(node: fx.Node) -> bool:
    # ATOM use mark_spliting_op to mark the attn as splitting op
    if node.op == "call_function" and (
        hasattr(node.target, "spliting_op") and (node.target.spliting_op)
    ):
        return True

    # When plugin mode(vLLM), the attention impl op is registered
    # as unified_attention
    from atom.plugin import is_vllm

    if is_vllm() and "unified_attention" in node.name:
        return True

    return False


def split_graph(
    graph: fx.GraphModule, ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if _split_judge_func(node):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by integer graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: Config,
        vllm_backend: "VllmBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        gm = getattr(self.module, target)
        outputs = gm.graph.output_node().args[0]
        output = fx.map_arg(outputs, lambda node: node.meta["example_value"])

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            # Lazy import here to avoid circular import
            from torch._inductor.compile_fx import graph_returns_tuple

            from .cuda_piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                self.vllm_backend,
                graph_returns_tuple(submod),
                submod_name=target,
            )

            # if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            #     # resolve the static graph wrapper class (e.g. CUDAGraphWrapper
            #     # class) as platform dependent.
            #     static_graph_wrapper_class = resolve_obj_by_qualname(
            #         get_static_graph_wrapper_cls())

            #     # Always assign PIECEWISE runtime mode to the
            #     # CUDAGraphWrapper for piecewise_backend, to distinguish
            #     # it from the FULL cudagraph runtime mode, no matter it
            #     # is wrapped on a full or piecewise fx graph.
            #     self.module.__dict__[target] = static_graph_wrapper_class(
            #         runnable=piecewise_backend,
            #         vllm_config=self.vllm_config,
            #         runtime_mode=CUDAGraphMode.PIECEWISE,
            #         cudagraph_options=CUDAGraphOptions(
            #             debug_log_enable=piecewise_backend.is_first_graph,
            #             gc_disable=not piecewise_backend.is_first_graph,
            #             weak_ref_output=piecewise_backend.is_last_graph))
            # else:
            self.module.__dict__[target] = piecewise_backend

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


# the tag for the part of model being compiled,
# e.g. backbone/eagle_head
model_tag: str = "backbone"


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert (
        tag != model_tag
    ), f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


VLLM_CACHE_ROOT = os.path.expanduser("~/.cache/atom")


class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation level of `CompilationLevel.PIECEWISE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: Config
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager
    inductor_config: dict[str, Any]

    def __init__(
        self,
        vllm_config: Config,
        prefix: str = "",
    ):

        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag

        # Passes to run on the graph post-grad.
        # self.post_grad_pass_manager = PostGradPassManager()

        self.sym_tensor_indices = []
        self.input_buffers = []

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

        self.inductor_config = copy.deepcopy(self.compilation_config.inductor_compile_config)

        self.is_encoder = False

        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    # def configure_post_pass(self):
    #     config = self.compilation_config
    #     self.post_grad_pass_manager.configure(self.vllm_config)

    #     # Post-grad custom passes are run using the post_grad_custom_post_pass
    #     # hook. If a pass for that hook exists, add it to the pass manager.
    #     inductor_config = config.inductor_compile_config
    #     PASS_KEY = "post_grad_custom_post_pass"
    #     if PASS_KEY in inductor_config:
    #         # Config should automatically wrap all inductor passes
    #         if isinstance(inductor_config[PASS_KEY], PostGradPassManager):
    #             assert (inductor_config[PASS_KEY].uuid() ==
    #                     self.post_grad_pass_manager.uuid())
    #         else:
    #             assert isinstance(inductor_config[PASS_KEY], InductorPass)
    #             self.post_grad_pass_manager.add(inductor_config[PASS_KEY])
    #     inductor_config[PASS_KEY] = self.post_grad_pass_manager

    def collect_standalone_compile_artifacts(
        self,
    ) -> tuple[Any, dict[str, list[int]] | None, dict[str, bool] | None]:
        """Collect inductor cache artifacts from all piecewise backends.

        Returns:
            tuple: (standalone_compile_artifacts, sym_shape_indices_map,
                    returns_tuple_map)
                - standalone_compile_artifacts: StandaloneCompiledArtifacts
                  with compiled artifacts
                - sym_shape_indices_map: dict mapping submod_name to
                  sym_shape_indices
                - returns_tuple_map: dict mapping submod_name to
                  returns_tuple
        """

        if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            return None, None, None

        from .caching import StandaloneCompiledArtifacts
        from .cuda_piecewise_backend import PiecewiseBackend

        standalone_compile_artifacts = StandaloneCompiledArtifacts()
        sym_shape_indices_map = {}
        returns_tuple_map = {}

        for name, _ in self.split_gm.named_children():
            # get the actual attribute (shadowed by PiecewiseBackend in __dict__)
            child = getattr(self.split_gm, name)
            # unwrap the static graph wrapper class if applicable
            piecewise_backend = child.runnable if hasattr(child, "runnable") else child

            if not isinstance(piecewise_backend, PiecewiseBackend):
                continue

            submod_name = name
            sym_shape_indices_map[submod_name] = piecewise_backend.sym_shape_indices
            returns_tuple_map[submod_name] = piecewise_backend.returns_tuple

            for shape_str, bytes_data in piecewise_backend.to_bytes().items():
                standalone_compile_artifacts.insert(submod_name, shape_str, bytes_data)
                logger.debug(
                    "collected artifact for %s shape %s (%d bytes)",
                    submod_name,
                    shape_str,
                    len(bytes_data),
                )

        logger.info(
            "collected artifacts: %d entries, %d artifacts, %d bytes total",
            standalone_compile_artifacts.num_entries(),
            standalone_compile_artifacts.num_artifacts(),
            standalone_compile_artifacts.size_bytes(),
        )

        logger.debug(
            "standalone compile artifact keys: %s",
            list(standalone_compile_artifacts.submodule_bytes.keys()),
        )

        return standalone_compile_artifacts, sym_shape_indices_map, returns_tuple_map


    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        from .caching import (
            VllmSerializableFunction,
        )

        vllm_config = self.vllm_config

        # self._log_compilation_config()

        # Minimal hashing here with existing utilities, reused below.

        # env_factors = envs.compile_factors()
        # env_hash = hash_factors(env_factors)
        # Compute config/compiler/code hashes once and reuse
        config_hash = vllm_config.compute_hash()
        compiler_hash = self.compiler_manager.compute_hash(vllm_config)
        forward_code_files = list(sorted(self.compilation_config.traced_files))

        # logger.debug(
        #     "Traced files (to be considered for compilation cache):\n%s",
        #     lazy(lambda: "\n".join(forward_code_files)),
        # )
        hash_content = []
        for filepath in forward_code_files:
            hash_content.append(filepath)
            if filepath == "<string>":
                # This means the function was dynamically generated, with
                # e.g. exec(). We can't actually check these.
                continue
            try:
                with open(filepath) as f:
                    hash_content.append(f.read())
            except (OSError, UnicodeDecodeError):
                logger.warning("Failed to read file %s", filepath)
                continue
        code_hash = hashlib.sha256("\n".join(hash_content).encode()).hexdigest()
        # Clear after consumption
        self.compilation_config.traced_files.clear()
        if not self.compilation_config.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.
            factors = [config_hash, code_hash, compiler_hash]
            # Use SHA-256 for cache key hashing to be consistent across
            # compute_hash functions. Truncate for a short cache dir name.
            hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
            cache_dir = os.path.join(
                VLLM_CACHE_ROOT, "torch_compile_cache", hash_key
            )
            self.compilation_config.cache_dir = cache_dir

        cache_dir = self.compilation_config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.compilation_config.cache_dir = cache_dir
        # rank = vllm_config.parallel_config.rank
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_cache_dir = os.path.join(cache_dir, f"rank_{dp_rank}", self.prefix)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compilation_config.local_cache_dir = local_cache_dir

        # Honors opt-outs such as CompilationMode.NONE or VLLM_DISABLE_COMPILE_CACHE.
        disable_cache = False

        if disable_cache:
            logger.info("vLLM's torch.compile cache is disabled.")
        else:
            logger.info(
                "Using cache directory: %s for vLLM's torch.compile",
                local_cache_dir
            )

        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache, self.prefix
        )

        # Reuses existing cache key

        logger.debug(
            "torch.compile cache factors: cfg=%s comp=%s code=%s dir=%s",
            # env_hash,
            config_hash,
            compiler_hash,
            code_hash,
            local_cache_dir,
        )

        # Persist and log only hash-relevant factors together.
        try:
            # logger.debug(
            #     "Compile env factors (raw):\n%s\nVllm config hash: %s",
            #     lazy(partial(pprint.pformat, env_factors, width=120)),
            #     config_hash,
            # )
            meta_path = os.path.join(local_cache_dir, "cache_key_factors.json")
            if not os.path.exists(meta_path):
                with open(meta_path, "w") as f:
                    json.dump(
                        {
                            # "env": env_factors,  # raw factors used for env_hash
                            "config_hash": config_hash,
                            "code_hash": code_hash,
                            "compiler_hash": compiler_hash,
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
        except Exception:
            # Best-effort only; metadata write failures are non-fatal.
            logger.warning(
                (
                    "Could not write compile cache metadata at %s; continuing without "
                    "metadata. Compiled cache remains valid; diagnostics may be "
                    "limited."
                ),
                local_cache_dir,
                exc_info=True,
            )

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        from .decorators import torch_compile_start_time

        dynamo_time = time.time() - torch_compile_start_time
        logger.info(
            "Dynamo bytecode transform time: %.2f s", dynamo_time
        )
        self.compilation_config.compilation_time += dynamo_time

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # self.configure_post_pass()

        # if self.compilation_config.use_inductor_graph_partition:
        #     # Let Inductor decide partitioning; avoid FX-level pre-splitting.
        #     fx_split_ops: list[str] = []
        # else:
        fx_split_ops = self.compilation_config.splitting_ops or []

        self.split_gm, self.piecewise_graphs = split_graph(graph, fx_split_ops)

        # keep a split_gm copy from BEFORE the interpreter replaces
        # submodules with PiecewiseBackend -- used for serialization
        original_split_gm = None
        if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            original_split_gm = copy.deepcopy(self.split_gm)

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        # Log the piecewise split graph for TORCH_TRACE/tlparse
        trace_structured(
            "graph_dump",
            metadata_fn=lambda: {"name": "vllm_piecewise_split_graph"},
            payload_fn=lambda: self.split_gm.print_readable(print_output=False),
        )

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # Extract fake values from the graph to use them when needed.
        all_fake_values = []
        for i in graph.graph.find_nodes(op="placeholder"):
            all_fake_values.append(i.meta["example_value"])

        fake_args = [
            all_fake_values[i] if isinstance(t, torch.Tensor) else t
            for i, t in enumerate(example_inputs)
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*fake_args)

        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()

        if (
            self.compilation_config.dynamic_shapes_config.evaluate_guards
            and self.compilation_config.dynamic_shapes_config.type
            == DynamicShapesType.BACKED
        ):
            from torch.utils._sympy.value_ranges import ValueRanges

            # Drop counter-0/1 specializations guards; for backed dynamic shapes,
            # torch.compile will specialize for 0/1 inputs or otherwise guards that
            # shape is >= 2. This is because it's really hard not to hit a check
            # against 0/1. When we evaluate shape guards, we exclude checking those
            # guards (We would fail always otherwise).

            # We avoid that by updating the ranges of backed sizes when the min is
            # 2 for any, we assume it's 0.
            for s, r in fake_mode.shape_env.var_to_range.items():
                if r.lower == 2:
                    fake_mode.shape_env.var_to_range[s] = ValueRanges(0, r.upper)

        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from
            # https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30
            # use `print_readable` because it can include submodules
            src = (
                "from __future__ import annotations\nimport torch\n"
                + self.split_gm.print_readable(print_output=False)
            )
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.debug(
                "Computation graph saved to %s", graph_path
            )

        self._called = True
        graph_to_serialize = (
            original_split_gm if envs.VLLM_USE_MEGA_AOT_ARTIFACT else self.graph
        )

        if (
            self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or not self.compilation_config.cudagraph_copy_inputs
        ):
            return VllmSerializableFunction(
                graph_to_serialize,
                example_inputs,
                self.prefix,
                self.split_gm,
                is_encoder=self.is_encoder,
                vllm_backend=self,
            )

        # index of tensors that have symbolic shapes (batch size)
        # for weights and static buffers, they will have concrete shapes.
        # symbolic shape only happens for input tensors.
        from torch.fx.experimental.symbolic_shapes import is_symbolic

        sym_tensor_indices = [
            i
            for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor)
            and any(is_symbolic(d) for d in x.size())
        ]

        # compiler managed cudagraph input buffers
        # we assume the first run with symbolic shapes
        # has the maximum size among all the tensors
        copy_and_call = make_copy_and_call(
            sym_tensor_indices,
            [example_inputs[x].clone() for x in sym_tensor_indices],
            self.split_gm,
        )

        return VllmSerializableFunction(
            graph_to_serialize,
            example_inputs,
            self.prefix,
            copy_and_call,
            is_encoder=self.is_encoder,
            vllm_backend=self,
            sym_tensor_indices=sym_tensor_indices,
        )
