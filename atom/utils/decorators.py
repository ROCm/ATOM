# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Callable, Optional, TypeVar, Union, TYPE_CHECKING
import inspect
import os
import sys
from types import CodeType
from abc import abstractmethod
from contextlib import contextmanager
from unittest.mock import patch
import logging

from atom.models.utils import IntermediateTensors
from atom.utils import envs, is_torch_equal_or_newer
from atom.utils.wrapper import TorchCompileWithNoGuardsWrapper
from torch._dynamo.symbolic_convert import InliningInstructionTranslator
import torch
import torch.nn as nn
import time
from collections.abc import Callable, Generator
if TYPE_CHECKING:
    # Only added on nightly/2.10 so wrap
    try:
        from torch._dynamo.package import SourceInfo
    except ImportError:
        # Fallback for old versions not supporting
        SourceInfo = Any


logger = logging.getLogger("atom")

from atom.config import CompilationConfig, Config, CompilationLevel, DynamicShapesType, set_current_atom_config

# from atom.utils import start_monitoring_torch_compile

_T = TypeVar("_T", bound=type[nn.Module])

context_manager = None
torch_compile_start_time: float = 0.0


# We remove it from utils/__init__.py to avoid circular import
def start_monitoring_torch_compile(vllm_config: Config):
    global torch_compile_start_time
    torch_compile_start_time = time.time()

    compilation_config: CompilationConfig = vllm_config.compilation_config
    if (
        compilation_config.level == CompilationLevel.PIECEWISE
        and compilation_config.debug_dump_path
    ):
        import depyf

        path = os.path.join(compilation_config.debug_dump_path, "rank_0")
        # f"rank_{vllm_config.parallel_config.rank}")
        global context_manager
        context_manager = depyf.prepare_debug(path)
        context_manager.__enter__()


def end_monitoring_torch_compile(vllm_config: Config):
    compilation_config: CompilationConfig = vllm_config.compilation_config
    if compilation_config.level == CompilationLevel.PIECEWISE:
        global context_manager
        if context_manager is not None:
            context_manager.__exit__(None, None, None)
            context_manager = None


def init_backend(config: Config):
    from .backends import VllmBackend

    return VllmBackend(config)


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(
        self,
        vllm_config: Config,
        compiled_callable: Optional[Callable] = None,
        compilation_level: int = 0,
    ):
        self.vllm_config = vllm_config

        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method
            options = None
            backend = init_backend(vllm_config)
            compiled_callable = torch.compile(
                self.forward,
                # fullgraph=True,
                backend=backend,
                # dynamic=True,
                options=options,
            )

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = (
            compilation_level >= CompilationLevel.DYNAMO_ONCE
        )

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        # print('compiled_callable=====================')
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return
        # print("new_code", new_code)
        self.compiled_codes.append(new_code)
        debug_dump_dir = self.vllm_config.compilation_config.debug_dump_path
        if isinstance(debug_dump_dir, str) and debug_dump_dir != "":
            # rank = self.vllm_config.parallel_config.rank
            rank = 0
            decompiled_file = os.path.join(
                debug_dump_dir, f"rank_{rank}", "transformed_code.py"
            )
            if not os.path.exists(decompiled_file):
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    import depyf

                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)
                except Exception:
                    pass

        if (
            self.vllm_config.compilation_config.use_cudagraph
            and "update" in new_code.co_names
        ):
            import depyf

            src = depyf.decompile(new_code)
            msg = (
                "Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler because it will cause silent errors. Please use eager mode or fix the code. The following code contains clues about which buffer is being modified (please search for the usage of the function `update`):\n"
                + src
            )  # noqa
            raise RuntimeError(msg)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """  # noqa
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object

import contextvars
import hashlib
import json

_on_compilation_complete_callback: contextvars.ContextVar[Callable[[], None] | None] = (
    contextvars.ContextVar("on_compilation_complete_callback", default=None)
)

def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def get_inductor_factors() -> list[Any]:
    factors: list[Any] = []
    # summarize system state
    from torch._inductor.codecache import CacheBase

    system_factors = CacheBase.get_system()
    factors.append(system_factors)

    # summarize pytorch state
    from torch._inductor.codecache import torch_key

    torch_factors = torch_key()
    factors.append(torch_factors)
    return factors


def aot_compile_hash_factors(vllm_config: Config) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = hash_factors(envs.compile_factors())
    factors.append(env_hash)

    # 1. factors come from the vllm_config (it mainly summarizes how the
    #    model is created)
    config_hash = vllm_config.compute_hash()
    factors.append(config_hash)

    # 2. inductor factors if applicable
    if envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        factors.extend(get_inductor_factors())

    return factors



@contextmanager
def set_on_compilation_complete(
    callback: Callable[[], None],
) -> Generator[None, None, None]:
    token = _on_compilation_complete_callback.set(callback)
    try:
        yield
    finally:
        _on_compilation_complete_callback.reset(token)

def _model_hash_key(fn: Callable[..., Any]) -> str:
    import atom
    sha256_hash = hashlib.sha256()
    sha256_hash.update(atom.__version__.encode())
    sha256_hash.update(fn.__qualname__.encode())
    sha256_hash.update(str(fn.__code__.co_firstlineno).encode())
    return sha256_hash.hexdigest()



def _verify_source_unchanged(
    source_info: "SourceInfo", vllm_config: Config
) -> None:
    from atom.utils.caching import _compute_code_hash_with_content, _compute_code_hash

    file_contents = {}
    for source in source_info.inlined_sources:
        module = sys.modules[source.module]
        file = inspect.getfile(module)
        vllm_config.compilation_config.traced_files.add(file)
        file_contents[file] = source.content
    expected_checksum = _compute_code_hash_with_content(file_contents)
    actual_checksum = _compute_code_hash(set(file_contents.keys()))
    if expected_checksum != actual_checksum:
        raise RuntimeError(
            "Source code has changed since the last compilation. Recompiling the model."
        )


def support_torch_compile(
    cls: Optional[_T] = None,
    *,
    dynamic_arg_dims: Optional[dict[str, Union[int, list[int]]]] = None,
) -> Union[Callable[[_T], _T], _T]:
    def cls_decorator_helper(cls: _T) -> _T:
        # helper to pass `dynamic_arg_dims`` to `_support_torch_compile``
        # to avoid too much indentation for `_support_torch_compile``
        if not hasattr(cls, "forward"):
            raise TypeError("decorated class should have a forward method.")
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [
                    torch.Tensor,
                    Optional[torch.Tensor],
                ]:
                    inferred_dynamic_arg_dims[k] = 0

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly."
            )

        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}"
                )
        return _support_torch_compile(cls, inferred_dynamic_arg_dims)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, Union[int, list[int]]],
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWithNoGuardsWrapper in cls.__bases__:
        # support decorating multiple times
        return cls
    # print("_support_torch_compile")
    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)

    old_init = cls.__init__

    def __init__(self, atom_config: Config, **kwargs):
        old_init(self, atom_config=atom_config, **kwargs)
        self.atom_config = atom_config
        # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = atom_config.compilation_config.level in [
            CompilationLevel.NO_COMPILATION,
            CompilationLevel.DYNAMO_AS_IS,
        ]
        # print("self.do_not_compile",self.do_not_compile)
        if self.do_not_compile:
            return
        
        self.compilation_config = self.atom_config.compilation_config

        self.was_aot_compile_fn_loaded_from_disk = False
        # compilation_counter.num_models_seen += 1
        self.compiled = False

        # Handled by monkeypatching `TorchCompileWithNoGuardsWrapper` into base class
        TorchCompileWithNoGuardsWrapper.__init__(self)

    cls.__init__ = __init__

    def _mark_dynamic_inputs(
        mod: type[_T], ds_type: DynamicShapesType, *args: Any, **kwargs: Any
    ) -> None:
        def mark_dynamic(arg: torch.Tensor, dims: list[int]) -> None:
            if ds_type == DynamicShapesType.UNBACKED:
                if is_torch_equal_or_newer("2.10.0"):
                    for dim in dims:
                        torch._dynamo.decorators.mark_unbacked(
                            arg, dim, hint_override=arg.size()[dim]
                        )
                else:
                    torch._dynamo.decorators.mark_unbacked(arg, dims)
            else:
                torch._dynamo.mark_dynamic(arg, dims)

        sig = inspect.signature(mod.__class__.forward)  # type: ignore[attr-defined]
        bound_args = sig.bind(mod, *args, **kwargs)
        bound_args.apply_defaults()
        for k, dims in dynamic_arg_dims.items():
            arg = bound_args.arguments.get(k)

            if arg is not None:
                dims = [dims] if isinstance(dims, int) else dims
                if isinstance(arg, torch.Tensor):
                    # In case dims is specified with negative indexing
                    dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                    mark_dynamic(arg, dims)
                elif isinstance(arg, IntermediateTensors):
                    for tensor in arg.tensors.values():
                        # In case dims is specified with negative indexing
                        dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
                        mark_dynamic(tensor, dims)
                else:
                    raise ValueError(
                        "Unsupported dynamic dimensions"
                        f" {dims} for argument {k} with type {type(arg)}."
                    )
        if mark_unbacked_dims:
            for k, dims in mark_unbacked_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        if is_torch_equal_or_newer("2.10.0"):
                            for dim in dims:
                                torch._dynamo.decorators.mark_unbacked(
                                    arg, dim, hint_override=arg.size()[dim]
                                )
                        else:
                            torch._dynamo.decorators.mark_unbacked(arg, dims)

    def __call__(self: type[_T], *args: Any, **kwargs: Any) -> Any:
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # If skip_compiled is set, bypass compiled model call. This is used e.g. for
        # enc-dec models where tensor shapes/types vary across invocations, preventing
        # the capture of a single computational graph.
        # if is_forward_context_available() and get_forward_context().skip_compiled:
        #     return self.forward(*args, **kwargs)

        # if aot_compiled_fn is set, call it with partition wrapper context.
        # The partition wrapper must be active at runtime for CUDA graph
        # capture to work correctly with inductor graph partitioning.
        if getattr(self, "aot_compiled_fn", None) is not None:
            # with maybe_use_cudagraph_partition_wrapper(self.vllm_config):
            return self.aot_compiled_fn(self, *args, **kwargs)

        ds_type = self.compilation_config.dynamic_shapes_config.type
        cache_dir = None
        aot_compilation_path = None
        if envs.VLLM_USE_AOT_COMPILE:
            """
            When using torch.compile in AOT mode, we store the cache artifacts
            under VLLM_CACHE_ROOT/torch_aot_compile/{hash}/rank_i_j. The {hash}
            contains all of the factors except for the source files being
            traced through, because we don't actually know which source files
            to check at this point (before dynamo runs).
            On loading we will actually look at the source files being traced
            through. If any source file have changed (compared with the
            serialized backend artifacts), then we need to generate a new AOT
            compile artifact from scratch.
            """

            factors: list[str] = aot_compile_hash_factors(self.vllm_config)

            factors.append(_model_hash_key(self.forward))
            hash_key = hashlib.sha256(str(factors).encode()).hexdigest()
            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_aot_compile",
                hash_key,
            )
            from aiter.dist.parallel_state import get_tp_group
            rank = get_tp_group().rank
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")
            aot_compilation_path = os.path.join(cache_dir, "model")
            try:
                with (
                    set_current_atom_config(self.vllm_config),
                    open(aot_compilation_path, "rb") as f,
                ):
                    start_monitoring_torch_compile(self.vllm_config)
                    loaded_fn = torch.compiler.load_compiled_function(
                        f, f_globals=self.forward.__globals__
                    )
                _verify_source_unchanged(loaded_fn.source_info(), self.vllm_config)
                if not self.compilation_config.dynamic_shapes_config.evaluate_guards:
                    loaded_fn.disable_guard_check()
                self.aot_compiled_fn = loaded_fn
                self.was_aot_compile_fn_loaded_from_disk = True
            except Exception as e:
                if os.path.exists(aot_compilation_path):
                    logger.warning(
                        "Cannot load aot compilation from path %s, error: %s",
                        aot_compilation_path,
                        str(e),
                    )
                if envs.VLLM_FORCE_AOT_LOAD:
                    raise e
            if getattr(self, "aot_compiled_fn", None) is not None:
                logger.info(
                    "Directly load AOT compilation from path %s", aot_compilation_path
                )
                # Apply partition wrapper context for proper CUDA graph capture
                # with maybe_use_cudagraph_partition_wrapper(self.vllm_config):
                return self.aot_compiled_fn(self, *args, **kwargs)

        if self.compiled:
            assert (
                not envs.VLLM_USE_AOT_COMPILE
                or self.vllm_config.compilation_config.backend == "eager"
            )
            return TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)  # type: ignore[arg-type]

        # This is the path for the first compilation.
        # the first compilation needs to have dynamic shapes marked
        _mark_dynamic_inputs(
            self,
            ds_type,
            *args,
            **kwargs,
        )

        # here, it is the starting point of the `torch.compile` process
        start_monitoring_torch_compile(self.vllm_config)
        original_code_object = self.original_code_object()
        logger.debug("Start compiling function %s", original_code_object)

        # we do not want tp delete the original code object entries since
        # we depend on them now to look up cached compiled functions.
        # torch._dynamo.eval_frame.remove_from_cache(original_code_object)

        # collect all relevant files traced by Dynamo,
        # so that the compilation cache can trigger re-compilation
        # properly when any of these files change.

        # 1. the file containing the top-level forward function
        self.compilation_config.traced_files.add(original_code_object.co_filename)

        # 2. every time Dynamo sees a function call, it will inline
        # the function by calling InliningInstructionTranslator.inline_call_
        # we hijack this function to know all the functions called
        # during Dynamo tracing, and their corresponding files
        inline_call = InliningInstructionTranslator.inline_call_

        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)

        # Disable the C++ compilation of symbolic shape guards. C++-fication
        # of symbolic shape guards can improve guard overhead. But, since
        # vllm skip guards anyways, setting this flag to False can improve
        # compile time.
        dynamo_config_patches = {}
        try:
            _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
            dynamo_config_patches["enable_cpp_symbolic_shape_guards"] = False
        except AttributeError:
            # Note: this config is not available in torch 2.6, we can skip
            # if the config doesn't exist
            logger.debug("enable_cpp_symbolic_shape_guards config not available")

        # Prepare backed_size_oblivious config patch if needed
        fx_config_patches = {}
        if ds_type == DynamicShapesType.BACKED_SIZE_OBLIVIOUS:
            fx_config_patches["backed_size_oblivious"] = True

        # Prepare inductor config patches
        # assume_32bit_indexing is only available in torch 2.10.0+
        inductor_config_patches = {}
        if is_torch_equal_or_newer("2.10.0"):
            inductor_config_patches["assume_32bit_indexing"] = (
                self.compilation_config.dynamic_shapes_config.assume_32_bit_indexing
            )

        with (
            patch.object(
                InliningInstructionTranslator, "inline_call_", patched_inline_call
            ),
            torch._dynamo.config.patch(**dynamo_config_patches),
            # maybe_use_cudagraph_partition_wrapper(self.vllm_config),
            torch.fx.experimental._config.patch(**fx_config_patches),
            torch._inductor.config.patch(**inductor_config_patches),
        ):
            use_aot_compile = envs.VLLM_USE_AOT_COMPILE
            if self.vllm_config.compilation_config.backend == "eager":
                logger.warning("Detected eager backend, disabling AOT compile.")
                use_aot_compile = False
            if use_aot_compile:

                # store the path for saving after warmup
                self._aot_compilation_path = aot_compilation_path
                self._aot_cache_dir = cache_dir
                # set callback in context so it's available when compilation completes
                with set_on_compilation_complete(self.save_aot_compiled_function):
                    self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
                    output = self.aot_compiled_fn(self, *args, **kwargs)
            else:
                output = TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)  # type: ignore[arg-type]

        self.compiled = True
        return output

    # triggers VllmSerializableFunction.serialize()
    def save_aot_compiled_function(self: type[_T]) -> None:
        if self.was_aot_compile_fn_loaded_from_disk:
            logger.debug("AOT compiled function was loaded from cache, skipping save")
            return

        assert (
            self.aot_compiled_fn and self._aot_compilation_path and self._aot_cache_dir
        )

        logger.info("saving AOT compiled function to %s", self._aot_compilation_path)
        try:
            os.makedirs(self._aot_cache_dir, exist_ok=True)
            self.aot_compiled_fn.save_compiled_function(self._aot_compilation_path)
            logger.info("saved AOT compiled function to %s", self._aot_compilation_path)
        except Exception as e:
            logger.warning(
                "unable to save AOT compiled function to %s: %s",
                self._aot_compilation_path,
                e,
            )

    cls.__call__ = __call__
    cls.save_aot_compiled_function = save_aot_compiled_function
    return cls

