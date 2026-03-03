# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional, TypeVar, Union
import inspect
import os
import sys
from types import CodeType
from abc import abstractmethod
from contextlib import contextmanager
from unittest.mock import patch

from torch._dynamo.symbolic_convert import InliningInstructionTranslator
import torch
import torch.nn as nn
import time

from atom.config import CompilationConfig, Config, CompilationLevel

from atom.utils.graph_marker import graph_marker
# from atom.utils import start_monitoring_torch_compile

_T = TypeVar("_T", bound=type[nn.Module])

context_manager = None
torch_compile_start_time: float = 0.0


def _graph_marker_first_tensor(obj, name: str):
    if torch.is_tensor(obj):
        return graph_marker(obj, name=name), True
    if isinstance(obj, tuple):
        out = []
        marked_any = False
        for v in obj:
            if marked_any:
                out.append(v)
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out.append(vv)
        out_t = tuple(out)
        # namedtuple support
        if hasattr(obj, "_fields"):
            return obj.__class__(*out_t), marked_any
        return out_t, marked_any
    if isinstance(obj, list):
        out = []
        marked_any = False
        for v in obj:
            if marked_any:
                out.append(v)
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out.append(vv)
        return out, marked_any
    if isinstance(obj, dict):
        out = {}
        marked_any = False
        for k, v in obj.items():
            if marked_any:
                out[k] = v
                continue
            vv, marked_any = _graph_marker_first_tensor(v, name)
            out[k] = vv
        return out, marked_any
    return obj, False


def mark_trace(cls):
    forward = getattr(cls, "forward", None)
    if forward is None:
        return cls
    if getattr(forward, "__mark_trace_wrapped__", False):
        return cls

    from atom.utils.graph_marker import is_graph_marker_enabled

    def wrapped_forward(self, *args, **kwargs):
        # When mark-trace is disabled, bypass all wrapping logic entirely
        if not is_graph_marker_enabled():
            return forward(self, *args, **kwargs)
        
        prefix = getattr(self, "prefix", cls.__name__)
        # Mark only the first tensor across args/kwargs, keeping names stable.
        args_l = list(args)
        marked = False
        for i, a in enumerate(args_l):
            if marked:
                break
            aa, marked = _graph_marker_first_tensor(a, f"{prefix}_start")
            args_l[i] = aa
        if not marked:
            for k, v in list(kwargs.items()):
                if marked:
                    break
                vv, marked = _graph_marker_first_tensor(v, f"{prefix}_start")
                kwargs[k] = vv
        args = tuple(args_l)
        y = forward(self, *args, **kwargs)
        yy, _ = _graph_marker_first_tensor(y, f"{prefix}_end")
        return yy

    wrapped_forward.__mark_trace_wrapped__ = True
    cls.forward = wrapped_forward
    return cls

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
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWrapperWithCustomDispatcher in cls.__bases__:
        # support decorating multiple times
        return cls
    # print("_support_torch_compile")
    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher,)

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

        TorchCompileWrapperWithCustomDispatcher.__init__(
            self,
            vllm_config=atom_config,
            compilation_level=atom_config.compilation_config.level,
        )

    cls.__init__ = __init__

    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # print("self.compiled_codes", self.compiled_codes)
        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            sig = inspect.signature(self.__class__.forward)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            for k, dims in dynamic_arg_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        # print(arg.shape)
                        # print(f"torch._dynamo.mark_dynamic({arg, dims})")
                        torch._dynamo.mark_dynamic(arg, dims)
                    else:
                        raise ValueError(
                            "Unsupported dynamic dimensions"
                            f" {dims} for argument {k} with type {type(arg)}."
                        )
            # here, it is the starting point of the `torch.compile` process
            start_monitoring_torch_compile(self.atom_config)
            # print("Start compiling function %s",
            #              self.original_code_object)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            # it seems Dynamo reuse the compilation across instances,
            # while we need to make sure the compiled code is not reused.
            # we need to control all the compilation of the model.
            torch._dynamo.eval_frame.remove_from_cache(self.original_code_object)

            # collect all relevant files traced by Dynamo,
            # so that the compilation cache can trigger re-compilation
            # properly when any of these files change.

            # 1. the file containing the top-level forward function
            self.vllm_config.compilation_config.traced_files.add(
                self.original_code_object.co_filename
            )

            # 2. every time Dynamo sees a function call, it will inline
            # the function by calling InliningInstructionTranslator.inline_call
            # we hijack this function to know all the functions called
            # during Dynamo tracing, and their corresponding files
            inline_call = InliningInstructionTranslator.inline_call

            def patched_inline_call(parent, func, args, kwargs):
                code = func.get_code()
                self.vllm_config.compilation_config.traced_files.add(code.co_filename)
                return inline_call(parent, func, args, kwargs)

            with patch.object(
                InliningInstructionTranslator, "inline_call", patched_inline_call
            ):
                # print("self.compiled_callable to call torch compile")
                output = self.compiled_callable(*args, **kwargs)
            return output

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(*args, **kwargs)
            return model_output

    cls.__call__ = __call__
    return cls
