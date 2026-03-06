from typing import Any, TypeVar, ParamSpec
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from atom.config import CUDAGraphMode, CompilationLevel, DynamicShapesType, get_current_atom_config
from atom.utils import envs
import torch
import torch.cuda.nvtx as nvtx
from types import CodeType
from abc import abstractmethod
import os
import sys
import logging
logger = logging.getLogger(__name__)

R = TypeVar("R")
P = ParamSpec("P")


def _noop_add_global_state_guard(
    self: torch._C._dynamo.guards.GuardManager, *args: Any, **kwargs: Any
) -> None:
    """No-op to skip the GLOBAL_STATE guard entirely"""
    pass


def _noop_add_torch_function_mode_stack_guard(
    self: torch._C._dynamo.guards.GuardManager, *args: Any, **kwargs: Any
) -> None:
    """No-op to skip the TORCH_FUNCTION_MODE_STACK guard entirely"""
    pass

def print_tensor(tensor_obj, prefix, tensor_list=None):
    """Descends iterators that contains Tensors and prints the Tensor.
    Recursive function that descends iterator type arguments until
    it finds a Tensor object.
    """
    if tensor_list is None:
        tensor_list = []

    if isinstance(tensor_obj, (list, tuple)):
        for ten in tensor_obj:
            tensor_list = print_tensor(ten, prefix, tensor_list)
    elif isinstance(tensor_obj, torch.Tensor):
        tensor_dims = list(tensor_obj.size())
        tensor_list.append(tensor_dims)
    return tensor_list


def process_layer_params(module_obj):
    """Extract the static parameters from LLM and VLM relevant layer types"""
    param_info = {}
    # Extract parameters for layers commonly used in LLMs and VLMs
    if isinstance(module_obj, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        conv_params = {}
        conv_params["in_chan"] = module_obj.in_channels
        conv_params["out_chan"] = module_obj.out_channels
        conv_params["filter_dim"] = module_obj.kernel_size
        conv_params["stride"] = module_obj.stride
        conv_params["padding"] = module_obj.padding
        conv_params["dilation"] = module_obj.dilation
        conv_params["transposed"] = module_obj.transposed
        conv_params["output_padding"] = module_obj.output_padding
        conv_params["groups"] = module_obj.groups
        conv_params["padding_mode"] = module_obj.padding_mode
        param_info = conv_params
    elif isinstance(
        module_obj,
        (
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        ),
    ):
        convtranspose_params = {}
        convtranspose_params["in_chan"] = module_obj.in_channels
        convtranspose_params["out_chan"] = module_obj.out_channels
        convtranspose_params["filter_dim"] = module_obj.kernel_size
        convtranspose_params["stride"] = module_obj.stride
        convtranspose_params["padding"] = module_obj.padding
        convtranspose_params["dilation"] = module_obj.dilation
        convtranspose_params["transposed"] = module_obj.transposed
        convtranspose_params["output_padding"] = module_obj.output_padding
        convtranspose_params["groups"] = module_obj.groups
        convtranspose_params["padding_mode"] = module_obj.padding_mode
        param_info = convtranspose_params
    elif isinstance(
        module_obj, (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)
    ):

        def _handle_int_or_tuple(parameter):
            if isinstance(parameter, tuple):
                return list(parameter)
            elif isinstance(parameter, int):
                return [parameter, parameter]

        pooling_params = {}
        pooling_params["filter_dim"] = _handle_int_or_tuple(module_obj.kernel_size)
        pooling_params["stride"] = _handle_int_or_tuple(module_obj.stride)
        pooling_params["padding"] = _handle_int_or_tuple(module_obj.padding)
        pooling_params["dilation"] = _handle_int_or_tuple(module_obj.dilation)
        param_info = pooling_params
    elif isinstance(
        module_obj, (torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d)
    ):
        pooling_params = {}
        pooling_params["filter_dim"] = [
            module_obj.kernel_size,
            module_obj.kernel_size,
        ]
        pooling_params["stride"] = [module_obj.stride, module_obj.stride]
        pooling_params["padding"] = [module_obj.padding, module_obj.padding]
        pooling_params["ceil_mode"] = module_obj.ceil_mode
        pooling_params["count_include_pad"] = module_obj.count_include_pad
        param_info = pooling_params
    elif isinstance(
        module_obj,
        (
            torch.nn.AdaptiveAvgPool1d,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.AdaptiveAvgPool3d,
        ),
    ):
        pooling_params = {}
        pooling_params["output_size"] = [
            module_obj.output_size,
            module_obj.output_size,
        ]
        param_info = pooling_params
    elif isinstance(module_obj, torch.nn.Linear):
        param_info["in_features"] = module_obj.in_features
        param_info["out_features"] = module_obj.out_features
    elif isinstance(
        module_obj,
        (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
    ):
        param_info["num_features"] = module_obj.num_features
        param_info["epsilon"] = module_obj.eps
        param_info["momentum"] = module_obj.momentum
    elif isinstance(module_obj, torch.nn.ReLU):
        param_info["in_place"] = module_obj.inplace
    elif isinstance(module_obj, torch.nn.Dropout):
        param_info["p"] = module_obj.p
        param_info["in_place"] = module_obj.inplace
    elif isinstance(module_obj, torch.nn.Embedding):
        param_info["num_embeddings"] = module_obj.num_embeddings
        param_info["embedding_dim"] = module_obj.embedding_dim
    elif isinstance(
        module_obj,
        (
            torch.nn.Upsample,
            torch.nn.UpsamplingNearest2d,
            torch.nn.UpsamplingBilinear2d,
        ),
    ):
        param_info["scale_factor"] = module_obj.scale_factor

    return param_info


class ResultHolder:
    """Holder for storing results from within a context manager."""

    result = None

def construct_marker_dict_and_push(
    module_name, module_obj, in_tensor, kwargs=None, out_tensor=None
):
    marker_dict = {}
    marker_dict["Module"] = module_name

    ## Get trainable parameters like weights and bias
    module_params = module_obj.named_parameters(recurse=False)
    for idx, (param_name, param_obj) in enumerate(module_params):
        if idx == 0:
            marker_dict["TrainableParams"] = {}
        marker_dict["TrainableParams"][param_name] = list(param_obj.size())

    in_tensor_list = print_tensor(in_tensor, "Input")
    if in_tensor_list:
        marker_dict["Inputs"] = in_tensor_list

    out_tensor_list = print_tensor(out_tensor, "Output")
    if out_tensor_list:
        marker_dict["Outputs"] = out_tensor_list

    ## Get Kwargs like input_ids and positions for the top module
    if kwargs:
        for key, value in kwargs.items():
            if isinstance(value, (torch.Tensor, list, tuple)):
                tensor_list = print_tensor(value, key)
                if tensor_list:
                    marker_dict[key] = tensor_list

    param_info = process_layer_params(module_obj)
    if param_info:
        marker_dict["StaticParams"] = param_info
    nvtx.range_push("{}".format(marker_dict))


@contextmanager
def layerwise_nvtx_marker_context(module_name, module_obj, in_tensor=None, kwargs=None):
    """Context manager for NVTX markers that automatically pushes on enter
    and pops on exit.

    Example:
        with nvtx_marker_context("Module:MyModule", module, in_tensor=args,
                                 kwargs=kwargs) as ctx:
            ctx.result = module(*args, **kwargs)
        return ctx.result
    """
    holder = ResultHolder()

    # Push input marker
    construct_marker_dict_and_push(
        module_name,
        module_obj,
        in_tensor=in_tensor,
        kwargs=kwargs,
    )
    try:
        yield holder
    finally:
        # Pop input marker
        nvtx.range_pop()
        # Push and pop output marker
        output_name = module_name.replace("(input)", "(output)")
        construct_marker_dict_and_push(
            output_name,
            module_obj,
            in_tensor=None,
            kwargs=None,
            out_tensor=holder.result,
        )
        nvtx.range_pop()

@contextmanager
def _compilation_context() -> Generator[None, None, None]:
    """Context manager for compilation settings and patches.

    This manager:
    1. Sets higher dynamo cache limits for compilation. (Needed for
        qwen2_5_vl see test_qwen2_5_vl_evs_functionality).
        Generally a recompilation can happen whenever we use a new
        backend instance in torch.compile.
    2. Patches out add_global_state_guard to skip GLOBAL_STATE guards
    3. Patches out add_torch_function_mode_stack_guard to skip
        TORCH_FUNCTION_MODE_STACK guards.
    4. Restores everything when compilation completes
    """
    # Save original values
    original_global_state_guard = (
        torch._C._dynamo.guards.GuardManager.add_global_state_guard
    )
    original_torch_function_mode_stack_guard = (
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard
    )
    original_cache_size = torch._dynamo.config.cache_size_limit
    original_accumulated_cache = torch._dynamo.config.accumulated_cache_size_limit

    try:
        # Set higher cache limits for compilation
        torch._dynamo.config.cache_size_limit = 2048
        torch._dynamo.config.accumulated_cache_size_limit = 8192

        # Patch guard manager
        torch._C._dynamo.guards.GuardManager.add_global_state_guard = (
            _noop_add_global_state_guard
        )
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard = (
            _noop_add_torch_function_mode_stack_guard
        )
        yield
    finally:
        # Restore original values
        torch._C._dynamo.guards.GuardManager.add_global_state_guard = (
            original_global_state_guard
        )
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard = (
            original_torch_function_mode_stack_guard
        )
        torch._dynamo.config.cache_size_limit = original_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = original_accumulated_cache



class TorchCompileWithNoGuardsWrapper:
    """
    A wrapper class for torch.compile, it ensures that all guards are dropped
    when CompilationMode is not CompilationMode.STOCK_TORCH_COMPILE.
    When guards are dropped, the first time __call__ is invoked, a single
    compilation is triggered. Dynamo should never be traced again after that
    since we drop all guards.
    """

    def check_invariants_and_forward(self, *args: Any, **kwargs: Any) -> Any:
        assert hasattr(self, "_check_shape_invariants")
        self._check_shape_invariants(*args, **kwargs)

        return self.forward(*args, **kwargs)

    def _call_with_optional_nvtx_range(
        self, callable_fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Any:
        if self.layerwise_nvtx_tracing_enabled:
            args_list = list(args)
            kwargs_dict = dict(kwargs)
            with layerwise_nvtx_marker_context(
                "Torch Compiled Module (input):{}".format(self.__class__.__name__),
                self,
                in_tensor=args_list,
                kwargs=kwargs_dict,
            ) as ctx:
                ctx.result = callable_fn(*args, **kwargs)
            return ctx.result
        return callable_fn(*args, **kwargs)

    def __init__(self) -> None:
        self.compiled = False

        vllm_config = get_current_atom_config()
        self.vllm_config = vllm_config
        mode = vllm_config.compilation_config.level
        self.layerwise_nvtx_tracing_enabled = False
        if mode is None:
            raise RuntimeError("Compilation mode cannot be NO_COMPILATION")

        backend = vllm_config.compilation_config.init_backend(vllm_config)
        options = {}

        if isinstance(backend, str) and backend == "inductor":
            options = vllm_config.compilation_config.inductor_compile_config

        self.first_compile = True
        self.evaluate_guards = (
            vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
        )

        ds_type = vllm_config.compilation_config.dynamic_shapes_config.type

        if mode != CompilationLevel.DYNAMO_AS_IS:
            # Drop all the guards.
            if self.evaluate_guards:

                options["guard_filter_fn"] = lambda x: [
                    entry.guard_type == "SHAPE_ENV" for entry in x
                ]
            else:
                options["guard_filter_fn"] = lambda x: [False for _ in x]

        compiled_ptr: Any = self.forward
        # Validate that unbacked dynamic shapes require VLLM_USE_BYTECODE_HOOK=False

        if ds_type == DynamicShapesType.UNBACKED:
            # reason is that bytecode does torch._dynamo.eval_frame.
            # remove_from_cache(self.original_code_object()) to force a new
            # re-compilation. And if we use
            # compiled_ptr = self.check_invariants_and_forward
            # it will reset all entries.
            assert not self.evaluate_guards, "UNBACKED dynamic shapes do not add guards"

            compiled_ptr = self.check_invariants_and_forward

        aot_context = nullcontext()
        if envs.VLLM_USE_AOT_COMPILE:
            if hasattr(torch._dynamo.config, "enable_aot_compile"):
                aot_context = torch._dynamo.config.patch(enable_aot_compile=True)
            else:
                msg = "torch._dynamo.config.enable_aot_compile is not "
                msg += "available. AOT compile is disabled and please "
                msg += "upgrade PyTorch version to use AOT compile."
                print(msg)

        with aot_context:
            self._compiled_callable = torch.compile(
                compiled_ptr,
                fullgraph=True,
                dynamic=False,
                backend=backend,
                options=options,
            )

        if envs.VLLM_USE_BYTECODE_HOOK and mode != CompilationLevel.DYNAMO_AS_IS:
            torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)
            self._compiled_bytecode: CodeType | None = None

    def aot_compile(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self._compiled_callable, "aot_compile"):
            raise RuntimeError(
                "aot_compile is not supported by the current configuration. "
                "Please make sure torch.compile is enabled with the latest "
                f"version of PyTorch (current using torch: {torch.__version__})"
            )
        return self._compiled_callable.aot_compile((args, kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if envs.VLLM_USE_BYTECODE_HOOK:
            if (
                self.vllm_config.compilation_config.level
                == CompilationLevel.DYNAMO_AS_IS
            ):
                return self._compiled_callable(*args, **kwargs)

            if not self._compiled_bytecode:
                # Make sure a compilation is triggered by clearing dynamo
                # cache.
                torch._dynamo.eval_frame.remove_from_cache(self.original_code_object())
                return self._call_with_optional_nvtx_range(
                    self._compiled_callable, *args, **kwargs
                )
            else:
                with self._dispatch_to_compiled_code():
                    return self._call_with_optional_nvtx_range(
                        self.forward, *args, **kwargs
                    )
        else:
            ctx = (
                nullcontext()
                if self.first_compile or not self.evaluate_guards
                else torch.compiler.set_stance("fail_on_recompile")
            )
            self.first_compile = False
            with _compilation_context(), ctx:
                return self._call_with_optional_nvtx_range(
                    self._compiled_callable, *args, **kwargs
                )

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def original_code_object(self) -> CodeType:
        """Return the original code object of the forward method."""
        return self.__class__.forward.__code__

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType) -> None:
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object():
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

        self._compiled_bytecode = new_code

        path = self.vllm_config.compile_debug_dump_path()
        if path:
            decompiled_file = path / "transformed_code.py"
            if not decompiled_file.exists():
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    import depyf

                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)

                    logger.debug("Dynamo transformed code saved to %s", decompiled_file)
                except Exception:
                    pass

        if (
            self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and "update" in new_code.co_names
        ):
            import depyf

            src = depyf.decompile(new_code)
            msg = (
                "Assigning / modifying buffers of nn.Module during forward pass is not "
                "allowed when using cudagraph inside the compiler because it will "
                "cause silent errors. Please use eager mode or fix the code. The "
                "following code contains clues about which buffer is being modified "
                f"(please search for the usage of the function `update`):\n{src}"
            )
            raise RuntimeError(msg)

    @contextmanager
    def _dispatch_to_compiled_code(self) -> Generator[None, None, None]:
        # noqa: E501
        """
        Context manager to dispatch to internally compiled code for torch<2.8.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """  # noqa: E501 line too long
        original = self.original_code_object()
        assert self._compiled_bytecode is not None
        self.__class__.forward.__code__ = self._compiled_bytecode
        try:
            yield
        finally:
            self.__class__.forward.__code__ = original
