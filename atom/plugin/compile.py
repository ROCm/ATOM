from typing import Type, TypeVar, Union, Optional, Callable

T = TypeVar('T')


def compile_decorator_for_plugin_mode(
    cls: Optional[Type[T]] = None,
    **kwargs
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """
    Decorator that conditionally applies torch.compile support to model classes based on the framework

    - If using vLLM: applies vLLM's support_torch_compile decorator with parameters
    - If using SGLang: returns the class as-is (no-op decorator) for now
    """

    def compile_decorator(cls: Type[T]) -> Type[T]:
        from atom.plugin import is_vllm, is_sglang
        if is_vllm():
            from vllm.compilation.decorators import support_torch_compile
            return support_torch_compile(**kwargs)(cls)
        elif is_sglang():
            # For SGLang, return the class unchanged (no-op decorator)
            # as SGLang uses torch.compile(model) instead of decorator
            return cls

    if cls is not None:
        return compile_decorator(cls)

    return compile_decorator
