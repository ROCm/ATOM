# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import TypeVar
from typing import (
    Dict,
    List,
    Protocol,
    Tuple,
    Union,
)
from contextlib import contextmanager
from typing_extensions import overload
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_module_registration_hook
import os


import logging

T = TypeVar("T")
logger = logging.getLogger(__name__)


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input


class StageMissingLayer(nn.Module):
    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None:
        super().__init__()

        self.stage_name = stage_name

        # Don't register this as a child module in order to
        # avoid missing keys when loading weights
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        return getattr(self.__dict__["module"], name)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self} should not be called")

    def extra_repr(self) -> str:
        return f"stage_name={self.stage_name!r}"


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    partition_list_str = os.getenv("VLLM_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
    else:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]

        if remaining_layers := num_hidden_layers % pp_size:
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1
            logger.info(
                "Hidden layers were unevenly partitioned: [%s]. "
                "This can be manually overridden using the "
                "VLLM_PP_LAYER_PARTITION environment variable",
                ",".join(str(p) for p in partitions),
            )

    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
    layer_num_offset: int = 0,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from aiter.dist.parallel_state import get_pp_group

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            layer_fn(prefix=f"{prefix}.{idx}", layer_num=layer_num_offset + idx)
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: Dict[int, List[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> List[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + ".")
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict[str, torch.Tensor]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


def make_empty_intermediate_tensors_factory(keys: List[str], hidden_size: int):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert (
            len(int_vals) == 1
        ), f"layer name {layer_name} should only contain one integer"

        return int_vals[0]
    else:
        assert (
            len(int_vals) <= 2
        ), f"layer name {layer_name} should contain at most two integers"
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


@contextmanager
def collect_children(
    module: nn.Module,
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    Within this context, collect all direct child assignments to `module`,
    returning a list of children names that is internally updated until the
    context is exited.

    If `targets` is set, instead collect descendents of `module`
    that are an instance of `targets`, even if they aren't direct children.
    """
    children_names = list[str]()

    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                children_names.append(name)

        with register_module_module_registration_hook(hook):
            yield children_names
    else:
        yield children_names

        for name, module_ in module.named_modules():
            if isinstance(module_, targets):
                children_names.append(name)


@contextmanager
def no_init_weights(
    module: nn.Module,
    placeholder: Callable[[nn.Module], nn.Module],
    *,
    targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    Within this context, prevent weight initialization from using device memory and
    replace direct child assignments to `module` with the result of `placeholder()`.

    If `targets` is set, instead prevent weight initialization and
    replace assignments where the child is an instance of `targets`,
    even if they aren't direct children of `module`.
    """
    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if module_ is module:
                return placeholder(submodule)

            return submodule

        with register_module_module_registration_hook(hook), torch.device("meta"):
            yield
    else:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            if isinstance(module_, targets):
                submodule.to("meta")  # Free memory
            if isinstance(submodule, targets):
                submodule.to("meta")  # Free memory
                return placeholder(submodule)

            return submodule

        # Not all descendents are targeted, so we can't use a blanket
        # `torch.device("meta")` context
        with register_module_module_registration_hook(hook):
            yield


@overload
def common_prefix(items: Sequence[str]) -> str: ...


@overload
def common_prefix(items: Sequence[Sequence[T]]) -> Sequence[T]: ...


def common_prefix(items: Sequence[Sequence[T] | str]) -> Sequence[T] | str:
    """Find the longest prefix common to all items."""
    if len(items) == 0:
        return []
    if len(items) == 1:
        return items[0]

    shortest = min(items, key=len)
    if not shortest:
        return shortest[:0]

    for match_len in range(1, len(shortest) + 1):
        match = shortest[:match_len]
        for item in items:
            if item[:match_len] != match:
                return shortest[: match_len - 1]

    return shortest
