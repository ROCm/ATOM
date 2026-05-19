import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

logger = logging.getLogger("atom")


def apply_vllm_lora_bridge() -> None:
    try:
        try:
            import vllm.lora.model_manager as model_manager
        except ModuleNotFoundError:
            import vllm.lora.models as model_manager
        import vllm.lora.lora_model as lora_model_module
        import vllm.lora.utils as lora_utils
        from vllm.config.lora import LoRAConfig
        from vllm.lora.layers.base import BaseLayerWithLoRA
        from vllm.lora.utils import parse_fine_tuned_lora_name

        from atom.model_ops.moe import FusedMoE as AtomFusedMoE
        from atom.model_ops.linear import LinearBase as AtomLinearBase
    except Exception as exc:
        logger.warning("ATOM plugin: failed to install vLLM LoRA bridge: %s", exc)
        return

    if getattr(lora_utils, "_atom_lora_bridge_patched", False):
        return

    original_get_supported_lora_modules = lora_utils.get_supported_lora_modules
    original_from_layer = lora_utils.from_layer
    original_is_moe_model = lora_utils.is_moe_model
    original_process_packed_modules_mapping = lora_utils.process_packed_modules_mapping
    original_get_lora_layer_weights = model_manager.LoRAModelManager._get_lora_layer_weights
    original_from_local_checkpoint = lora_model_module.LoRAModel.from_local_checkpoint

    class AtomLinearWithLoRA(BaseLayerWithLoRA):
        def __init__(
            self,
            base_layer: AtomLinearBase,
            packed_modules_list: list[str] | None = None,
        ) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.packed_modules_list = packed_modules_list or []
            self.input_size = base_layer.input_size
            self.output_size = base_layer.output_size
            self.tp_size = base_layer.tp_size
            self.tp_rank = base_layer.tp_rank
            self.n_slices = max(1, len(self.packed_modules_list))
            if self.n_slices > 1:
                output_sizes = getattr(base_layer, "output_sizes", None)
                if output_sizes is not None and base_layer.tp_dim != 0:
                    output_partitions = tuple(int(size) for size in output_sizes)
                else:
                    output_partitions = tuple(
                        int(size)
                        for size in getattr(
                            base_layer, "output_partition_sizes", [self.output_size]
                        )
                    )
                if len(output_partitions) >= self.n_slices:
                    self.output_slices = output_partitions[: self.n_slices]
                else:
                    self.output_slices = (self.output_size // self.n_slices,) * (
                        self.n_slices
                    )
            else:
                self.output_slices = (self.output_size,)
            self.device = base_layer.weight.device

        @classmethod
        def can_replace_layer(
            cls,
            source_layer: nn.Module,
            lora_config: LoRAConfig,
            packed_modules_list: list,
            model_config: PretrainedConfig | None = None,
        ) -> bool:
            return isinstance(source_layer, AtomLinearBase)

        def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: PretrainedConfig | None = None,
        ) -> None:
            self.lora_config = lora_config
            self.lora_a_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    lora_config.max_lora_rank,
                    self.input_size,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                )
                for _ in range(self.n_slices)
            )
            self.lora_b_stacked = tuple(
                torch.zeros(
                    max_loras,
                    1,
                    output_size,
                    lora_config.max_lora_rank,
                    dtype=lora_config.lora_dtype,
                    device=self.device,
                )
                for output_size in self.output_slices
            )

        def reset_lora(self, index: int):
            for lora_a, lora_b in zip(self.lora_a_stacked, self.lora_b_stacked):
                lora_a[index] = 0
                lora_b[index] = 0

        def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
            if self.tp_size <= 1 or self.base_layer.tp_dim != 1:
                return lora_a
            shard_size = self.input_size
            start_idx = self.tp_rank * shard_size
            return lora_a[:, start_idx : start_idx + shard_size]

        def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
            if self.tp_size <= 1 or self.base_layer.tp_dim != 0:
                return lora_b
            shard_size = self.output_size
            start_idx = self.tp_rank * shard_size
            return lora_b[start_idx : start_idx + shard_size, :]

        def _output_shard_id(self, slice_index: int) -> int:
            if (
                slice_index in (1, 2)
                and hasattr(self.base_layer, "num_kv_head_replicas")
            ):
                return self.tp_rank // int(self.base_layer.num_kv_head_replicas)
            return self.tp_rank

        def _slice_lora_b_for_slice(
            self, lora_b: torch.Tensor, slice_index: int
        ) -> torch.Tensor:
            if self.tp_size <= 1 or self.base_layer.tp_dim != 0:
                return lora_b
            shard_size = self.output_slices[slice_index]
            shard_id = self._output_shard_id(slice_index)
            start_idx = shard_id * shard_size
            return lora_b[start_idx : start_idx + shard_size, :]

        def set_lora(
            self,
            index: int,
            lora_a: torch.Tensor | list[torch.Tensor],
            lora_b: torch.Tensor | list[torch.Tensor],
        ):
            self.reset_lora(index)
            if isinstance(lora_a, torch.Tensor) and isinstance(lora_b, torch.Tensor):
                lora_a = self.slice_lora_a(lora_a)
                lora_b = self.slice_lora_b(lora_b)
                self.lora_a_stacked[0][
                    index, 0, : lora_a.shape[0], : lora_a.shape[1]
                ].copy_(lora_a, non_blocking=True)
                self.lora_b_stacked[0][
                    index, 0, : lora_b.shape[0], : lora_b.shape[1]
                ].copy_(lora_b, non_blocking=True)
                return

            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            for slice_index in range(self.n_slices):
                lora_a_i = lora_a[slice_index]
                lora_b_i = lora_b[slice_index]
                if lora_a_i is not None:
                    lora_a_i = self.slice_lora_a(lora_a_i)
                    self.lora_a_stacked[slice_index][
                        index, 0, : lora_a_i.shape[0], : lora_a_i.shape[1]
                    ].copy_(lora_a_i, non_blocking=True)
                if lora_b_i is not None:
                    lora_b_i = self._slice_lora_b_for_slice(lora_b_i, slice_index)
                    self.lora_b_stacked[slice_index][
                        index, 0, : lora_b_i.shape[0], : lora_b_i.shape[1]
                    ].copy_(lora_b_i, non_blocking=True)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            output = self.base_layer(x, *args, **kwargs)
            original_shape = output.shape if output.ndim == 3 else None
            lora_x = x
            if x.ndim == 3 and output.ndim == 3:
                output = output.flatten(0, 1)
                lora_x = x.flatten(0, 1)
            lora_dtype = self.lora_a_stacked[0].dtype
            if lora_x.dtype != lora_dtype:
                lora_x = lora_x.to(lora_dtype)
            lora_output = self.punica_wrapper.add_lora_linear(
                output,
                lora_x,
                self.lora_a_stacked,
                self.lora_b_stacked,
                1.0,
                self.output_slices,
            )
            if lora_output is not None:
                output = lora_output
            if original_shape is not None:
                output = output.reshape(original_shape)
            return output

    class AtomFusedMoEWithLoRA(BaseLayerWithLoRA):
        def __init__(self, base_layer: AtomFusedMoE) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.device = base_layer.w13_weight.device
            self.tp_size = base_layer.tp_size
            self.tp_rank = base_layer.tp_rank
            self._w13_slices = 2

        @classmethod
        def can_replace_layer(
            cls,
            source_layer: nn.Module,
            lora_config: LoRAConfig,
            packed_modules_list: list,
            model_config: PretrainedConfig | None = None,
        ) -> bool:
            return isinstance(source_layer, AtomFusedMoE) and len(packed_modules_list) > 1

        def set_mapping(self, punica_wrapper):
            super().set_mapping(punica_wrapper)
            self.base_layer._dynamic_routed_lora_punica_wrapper = punica_wrapper

        def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: PretrainedConfig | None = None,
        ) -> None:
            self.max_loras = max_loras
            self.fully_sharded = lora_config.fully_sharded_loras
            layer = self.base_layer
            rank = lora_config.max_lora_rank
            dtype = lora_config.lora_dtype
            self.adapter_enabled = torch.zeros(
                max_loras + 1, dtype=torch.int32, device=self.device
            )
            self.w13_lora_a_stacked = tuple(
                torch.zeros(
                    max_loras,
                    layer.local_num_experts,
                    rank,
                    layer.hidden_size,
                    dtype=dtype,
                    device=self.device,
                )
                for _ in range(self._w13_slices)
            )
            self.w13_lora_b_stacked = tuple(
                torch.zeros(
                    max_loras,
                    layer.local_num_experts,
                    layer.intermediate_size_per_partition,
                    rank,
                    dtype=dtype,
                    device=self.device,
                )
                for _ in range(self._w13_slices)
            )
            self.w2_lora_a_stacked = (
                torch.zeros(
                    max_loras,
                    layer.local_num_experts,
                    rank,
                    layer.intermediate_size_per_partition,
                    dtype=dtype,
                    device=self.device,
                ),
            )
            self.w2_lora_b_stacked = (
                torch.zeros(
                    max_loras,
                    layer.local_num_experts,
                    layer.hidden_size,
                    rank,
                    dtype=dtype,
                    device=self.device,
                ),
            )
            self.lora_a_stacked = self.w13_lora_a_stacked + self.w2_lora_a_stacked
            self.lora_b_stacked = self.w13_lora_b_stacked + self.w2_lora_b_stacked
            layer._dynamic_routed_lora = True
            layer._dynamic_routed_lora_w13_a_stacked = self.w13_lora_a_stacked
            layer._dynamic_routed_lora_w13_b_stacked = self.w13_lora_b_stacked
            layer._dynamic_routed_lora_w2_a_stacked = self.w2_lora_a_stacked
            layer._dynamic_routed_lora_w2_b_stacked = self.w2_lora_b_stacked
            layer._dynamic_routed_lora_adapter_enabled = self.adapter_enabled
            layer._dynamic_routed_lora_max_loras = max_loras

        def reset_lora(self, index: int):
            for tensor in self.w13_lora_a_stacked + self.w13_lora_b_stacked:
                tensor[index] = 0
            self.w2_lora_a_stacked[0][index] = 0
            self.w2_lora_b_stacked[0][index] = 0
            self.adapter_enabled[index] = 0

        def _localize_experts(self, tensor: torch.Tensor) -> torch.Tensor:
            layer = self.base_layer
            if tensor.shape[0] == layer.local_num_experts:
                return tensor
            expert_map = getattr(layer, "expert_map", None)
            if expert_map is None:
                return tensor[: layer.local_num_experts]
            local = torch.zeros(
                (layer.local_num_experts, *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for global_expert_id in range(min(int(expert_map.numel()), tensor.shape[0])):
                local_expert_id = int(expert_map[global_expert_id].item())
                if local_expert_id >= 0:
                    local[local_expert_id].copy_(tensor[global_expert_id])
            return local

        def _slice_w13_b(self, lora_b: torch.Tensor) -> torch.Tensor:
            if self.tp_size <= 1:
                return lora_b
            shard_size = self.base_layer.intermediate_size_per_partition
            start_idx = self.tp_rank * shard_size
            return lora_b[:, start_idx : start_idx + shard_size, :]

        def _slice_w2_a(self, lora_a: torch.Tensor) -> torch.Tensor:
            if self.tp_size <= 1:
                return lora_a
            shard_size = self.base_layer.intermediate_size_per_partition
            start_idx = self.tp_rank * shard_size
            return lora_a[:, :, start_idx : start_idx + shard_size]

        def set_lora(
            self,
            index: int,
            lora_a: torch.Tensor | list[torch.Tensor],
            lora_b: torch.Tensor | list[torch.Tensor],
        ):
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            w1_lora_a, w2_lora_a, w3_lora_a = lora_a
            w1_lora_b, w2_lora_b, w3_lora_b = lora_b
            self.reset_lora(index)
            self.adapter_enabled[index] = 1

            w1_lora_a = self._localize_experts(w1_lora_a)
            w2_lora_a = self._localize_experts(self._slice_w2_a(w2_lora_a))
            w3_lora_a = self._localize_experts(w3_lora_a)
            w1_lora_b = self._localize_experts(self._slice_w13_b(w1_lora_b))
            w2_lora_b = self._localize_experts(w2_lora_b)
            w3_lora_b = self._localize_experts(self._slice_w13_b(w3_lora_b))

            self.w13_lora_a_stacked[0][index, :, : w1_lora_a.shape[1], :].copy_(
                w1_lora_a, non_blocking=True
            )
            self.w13_lora_a_stacked[1][index, :, : w3_lora_a.shape[1], :].copy_(
                w3_lora_a, non_blocking=True
            )
            self.w13_lora_b_stacked[0][
                index, :, : w1_lora_b.shape[1], : w1_lora_b.shape[2]
            ].copy_(w1_lora_b, non_blocking=True)
            self.w13_lora_b_stacked[1][
                index, :, : w3_lora_b.shape[1], : w3_lora_b.shape[2]
            ].copy_(w3_lora_b, non_blocking=True)
            self.w2_lora_a_stacked[0][
                index, :, : w2_lora_a.shape[1], : w2_lora_a.shape[2]
            ].copy_(w2_lora_a, non_blocking=True)
            self.w2_lora_b_stacked[0][
                index, :, : w2_lora_b.shape[1], : w2_lora_b.shape[2]
            ].copy_(w2_lora_b, non_blocking=True)

        def forward(self, *args, **kwargs):
            return self.base_layer.forward(*args, **kwargs)

        def maybe_all_reduce_tensor_model_parallel(self, *args, **kwargs):
            return self.base_layer.maybe_all_reduce_tensor_model_parallel(
                *args, **kwargs
            )

        @property
        def quant_method(self):
            return self.base_layer.quant_method

        @property
        def use_ep(self):
            return self.base_layer.use_ep

    def atom_get_supported_lora_modules(model: nn.Module) -> list[str]:
        supported = set(original_get_supported_lora_modules(model))
        for name, module in model.named_modules():
            if isinstance(module, AtomLinearBase):
                supported.add(name.split(".")[-1])
            if isinstance(module, AtomFusedMoE):
                supported.add(name.split(".")[-1])
        return list(supported)

    def atom_from_layer(
        layer: nn.Module,
        max_loras: int,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> nn.Module:
        if AtomFusedMoEWithLoRA.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance_layer = AtomFusedMoEWithLoRA(layer)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
        if AtomLinearWithLoRA.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance_layer = AtomLinearWithLoRA(layer, packed_modules_list)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
        return original_from_layer(
            layer, max_loras, lora_config, packed_modules_list, model_config
        )

    def atom_is_moe_model(model: nn.Module) -> bool:
        return original_is_moe_model(model) or any(
            isinstance(module, AtomFusedMoE) for module in model.modules()
        )

    def _convert_atom_packed_modules_mapping(
        mapping: dict[str, Any],
    ) -> dict[str, list[str]]:
        converted: dict[str, list[str]] = {}
        shard_entries: dict[str, list[tuple[Any, str]]] = {}
        for checkpoint_name, packed_value in mapping.items():
            if isinstance(packed_value, list):
                converted[checkpoint_name] = list(packed_value)
            elif isinstance(packed_value, tuple) and len(packed_value) >= 2:
                packed_name, shard_id = packed_value[:2]
                shard_entries.setdefault(packed_name, []).append(
                    (shard_id, checkpoint_name)
                )
            else:
                converted[checkpoint_name] = packed_value

        def shard_sort_key(entry: tuple[Any, str]) -> tuple[int, str]:
            shard_id, checkpoint_name = entry
            shard_order = {
                "q": 0,
                "k": 1,
                "v": 2,
                "qkvz": 0,
                "ba": 1,
            }
            if isinstance(shard_id, int):
                return shard_id, checkpoint_name
            if shard_id in shard_order:
                return shard_order[shard_id], checkpoint_name
            return 100, str(shard_id)

        for packed_name, entries in shard_entries.items():
            converted[packed_name] = [
                checkpoint_name
                for _, checkpoint_name in sorted(entries, key=shard_sort_key)
            ]
        return converted

    def _get_atom_expert_mapping(
        model: nn.Module,
    ) -> list[tuple[str, str, int, str]]:
        parent_map = getattr(model, "get_expert_mapping", None)
        if parent_map is not None:
            return parent_map()
        for child in model.children():
            child_map = getattr(child, "get_expert_mapping", None)
            if child_map is not None:
                return child_map()
        return []

    def atom_process_packed_modules_mapping(model: nn.Module) -> dict[str, list[str]]:
        try:
            mapping = original_process_packed_modules_mapping(model)
        except AttributeError:
            mapping = getattr(model, "packed_modules_mapping", {})
        converted = _convert_atom_packed_modules_mapping(mapping)
        if atom_is_moe_model(model) and "experts" not in converted:
            if moe_mapping := _get_atom_expert_mapping(model):
                converted["experts"] = [
                    weight_name.rstrip(".") for _, weight_name, _, _ in moe_mapping
                ]
        return converted

    def atom_get_lora_layer_weights(self, lora_model, module_name: str):
        lora = original_get_lora_layer_weights(self, lora_model, module_name)
        if lora is not None:
            return lora
        if module_name.startswith("model.model."):
            return original_get_lora_layer_weights(
                self,
                lora_model,
                "model." + module_name[len("model.model.") :],
            )
        return None

    def _iter_lora_module_names(lora_dir: str, weights_mapper=None):
        import os

        tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.isfile(tensor_path):
            import safetensors

            with safetensors.safe_open(tensor_path, framework="pt") as handle:
                keys = list(handle.keys())
        else:
            return
        for key in keys:
            module_name, _ = parse_fine_tuned_lora_name(key, weights_mapper)
            yield module_name

    def atom_from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper,
        *args,
        **kwargs,
    ):
        expanded_expected = set(expected_lora_modules)
        target_modules = getattr(peft_helper, "target_modules", None)
        if target_modules:
            expanded_expected.update(target_modules)
        weights_mapper = kwargs.get("weights_mapper")
        try:
            for module_name in _iter_lora_module_names(lora_dir, weights_mapper):
                short_name = module_name.rsplit(".", 1)[-1]
                if short_name in expanded_expected:
                    if ".experts" in module_name:
                        expert_idx = module_name.find(".experts")
                        expanded_expected.add(module_name[expert_idx + 1 :])
                    else:
                        expanded_expected.add(short_name)
        except Exception as exc:
            logger.warning(
                "ATOM plugin: failed to expand LoRA target modules for %s: %s",
                lora_dir,
                exc,
            )
        return original_from_local_checkpoint(
            lora_dir,
            expanded_expected,
            peft_helper,
            *args,
            **kwargs,
        )

    lora_utils.get_supported_lora_modules = atom_get_supported_lora_modules
    lora_utils.from_layer = atom_from_layer
    lora_utils.is_moe_model = atom_is_moe_model
    lora_utils.process_packed_modules_mapping = atom_process_packed_modules_mapping
    model_manager.get_supported_lora_modules = atom_get_supported_lora_modules
    model_manager.from_layer = atom_from_layer
    model_manager.process_packed_modules_mapping = atom_process_packed_modules_mapping
    model_manager.is_moe_model = atom_is_moe_model
    model_manager.LoRAModelManager._get_lora_layer_weights = atom_get_lora_layer_weights
    lora_model_module.LoRAModel.from_local_checkpoint = classmethod(
        atom_from_local_checkpoint
    )
    lora_utils._atom_lora_bridge_patched = True
    logger.info("ATOM plugin: installed vLLM dynamic LoRA bridge for ATOM LinearBase")
