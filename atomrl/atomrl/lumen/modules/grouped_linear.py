# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Grouped-linear (MoE) modules with tensor-parallel support.

Parameter names and dist-checkpoint sharding follow Megatron TE grouped linear
(``weight0``, ``weight1``, …) so torch_dist checkpoints from the Miles DSV4
spec load without key remapping.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_pg_rank, get_pg_size
from torch import nn
from torch.nn.parameter import Parameter

from atomrl.lumen.modules._megatron_compat import ensure_metadata_has_dp_cp_group
from atomrl.lumen.modules.parallel_linear import _DeferredWgrad, _get_tp_group, _pg_size

__all__ = [
    "LumenColumnParallelGroupedLinear",
    "LumenGroupedLinear",
    "LumenRowParallelGroupedLinear",
]


class LumenGroupedLinear(nn.Module):
    """Grouped linear for MoE experts (TE-compatible checkpoint layout)."""

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str | None,
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__()
        del kwargs, tp_comm_buffer_name

        self.config = config
        self.num_gemms = num_gemms
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.use_bias = bias

        if pg_collection is None:
            # use_mpu_process_groups() asserts on partial DP groups that exist
            # only after Megatron parallel state is initialized.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            else:
                pg_collection = ProcessGroupCollection()
        self._pg_collection = pg_collection

        self.tp_group = _get_tp_group(tp_group, is_expert, pg_collection)
        self._tp_group = self.tp_group
        tp_size = _pg_size(self.tp_group)

        self.expert_parallel = getattr(config, "expert_model_parallel_size", 1) > 1
        self.explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

        if self.explicit_expert_comm:
            if parallel_mode == "column":
                output_size = divide(output_size, tp_size)
            elif parallel_mode == "row":
                input_size = divide(input_size, tp_size)

        self.in_features = input_size
        self.out_features = output_size

        self.scaling_type = "none"
        self.scaling_manager = None
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self.fp8_activation_store = False
        self._deferred_wgrad = _DeferredWgrad()
        self.use_gemm_bf16 = False

        for gemm_idx in range(num_gemms):
            weight = Parameter(
                torch.empty(
                    output_size,
                    input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            setattr(self, f"weight{gemm_idx}", weight)
            if bias:
                bias_param = Parameter(
                    torch.zeros(
                        output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                setattr(self, f"bias{gemm_idx}", bias_param)

        if getattr(config, "perform_initialization", True):
            for gemm_idx in range(num_gemms):
                init_method(getattr(self, f"weight{gemm_idx}"))

        for param in self.parameters():
            param.allreduce = not (is_expert and self.expert_parallel)

    @property
    def weights(self):
        return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

    @property
    def biases(self):
        if not self.use_bias:
            return None
        return [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return self._sharded_state_dict_grouped({}, prefix, sharded_offsets, metadata)

    def forward(self, x: torch.Tensor, m_splits, m_splits_gpu=None):
        outputs = []
        offset = 0
        for i in range(self.num_gemms):
            count = (
                int(m_splits[i]) if not isinstance(m_splits[i], int) else m_splits[i]
            )
            if count == 0:
                continue
            xi = x[offset : offset + count]
            bias_i = None
            if self.use_bias and not self.skip_bias_add:
                bias_i = getattr(self, f"bias{i}")
            weight = getattr(self, f"weight{i}")
            if self.scaling_type != "none" or self.delay_wgrad or self.use_gemm_bf16:
                from atomrl.lumen.ops.quantize.linear import quantized_linear

                yi = quantized_linear(
                    xi,
                    weight,
                    bias_i,
                    scaling_manager=self.scaling_manager,
                    scaling_type=self.scaling_type,
                    fp8_dtype=self.fp8_dtype,
                    block_size=self.block_size,
                    gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                    delay_wgrad=self.delay_wgrad,
                    deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
                )
            else:
                yi = F.linear(xi, weight, bias_i)
            outputs.append(yi)
            offset += count

        if not outputs:
            output = x.new_empty(0, self.out_features)
        else:
            output = torch.cat(outputs, dim=0)

        if self.skip_bias_add and self.use_bias:
            output_bias = torch.stack(
                [getattr(self, f"bias{i}") for i in range(self.num_gemms)], dim=0
            )
        else:
            output_bias = None
        return output, output_bias

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        from atomrl.lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    @staticmethod
    def _empty_extra_state():
        return torch.empty(0, dtype=torch.uint8)

    def _split_extra_state(self, state):
        return [state] * self.num_gemms

    def _sharded_state_dict_grouped(
        self,
        tp_axis_map,
        prefix="",
        sharded_offsets=(),
        metadata=None,
    ):
        metadata = ensure_metadata_has_dp_cp_group(metadata) or {}
        sharded_state_dict = {}
        full_state_dict = self.state_dict(prefix="", keep_vars=True)
        ep_group = getattr(self._pg_collection, "ep", None)
        if ep_group is None or not torch.distributed.is_initialized():
            ep_size, ep_rank = 1, 0
        else:
            ep_size, ep_rank = get_pg_size(ep_group), get_pg_rank(ep_group)
        num_global_experts = ep_size * self.num_gemms
        local_expert_indices_offset = ep_rank * self.num_gemms
        ep_axis = len(sharded_offsets)
        extra_state = full_state_dict.get("_extra_state", self._empty_extra_state())
        extra_states = self._split_extra_state(extra_state)
        singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)

        for gemm_idx in range(self.num_gemms):
            global_expert_idx = local_expert_indices_offset + gemm_idx
            state_dict = {
                f"{gemm_idx}.weight": full_state_dict[f"weight{gemm_idx}"],
                f"{gemm_idx}._extra_state": extra_states[gemm_idx],
            }
            if self.use_bias:
                state_dict[f"{gemm_idx}.bias"] = full_state_dict[f"bias{gemm_idx}"]
            if singleton_local_shards:
                expert_prefix = f"{global_expert_idx}.{prefix}"
                new_sharded_offsets = sharded_offsets
            else:
                expert_prefix = prefix
                new_sharded_offsets = (
                    *sharded_offsets,
                    (ep_axis, global_expert_idx, num_global_experts),
                )
            sub_sd = make_sharded_tensors_for_checkpoint(
                state_dict,
                "",
                tp_axis_map,
                new_sharded_offsets,
                tp_group=self._tp_group,
                dp_cp_group=metadata.get("dp_cp_group"),
            )
            replace_prefix_for_sharding(sub_sd, f"{gemm_idx}.", expert_prefix)
            sharded_state_dict.update(
                {
                    f"{prefix}weight{gemm_idx}": sub_sd[f"{gemm_idx}.weight"],
                    f"{prefix}_extra_state{'' if gemm_idx == 0 else gemm_idx}": sub_sd[
                        f"{gemm_idx}._extra_state"
                    ],
                }
            )
            if self.use_bias:
                sharded_state_dict[f"{prefix}bias{gemm_idx}"] = sub_sd[
                    f"{gemm_idx}.bias"
                ]

        expt_dp = getattr(self._pg_collection, "expt_dp", None)
        if expt_dp is None or not torch.distributed.is_initialized():
            edp_rank = 0
        else:
            edp_rank = get_pg_rank(expt_dp)
        for sh_ten in sharded_state_dict.values():
            replica_id = getattr(sh_ten, "replica_id", None)
            if replica_id is None:
                continue
            assert len(replica_id) == 3, f"Unexpected replica_id: {replica_id}"
            if getattr(sh_ten, "is_data_parallel_fully_shard", False):
                edp_replica_id = 0
            else:
                edp_replica_id = edp_rank
            sh_ten.replica_id = (*replica_id[:2], edp_replica_id)
        return sharded_state_dict

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return self._empty_extra_state()

    def execute_deferred_wgrad(self):
        self._deferred_wgrad.execute()

    def backward_dw(self):
        self._deferred_wgrad.execute()


class LumenColumnParallelGroupedLinear(LumenGroupedLinear):
    """Column-parallel grouped linear for MoE."""

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            pg_collection=pg_collection,
            **kwargs,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        tp_axis_map = {}
        for gemm_idx in range(self.num_gemms):
            tp_axis_map.update({f"{gemm_idx}.weight": 0, f"{gemm_idx}.bias": 0})
        return self._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )


class LumenRowParallelGroupedLinear(LumenGroupedLinear):
    """Row-parallel grouped linear for MoE."""

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(
            num_gemms=num_gemms,
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            pg_collection=pg_collection,
            **kwargs,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        tp_axis_map = {f"{gemm_idx}.weight": 1 for gemm_idx in range(self.num_gemms)}
        return self._sharded_state_dict_grouped(
            tp_axis_map, prefix, sharded_offsets, metadata
        )
