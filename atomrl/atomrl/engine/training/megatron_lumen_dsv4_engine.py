# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Megatron-Core training engine for DeepSeek-V4-Flash (Lumen DSV4 spec).

Subclasses :class:`MegatronEngine` to use the Lumen DSV4 layer spec
(MLA attention, Hyper-Connection, compressor/indexer, MoE) instead of
the standard Qwen3 GPT layer spec.  Registered as ``backend="megatron_lumen_dsv4"``.

Key differences from the Qwen3 parent:
  - ``DSV4Dims`` replaces ``GPTDims``
  - ``get_dsv4_spec()`` replaces ``get_gpt_layer_local_spec()``
  - ``bridges.dsv4.hf_to_megatron()`` / ``bridges.dsv4.megatron_to_hf()`` for weight I/O
  - MoE router gates, expert bias, and tid2eid hash table are frozen
  - FP32 parameters (HC, attn_sink, compressor APE/norm) are preserved
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import SimpleNamespace
from typing import Any, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F

from atomrl.engine.training.base_engine import EngineRegistry
from atomrl.engine.training.bridges import dsv4
from atomrl.engine.training.bridges.core import load_hf_safetensors
from atomrl.engine.training.megatron_base_engine import moe_dispatcher_kwargs
from atomrl.engine.training.megatron_engine import (
    MegatronEngine,
    _clear_stale_router_replay_instances,
    _gather_with_stride,
    _shard_with_stride,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "INFO"))

_STREAMED_ADAM_CAPABILITY = "ATOMRL_DSV4_CAPABILITY_STREAMED_ADAM"
_HYBRID_OPTIMIZER_MODULE = "megatron.core.optimizer.cpu_offloading.hybrid_optimizer"
_MAX_STREAMED_OPTIMIZER_CHUNK_MIB = 1024

T = TypeVar("T")


def _lockstep_stream(
    stream: Iterable[T],
    synchronize: Callable[[], None],
) -> Iterator[T]:
    """Prevent consumers from requesting the next item before all ranks agree.

    All distributed ranks must consume through ``StopIteration``. Closing early
    or a consumer failure may leave peers waiting at the final barrier.
    """
    for item in stream:
        yield item
        synchronize()


def _safe_optimizer_attr(optimizer, name):
    try:
        return getattr(optimizer, name, None)
    except (AssertionError, AttributeError, RuntimeError):
        return None


def _find_hybrid_device_optimizers(optimizer):
    """Find every unique HDO through wrappers in deterministic traversal order."""
    pending = [optimizer]
    seen: set[int] = set()
    hybrids = []
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            pending.extend(reversed(tuple(current.values())))
            continue
        if isinstance(current, (list, tuple)):
            pending.extend(reversed(current))
            continue
        cpu_optimizers = _safe_optimizer_attr(current, "cpu_optimizers")
        offload_fraction = _safe_optimizer_attr(current, "offload_fraction")
        if cpu_optimizers is not None and offload_fraction is not None:
            hybrids.append(current)
            continue

        children = []
        for name in (
            "chained_optimizers",
            "optimizers",
            "inner_optimizer",
            "optimizer",
        ):
            child = _safe_optimizer_attr(current, name)
            if child is not None:
                children.append(child)
        pending.extend(reversed(children))
    return hybrids


def _streamed_optimizer_settings(
    engine_config: Mapping[str, Any],
) -> tuple[str, int]:
    mode = engine_config.get("streamed_optimizer_mode", "off")
    if not isinstance(mode, str) or mode not in {"off", "sgd", "adam"}:
        raise ValueError(
            "streamed_optimizer_mode must be exactly one of off, sgd, or adam; "
            f"got {mode!r}"
        )
    chunk_mib = engine_config.get(
        "streamed_optimizer_chunk_size_mib",
        256,
    )
    if (
        isinstance(chunk_mib, bool)
        or not isinstance(chunk_mib, int)
        or chunk_mib <= 0
        or chunk_mib > _MAX_STREAMED_OPTIMIZER_CHUNK_MIB
    ):
        raise ValueError(
            "streamed_optimizer_chunk_size_mib must be a positive integer "
            f"no greater than {_MAX_STREAMED_OPTIMIZER_CHUNK_MIB}; "
            f"got {chunk_mib!r}"
        )
    return mode, chunk_mib


def _streamed_optimizer_staging_diagnostic(
    mode: str,
    chunk_mib: int,
    moment_dtype: torch.dtype = torch.float32,
) -> str:
    if mode == "adam":
        buffers = 4 if moment_dtype == torch.bfloat16 else 2
        return f"{buffers * chunk_mib} MiB"
    if mode == "off":
        return "0 MiB (disabled)"
    return "runtime-sized/unknown"


def _streamed_optimizer_moment_dtype(
    engine_config: Mapping[str, Any],
) -> torch.dtype:
    value = engine_config.get("streamed_optimizer_moment_dtype", "fp32")
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(
        "streamed_optimizer_moment_dtype must be exactly fp32 or bf16; "
        f"got {value!r}"
    )


def _streamed_adam_runtime_capability() -> bool:
    try:
        module = importlib.import_module(_HYBRID_OPTIMIZER_MODULE)
    except (ImportError, ModuleNotFoundError):
        return False
    return getattr(module, _STREAMED_ADAM_CAPABILITY, False) is True


def _validate_streamed_optimizer_preconstruction(engine_config):
    """Validate settings that do not depend on a constructed optimizer."""
    mode, _ = _streamed_optimizer_settings(engine_config)
    _streamed_optimizer_moment_dtype(engine_config)
    capability = _streamed_adam_runtime_capability()
    if mode != "adam":
        return mode, capability
    if engine_config.get("optimizer_cpu_offload", False) is not True:
        raise RuntimeError("streamed Adam requires optimizer_cpu_offload=True")
    try:
        configured_fraction = float(
            engine_config.get("optimizer_offload_fraction", 1.0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "streamed Adam requires full CPU offload with "
            "optimizer_offload_fraction=1.0"
        ) from exc
    if configured_fraction != 1.0:
        raise RuntimeError(
            "streamed Adam requires full CPU offload with "
            "optimizer_offload_fraction=1.0"
        )
    if not capability:
        raise RuntimeError(
            "streamed Adam requires patched Megatron runtime capability "
            f"{_STREAMED_ADAM_CAPABILITY} is True"
        )
    return mode, True


def _validate_streamed_optimizer_request(engine_config, optimizer):
    """Validate a requested streamed optimizer before the first training step."""
    mode, _ = _validate_streamed_optimizer_preconstruction(engine_config)
    if mode == "off":
        return None
    hybrids = _find_hybrid_device_optimizers(optimizer)
    if not hybrids:
        raise RuntimeError(
            f"streamed optimizer mode={mode} requires at least one "
            "Megatron HybridDeviceOptimizer"
        )
    if mode == "sgd":
        # The existing patched-SGD dispatch also supports an unset mode. Return
        # all HDOs so callers can make the explicit request visible.
        return hybrids

    incompatible_flags = (
        "amsgrad",
        "differentiable",
        "capturable",
        "foreach",
    )
    for hybrid_index, hybrid in enumerate(hybrids):
        try:
            runtime_fraction = float(hybrid.offload_fraction)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "streamed Adam requires every HybridDeviceOptimizer to use "
                f"full CPU offload; hybrid={hybrid_index}"
            ) from exc
        if runtime_fraction != 1.0:
            raise RuntimeError(
                "streamed Adam requires every HybridDeviceOptimizer to use "
                f"full CPU offload; hybrid={hybrid_index}"
            )
        if getattr(hybrid, "gpu_optimizer", None) is not None:
            raise RuntimeError(
                "streamed Adam full CPU offload requires no GPU optimizer; "
                f"hybrid={hybrid_index}"
            )

        cpu_optimizers = getattr(hybrid, "cpu_optimizers", None)
        if not cpu_optimizers:
            raise RuntimeError(
                "streamed Adam requires at least one CPU Adam or AdamW "
                f"optimizer per HDO; hybrid={hybrid_index}"
            )
        for optimizer_index, cpu_optimizer in enumerate(cpu_optimizers):
            if not isinstance(
                cpu_optimizer,
                (torch.optim.Adam, torch.optim.AdamW),
            ):
                raise RuntimeError(  # noqa: TRY004
                    "streamed Adam requires every CPU optimizer to be "
                    "torch.optim.Adam or AdamW; "
                    f"hybrid={hybrid_index} optimizer={optimizer_index}"
                )
            for group_index, group in enumerate(cpu_optimizer.param_groups):
                for flag in incompatible_flags:
                    if bool(group.get(flag, False)):
                        raise RuntimeError(
                            "streamed Adam has an incompatible optimizer flag: "
                            f"hybrid={hybrid_index} "
                            f"optimizer={optimizer_index} "
                            f"parameter_group={group_index} {flag}=True"
                        )
    return hybrids


def _configure_streamed_optimizer_request(engine_config, optimizer):
    """Validate all HDOs, then attach Lumen-only settings atomically."""
    hybrids = _validate_streamed_optimizer_request(engine_config, optimizer)
    if hybrids is None:
        return None
    mode, chunk_mib = _streamed_optimizer_settings(engine_config)
    moment_dtype = _streamed_optimizer_moment_dtype(engine_config)
    chunk_numel = chunk_mib * (1024 * 1024 // 4)
    for hybrid in hybrids:
        hybrid._lumen_streamed_optimizer_mode = mode
        if mode == "adam":
            hybrid._lumen_streamed_adam_chunk_numel = chunk_numel
            hybrid._lumen_streamed_adam_moment_dtype = moment_dtype
    return hybrids


def _rss_gib() -> float | None:
    """Return this rank's peak RSS in GiB where getrusage is available."""
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _resolve_dsv4_topk_backend(engine_config: Mapping[str, Any]) -> str:
    return str(
        engine_config.get(
            "dsv4_dsa_topk_backend",
            engine_config.get("miles_dsa_topk_backend", "torch"),
        )
    )


def _dsv4_router_kwargs(
    hf: Mapping[str, Any],
    ec: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
    }
    for name, value in expected.items():
        if hf.get(name) != value:
            raise ValueError(
                f"DSV4 Megatron requires checkpoint {name}={value!r}, "
                f"got {hf.get(name)!r}"
            )

    scaling_factor = float(hf.get("routed_scaling_factor", 1.0))
    scaling_override = ec.get("moe_router_topk_scaling_factor")
    if scaling_override is not None and float(scaling_override) != scaling_factor:
        raise ValueError(
            "engine moe_router_topk_scaling_factor conflicts with checkpoint "
            f"routed_scaling_factor={scaling_factor}"
        )
    return {
        "moe_router_load_balancing_type": "none",
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_dtype": "fp32",
        "moe_router_topk_scaling_factor": scaling_factor,
        "moe_router_enable_expert_bias": True,
        # DSV4's checkpoint bias is part of the frozen routing policy.
        "moe_router_bias_update_rate": 0.0,
    }


def _optimizer_precision_kwargs(ec: Mapping[str, Any]) -> dict[str, Any]:
    enabled = bool(ec.get("use_precision_aware_optimizer", False))
    if not enabled:
        return {"use_precision_aware_optimizer": False}
    moment_dtype = _streamed_optimizer_moment_dtype(ec)
    return {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": torch.float32,
        "main_params_dtype": torch.float32,
        "exp_avg_dtype": moment_dtype,
        "exp_avg_sq_dtype": moment_dtype,
    }


def _register_checkpoint_static_buffers(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Materialize checkpoint-only router tables omitted by the model spec."""
    for name, tensor in state.items():
        if not name.endswith(".tid2eid"):
            continue
        module_path, leaf = name.rsplit(".", 1)
        module = model.get_submodule(module_path)
        if leaf in module._buffers:
            continue
        if hasattr(module, leaf):
            delattr(module, leaf)
        module.register_buffer(leaf, torch.empty_like(tensor), persistent=True)


def _named_export_tensors(module) -> list[tuple[str, torch.Tensor]]:
    return list(module.named_parameters())


class _StreamingGatheredParamMapping(Mapping[str, torch.Tensor]):
    """Materialize only the Megatron parameter currently requested by the bridge."""

    _EXPERT_PATTERN = re.compile(
        r"^(?P<prefix>.*\.experts\.linear_fc[12]\.weight)(?P<index>\d+)$"
    )

    def __init__(
        self,
        named_parameters,
        *,
        tp_size,
        tp_group,
        ep_size,
        ep_group,
        etp_size,
        etp_group,
        num_experts,
        metadata=False,
    ):
        self._params = {}
        for raw_name, param in named_parameters:
            name = raw_name
            for prefix in ("module.module.", "module."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            self._params[name] = param
        self._tp_size = int(tp_size)
        self._tp_group = tp_group
        self._ep_size = int(ep_size)
        self._ep_group = ep_group
        self._etp_size = int(etp_size)
        self._etp_group = etp_group
        self._num_experts = int(num_experts)
        self._num_local_experts = max(1, self._num_experts // max(1, self._ep_size))
        self._metadata = bool(metadata)

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def _expert_location(self, key):
        match = self._EXPERT_PATTERN.match(key)
        if match is None or self._num_experts <= 0:
            return None
        global_index = int(match.group("index"))
        if global_index >= self._num_experts:
            return None
        local_index = global_index % self._num_local_experts
        source_ep_rank = global_index // self._num_local_experts
        local_key = f"{match.group('prefix')}{local_index}"
        return local_key, source_ep_rank

    def __contains__(self, key) -> bool:
        if key in self._params:
            return True
        location = self._expert_location(key)
        return location is not None and location[0] in self._params

    @staticmethod
    def _partition_dim(param):
        if hasattr(param, "_lumen_weight_partition_dim"):
            return param._lumen_weight_partition_dim
        if getattr(param, "tensor_model_parallel", False):
            return int(getattr(param, "partition_dim", 0))
        return None

    @classmethod
    def _meta_tensor(cls, param, group_size):
        shape = list(param.shape)
        partition_dim = cls._partition_dim(param)
        if partition_dim is not None and group_size > 1:
            shape[partition_dim] *= int(group_size)
        return torch.empty(tuple(shape), dtype=param.dtype, device="meta")

    def _materialize_local(self, param, *, expert):
        group_size = self._etp_size if expert else self._tp_size
        group = self._etp_group if expert else self._tp_group
        if self._metadata:
            return self._meta_tensor(param, group_size)
        tensor = param.data
        partition_dim = self._partition_dim(param)
        if partition_dim is not None and group_size > 1 and group is not None:
            parts = [torch.empty_like(tensor) for _ in range(group_size)]
            dist.all_gather(parts, tensor, group=group)
            tensor = _gather_with_stride(
                parts,
                partition_dim,
                int(getattr(param, "partition_stride", 1)),
            )
        return tensor

    def __getitem__(self, key):
        location = self._expert_location(key)
        if location is None:
            try:
                param = self._params[key]
            except KeyError:
                raise KeyError(key) from None
            return self._materialize_local(param, expert=False)

        local_key, source_ep_rank = location
        try:
            param = self._params[local_key]
        except KeyError:
            raise KeyError(key) from None
        tensor = self._materialize_local(param, expert=True)
        if self._metadata or self._ep_size <= 1 or self._ep_group is None:
            return tensor
        tensor = tensor.contiguous()
        gathered = [
            torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for _ in range(self._ep_size)
        ]
        try:
            dist.all_gather(gathered, tensor, group=self._ep_group)
        except BaseException as exc:
            try:
                group_ranks = dist.get_process_group_ranks(self._ep_group)
            except (RuntimeError, ValueError):
                group_ranks = "<unavailable>"
            raise RuntimeError(
                "DSV4 expert export EP gather failed: "
                f"key={key}, local_key={local_key}, "
                f"source_ep_rank={source_ep_rank}, "
                f"device={tensor.device}, shape={tuple(tensor.shape)}, "
                f"stride={tensor.stride()}, dtype={tensor.dtype}, "
                f"contiguous={tensor.is_contiguous()}, "
                f"group_ranks={group_ranks}"
            ) from exc
        return gathered[source_ep_rank]


def _configure_dsv4_indexer_environment(engine_config: dict[str, Any]) -> None:
    """Export optional indexer tuning before DSV4 modules are imported."""
    settings = (
        ("v4_indexer_block_n", "V4_INDEXER_BLOCK_N"),
        ("v4_indexer_num_stages", "V4_INDEXER_NUM_STAGES"),
    )
    for config_key, environment_key in settings:
        if engine_config.get(config_key) is not None:
            os.environ[environment_key] = str(engine_config[config_key])


def _dsv4_sequence_alignment(
    tensor_parallel_size: int, compress_ratios: list[int]
) -> int:
    """Return an input length accepted by TP and every DSV4 compressor."""
    divisors = [max(1, int(tensor_parallel_size))]
    divisors.extend(int(ratio) for ratio in compress_ratios if int(ratio) > 0)
    return math.lcm(*divisors)


class MegatronLumenDSV4Engine(MegatronEngine):
    """Megatron-Core GPTModel engine for DeepSeek-V4-Flash (BF16, TP/PP/EP/DP)."""

    def initialize(self) -> None:
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        ec = self.engine_config
        _configure_dsv4_indexer_environment(ec)
        tp = int(ec.get("tensor_model_parallel_size", 1))
        pp = int(ec.get("pipeline_model_parallel_size", 1))
        cp = int(ec.get("context_parallel_size", 1))
        ep = int(ec.get("expert_model_parallel_size", 1))
        seed = int(ec.get("seed", 42))
        self._attention_backend = str(ec.get("attention_backend") or "unfused").lower()
        self._use_packed_sequences = bool(ec.get("use_packed_sequences", True))
        self._logprob_chunk_size = int(ec.get("log_probs_chunk_size") or 0)
        self._r3_enabled = bool(ec.get("moe_enable_routing_replay", False))

        etp = ec.get("expert_tensor_parallel_size", None)
        if etp is not None:
            etp = int(etp)

        def _early_diag(tag):
            if not torch.cuda.is_available():
                return
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            print(
                f"[MEM_DIAG rank={self._rank()}] {tag}: "
                f"total={total/2**30:.2f} GiB, free={free/2**30:.2f} GiB, "
                f"alloc={alloc/2**30:.2f} GiB, "
                f"non_torch={(total - free - alloc)/2**30:.2f} GiB",
                file=sys.stderr,
                flush=True,
            )

        _early_diag("BEFORE mpu.initialize_model_parallel")
        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                expert_model_parallel_size=ep,
                expert_tensor_parallel_size=etp,
            )
        _early_diag("AFTER mpu.initialize_model_parallel")
        model_parallel_cuda_manual_seed(seed)

        # ---- HF config -> DSV4 dims / TransformerConfig ----
        cfg_path = os.path.join(self.model_name, "config.json")
        with open(cfg_path) as fh:
            hf = json.load(fh)

        # HF config field names differ between model families. DSV4 uses:
        #   head_dim (not kv_lora_rank), qk_rope_head_dim (not qk_pos_emb_head_dim),
        #   n_routed_experts (not num_experts), compress_ratios (not dsv4_compress_ratios),
        #   sliding_window (not dsv4_window_size), etc.
        num_experts = int(
            hf.get("n_routed_experts")
            or hf.get("num_experts", 0)
            or ec.get("num_experts", 0)
        )
        moe_ffn = int(
            hf.get("moe_intermediate_size", 0) or hf.get("moe_ffn_hidden_size", 0) or 0
        )
        n_shared = int(hf.get("n_shared_experts", 0) or 0)
        shared_ffn = int(
            hf.get("shared_expert_intermediate_size", 0)
            or (moe_ffn * n_shared if n_shared else 0)
        )
        head_dim = int(hf.get("head_dim") or hf.get("kv_lora_rank", 512))
        q_lora_rank = int(hf.get("q_lora_rank", 1024))
        qk_pos_emb_head_dim = int(
            hf.get("qk_rope_head_dim") or hf.get("qk_pos_emb_head_dim", 64)
        )
        v_head_dim = int(hf.get("v_head_dim") or head_dim)

        compress_ratios_raw = (
            ec.get("dsv4_compress_ratios")
            or hf.get("compress_ratios")
            or hf.get("dsv4_compress_ratios")
        )
        if compress_ratios_raw is not None:
            if isinstance(compress_ratios_raw, str):
                compress_ratios = [int(x) for x in compress_ratios_raw.split()]
            else:
                compress_ratios = [int(x) for x in compress_ratios_raw]
        else:
            compress_ratios = list(dsv4.DSV4_FLASH_COMPRESS_RATIOS)
        self._input_sequence_alignment = _dsv4_sequence_alignment(tp, compress_ratios)

        hc_mult = int(
            ec.get("dsv4_hc_mult") or hf.get("hc_mult") or hf.get("dsv4_hc_mult", 4)
        )
        o_groups = int(
            ec.get("dsv4_o_groups") or hf.get("o_groups") or hf.get("dsv4_o_groups", 8)
        )
        o_lora_rank = int(
            ec.get("dsv4_o_lora_rank")
            or hf.get("o_lora_rank")
            or hf.get("dsv4_o_lora_rank", 1024)
        )
        n_hash_layers = int(
            ec.get("dsv4_n_hash_layers")
            or hf.get("num_hash_layers")
            or hf.get("dsv4_n_hash_layers", 3)
        )
        window_size = int(
            ec.get("dsv4_window_size")
            or hf.get("sliding_window")
            or hf.get("dsv4_window_size", 128)
        )
        moe_topk = int(hf.get("num_experts_per_tok", ec.get("moe_router_topk", 6)))

        self._dims = dsv4.DSV4Dims(
            num_layers=hf["num_hidden_layers"],
            hidden=hf["hidden_size"],
            num_heads=hf["num_attention_heads"],
            num_kv_groups=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            head_dim=head_dim,
            ffn=hf.get("intermediate_size", moe_ffn),
            vocab=hf["vocab_size"],
            num_experts=num_experts,
            moe_ffn=moe_ffn,
            shared_expert_ffn=shared_ffn,
            shared_expert_gate=bool(hf.get("shared_expert_gate", False)),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            hc_mult=hc_mult,
            n_hash_layers=n_hash_layers,
            window_size=window_size,
            compress_ratios=compress_ratios,
            moe_topk=moe_topk,
        )

        # ---- Build MLATransformerConfig directly ----
        # Bypass core_transformer_config_from_args (too many version-dependent args).
        # MLATransformerConfig is a dataclass — pass only the fields it declares.
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        compress_rope_theta = float(
            ec.get("dsv4_compress_rope_theta")
            or hf.get("compress_rope_theta")
            or hf.get("dsv4_compress_rope_theta", 160000)
        )
        first_pp_layers = ec.get("num_layers_in_first_pipeline_stage")
        last_pp_layers = ec.get("num_layers_in_last_pipeline_stage")

        recompute_kwargs = {}
        if ec.get("recompute_granularity"):
            recompute_kwargs["recompute_granularity"] = ec["recompute_granularity"]
            recompute_kwargs["recompute_method"] = (
                ec.get("recompute_method") or "uniform"
            )
            recompute_kwargs["recompute_num_layers"] = int(
                ec.get("recompute_num_layers") or 1
            )

        pp_kwargs = {}
        if first_pp_layers is not None:
            pp_kwargs["num_layers_in_first_pipeline_stage"] = int(first_pp_layers)
        if last_pp_layers is not None:
            pp_kwargs["num_layers_in_last_pipeline_stage"] = int(last_pp_layers)
        if pp > 1:
            pp_kwargs["variable_seq_lengths"] = True
            pp_kwargs["batch_p2p_comm"] = False

        moe_kwargs = {}
        if num_experts > 0:
            moe_kwargs.update(
                num_moe_experts=num_experts,
                moe_ffn_hidden_size=(
                    moe_ffn if moe_ffn > 0 else hf.get("intermediate_size", 2048)
                ),
                moe_router_topk=moe_topk,
                moe_grouped_gemm=bool(ec.get("moe_grouped_gemm", False)),
                expert_model_parallel_size=ep,
                moe_enable_routing_replay=self._r3_enabled,
            )
            moe_kwargs.update(_dsv4_router_kwargs(hf, ec))
            if shared_ffn > 0:
                moe_kwargs["moe_shared_expert_intermediate_size"] = shared_ffn

        tfcfg = MLATransformerConfig(
            # Core
            num_layers=hf["num_hidden_layers"],
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_query_groups=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            ffn_hidden_size=hf.get("intermediate_size", moe_ffn),
            kv_channels=head_dim,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            normalization="RMSNorm",
            layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
            qk_layernorm=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_softmax_in_fp32=bool(ec.get("attention_softmax_in_fp32", True)),
            bf16=True,
            fp8=None,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            sequence_parallel=bool(ec.get("sequence_parallel", False)),
            use_cpu_initialization=True,
            perform_initialization=True,
            # MLA
            q_lora_rank=q_lora_rank,
            kv_lora_rank=head_dim,
            qk_head_dim=head_dim,
            qk_pos_emb_head_dim=qk_pos_emb_head_dim,
            v_head_dim=v_head_dim,
            rotary_base=hf.get("rope_theta", 10000.0),
            rotary_scaling_factor=hf.get("rotary_scaling_factor", 16),
            original_max_position_embeddings=hf.get(
                "original_max_position_embeddings", 65536
            ),
            beta_fast=32,
            beta_slow=1,
            # DSV4 (patch-added fields)
            experimental_attention_variant="dsv4",
            dsv4_hc_mult=hc_mult,
            dsv4_hc_sinkhorn_iters=int(ec.get("dsv4_hc_sinkhorn_iters", 20)),
            dsv4_hc_eps=float(ec.get("dsv4_hc_eps", 1e-6)),
            dsv4_o_groups=o_groups,
            dsv4_o_lora_rank=o_lora_rank,
            dsv4_window_size=window_size,
            dsv4_n_hash_layers=n_hash_layers,
            dsv4_compress_ratios=compress_ratios,
            dsv4_compress_rope_theta=compress_rope_theta,
            dsa_indexer_n_heads=int(
                ec.get("dsa_indexer_n_heads")
                or hf.get("index_n_heads")
                or hf.get("dsa_indexer_n_heads", 64)
            ),
            dsa_indexer_head_dim=int(
                ec.get("dsa_indexer_head_dim")
                or hf.get("index_head_dim")
                or hf.get("dsa_indexer_head_dim", 128)
            ),
            dsa_indexer_topk=int(
                ec.get("dsa_indexer_topk")
                or hf.get("index_topk")
                or hf.get("dsa_indexer_topk", 512)
            ),
            **moe_dispatcher_kwargs(
                ec,
                tp=tp,
                cp=cp,
                sp=bool(ec.get("sequence_parallel", False)),
                max_tokens_per_gpu=int(ec.get("max_tokens_per_gpu") or 0),
            ),
            **recompute_kwargs,
            **moe_kwargs,
            **pp_kwargs,
        )
        # This is a Lumen extension rather than an MLATransformerConfig field.
        tfcfg.dsv4_dsa_topk_backend = _resolve_dsv4_topk_backend(ec)

        # get_dsv4_spec copies this canonical Lumen argument onto the config.
        mock_args = SimpleNamespace(
            dsv4_dsa_topk_backend=tfcfg.dsv4_dsa_topk_backend,
        )

        # ---- Build GPTModel using Lumen's get_dsv4_spec ----
        from atomrl.lumen.models.dsv4.megatron.spec import get_dsv4_spec

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._pp_rank = pp_rank
        self._pp_size = pp_size
        self._tp_rank = mpu.get_tensor_model_parallel_rank()
        self._tp_size = tp

        spec = get_dsv4_spec(mock_args, tfcfg, vp_stage=0)

        # Compute per-PP-rank layer counts for the bridge.
        self._layers_per_pp_rank: list[int] | None = None
        if pp_size > 1:
            total = hf["num_hidden_layers"]
            first = int(first_pp_layers) if first_pp_layers is not None else None
            last = int(last_pp_layers) if last_pp_layers is not None else None
            if first is not None or last is not None:
                remaining = total - (first or 0) - (last or 0)
                mid_stages = (
                    pp_size
                    - (1 if first is not None else 0)
                    - (1 if last is not None else 0)
                )
                per_mid = remaining // mid_stages if mid_stages > 0 else 0
                lpp: list[int] = []
                for s in range(pp_size):
                    if s == 0 and first is not None:
                        lpp.append(first)
                    elif s == pp_size - 1 and last is not None:
                        lpp.append(last)
                    else:
                        lpp.append(per_mid)
                self._layers_per_pp_rank = lpp
            else:
                per_stage = total // pp_size
                self._layers_per_pp_rank = [per_stage] * pp_size

        _clear_stale_router_replay_instances(
            self._r3_enabled,
            dsv4_enabled=True,
        )
        model = GPTModel(
            config=tfcfg,
            transformer_layer_spec=spec,
            vocab_size=hf["vocab_size"],
            max_sequence_length=hf.get("max_position_embeddings", 65536),
            pre_process=(pp_rank == 0),
            post_process=(pp_rank == pp_size - 1),
            parallel_output=False,
            position_embedding_type="rope",
            rotary_base=hf.get("rope_theta", 10000.0),
            share_embeddings_and_output_weights=hf.get("tie_word_embeddings", False),
        )

        # ---- Load HF weights ----
        ep_rank = mpu.get_expert_model_parallel_rank() if num_experts > 0 else 0
        ep_size = mpu.get_expert_model_parallel_world_size() if num_experts > 0 else 1
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        logger.info(
            "MegatronLumenDSV4Engine[%d]: loading HF weights from %s (ep_rank=%d/%d, experts=%d)",
            self._rank(),
            self.model_name,
            ep_rank,
            ep_size,
            num_experts,
        )
        hf_state = load_hf_safetensors(self.model_name)
        meg_state = dsv4.hf_to_megatron(
            hf_state,
            self._dims,
            ep_rank=ep_rank,
            ep_size=ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            layers_per_pp_rank=self._layers_per_pp_rank,
            use_grouped_mlp=True,
        )
        del hf_state

        # TP shard: dsv4.hf_to_megatron returns full tensors; the GPTModel's
        # parallel linears only hold 1/tp of each weight.
        #
        # Detect which params need TP sharding by comparing bridge output shape
        # vs model param shape. This is more reliable than tensor_model_parallel
        # because Lumen's LumenDuplicatedLinear sets tensor_model_parallel=True
        # even though it's a full-size replicated weight (used for grad allreduce
        # policy, not for weight partitioning).
        tp_rank = mpu.get_tensor_model_parallel_rank()
        etp_val = etp if etp is not None else 1
        etp_rank = mpu.get_expert_tensor_parallel_rank() if etp_val > 1 else 0
        if tp > 1 or etp_val > 1:
            for name, param in model.named_parameters():
                if name not in meg_state:
                    continue
                full = meg_state[name]
                if full.shape == param.shape:
                    param._lumen_weight_partition_dim = None
                    continue  # same shape → no sharding needed (duplicated)
                is_expert = ".experts." in name and ".shared_experts." not in name
                shard_size = etp_val if is_expert else tp
                shard_rank = etp_rank if is_expert else tp_rank
                if shard_size <= 1:
                    continue
                # Detect partition dim from shape difference
                pdim = 0
                for d in range(min(full.ndim, param.ndim)):
                    if full.shape[d] != param.shape[d]:
                        pdim = d
                        break
                param._lumen_weight_partition_dim = pdim
                pstride = getattr(param, "partition_stride", 1)
                meg_state[name] = _shard_with_stride(
                    meg_state[name],
                    pdim,
                    pstride,
                    shard_rank,
                    shard_size,
                )

        _register_checkpoint_static_buffers(model, meg_state)
        incompat = model.load_state_dict(meg_state, strict=False)
        real_missing = [k for k in incompat.missing_keys if "_extra_state" not in k]
        if real_missing:
            raise RuntimeError(
                f"Megatron DSV4 load missing keys: {real_missing[:10]} ..."
            )
        if incompat.unexpected_keys:
            logger.warning(
                "Megatron DSV4 load unexpected keys: %s", incompat.unexpected_keys[:6]
            )
        del meg_state
        import gc

        gc.collect()

        def _gpu_diag(tag: str) -> None:
            if not torch.cuda.is_available():
                return
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            alloc = torch.cuda.memory_allocated()
            resv = torch.cuda.memory_reserved()
            msg = (
                f"[GPU_DIAG rank={self._rank()}] {tag}: "
                f"total={total / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB, "
                f"alloc={alloc / 2**30:.2f} GiB, reserved={resv / 2**30:.2f} GiB"
            )
            print(msg, file=sys.stderr, flush=True)
            logger.info(msg)

        _gpu_diag("BEFORE model.cuda()")
        # Move to GPU. Preserve FP32 params that have _keep_fp32 marker.
        model = model.cuda()
        for name, param in model.named_parameters():
            if getattr(param, "_keep_fp32", False):
                continue
            if param.dtype != torch.bfloat16 and param.is_floating_point():
                param.data = param.data.bfloat16()
        self.module = model
        self._tfcfg = tfcfg
        _gpu_diag("AFTER model.cuda()")

        # ---- Freeze non-trainable params ----
        # MoE router gates, expert bias, and tid2eid hash table
        for name, param in self.module.named_parameters():
            if any(
                key in name
                for key in (
                    "mlp.router.weight",
                    "mlp.router.expert_bias",
                    "mlp.router.tid2eid",
                )
            ):
                param.requires_grad_(False)

        n_params = sum(p.numel() for p in self.module.parameters())
        n_grad = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        print(
            f"[GPU_DIAG rank={self._rank()}] params={n_params:,} "
            f"({n_params * 2 / 2**30:.2f} GiB bf16), requires_grad={n_grad:,}",
            file=sys.stderr,
            flush=True,
        )

        if os.environ.get("ATOMRL_FORWARD_ONLY_INIT", "0") == "1":
            self._ddp = None
            self.optimizer = None
            self.lr_scheduler = None
            logger.info(
                "Forward-only Megatron initialization: skipped DDP, gradient "
                "buffers, optimizer, and scheduler"
            )
            return

        # ---- Megatron DistributedDataParallel + optimizer ----
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        oc = self.optimizer_config
        self._clip = float(oc.get("clip_grad", 1.0))
        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=bool(ec.get("grad_reduce_in_fp32", False)),
            overlap_grad_reduce=bool(ec.get("overlap_grad_reduce", False)),
            use_distributed_optimizer=bool(ec.get("use_distributed_optimizer", True)),
            average_in_collective=True,
            bucket_size=int(ec.get("bucket_size", None) or 500_000_000),
        )
        _gpu_diag("BEFORE DDP init")
        self._ddp = DDP(config=tfcfg, ddp_config=ddp_cfg, module=self.module)
        _gpu_diag("AFTER DDP init")

        cpu_offload = bool(ec.get("optimizer_cpu_offload", False))
        offload_frac = float(ec.get("optimizer_offload_fraction", 1.0))
        streamed_mode, streamed_capability = (
            _validate_streamed_optimizer_preconstruction(ec)
        )
        _, streamed_chunk_mib = _streamed_optimizer_settings(ec)
        streamed_moment_dtype = _streamed_optimizer_moment_dtype(ec)
        streamed_staging = _streamed_optimizer_staging_diagnostic(
            streamed_mode,
            streamed_chunk_mib,
            streamed_moment_dtype,
        )
        startup_message = (
            "[AtomRL streamed optimizer] "
            f"mode={streamed_mode} chunk={streamed_chunk_mib} MiB "
            f"moments={str(streamed_moment_dtype).removeprefix('torch.')} "
            f"staging={streamed_staging} "
            f"capability={streamed_capability}"
        )
        print(startup_message, file=sys.stderr, flush=True)
        logger.info(startup_message)
        optimizer_name = str(
            oc.get("optimizer_type", oc.get("optimizer", "adam"))
        ).lower()
        if optimizer_name == "adamw":
            optimizer_name = "adam"
        opt_cfg = OptimizerConfig(
            optimizer=optimizer_name,
            lr=float(oc.get("lr", 1e-6)),
            weight_decay=float(oc.get("weight_decay", 0.1)),
            adam_beta1=float(oc.get("adam_beta1", 0.9)),
            adam_beta2=float(oc.get("adam_beta2", 0.95)),
            adam_eps=float(oc.get("adam_eps", 1e-8)),
            sgd_momentum=float(oc.get("sgd_momentum", 0.0)),
            clip_grad=self._clip,
            bf16=True,
            fp16=False,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=bool(ec.get("use_distributed_optimizer", True)),
            optimizer_cpu_offload=cpu_offload,
            optimizer_offload_fraction=offload_frac,
            pin_cpu_grads=False,
            pin_cpu_params=False,
            **_optimizer_precision_kwargs(ec),
        )
        rss_gib = _rss_gib()
        rss_value = "unavailable" if rss_gib is None else f"{rss_gib:.2f} GiB"
        rss_message = (
            f"[AtomRL RSS rank={self._rank()}] "
            f"before_optimizer_state_allocation={rss_value}"
        )
        print(rss_message, file=sys.stderr, flush=True)
        logger.info(rss_message)
        _gpu_diag("BEFORE optimizer init")
        self.optimizer = get_megatron_optimizer(opt_cfg, model_chunks=[self._ddp])
        _configure_streamed_optimizer_request(ec, self.optimizer)

        warmup = int(oc.get("lr_warmup_steps", 10))
        base_lr = float(oc.get("lr", 1e-6))
        wd = float(oc.get("weight_decay", 0.1))
        self.lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0,
            max_lr=base_lr,
            min_lr=base_lr,
            lr_warmup_steps=warmup,
            lr_decay_steps=max(warmup + 1, 1000),
            lr_decay_style="constant",
            start_wd=wd,
            end_wd=wd,
            wd_incr_steps=0,
            wd_incr_style="constant",
        )
        if self._rank() == 0:
            n = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
            logger.info(
                "MegatronLumenDSV4Engine: model+distributed-optimizer ready, %d params, "
                "dp_size=%d, MILES-R3=%s",
                n,
                self.get_data_parallel_size(),
                self._r3_enabled,
            )

    # ---- weight sync: Megatron -> HF named tensors ----
    def get_per_tensor_param(self, **kwargs):
        """Yield HF tensors while bounding TP/EP/PP gather memory to one parameter."""
        assert self.module is not None
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group() if tp_size > 1 else None
        ep_size = getattr(self, "_ep_size", 1)
        ep_group = mpu.get_expert_model_parallel_group() if ep_size > 1 else None
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        etp_group = mpu.get_expert_tensor_parallel_group() if etp_size > 1 else None
        pp_size = getattr(self, "_pp_size", 1)
        pp_rank = getattr(self, "_pp_rank", 0)
        named_parameters = _named_export_tensors(self.module)

        def make_mapping(metadata):
            return _StreamingGatheredParamMapping(
                named_parameters,
                tp_size=tp_size,
                tp_group=tp_group,
                ep_size=ep_size,
                ep_group=ep_group,
                etp_size=etp_size,
                etp_group=etp_group,
                num_experts=self._dims.num_experts,
                metadata=metadata,
            )

        actual_mapping = make_mapping(metadata=False)
        convert_kwargs = {
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "layers_per_pp_rank": getattr(self, "_layers_per_pp_rank", None),
            "use_grouped_mlp": True,
        }

        def convert_for_rollout(mapping):
            for key, tensor in dsv4.megatron_to_hf(
                mapping,
                self._dims,
                **convert_kwargs,
            ):
                yield dsv4.denormalize_redhat_key(key), tensor

        # PP=1 still returns a lazy bridge generator: TP/EP collectives occur
        # only when the caller requests the next HF tensor.
        if pp_size <= 1:
            return convert_for_rollout(actual_mapping), None

        # Build only shape/dtype metadata on the meta device. This establishes
        # a common deterministic PP order without materializing gathered data.
        metadata_mapping = make_mapping(metadata=True)
        local_meta_tensors = convert_for_rollout(metadata_mapping)
        my_meta = {
            key: (tensor.shape, tensor.dtype) for key, tensor in local_meta_tensors
        }

        # Each PP group connects ranks with matching TP/EP coordinates.
        # Share stage metadata once, then broadcast one real tensor at a time.
        pp_group = mpu.get_pipeline_model_parallel_group()
        pp_global_ranks = dist.get_process_group_ranks(pp_group)
        all_meta: list = [None] * pp_size
        dist.all_gather_object(all_meta, (pp_rank, my_meta), group=pp_group)
        all_meta.sort(key=lambda item: item[0])

        def _streaming_pp_gen():
            for src_pp, meta in all_meta:
                src_global = pp_global_ranks[src_pp]
                source_iter = None
                if src_pp == pp_rank:
                    source_iter = iter(convert_for_rollout(actual_mapping))
                for tensor_index, (expected_key, (shape, dtype)) in enumerate(
                    meta.items()
                ):
                    if source_iter is not None:
                        actual_key, tensor = next(source_iter)
                        if actual_key != expected_key:
                            raise RuntimeError(
                                "DSV4 streaming metadata order mismatch: "
                                f"expected {expected_key}, got {actual_key}"
                            )
                        expected_shape = tuple(shape)
                        actual_shape = tuple(tensor.shape)
                        if actual_shape != expected_shape or tensor.dtype != dtype:
                            raise RuntimeError(
                                "DSV4 streaming metadata mismatch for "
                                f"{expected_key} at src_pp={src_pp} "
                                f"tensor_index={tensor_index}: "
                                f"expected shape={expected_shape} dtype={dtype}, "
                                f"actual shape={actual_shape} dtype={tensor.dtype}"
                            )
                    else:
                        tensor = torch.empty(shape, dtype=dtype, device="cuda")
                    work = dist.broadcast(
                        tensor,
                        src=src_global,
                        group=pp_group,
                        async_op=True,
                    )
                    work.wait()
                    torch.cuda.synchronize(tensor.device)
                    yield expected_key, tensor
                    del tensor
                if source_iter is not None:
                    try:
                        extra_key, _ = next(source_iter)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError(
                            f"DSV4 streaming metadata omitted tensor {extra_key}"
                        )

        return _lockstep_stream(_streaming_pp_gen(), dist.barrier), None


@EngineRegistry.register(model_type="language_model", backend="megatron_lumen_dsv4")
class MegatronLumenDSV4EngineWithLMHead(MegatronLumenDSV4Engine):
    pass


@EngineRegistry.register(model_type="value_model", backend="megatron_lumen_dsv4")
class MegatronLumenDSV4EngineWithValueHead(MegatronLumenDSV4Engine):
    pass
