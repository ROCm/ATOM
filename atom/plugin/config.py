from typing import Any, Optional
from dataclasses import dataclass

import torch
import logging

from atom.utils import envs

logger = logging.getLogger("atom")


@dataclass
class PluginConfig:
    # common config for both framework
    model_config: Any = None
    rank: int = 0
    is_plugin_mode: bool = False
    is_vllm: bool = False
    is_sglang: bool = False

    # vllm specific
    vllm_config: Any = None
    vllm_scheduler_config: Any = None
    vllm_cache_config: Any = None
    vllm_quant_config: Any = None
    vllm_use_atom_attention: bool = False

    # sglang specific
    sglang_model_opt_config: Any = None
    sglang_load_config: Any = None
    sglang_enable_torch_compile: bool = False
    sglang_disable_cuda_graph: bool = False
    sglang_enable_dp_attention: bool = False
    sglang_aiter_rank_id: int = 0
    sglang_dist_init_addr: Optional[str] = None
    sglang_port_args: Any = None


def _normalize_sglang_parallel_config(
    tp_size: int,
    dp_size: int,
    tp_rank: int,
    enable_dp_attention: bool,
) -> tuple[int, int, int, int]:
    """Translate SGLang parallel args into the runtime layout ATOM expects.

    SGLang's ``tp_size`` is the whole world used by the model runner, while
    ``dp_size`` under dp-attention is only an attention-layout factor inside
    that world. For pure-DP, SGLang launches multiple independent TP workers,
    so ATOM should treat that DP dimension as external scheduling rather than
    a model-internal communication group.
    """

    if enable_dp_attention:
        if dp_size < 1:
            raise ValueError(f"SGLang dp_size must be >= 1, got {dp_size}")
        if tp_size % dp_size != 0:
            raise ValueError(
                "SGLang tp_size must be divisible by dp_size when "
                f"enable_dp_attention=True, got tp_size={tp_size}, dp_size={dp_size}"
            )

        runtime_tp_size = 1
        runtime_dp_size = tp_size
        runtime_dp_rank = tp_rank
        aiter_rank_id = 0
        return runtime_tp_size, runtime_dp_size, runtime_dp_rank, aiter_rank_id

    # Without dp-attention, SGLang's DP workers are external replicas. Keep
    # ATOM/aiter on the per-worker TP world and do not create an internal DP
    # communication group.
    return tp_size, 1, 0, tp_rank


def _generate_atom_config_from_vllm_config(config: Any) -> PluginConfig:
    from atom.config import Config, CompilationConfig

    vllm_model_config = config.model_config
    vllm_scheduler_config = config.scheduler_config
    vllm_cache_config = config.cache_config
    vllm_parallel_config = config.parallel_config
    vllm_use_atom_attention = not envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

    # here use the ATOM compilation config, as the ATOM compile policy is used
    # instead of vLLM one for torch compile, while for cuda graph capture,
    # still use the vLLM because it has FULL_AND_PIECEWISE feature
    # when you don't want to use atom torch compile, you can also use
    # --enforce-eager to disable the atom torch compile when launch vllm server
    compilation_config = config.compilation_config
    vllm_compilation_config = CompilationConfig(
        # use mode because vllm level argument is deprecated
        level=compilation_config.mode,
        use_cudagraph=False,
        cudagraph_mode=None,
    )

    vllm_quant_config = config.quant_config

    plugin_config = PluginConfig(
        # common config
        model_config=vllm_model_config,
        rank=vllm_parallel_config.rank,
        is_plugin_mode=True,
        is_vllm=True,
        is_sglang=False,
        # vllm specific
        vllm_config=config,
        vllm_scheduler_config=vllm_scheduler_config,
        vllm_cache_config=vllm_cache_config,
        vllm_quant_config=vllm_quant_config,
        vllm_use_atom_attention=vllm_use_atom_attention,
    )

    # specific
    max_model_len = vllm_model_config.max_model_len
    if hasattr(vllm_scheduler_config, "max_model_len"):
        max_model_len = vllm_scheduler_config.max_model_len

    max_num_batched_tokens = vllm_scheduler_config.max_num_batched_tokens

    return Config(
        model=vllm_model_config.model,
        trust_remote_code=getattr(vllm_model_config, "trust_remote_code", False),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=vllm_scheduler_config.max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=vllm_cache_config.gpu_memory_utilization,
        tensor_parallel_size=vllm_parallel_config.tensor_parallel_size,
        enforce_eager=True,  # disable using atom cuda graph
        parallel_config=vllm_parallel_config,
        kv_cache_block_size=vllm_cache_config.block_size,
        num_kvcache_blocks=vllm_cache_config.num_gpu_blocks,
        kv_cache_dtype=vllm_cache_config.cache_dtype,
        enable_prefix_caching=vllm_cache_config.enable_prefix_caching,
        port=None,
        torch_profiler_dir=None,
        compilation_config=vllm_compilation_config,
        asyncio_mode=False,
        load_dummy=False,
        enable_expert_parallel=vllm_parallel_config.enable_expert_parallel,
        master_addr=None,
        enable_dp_attention=False,
        plugin_config=plugin_config,
    )


def _generate_atom_config_from_sglang_config(config: Any):
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.server_args import (
        get_global_server_args,
        PortArgs,
        ZMQ_TCP_PORT_DELTA,
    )
    from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig
    from sglang.srt.configs.modelopt_config import ModelOptConfig
    from sglang.srt.configs.load_config import LoadConfig
    from atom.config import Config, ParallelConfig, CompilationConfig

    # sglang's ModelRunner already parsed and stored ServerArgs globally
    # before OOT model loading, so we can retrieve it directly.
    try:
        server_args = get_global_server_args()
    except Exception as exc:
        raise RuntimeError(
            "Failed to retrieve SGLang global ServerArgs. Ensure this "
            "function is called after SGLang has initialized its server "
            "arguments."
        ) from exc

    if server_args is None:
        raise RuntimeError(
            "SGLang global ServerArgs are not initialized. Ensure this "
            "function is called after SGLang has parsed and set its "
            "server arguments."
        )

    sgl_model_config = SglangModelConfig.from_server_args(server_args)
    sgl_model_opt_config = ModelOptConfig(
        quant=server_args.modelopt_quant,
        checkpoint_restore_path=server_args.modelopt_checkpoint_restore_path,
        checkpoint_save_path=server_args.modelopt_checkpoint_save_path,
        export_path=server_args.modelopt_export_path,
    )

    sgl_load_config = LoadConfig(
        load_format=server_args.load_format,
        download_dir=server_args.download_dir,
        model_loader_extra_config=server_args.model_loader_extra_config,
        remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
        remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
        remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        remote_instance_weight_loader_backend=server_args.remote_instance_weight_loader_backend,
        modelopt_config=sgl_model_opt_config,
        rl_quant_profile=server_args.rl_quant_profile,
    )

    # sglang doesn't passed the rank number in config, so ATOM plugin
    # get rank number through the torch.distributed.get_rank()
    rank = torch.distributed.get_rank()

    tp_rank = get_tensor_model_parallel_rank()
    (
        atom_tensor_parallel_size,
        atom_data_parallel_size,
        atom_data_parallel_rank,
        sglang_aiter_rank_id,
    ) = _normalize_sglang_parallel_config(
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        tp_rank=tp_rank,
        enable_dp_attention=server_args.enable_dp_attention,
    )

    # sglang uses the atom parallel config
    sgl_parallel_config = ParallelConfig(
        data_parallel_size=atom_data_parallel_size,
        data_parallel_size_local=atom_data_parallel_size,
        data_parallel_rank=atom_data_parallel_rank,
        data_parallel_rank_local=atom_data_parallel_rank,
    )

    # use sglang torch compile policy and cuda graph policy
    # because sglang doesn't use the compile decorator for model,
    # we have no method to define self policy
    sgl_compilation_config = CompilationConfig(
        level=0,
        use_cudagraph=False,
        cudagraph_mode=None,
    )

    sglang_dist_init_addr = server_args.dist_init_addr
    # In single-node DP attention, synthesize the same TCP base address that
    # SGLang uses for its DP-attention TCP port family. The primary purpose is
    # to avoid calling PortArgs.init_new() again in ATOM plugin mode, because a
    # second call would probe that fixed TCP range again and conflict with
    # SGLang's existing allocation. In the current plugin path, this value
    # should be treated as a compatibility/fallback hint rather than a
    # guaranteed representation of the runtime default torch.distributed world
    # rendezvous endpoint.
    if (
        sglang_dist_init_addr is None
        and server_args.enable_dp_attention
        and server_args.nnodes == 1
    ):
        sglang_dist_init_addr = f"127.0.0.1:{server_args.port + ZMQ_TCP_PORT_DELTA}"

    sglang_port_args = None
    if sglang_dist_init_addr is None:
        sglang_port_args = PortArgs.init_new(server_args)

    plugin_config = PluginConfig(
        # common config
        model_config=sgl_model_config,
        rank=rank,
        is_plugin_mode=True,
        is_vllm=False,
        is_sglang=True,
        # sglang specific
        sglang_model_opt_config=sgl_model_opt_config,
        sglang_load_config=sgl_load_config,
        sglang_enable_torch_compile=server_args.enable_torch_compile,
        sglang_disable_cuda_graph=server_args.disable_cuda_graph,
        sglang_enable_dp_attention=server_args.enable_dp_attention,
        sglang_aiter_rank_id=sglang_aiter_rank_id,
        sglang_dist_init_addr=sglang_dist_init_addr,
        sglang_port_args=sglang_port_args,
    )

    # force max num batched tokens to 16K because sgl doesn't have
    # concept for max num batched tokens
    return Config(
        model=server_args.model_path,
        max_num_batched_tokens=16384,
        max_num_seqs=server_args.max_running_requests,
        max_model_len=server_args.context_length,
        gpu_memory_utilization=server_args.mem_fraction_static,
        tensor_parallel_size=atom_tensor_parallel_size,
        # Disable ATOM's own torch.compile and CUDA graph capture —
        # sglang manages its own compilation/graph strategy, and the
        # @support_torch_compile decorator checks enforce_eager to skip,
        # preventing double-compile.
        enforce_eager=True,
        parallel_config=sgl_parallel_config,
        kv_cache_dtype=server_args.kv_cache_dtype,
        enable_prefix_caching=False,
        port=None,
        torch_profiler_dir=None,
        compilation_config=sgl_compilation_config,
        asyncio_mode=False,
        load_dummy=False,
        enable_expert_parallel=bool(server_args.ep_size > 1),
        master_addr=None,
        enable_dp_attention=server_args.enable_dp_attention,
        plugin_config=plugin_config,
    )


def generate_atom_config_for_plugin_mode(config: Any = None):
    """
    Generate the atom config in plugin mode, be called when create the custom model
    config:
        - for vllm: config is VllmConfig and contains all config value from vllm
        - for sglang: config is only model specific config passed from sglang, so the
                      server args is used
    """

    logger.info("Generate atom config for plugin mode from passed config")
    atom_config = None
    from atom.plugin import is_vllm, is_sglang
    from atom.config import set_current_atom_config

    if is_vllm():
        atom_config = _generate_atom_config_from_vllm_config(config)
    elif is_sglang():
        atom_config = _generate_atom_config_from_sglang_config(config)
    else:
        raise ValueError(
            "Make sure ATOM is running in plugin mode; "
            "generate_atom_config_for_plugin_mode should be called in plugin mode."
        )

    # set the current atom config for the custom model
    set_current_atom_config(atom_config)

    return atom_config
