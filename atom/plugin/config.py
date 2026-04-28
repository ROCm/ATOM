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
    is_rtpllm: bool = False

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
    sglang_dist_init_addr: Optional[str] = None
    sglang_port_args: Any = None

    # rtp-llm specific
    rtpllm_model_config: Any = None
    rtpllm_parallelism_config: Any = None


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
        is_rtpllm=False,
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

    # Derive DP rank from SGLang's TP-local rank rather than the global
    # distributed rank so PP/multi-stage layouts do not skew the result.
    data_parallel_rank = 0
    if server_args.dp_size > 1:
        tp_rank = get_tensor_model_parallel_rank()
        tp_group_size = max(1, server_args.tp_size // server_args.dp_size)
        data_parallel_rank = tp_rank // tp_group_size

    # sglang uses the atom parallel config
    sgl_parallel_config = ParallelConfig(
        data_parallel_size=server_args.dp_size,
        data_parallel_rank=data_parallel_rank,
    )

    # use sglang torch compile policy and cuda graph policy
    # because sglang doesn't use the compile decorator for model,
    # we have no method to define self policy
    sgl_compilation_config = CompilationConfig(
        level=0,
        use_cudagraph=False,
        cudagraph_mode=None,
    )

    plugin_config = PluginConfig(
        # common config
        model_config=sgl_model_config,
        rank=rank,
        is_plugin_mode=True,
        is_vllm=False,
        is_sglang=True,
        is_rtpllm=False,
        # sglang specific
        sglang_model_opt_config=sgl_model_opt_config,
        sglang_load_config=sgl_load_config,
        sglang_enable_torch_compile=server_args.enable_torch_compile,
        sglang_disable_cuda_graph=server_args.disable_cuda_graph,
        sglang_enable_dp_attention=server_args.enable_dp_attention,
        sglang_dist_init_addr=server_args.dist_init_addr,
        sglang_port_args=PortArgs.init_new(server_args),
    )

    # force max num batched tokens to 16K because sgl doesn't have
    # concept for max num batched tokens
    return Config(
        model=server_args.model_path,
        max_num_batched_tokens=16384,
        max_num_seqs=server_args.max_running_requests,
        max_model_len=server_args.context_length,
        gpu_memory_utilization=server_args.mem_fraction_static,
        tensor_parallel_size=server_args.tp_size,
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


def _generate_atom_config_from_rtpllm_config(config: Any):
    from atom.config import Config, ParallelConfig, CompilationConfig

    rtpllm_model_config = getattr(config, "model_config", None)
    rtpllm_parallelism_config = getattr(config, "parallelism_config", None)
    if rtpllm_model_config is None:
        raise ValueError(
            "rtpllm plugin expects config.model_config to be available "
            "(BaseModel instance is recommended)."
        )

    tp_size = getattr(rtpllm_parallelism_config, "tp_size", 1)
    tp_rank = getattr(rtpllm_parallelism_config, "tp_rank", 0)
    max_generate_batch_size = getattr(config, "max_generate_batch_size", 512)
    max_model_len = getattr(rtpllm_model_config, "max_seq_len", None) or 8192

    # rtp-llm plugin path follows ATOM plugin-mode execution, so ATOM should not
    # perform its own torch compile/cudagraph policy.
    rtpllm_compilation_config = CompilationConfig(
        level=0,
        use_cudagraph=False,
        cudagraph_mode=None,
    )

    plugin_config = PluginConfig(
        # common config
        model_config=rtpllm_model_config,
        rank=tp_rank,
        is_plugin_mode=True,
        is_vllm=False,
        is_sglang=False,
        is_rtpllm=True,
        # rtp-llm specific
        rtpllm_model_config=rtpllm_model_config,
        rtpllm_parallelism_config=rtpllm_parallelism_config,
    )

    kv_cache_dtype = "bf16"
    if hasattr(rtpllm_model_config, "attn_config") and hasattr(
        rtpllm_model_config.attn_config, "kv_cache_dtype"
    ):
        raw_kv_dtype = str(rtpllm_model_config.attn_config.kv_cache_dtype).lower()
        if "fp8" in raw_kv_dtype:
            kv_cache_dtype = "fp8"
        elif "int8" in raw_kv_dtype:
            kv_cache_dtype = "int8"

    # Keep RTP behavior aligned with SGLang plugin semantics:
    # only enable EP when ep_size > 1; pure TP (ep_size == 1) must not use EP.
    rtpllm_ep_size = getattr(rtpllm_parallelism_config, "ep_size", 1)

    return Config(
        model=rtpllm_model_config.ckpt_path,
        max_num_batched_tokens=max(16384, max_generate_batch_size),
        max_num_seqs=max_generate_batch_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        parallel_config=ParallelConfig(data_parallel_size=1, data_parallel_rank=0),
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=False,
        port=None,
        torch_profiler_dir=None,
        compilation_config=rtpllm_compilation_config,
        asyncio_mode=False,
        load_dummy=False,
        enable_expert_parallel=bool(rtpllm_ep_size > 1),
        master_addr=None,
        enable_dp_attention=False,
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
    from atom.plugin import is_vllm, is_sglang, is_rtpllm
    from atom.config import set_current_atom_config

    if is_vllm():
        atom_config = _generate_atom_config_from_vllm_config(config)
    elif is_sglang():
        atom_config = _generate_atom_config_from_sglang_config(config)
    elif is_rtpllm():
        atom_config = _generate_atom_config_from_rtpllm_config(config)
    else:
        raise ValueError(
            "Make sure ATOM is running in plugin mode; "
            "generate_atom_config_for_plugin_mode should be called in plugin mode."
        )

    # set the current atom config for the custom model
    set_current_atom_config(atom_config)

    return atom_config
