import logging

from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.config import Config
from atom.plugin.prepare import is_vllm, is_sglang

logger = logging.getLogger("atom")

_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


def _register_custom_attention_to_vllm() -> None:
    from vllm.v1.attention.backends.registry import (
        register_backend,
        AttentionBackendEnum,
    )

    logger.info("Register custom attention backend AiterBackend to vLLM")
    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        is_mamba=False,
        class_path="atom.model_ops.attentions.aiter_attention.AiterBackend",
    )


def _register_custom_attention_to_sglang() -> None:

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    # here register the custom attention backend with the name "aiter"
    # as sglang defines the fixed attention backend choices, which must be
    # in-tree
    logger.info("Register custom attention backend AiterBackend to SGLang")

    @register_attention_backend("aiter")
    def create_atom_backend(runner):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        return AiterAttnBackend(runner)


def register_ops_to_vllm(atom_config: Config) -> None:
    """
    Register custom ops to vllm, including attention
    """
    if atom_config.plugin_config.vllm_use_custom_attention:
        _register_custom_attention_to_vllm()
    else:
        logger.warning(
            "Please export VLLM_ATTENTION_BACKEND=CUSTOM to use atom attention"
        )


def register_ops_to_sglang(atom_config: Config) -> None:
    """
    Register custom ops to sglang, including attention
    """
    _register_custom_attention_to_sglang()


def set_attn_cls() -> None:
    """
    Set the attention class for constructing the model based on the framework
    """
    import atom.model_ops as ops

    if is_vllm():
        ops.ATTN_CLS = ops.PagedAttention
        logger.info("Set ATTN_CLS to PagedAttention for vLLM")
    elif is_sglang():
        ops.ATTN_CLS = ops.RadixAttention
        logger.info("Set ATTN_CLS to RadixAttention for SGLang")


def init_aiter_dist(config: Config) -> None:
    """
    Initialize aiter dist for using aiter custom collective op
    """
    logger.info(
        "Initialize aiter dist for using aiter custom collective op for plugin mode"
    )

    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    rank = config.plugin_config.rank
    tensor_parallel_size = config.tensor_parallel_size

    assert (
        config.plugin_config.is_plugin_mode
    ), "Make sure ATOM is running in plugin mode"

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.sglang_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = f"127.0.0.1"
            dp_master_port = config.plugin_config.sglang_port_args.nccl_port

    distributed_init_method = get_distributed_init_method(dp_master_ip, dp_master_port)

    logger.info(
        f"Initialize aiter dist for using aiter custom collective op for plugin mode, rank:{rank}"
    )
    init_dist_env(
        tensor_model_parallel_size=tensor_parallel_size,
        rankID=rank,
        backend="nccl",
        distributed_init_method=distributed_init_method,
        data_parallel_size=config.parallel_config.data_parallel_size,
        data_parallel_rank=config.parallel_config.data_parallel_rank,
    )
