import logging

from atom.models.qwen3 import Qwen3ForCausalLM
from atom.models.qwen3_moe import Qwen3MoeForCausalLM
from atom.models.glm4_moe import Glm4MoeForCausalLM
from atom.models.deepseek_v2 import DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM
from atom.models.minimax_m2 import MiniMaxM2ForCausalLM
from atom.models.minimax_m3 import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from atom.config import Config
from atom.plugin.prepare import is_vllm, is_sglang

logger = logging.getLogger("atom")

_ATOM_SUPPORTED_MODELS = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "Glm4MoeForCausalLM": Glm4MoeForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "GlmMoeDsaForCausalLM": GlmMoeDsaForCausalLM,
    "MiniMaxM2ForCausalLM": MiniMaxM2ForCausalLM,
    "MiniMaxM3SparseForCausalLM": MiniMaxM3SparseForCausalLM,
    "MiniMaxM3SparseForConditionalGeneration": MiniMaxM3SparseForConditionalGeneration,
}

if is_sglang():
    from atom.models.qwen3_next import Qwen3NextForCausalLM
    from atom.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )
    from atom.models.kimi_k25 import KimiK25ForCausalLM

    _ATOM_SUPPORTED_MODELS.update(
        {
            "Qwen3NextForCausalLM": Qwen3NextForCausalLM,
            "Qwen3_5ForConditionalGeneration": Qwen3_5ForCausalLM,
            "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForCausalLM,
            # ROCm/ATOM#1078: route Kimi-K2.x through ATOM's quant-aware model
            # path (KimiK25ForCausalLM -> DeepseekV2ForCausalLM). The standalone
            # engine already registers this in atom/model_engine/model_runner.py;
            # the SGLang plugin path was missing it, so launches fell back to
            # sglang's native model and failed weight loading on the excluded
            # (BF16) attention projections.
            "KimiK25ForConditionalGeneration": KimiK25ForCausalLM,
        }
    )


def _register_custom_attention_to_sglang() -> None:
    """Override sglang's built-in "aiter" attention backend with ATOM's implementation.

    sglang only accepts pre-registered backend names, so we reuse the "aiter"
    name to inject ATOMAttnBackendForSgl without modifying sglang source.
    """
    import sglang.srt.layers.attention.aiter_backend as sglang_aiter_backend

    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )
    from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
        ATOMAttnBackendForSgl,
    )

    # here register the custom attention backend with the name "aiter"
    # as sglang defines the fixed attention backend choices, which must be
    # in-tree
    logger.info("Register custom attention backend ATOMAttnBackendForSgl to SGLang")

    # Speculative draft paths instantiate AiterAttnBackend directly inside
    # AiterMultiStepDraftBackend, bypassing the attention registry. Rebind the
    # module symbol as well so both registry lookup and direct construction use
    # the plugin backend.
    sglang_aiter_backend.AiterAttnBackend = ATOMAttnBackendForSgl

    @register_attention_backend("aiter")
    def create_atom_backend(runner):
        return ATOMAttnBackendForSgl(runner)


def register_ops_to_sglang(atom_config: Config) -> None:
    """
    Register custom ops to sglang, including attention
    """
    _register_custom_attention_to_sglang()


def set_attn_cls() -> None:
    """Keep compatibility with old plugin init hooks.

    FIXME: This is a legacy no-op after attention construction moved to the
    frontend dispatcher. Remove it once downstream plugin init paths stop
    calling ``set_attn_cls`` for side effects.

    Attention selection now happens in ``atom.model_ops.base_attention.Attention``
    at construction time, so plugin init no longer mutates ``atom.model_ops``.
    """
    if is_vllm():
        logger.info("Use Attention dispatcher for vLLM")
    elif is_sglang():
        logger.info("Use Attention dispatcher for SGLang")


def init_aiter_dist(config: Config) -> None:
    """
    Initialize aiter dist for using aiter custom collective op.

    In vLLM plugin mode, tries to reuse vLLM's TP group and inject aiter's ca_comm
    first (single IPC init, avoids 2x reduce slowdown). Falls back to init_dist_env
    if reuse fails.
    """
    logger.info(
        "Initialize aiter dist for using aiter custom collective op for plugin mode"
    )

    rank = config.plugin_config.rank
    if getattr(config.plugin_config, "is_sglang", False):
        rank = getattr(config.plugin_config, "sglang_aiter_rank_id", rank)
    tensor_parallel_size = config.tensor_parallel_size

    assert (
        config.plugin_config.is_plugin_mode
    ), "Make sure ATOM is running in plugin mode"

    if config.plugin_config.is_vllm:
        from atom.plugin.vllm.tp_group_reuse import init_aiter_tp_from_vllm

        if init_aiter_tp_from_vllm(tensor_parallel_size):
            return

    # Fallback: create aiter's own groups (vLLM reuse failed or non-vLLM plugin)
    from aiter import init_dist_env
    from aiter.dist.utils import get_distributed_init_method

    if config.plugin_config.is_vllm:
        dp_master_ip = config.parallel_config.data_parallel_master_ip
        dp_master_port = config.parallel_config.data_parallel_master_port
    elif config.plugin_config.is_sglang:
        if config.plugin_config.sglang_dist_init_addr is not None:
            dp_master_ip, dp_master_port = (
                config.plugin_config.sglang_dist_init_addr.split(":")
            )
        else:
            dp_master_ip = "127.0.0.1"
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
