from typing import Optional
import logging
import torch

from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs

logger = logging.getLogger("atom")

disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN


_VLLM_OMNI_DIFFUSION_MODEL_REGISTRY_OVERRIDES = {}


def _ensure_atom_config_for_diffusion(od_config) -> None:
    """Set a minimal ATOM config if not already set, so LinearBase.__init__ can read torch_dtype.

    In the vLLM OOT LLM plugin, generate_atom_config_for_plugin_mode(vllm_config) sets this
    inside ATOMModelBase.__init__. For diffusion models, no full VllmConfig exists, so we
    construct a lightweight stand-in from od_config.dtype.

    Only torch_dtype is accessed from the config in the diffusion construction path
    (LinearBase.__init__ line 263, bias tensor allocation). A SimpleNamespace suffices.
    """
    import atom.config as _atom_cfg

    if _atom_cfg._current_atom_config is not None:
        return  # Already set (e.g. vLLM OOT LLM plugin ran first)

    import types

    torch_dtype = getattr(od_config, "dtype", torch.bfloat16)
    _atom_cfg.set_current_atom_config(types.SimpleNamespace(torch_dtype=torch_dtype))
    logger.info(f"ATOM: set minimal diffusion atom config (torch_dtype={torch_dtype})")


def _ensure_aiter_tp_initialized() -> None:
    """Reuse vLLM's TP group for aiter if not already initialized.

    Mirrors init_aiter_dist() in the vLLM OOT plugin (called from ATOMModelBase.__init__).
    Called lazily at model-load time via the wrapped initialize_model, so vLLM's TP
    group is guaranteed to be ready. One central call covers all diffusion models.
    """
    from aiter.dist import parallel_state as aiter_ps

    if aiter_ps._TP is not None:
        return  # Already initialized (e.g. regular vLLM plugin path ran first)

    import vllm.distributed.parallel_state as vllm_ps

    tp_size = vllm_ps.get_tensor_model_parallel_world_size()

    from atom.plugin.vllm.tp_group_reuse import init_aiter_tp_from_vllm

    if init_aiter_tp_from_vllm(tp_size):
        return  # TP>1: reused vLLM's group + aiter ca_comm (optimal path)

    # Fallback for TP=1 or no ca_comm: minimal adapter backed by vLLM's ProcessGroups.
    # LinearBase.forward() never calls all_reduce when tp_size==1 (guarded by tp_size>1).
    from aiter.dist.parallel_state import (
        GroupCoordinator as AiterGroupCoordinator,
        _register_group,
    )

    vllm_tp = vllm_ps.get_tp_group()

    class _AiterTPFromVllm(AiterGroupCoordinator):
        def __init__(self):
            # Skip GroupCoordinator.__init__ to avoid creating new ProcessGroups.
            self.unique_name = "tp:0"
            _register_group(self)
            self.rank = vllm_tp.rank
            self.local_rank = vllm_tp.local_rank
            self.ranks = vllm_tp.ranks
            self.world_size = vllm_tp.world_size
            self.rank_in_group = vllm_tp.rank_in_group
            self.cpu_group = vllm_tp.cpu_group
            self.device_group = vllm_tp.device_group
            self.device = vllm_tp.device
            self.use_device_communicator = False
            self.device_communicator = None
            self.mq_broadcaster = None

    aiter_ps._TP = _AiterTPFromVllm()
    logger.info(
        "ATOM: initialized aiter TP group from vLLM "
        f"(world_size={vllm_tp.world_size}, rank={vllm_tp.rank_in_group})"
    )


def register_omni_platform() -> Optional[str]:

    if disable_vllm_plugin:
        logger.info("Disable ATOM OOT plugin platforms (vllm-omni)")
        return None

    _set_framework_backbone("vllm")

    # return the ATOM omni platform to vllm-omni
    return "atom.plugin.vllm_omni.platform.ATOMOmniPlatform"


def register_omni_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register (vllm-omni)")
        return

    try:
        import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image as _qwen_t2i
        import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit as _qwen_edit
        import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus as _qwen_edit_plus
        import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered as _qwen_layered
        from atom.plugin.vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
            ATOMQwenImageTransformer2DModel,
        )

        for _m in [_qwen_t2i, _qwen_edit, _qwen_edit_plus, _qwen_layered]:
            _m.QwenImageTransformer2DModel = ATOMQwenImageTransformer2DModel
        logger.info(
            "Patched QwenImageTransformer2DModel → ATOMQwenImageTransformer2DModel in qwen_image pipelines"
        )
    except ImportError as e:
        logger.warning(
            f"Could not patch qwen_image pipelines with ATOM transformer: {e}"
        )

    try:
        from atom.plugin.vllm_omni.diffusion.models.flux2.flux2_transformer import (
            ATOMFlux2Transformer2DModel,
        )

        import vllm_omni.diffusion.models.flux2.pipeline_flux2 as pipeline_flux2

        pipeline_flux2.Flux2Transformer2DModel = ATOMFlux2Transformer2DModel
        logger.info(
            "Patched Flux2Transformer2DModel → ATOMFlux2Transformer2DModel in flux2 pipeline"
        )
    except ImportError as e:
        logger.warning(f"Could not patch flux2 pipeline with ATOM transformer: {e}")

    # Wrap initialize_model to call aiter TP init before every diffusion model is loaded.
    # Mirrors ATOMModelBase.__init__ → _prepare_env() in the vLLM OOT plugin:
    # one central point covers all diffusion models, no per-model initialization needed.
    #
    # Must patch diffusers_loader (the call site), not registry (the definition site):
    # diffusers_loader does `from vllm_omni.diffusion.registry import initialize_model`,
    # creating a local binding that is unaffected by patching the registry module.
    import vllm_omni.diffusion.model_loader.diffusers_loader as _diffusers_loader

    _orig_initialize_model = _diffusers_loader.initialize_model

    def _atom_initialize_model(od_config):
        _ensure_aiter_tp_initialized()
        _ensure_atom_config_for_diffusion(od_config)
        return _orig_initialize_model(od_config)

    _diffusers_loader.initialize_model = _atom_initialize_model
    logger.info("Wrapped vllm_omni initialize_model with ATOM aiter TP initialization")
