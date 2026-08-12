import logging

logger = logging.getLogger("atom.plugin.sglang.register")


def _is_atom_external_model_enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get() == "atom.plugin.sglang.models"
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return False


def _hf_quant_method(model_config) -> str:
    try:
        quant_cfg = model_config._parse_quant_hf_config()
    except Exception:  # noqa: BLE001 - tolerate absent or incompatible HF config
        quant_cfg = None
    if not quant_cfg:
        return ""
    return str(quant_cfg.get("quant_method", "")).lower()


def _install_model_config_quant_patch() -> None:
    from sglang.srt.configs.model_config import ModelConfig

    if getattr(ModelConfig, "_atom_sglang_quant_patch", False):
        return

    original_verify_quantization = ModelConfig._verify_quantization

    def verify_quantization_with_atom_external_bypass(self):
        try:
            return original_verify_quantization(self)
        except ValueError as exc:
            if (
                _is_atom_external_model_enabled()
                and _hf_quant_method(self) == "mxfp8"
                and "quantization is currently not supported in ROCm" in str(exc)
            ):
                logger.info(
                    "Skipping SGLang server-args quantization gate for ATOM "
                    "external MXFP8 model; ATOM owns quantized weight loading."
                )
                self.quantization = None
                return None
            raise

    ModelConfig._verify_quantization = verify_quantization_with_atom_external_bypass
    ModelConfig._atom_sglang_quant_patch = True


def _install_loader_quant_patch() -> None:
    from sglang.srt.model_loader import loader

    if getattr(loader, "_atom_sglang_quant_patch", False):
        return

    original_get_quantization_config = loader._get_quantization_config

    def get_quantization_config_with_atom_external_bypass(model_config, load_config):
        model_class, _ = loader.get_model_architecture(model_config)
        if getattr(model_class, "sglang_skip_quant_config", False):
            logger.info(
                "Skipping SGLang native quant_config for external model %s; "
                "the model wrapper owns quantized weight loading.",
                model_class.__name__,
            )
            return None
        return original_get_quantization_config(model_config, load_config)

    loader._get_quantization_config = get_quantization_config_with_atom_external_bypass
    loader._atom_sglang_quant_patch = True


def _register_k3_dspark_config() -> None:
    """Register the standalone Kimi-K3 DSpark draft config with transformers.

    The draft checkpoint (arch ``K3DSparkModel``, ``model_type: k3_dspark``)
    ships no remote config code, so transformers ``AutoConfig`` cannot load it
    and sglang's ``get_config`` fails during ``prepare_server_args``. ATOM treats
    ``k3_dspark`` as a plain config; mirror that with a generic config class
    whose ``__init__`` stores the checkpoint's top-level DSpark fields.
    """
    try:
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:  # noqa: BLE001 - transformers always present at runtime
        return

    if "k3_dspark" in CONFIG_MAPPING:
        return

    class K3DSparkConfig(PretrainedConfig):
        model_type = "k3_dspark"

    try:
        AutoConfig.register("k3_dspark", K3DSparkConfig)
        logger.info("Registered Kimi-K3 DSpark draft config (model_type=k3_dspark)")
    except (ValueError, KeyError):  # already registered by another import/rank
        pass


def register_plugin() -> None:
    """Install ATOM patches that must run before SGLang parses server args."""
    # Map sglang 0.5.15 import paths the plugin still uses onto their relocated
    # homes when running on sglang-main. Must precede any attention-backend
    # import below. No-op on a genuine 0.5.15 runtime.
    try:
        from atom.plugin.sglang.runtime.sglang_main_compat import (
            install as _install_sglang_main_compat,
        )

        _install_sglang_main_compat()
    except Exception:
        logger.exception("Failed to install sglang-main compat aliases")

    # Must run before prepare_server_args()'s speculative-decoding hook loads the
    # Kimi-K3 DSpark draft config (model_type=k3_dspark) via transformers.
    _register_k3_dspark_config()

    _install_model_config_quant_patch()
    _install_loader_quant_patch()
    from atom.plugin.sglang.models.kimi_k3_processor import (
        register_kimi_k3_text_only_processor,
    )

    register_kimi_k3_text_only_processor()

    try:
        from atom.plugin.sglang.runtime import apply_load_config_patch

        apply_load_config_patch()
    except Exception:
        logger.exception("Failed to install ATOM SGLang load-config patch")
