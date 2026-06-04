# all of the supported frameworks, including server mode and plugin mode
_SUPPORTED_FRAMEWORKS = ["vllm", "sglang", "sgl", "atom"]

# supported frameworks for plugin mode
_SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE = ["vllm", "sglang", "sgl"]

# default is atom for server mode
_CURRENT_FRAMEWORK = "atom"


def is_sglang() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["sglang", "sgl"])


def is_vllm() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in ["vllm"])


def is_plugin_mode() -> bool:
    global _CURRENT_FRAMEWORK
    return bool(_CURRENT_FRAMEWORK.lower() in _SUPPORTED_FRAMEWORKS_FOR_PLUGIN_MODE)


def _set_framework_backbone(framework: str) -> None:
    if framework.lower() not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework {framework} for ATOM to plug in")
    global _CURRENT_FRAMEWORK
    _CURRENT_FRAMEWORK = framework
