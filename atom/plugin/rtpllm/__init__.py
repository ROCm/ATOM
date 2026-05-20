try:
    from .models import base_model_wrapper as _base_model_wrapper
except ModuleNotFoundError as exc:
    if exc.name != "rtp_llm":
        raise
    _base_model_wrapper = None

__all__ = ["_base_model_wrapper"]
