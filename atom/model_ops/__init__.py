# This frontend class is used to construct the attention op in model files.
# It dispatches to the mode-specific attention implementation at construction
# time instead of mutating this module-level symbol during plugin init.
#
# Resolved lazily (PEP 562): importing it eagerly would pull AITER into every
# `atom.model_ops.*` submodule import, including the pure-Python ones the
# loader unit tests exercise on a runner with no AITER build.

__all__ = [
    "Attention",
]


def __getattr__(name: str):
    if name == "Attention":
        from .base_attention import Attention

        return Attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
