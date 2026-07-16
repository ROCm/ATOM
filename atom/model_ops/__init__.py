from .base_attention import Attention

# This frontend class is used to construct the attention op in model files.
# It dispatches to the mode-specific attention implementation at construction
# time instead of mutating this module-level symbol during plugin init.

__all__ = [
    "Attention",
]
