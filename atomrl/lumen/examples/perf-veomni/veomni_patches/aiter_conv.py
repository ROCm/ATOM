"""Compatibility imports for Lumen's in-package convolution implementation."""

from atomrl.lumen.ops.conv import conv, conv_calls


def load():
    """Return Lumen's public convolution entry point."""
    return conv


__all__ = ["conv", "conv_calls", "load"]
