"""Fastokens replacement for ATOM's final text encode only; no HTTP sidecar.

Keep the existing HuggingFace tokenizer object for chat rendering, tools,
reasoning controls and decoding. Install once before request executors start.
Never silently fall back or tokenize twice. The native tokenizer is loaded from
HF's effective backend serialization to preserve configured post-processor and
added-token state, including any BOS updates made at tokenizer initialization.
"""

import os
from importlib.metadata import version

PINNED_FASTOKENS_VERSION = "0.3.6"


def install_fastokens_final_encode(tokenizer, *, bpe_threads=1):
    if type(bpe_threads) is not int or bpe_threads < 1:
        raise ValueError("bpe_threads must be a positive integer")
    if getattr(tokenizer, "_atom_fastokens_installed", False):
        raise RuntimeError("fastokens final encode is already installed")
    os.environ["FASTOKENS_BPE_THREADS"] = str(bpe_threads)
    os.environ["FASTOKENS_INPUT_CACHE"] = "0"
    os.environ["FASTOKENS_SEGMENT_CACHE"] = "0"
    if version("fastokens") != PINNED_FASTOKENS_VERSION:
        raise RuntimeError("This adapter requires fastokens==0.3.6")
    from fastokens._native import Tokenizer

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise TypeError("A HuggingFace fast tokenizer backend is required")
    native = Tokenizer.from_json_str(backend.to_str())
    # HF encode(prompt) resets these two modes on each ordinary call. A backend
    # can retain state from an earlier padded/truncated call before installation.
    native.no_padding()
    native.no_truncation()

    def encode(text, text_pair=None, add_special_tokens=True, **kwargs):
        # This serving path calls encode(prompt) and must retain HF's True default.
        # Refuse additional modes rather than silently using a different backend.
        supported_defaults = {
            "padding": False,
            "truncation": False,
            "return_tensors": None,
            "is_split_into_words": False,
        }
        unsupported = {
            key: value
            for key, value in kwargs.items()
            if key not in supported_defaults or value != supported_defaults[key]
        }
        if text_pair is not None or not isinstance(text, str) or unsupported:
            raise ValueError(
                "fastokens final encode supports single rendered text "
                "without pair/truncation/padding/tensors; unsupported="
                + repr(unsupported)
            )
        return native.encode(text, add_special_tokens=add_special_tokens).ids

    tokenizer.encode = encode
    tokenizer._atom_fastokens_installed = True
    tokenizer._atom_fastokens_native = native
    return {
        "version": PINNED_FASTOKENS_VERSION,
        "bpe_threads": bpe_threads,
        "prefix_cache": False,
        "segment_cache": False,
        "default_add_special_tokens": True,
        "native_source": "effective HuggingFace backend JSON",
    }
