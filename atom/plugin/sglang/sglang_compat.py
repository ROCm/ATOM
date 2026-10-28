"""SGLang 0.5.22 moved a few helpers the plugin calls.

0.5.20 exported ``get_pp_group`` from ``sglang.srt.distributed`` and
``get_attn_tp_context`` from ``sglang.srt.layers.communicator``. 0.5.22 keeps
the same objects, but the pipeline group is read from ``get_parallel()`` and
the attention-TP context lives under ``layer_boundary``.
"""

from __future__ import annotations


def _distributed_symbol(name):
    """Import a SGLang parallel helper.

    0.5.22 still defines these on ``parallel_state`` but drops them from
    ``sglang.srt.distributed``'s star-export, so ``from sglang.srt.distributed
    import <name>`` raises ``ImportError``.
    """

    import importlib

    try:
        module = importlib.import_module("sglang.srt.distributed")
        return getattr(module, name)
    except (ImportError, AttributeError):
        module = importlib.import_module("sglang.srt.distributed.parallel_state")
        return getattr(module, name)


def get_pp_group():
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        return _distributed_symbol("get_pp_group")()
    return get_parallel().pp_group


def get_tp_group():
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        return _distributed_symbol("get_tp_group")()
    return get_parallel().tp_group


def get_tensor_model_parallel_rank():
    # Prefer the old getter. 0.5.22 keeps it on parallel_state and only
    # removes it from the distributed package export.
    return _distributed_symbol("get_tensor_model_parallel_rank")()


def get_attn_tp_context():
    try:
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            get_attn_tp_context as current,
        )
    except ImportError:
        from sglang.srt.layers.communicator import (
            get_attn_tp_context as current,
        )

    return current()
