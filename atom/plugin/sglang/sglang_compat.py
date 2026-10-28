"""SGLang 0.5.22 moved a few helpers the plugin calls.

0.5.20 exported ``get_pp_group`` from ``sglang.srt.distributed`` and
``get_attn_tp_context`` from ``sglang.srt.layers.communicator``. 0.5.22 keeps
the same objects, but the pipeline group is read from ``get_parallel()`` and
the attention-TP context lives under ``layer_boundary``.
"""

from __future__ import annotations


def get_pp_group():
    try:
        from sglang.srt.runtime_context import get_parallel
    except ImportError:
        from sglang.srt.distributed import get_pp_group as legacy_get_pp_group

        return legacy_get_pp_group()
    return get_parallel().pp_group


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
