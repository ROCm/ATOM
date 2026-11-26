# SPDX-License-Identifier: MIT
"""The SGLang the plugin's DeepSeek-V4.1 runtime rewiring is written against.

The bridge's own rewiring of SGLang internals has no SGLang path to fall back
to (``ForwardBatch.init_new``'s host mirror, the eager runner's view
carriers, the radix seam, DSpark's verify-commit hook), so a DeepSeek-V4.1
launch on any other version is refused first, before the bridge installs any
of it and before the weights load
(``deepseek_v41_bridge.install_deepseek_v41_proxy_pool_patch``). This is the
one version check of the plugin's V4.1 path: the latency patches under
``atom.plugin.sglang.patches`` install after it, so they only ever see this
version. Each still checks the source shape of every function it wraps
(``missing_source_needles``) before rewiring it; where that check fails, the
patch keeps SGLang's code and logs why.
"""

from __future__ import annotations

import inspect

SUPPORTED_SGLANG_VERSIONS = ("0.5.23",)


def sglang_version() -> str:
    import sglang

    return str(sglang.__version__)


def refuse_unsupported_sglang() -> None:
    """Refuse an SGLang other than ``SUPPORTED_SGLANG_VERSIONS`` (an exact
    match on ``sglang.__version__``)."""
    version = sglang_version()
    if version not in SUPPORTED_SGLANG_VERSIONS:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin does not support this SGLang "
            f"(sglang {version} not in {SUPPORTED_SGLANG_VERSIONS})"
        )


def missing_source_needles(function, needles) -> list[str] | None:
    """The ``needles`` missing from ``function``'s source, or None when it
    has no source to read."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return None
    return [needle for needle in needles if needle not in source]
