"""Lazy ``sys.modules`` aliases mapping sglang 0.5.15 import paths the ATOM
plugin still uses onto their relocated homes in ``sglang-main``.

Behavior-preserving and additive: an alias is installed ONLY when the old module
is genuinely absent (so real 0.5.15 runtimes are untouched), and each alias
resolves the requested symbol lazily from the new location on first access. This
avoids editing shared or other-model plugin files to chase sglang refactors.

``install()`` is called once from ``register_plugin`` -- early enough to precede
the plugin's attention-backend imports, late enough that importing sglang parent
packages is safe.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types

logger = logging.getLogger("atom.plugin.sglang.compat")

# old module path -> ordered candidate new source modules to search for symbols.
_ALIASES: dict[str, list[str]] = {
    "sglang.srt.layers.attention.utils": [
        "sglang.kernels.ops.kvcache.kv_indices",
        "sglang.kernels.ops.kvcache.cache_ops",
        "sglang.kernels.ops.attention.utils",
        "sglang.kernels.ops.attention.pad",
    ],
    "sglang.srt.layers.quantization.fp8_kernel": [
        "sglang.kernels.ops.quantization.fp8_kernel",
    ],
    "sglang.srt.model_executor.cuda_graph_runner": [
        "sglang.srt.model_executor.runner_utils.capture_mode",
        "sglang.srt.model_executor.runner",
    ],
}


def _parallel_attr_getter(attr: str):
    """Return a zero-arg getter reading ``attr`` off the current ParallelState.

    sglang-main removed the free ``get_attention_{tp,cp}_{rank,size}`` helpers in
    favor of ``get_parallel().attn_{tp,cp}_{rank,size}`` properties. These shims
    restore the old call contract with the correct (non-default) values.
    """

    def _getter() -> int:
        from sglang.srt.runtime_context import get_parallel

        return int(getattr(get_parallel(), attr))

    _getter.__name__ = f"_atom_compat_{attr}"
    return _getter


# existing module path -> {missing symbol name: value/factory} to inject.
# Used when a module still exists but a symbol the plugin imports was relocated.
_SYMBOL_SHIMS: dict[str, dict[str, object]] = {
    "sglang.srt.layers.dp_attention": {
        "get_attention_tp_rank": _parallel_attr_getter("attn_tp_rank"),
        "get_attention_tp_size": _parallel_attr_getter("attn_tp_size"),
        "get_attention_cp_rank": _parallel_attr_getter("attn_cp_rank"),
        "get_attention_cp_size": _parallel_attr_getter("attn_cp_size"),
    },
}

_INSTALLED = False


class _LazyAliasModule(types.ModuleType):
    """A module whose attributes resolve from the first candidate that has them."""

    def __init__(self, name: str, sources: list[str]) -> None:
        super().__init__(name)
        self.__dict__["_atom_sources"] = list(sources)

    def __getattr__(self, attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        for source in self.__dict__["_atom_sources"]:
            try:
                module = importlib.import_module(source)
            except Exception:  # noqa: BLE001,S112
                continue
            if hasattr(module, attr):
                value = getattr(module, attr)
                setattr(self, attr, value)
                return value
        raise AttributeError(
            f"{self.__name__!r} compat alias could not resolve {attr!r} from "
            f"{self.__dict__['_atom_sources']}"
        )


def _old_module_present(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - a missing parent raises here; treat as absent
        return False


def install() -> None:
    """Install compat aliases for any sglang module missing on this runtime."""
    global _INSTALLED
    if _INSTALLED:
        return
    installed = []
    for old_name, sources in _ALIASES.items():
        if _old_module_present(old_name):
            continue
        sys.modules[old_name] = _LazyAliasModule(old_name, sources)
        installed.append(old_name)
    # Inject relocated symbols back into modules that still exist.
    injected = []
    for mod_name, symbols in _SYMBOL_SHIMS.items():
        try:
            module = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001,S112 - module optional across versions
            continue
        for symbol, value in symbols.items():
            if not hasattr(module, symbol):
                setattr(module, symbol, value)
                injected.append(f"{mod_name}.{symbol}")
    _INSTALLED = True
    if installed:
        logger.info(
            "Installed sglang-main compat aliases for relocated modules: %s",
            ", ".join(installed),
        )
    if injected:
        logger.info(
            "Injected sglang-main compat symbols for relocated names: %s",
            ", ".join(injected),
        )
