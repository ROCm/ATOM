# Lazy subpackage re-exports (PEP 562).
# Avoids eagerly importing GPU kernel code when only a single subpackage
# (e.g. ops.quantize) is needed.

_SUBMODULES = (
    "attention",
    "normalization",
    "quantize",
    "gemm",
    "sdma",
    "mlp",
    "moe",
    "fused_residual_norm",
)


def __getattr__(name):
    import importlib

    for sub in _SUBMODULES:
        mod = importlib.import_module(f".{sub}", __name__)
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
