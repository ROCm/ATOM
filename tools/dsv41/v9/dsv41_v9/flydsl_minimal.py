"""AITER's mixed MoE stage-1 FlyDSL builder with the minimal async-copy wait.

The installed `mixed_moe_gemm_2stage_common` is loaded as a sibling module with
two edits: every async copy (not only heterogeneous-B ones) drains with wait 0
before the LDS read, and the kernel name gains a suffix so its compile cache
never aliases the stock kernel.
"""

import importlib.util
import sys

PACKAGE = "aiter.ops.flydsl.kernels"
SOURCE = PACKAGE + ".mixed_moe_gemm_2stage_common"
NAME = PACKAGE + ".mixed_moe_gemm_2stage_minimal_wait"
EDITS = (
    (
        '    ).replace("-", "_")\n',
        '    ).replace("-", "_") + "_asyncwait0_r1"\n',
        2,
    ),
    (
        "                    if const_expr(heterogeneous_b and use_async_copy):",
        "                    if const_expr(use_async_copy):",
        1,
    ),
)


def patched_source():
    spec = importlib.util.find_spec(SOURCE)
    text = spec.loader.get_source(SOURCE)
    for old, new, count in EDITS:
        if text.count(old) != count:
            raise RuntimeError(
                f"{spec.origin}: expected {count} of {old.strip()!r}; "
                "the installed AITER builder changed shape"
            )
        text = text.replace(old, new)
    return spec.origin, text


def load():
    if NAME in sys.modules:
        return sys.modules[NAME]
    origin, text = patched_source()
    spec = importlib.util.spec_from_loader(NAME, loader=None, origin=origin)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PACKAGE
    module.__file__ = origin
    sys.modules[NAME] = module
    try:
        code = compile(text, origin, "exec")
        exec(code, module.__dict__)  # noqa: S102 - the edited AITER builder
    except BaseException:
        del sys.modules[NAME]
        raise
    return module


def compile_mixed_moe_gemm1_common(**kwargs):
    return load().compile_mixed_moe_gemm1_common(**kwargs)
