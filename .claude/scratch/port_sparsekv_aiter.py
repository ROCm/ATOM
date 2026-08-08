#!/usr/bin/env python3
"""Port the module_sparsekv_swap aiter op onto the new-image aiter (/app/aiter-test).

Additive + idempotent: copies the 4 sparsekv source files and applies the
optCompilerConfig.json / rocm_ops.hpp / __init__.py registrations plus the
topk __threadfence correctness fix. Canonical sources come from the mounted
feature branch checkout.
"""

import json
import re
import shutil
import sys

SRC = "/it-share/yajizhan/code/aiter"
DST = "/app/aiter-test"

COPIES = [
    "csrc/py_itfs_cu/sparsekv_swap_kernels.cu",
    "csrc/include/sparsekv_swap.h",
    "csrc/pybind/sparsekv_swap_pybind.cu",
    "aiter/ops/sparsekv_swap.py",
]


def copy_files():
    for rel in COPIES:
        shutil.copyfile(f"{SRC}/{rel}", f"{DST}/{rel}")
        print(f"  copied {rel}")


def patch_optconfig():
    p = f"{DST}/aiter/jit/optCompilerConfig.json"
    with open(p) as f:
        dst = json.load(f)
    with open(f"{SRC}/aiter/jit/optCompilerConfig.json") as f:
        src = json.load(f)
    if "module_sparsekv_swap" in dst:
        print("  optCompilerConfig.json: already present")
        return
    dst["module_sparsekv_swap"] = src["module_sparsekv_swap"]
    with open(p, "w") as f:
        json.dump(dst, f, indent=4)
        f.write("\n")
    print("  optCompilerConfig.json: added module_sparsekv_swap")


def extract_macro(path):
    """Pull the SPARSEKV_SWAP_PYBIND macro block out of a rocm_ops.hpp."""
    with open(path) as f:
        lines = f.readlines()
    out, grabbing = [], False
    for ln in lines:
        if ln.startswith("#define SPARSEKV_SWAP_PYBIND"):
            grabbing = True
        if grabbing:
            out.append(ln)
            if not ln.rstrip("\n").endswith("\\"):
                break
    return "".join(out)


def patch_rocm_ops():
    p = f"{DST}/csrc/include/rocm_ops.hpp"
    with open(p) as f:
        text = f.read()
    macro = extract_macro(f"{SRC}/csrc/include/rocm_ops.hpp")
    if not macro:
        sys.exit("ERROR: SPARSEKV_SWAP_PYBIND macro not found in source rocm_ops.hpp")
    existing = extract_macro(p)
    if existing:
        # Refresh rather than skip: a newly registered op (the macro is the only
        # place a binding is declared) would otherwise never reach the image.
        if existing == macro:
            print("  rocm_ops.hpp: already up to date")
            return
        with open(p, "w") as f:
            f.write(text.replace(existing, macro, 1))
        print("  rocm_ops.hpp: refreshed SPARSEKV_SWAP_PYBIND macro")
        return
    block = "\n" + macro + "\n"
    # Insert before the final #endif if the header uses an include guard;
    # otherwise append (file uses #pragma once).
    idx = text.rfind("#endif")
    if idx != -1:
        text = text[:idx] + block + text[idx:]
    else:
        text = text + block
    with open(p, "w") as f:
        f.write(text)
    print("  rocm_ops.hpp: added SPARSEKV_SWAP_PYBIND macro")


def patch_init():
    p = f"{DST}/aiter/__init__.py"
    with open(p) as f:
        lines = f.readlines()
    line = "    from .ops.sparsekv_swap import *  # noqa: F403,E402\n"
    if any("ops.sparsekv_swap" in ln for ln in lines):
        print("  __init__.py: already present")
        return
    # Insert after the last `from .ops.<x> import *` inside the else block.
    last = -1
    pat = re.compile(r"^\s*from \.ops\.\w+ import \*")
    for i, ln in enumerate(lines):
        if pat.match(ln):
            last = i
    if last == -1:
        sys.exit("ERROR: no `from .ops.* import *` anchor found in __init__.py")
    indent = re.match(r"^(\s*)", lines[last]).group(1)
    lines.insert(
        last + 1, f"{indent}from .ops.sparsekv_swap import *  # noqa: F403,E402\n"
    )
    with open(p, "w") as f:
        f.writelines(lines)
    print(f"  __init__.py: added import after line {last + 1}")


def patch_topk():
    p = f"{DST}/csrc/kernels/topk_per_row_kernels.cu"
    with open(p) as f:
        text = f.read()
    anchor = "        __syncthreads();\n\n        // Cross-block barrier via atomicInc + spin-wait."
    fixed = "        __syncthreads();\n        __threadfence();\n\n        // Cross-block barrier via atomicInc + spin-wait."
    if "__threadfence();\n\n        // Cross-block barrier via atomicInc" in text:
        print("  topk_per_row_kernels.cu: fence already present")
        return
    if anchor not in text:
        print("  WARNING: topk anchor not found; skipping fence fix (verify manually)")
        return
    text = text.replace(anchor, fixed, 1)
    with open(p, "w") as f:
        f.write(text)
    print("  topk_per_row_kernels.cu: inserted __threadfence()")


GRID_DIM_OLD = """        IdxT num_blocks = std::min(
            max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
        IdxT items_per_thread  = ceildiv<IdxT>(len, num_blocks * BlockSize);"""

GRID_DIM_NEW = """        // items_per_thread below rounds up, so the recomputed block count is
        // <= the one asked for here. Once max_resident_blocks is the binding
        // bound the two stop agreeing, and the loop's only unconditional exit
        // (below) compares the recomputed count against max_num_blocks — a test
        // that then never fires. Keep the requested count to end the search on.
        const IdxT requested_blocks = std::min(
            max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
        IdxT num_blocks        = requested_blocks;
        IdxT items_per_thread  = ceildiv<IdxT>(len, num_blocks * BlockSize);"""

GRID_EXIT_OLD = """        if(num_blocks == max_num_blocks)
        {
            break;
        }"""

GRID_EXIT_NEW = """        if(requested_blocks == max_num_blocks)
        {
            break;
        }"""


def patch_topk_grid_dim():
    """Make the capped calc_grid_dim's wave search terminate.

    The resident-capacity cap turned the tail-wave search into an infinite host
    loop for len > (VECTORIZED_READ_SIZE / sizeof(T)) * BlockSize *
    active_blocks (1,048,576 for float/1024 on a 256-CU MI355X): the search
    saturates at max_num_blocks, items_per_thread rounds up, and the recomputed
    num_blocks lands strictly below the bound the exit test compares against.
    Only the capped overload is affected; the uncapped one further down keeps a
    self-consistent bound.
    """
    p = f"{DST}/csrc/kernels/topk_per_row_kernels.cu"
    with open(p) as f:
        text = f.read()
    if "requested_blocks" in text:
        print("  topk_per_row_kernels.cu: grid-dim termination already fixed")
        return
    # Both overloads share the loop body; scope the edit to the capped one by
    # starting from its max_resident_blocks line.
    start = text.find("const IdxT max_resident_blocks")
    if start == -1 or GRID_DIM_OLD not in text[start:]:
        print("  WARNING: capped calc_grid_dim not found; skipping grid-dim fix")
        return
    head, tail = text[:start], text[start:]
    tail = tail.replace(GRID_DIM_OLD, GRID_DIM_NEW, 1)
    if GRID_EXIT_OLD not in tail:
        print("  WARNING: grid-dim exit test not found; skipping grid-dim fix")
        return
    tail = tail.replace(GRID_EXIT_OLD, GRID_EXIT_NEW, 1)
    with open(p, "w") as f:
        f.write(head + tail)
    print("  topk_per_row_kernels.cu: grid-dim search now ends on saturation")


if __name__ == "__main__":
    print(">>> Porting module_sparsekv_swap into", DST)
    copy_files()
    patch_optconfig()
    patch_rocm_ops()
    patch_init()
    patch_topk()
    patch_topk_grid_dim()
    print(">>> Port complete.")
