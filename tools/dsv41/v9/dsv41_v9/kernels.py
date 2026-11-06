"""Build and load the V9 HIP communication kernels.

`ordered_rs.so` (ordered TP4 BF16 reduce-scatter) and `push_join.so` (routed +
shared push-join reduce-scatter) compile against the installed AITER custom
all-reduce header. The ordered kernel adds a kernel template and a dispatcher
member to `CustomAllreduce`, so it compiles against a copy of that header with
`csrc/ordered_rs_{kernel,dispatch}.cuh` inserted. Builds are cached by content
hash under `$DSV41_V9_KERNEL_DIR` (default `~/.cache/atom/dsv41_v9_kernels`)
and serialized with a file lock, so the TP ranks build once.
"""

import ctypes
import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
HEADER = "custom_all_reduce.cuh"
ORDERED_HEADER = "custom_all_reduce_ordered.cuh"
KERNEL_ANCHOR = (
    "// reduce_scatter, scatter on last dim — naive (non-vectorized) fallback."
)
DISPATCH_ANCHOR = "template <typename T>\nvoid dispatchReduceScatter("
SOURCES = {"ordered_rs": "ordered_rs.cu", "push_join": "push_join.cu"}


def aiter_root():
    spec = importlib.util.find_spec("aiter")
    for location in spec.submodule_search_locations or ():
        root = Path(location).resolve().parent
        if (root / "csrc/include" / HEADER).is_file():
            return root
    raise FileNotFoundError(f"no AITER checkout with csrc/include/{HEADER}")


def _insert_before(text, anchor, snippet):
    if text.count(anchor) != 1:
        raise RuntimeError(f"{HEADER}: anchor {anchor.splitlines()[0]!r} not unique")
    index = text.index(anchor)
    return text[:index] + snippet + text[index:]


def ordered_header(native):
    text = _insert_before(
        native, KERNEL_ANCHOR, "\n" + (CSRC / "ordered_rs_kernel.cuh").read_text()
    )
    return _insert_before(
        text, DISPATCH_ANCHOR, (CSRC / "ordered_rs_dispatch.cuh").read_text()
    )


def _build(name, aiter, arch, target):
    include = aiter / "csrc/include"
    work = target.with_suffix(".build")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / ORDERED_HEADER).write_text(ordered_header((include / HEADER).read_text()))
    source = work / SOURCES[name]
    shutil.copy2(CSRC / SOURCES[name], source)
    subprocess.run(
        [
            "hipcc",
            "-std=c++17",
            f"--offload-arch={arch}",
            "-O3",
            "-fPIC",
            "-shared",
            "-DUSE_ROCM",
            "-DENABLE_CK=0",
            f"-I{work}",
            f"-I{include}",
            str(source),
            "-o",
            str(work / target.name),
        ],
        check=True,
    )
    os.replace(work / target.name, target)
    shutil.rmtree(work)


def library_path(name, aiter=None, arch="gfx950", cache_dir=None):
    """The cached shared object for `name`, building it on first use."""
    assert name in SOURCES, name
    aiter = Path(aiter) if aiter else aiter_root()
    cache_dir = Path(
        cache_dir
        or os.environ.get("DSV41_V9_KERNEL_DIR")
        or Path.home() / ".cache/atom/dsv41_v9_kernels"
    )
    digest = hashlib.sha256(arch.encode())
    for path in (
        aiter / "csrc/include" / HEADER,
        CSRC / SOURCES[name],
        CSRC / "ordered_rs_kernel.cuh",
        CSRC / "ordered_rs_dispatch.cuh",
    ):
        digest.update(path.read_bytes())
    target = cache_dir / f"{name}_{arch}_{digest.hexdigest()[:16]}.so"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            _build(name, aiter, arch, target)
    return target


def load(name):
    return ctypes.CDLL(str(library_path(name)))
