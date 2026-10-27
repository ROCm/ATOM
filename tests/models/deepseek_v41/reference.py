# SPDX-License-Identifier: MIT
"""Load pinned official model code with test-only PyTorch kernels.

Each load gets a private module namespace: the upstream module-global attention
state must never leak between independent oracle models or into ATOM imports.
"""

import ast
import hashlib
import json
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import oracle_kernels

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "reference_manifest.json").read_text())


def check_reference_sources(model_dir):
    root = Path(model_dir)
    for filename, expected in MANIFEST["sha256"].items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference revision mismatch: {filename}: {actual}")


@contextmanager
def load_reference(model_dir):
    check_reference_sources(model_dir)
    root = Path(model_dir) / "inference"
    package_name = f"_atom_dsv41_reference_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    registered = [package_name]
    sys.modules[package_name] = package
    kernel_name = f"{package_name}.kernel"
    sys.modules[kernel_name] = oracle_kernels
    registered.append(kernel_name)
    try:
        for name in ("engram", "image_processor", "vision", "model"):
            full_name = f"{package_name}.{name}"
            path = root / f"{name}.py"
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in (
                    "kernel",
                    "engram",
                    "image_processor",
                    "vision",
                ):
                    node.module = f"{package_name}.{node.module}"
            module = types.ModuleType(full_name)
            module.__file__ = str(path)
            module.__package__ = package_name
            sys.modules[full_name] = module
            registered.append(full_name)
            # The pinned SHA256 check above precedes all source execution.
            exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
        yield sys.modules[f"{package_name}.model"]
    finally:
        for name in registered:
            sys.modules.pop(name, None)
