"""CPU source/ABI preparation check; does not claim GPU correctness or speed."""

import ast
import hashlib
import json
from itertools import pairwise
from pathlib import Path

from binding import build_ast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    source = (
        ROOT
        / "ATOM/atom/models/deepseek_v41/owner_service/owner_cost_attention_chain.py"
    )
    tree = ast.parse(source.read_text())
    native = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_owner"
    )
    for marked in (False, True):
        compile(build_ast(ast.unparse(native), marked=marked), "a0_binding", "exec")
    candidate = ast.parse((HERE / "candidate.py").read_text())
    calls = [node for node in ast.walk(candidate) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(n.func, ast.Attribute)
        and n.func.attr in ("cat", "contiguous", "copy_")
        for n in calls
    )
    assert (
        sum(isinstance(n.func, ast.Attribute) and n.func.attr == "empty" for n in calls)
        == 1
    )
    abi = HERE.parent / "aiter_fp8/csrc/kernels/mla/opus_mla_prefill_fwd.cu"
    native_abi = abi.read_text()
    for field, tensor, axis in (
        ("q_n", "q", 0),
        ("q_h", "q", 1),
        ("q_scale_n", "q_scale", 0),
        ("q_scale_h", "q_scale", 1),
        ("o_n", "out", 0),
        ("o_h", "out", 1),
    ):
        assert f"kargs.stride_{field}" in native_abi
        assert f"static_cast<int>({tensor}->stride({axis}))" in native_abi
    layouts = []
    for rows in (384, 221):
        for width, item_bytes, alignment in ((512, 1, 16), (16, 1, 4), (512, 2, 16)):
            row_bytes, head_bytes = 64 * width * item_bytes, width * item_bytes
            assert row_bytes % alignment == head_bytes % alignment == 0
            intervals = [
                (head * head_bytes, (head + 16) * head_bytes)
                for head in range(0, 64, 16)
            ]
            assert intervals[0][0] == 0 and intervals[-1][1] == row_bytes
            assert all(a[1] == b[0] for a, b in pairwise(intervals))
            layouts.append(
                {
                    "rows": rows,
                    "width": width,
                    "item_bytes": item_bytes,
                    "row_bytes": row_bytes,
                }
            )
    python_paths = sorted(HERE.glob("*.py"))
    for path in python_paths:
        compile(path.read_text(), str(path), "exec")
    paths = [source, abi, *python_paths]
    result = {
        "status": "CPU_SOURCE_AND_LAYOUT_PASS_GPU_UNRUN",
        "native_math_ast_preserved": True,
        "native_independent_stride_fields": True,
        "full_and_ragged_alignment": layouts,
        "files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    (HERE / "LOCAL_PREPARATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    main()
