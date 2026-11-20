"""CI-only controls for the frozen #3311 repair; never runs a GPU test."""

import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CANDIDATE = "dd1fd8ed0f60f887fed03770782e32a0509fb565"
FIRST_DELTA = "174bd09ed353845aef5ffd7346211123b9db895a"
BASE = "6dcaba1e0db9bb6956bfb2bca31b8d6794be61da"
ATTENTION = Path("atom/model_ops/attention_mla.py")
DCP = Path("atom/model_ops/dcp_ops.py")
INDEXER_SOURCE = Path("atom/model_ops/indexer/indexer_forward.py")
RESULTS = Path("diagnostic-results")
PLAN = "tests/test_sparse_mla_plan_dispatch.py"
LAYOUT = "tests/test_mla_v_up_output_layout.py"
INDEXER_TEST = "tests/test_indexer_exchange_context.py"


def git(*args):
    return subprocess.check_output(["git", *args])


def run_case(name, selection, *, expected_ids=None, witnesses=None):
    report = RESULTS / f"{name}.xml"
    with (RESULTS / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *selection, f"--junitxml={report}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    failures = sum(witness is not None for witness in (witnesses or {}).values())
    assert result.returncode == (1 if failures else 0), (name, result.returncode)
    tested = ET.parse(report).findall(".//testcase")
    ids = [f"{case.attrib['classname']}::{case.attrib['name']}" for case in tested]
    assert len(ids) == len(set(ids)), (name, "duplicate test IDs")
    assert tested and not any(case.find("error") is not None for case in tested), name
    assert not any(case.find("skipped") is not None for case in tested), name
    if expected_ids is not None:
        assert set(ids) == set(expected_ids), (name, ids)
    if witnesses:
        assert set(ids) == set(witnesses), (name, "unclassified test IDs")
    for case_id, case in zip(ids, tested):
        failure = case.find("failure")
        witness = witnesses[case_id] if witnesses else None
        if witness is None:
            assert failure is None, (name, case_id)
            continue
        assert failure is not None, (name, case_id)
        text = failure.text or ""
        location = f"{witness['path']}:{witness['line']}: {witness['error']}"
        assert location in text, (name, case_id, "wrong failure location", location)
        assert all(token in text for token in witness["tokens"]), (
            name,
            case_id,
            "wrong failure reason",
        )
    return {
        "name": name,
        "tests": len(tested),
        "failures": failures,
        "exit_code": result.returncode,
        "test_ids": sorted(ids),
        "witnesses": witnesses,
    }


def family_ids(baseline, path, function, count=8):
    prefix = path.removesuffix(".py").replace("/", ".") + "::" + function + "["
    ids = {case_id for case_id in baseline["test_ids"] if case_id.startswith(prefix)}
    assert len(ids) == count, (function, ids)
    return ids


def stride_witness(variable, shape):
    tree = ast.parse(Path(LAYOUT).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "as_strided"
    ]
    assert len(calls) == 1
    return {
        "path": LAYOUT,
        "line": calls[0].lineno,
        "error": "RuntimeError",
        "tokens": ["setStorage", "out of bounds", f"sizes {shape}"],
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    git("diff", "--exit-code", CANDIDATE, "--", "atom", "tests")
    original_attention, original_dcp = ATTENTION.read_bytes(), DCP.read_bytes()
    original_indexer = INDEXER_SOURCE.read_bytes()
    evidence = {
        "candidate": CANDIDATE,
        "pre_fix": BASE,
        "first_delta": FIRST_DELTA,
        "ci_head": git("rev-parse", "HEAD").decode().strip(),
        "cases": [],
    }
    try:
        baseline = run_case("candidate-before", [PLAN, LAYOUT, INDEXER_TEST])
        evidence["cases"].append(baseline)
        plan_ids = family_ids(
            baseline, PLAN, "test_each_owner_rebuilds_folded_or_unknown_plans"
        )
        layout_ids = family_ids(
            baseline, LAYOUT, "test_head_major_exchange_preserves_partials_and_fp32_lse"
        )
        native_ids = {
            case_id
            for case_id in plan_ids
            if re.search(r"(?:\[|-)True-True(?:-|\])", case_id)
        }
        assert len(native_ids) == 2
        assertions = [
            node
            for node in ast.walk(ast.parse(Path(PLAN).read_text()))
            if isinstance(node, ast.Assert)
        ]
        folded = [
            node
            for node in assertions
            if isinstance(node.msg, ast.Constant)
            and node.msg.value
            == "folded or unknown plans must rebuild, not patch row indices"
        ]
        queried = [
            node
            for node in assertions
            if isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "queries"
        ]
        assert len(folded) == len(queried) == 1
        plan_witnesses = {
            case_id: {
                "path": PLAN,
                "line": (queried[0] if case_id in native_ids else folded[0]).lineno,
                "error": "AssertionError",
                "tokens": [
                    (
                        "assert queries"
                        if case_id in native_ids
                        else "folded or unknown plans must rebuild"
                    )
                ],
            }
            for case_id in plan_ids
        }
        ATTENTION.write_bytes(git("show", f"{BASE}:{ATTENTION.as_posix()}"))
        evidence["cases"].append(
            run_case(
                "pre-fix-plan",
                [f"{PLAN}::test_each_owner_rebuilds_folded_or_unknown_plans"],
                expected_ids=plan_ids,
                witnesses=plan_witnesses,
            )
        )
        ATTENTION.write_bytes(original_attention)

        one_row_ids = family_ids(
            baseline,
            LAYOUT,
            "test_one_row_prefill_keeps_the_projection_transport",
            count=3,
        )
        bf16_ids = {case_id for case_id in one_row_ids if case_id.endswith("[bf16-2]")}
        assert len(bf16_ids) == 1
        one_row = next(
            node
            for node in ast.parse(Path(LAYOUT).read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_one_row_prefill_keeps_the_projection_transport"
        )
        count_assert = [
            node
            for node in ast.walk(one_row)
            if isinstance(node, ast.Assert)
            and ast.unparse(node.test) == "len(calls) == collectives"
        ]
        assert len(count_assert) == 1
        ATTENTION.write_bytes(git("show", f"{FIRST_DELTA}:{ATTENTION.as_posix()}"))
        evidence["cases"].append(
            run_case(
                "first-delta-one-row",
                [f"{LAYOUT}::test_one_row_prefill_keeps_the_projection_transport"],
                expected_ids=one_row_ids,
                witnesses={
                    case_id: (
                        None
                        if case_id in bf16_ids
                        else {
                            "path": LAYOUT,
                            "line": count_assert[0].lineno,
                            "error": "AssertionError",
                            "tokens": ["assert len(calls) == collectives"],
                        }
                    )
                    for case_id in one_row_ids
                },
            )
        )
        ATTENTION.write_bytes(original_attention)

        stream_ids = family_ids(
            baseline,
            INDEXER_TEST,
            "test_dcp_filter_enters_a_cuda_context_only_for_a_real_exchange",
            count=2,
        )
        live_stream_ids = {
            case_id for case_id in stream_ids if case_id.endswith("[True]")
        }
        assert len(live_stream_ids) == 1
        stream_assert = [
            node
            for node in ast.walk(ast.parse(Path(INDEXER_TEST).read_text()))
            if isinstance(node, ast.Assert)
            and isinstance(node.msg, ast.Constant)
            and node.msg.value == "streamless filters must not resolve a CUDA context"
        ]
        assert len(stream_assert) == 1
        INDEXER_SOURCE.write_bytes(
            git("show", f"{FIRST_DELTA}:{INDEXER_SOURCE.as_posix()}")
        )
        evidence["cases"].append(
            run_case(
                "first-delta-stream-context",
                [INDEXER_TEST],
                expected_ids=stream_ids,
                witnesses={
                    case_id: (
                        None
                        if case_id in live_stream_ids
                        else {
                            "path": INDEXER_TEST,
                            "line": stream_assert[0].lineno,
                            "error": "AssertionError",
                            "tokens": [
                                "streamless filters must not resolve a CUDA context"
                            ],
                        }
                    )
                    for case_id in stream_ids
                },
            )
        )
        INDEXER_SOURCE.write_bytes(original_indexer)

        text = original_dcp.decode()
        # The first combine call in cp_lse_a2a is the head-major path.
        start = text.index("def cp_lse_a2a(")
        old = (
            "recv.stride(0),\n            recv.stride(1),\n            recv.stride(2),"
        )
        new = (
            "recv.stride(0),\n            recv.stride(2),\n            recv.stride(1),"
        )
        at = text.index(old, start)
        DCP.write_text(text[:at] + text[at:].replace(old, new, 1), encoding="utf-8")
        evidence["cases"].append(
            run_case(
                "swapped-output-strides",
                [f"{LAYOUT}::test_head_major_exchange_preserves_partials_and_fp32_lse"],
                expected_ids=layout_ids,
                witnesses={
                    case_id: stride_witness("received", "[2, 3, 2, 8]")
                    for case_id in layout_ids
                },
            )
        )
        old = "recv_lse_stride_b=recv_lse.stride(1),\n            recv_lse_stride_h=recv_lse.stride(2),"
        new = "recv_lse_stride_b=recv_lse.stride(2),\n            recv_lse_stride_h=recv_lse.stride(1),"
        assert text.count(old) == 1
        DCP.write_text(text.replace(old, new), encoding="utf-8")
        evidence["cases"].append(
            run_case(
                "swapped-lse-strides",
                [f"{LAYOUT}::test_head_major_exchange_preserves_partials_and_fp32_lse"],
                expected_ids=layout_ids,
                witnesses={
                    case_id: stride_witness("received_lse", "[2, 3, 2]")
                    for case_id in layout_ids
                },
            )
        )
        DCP.write_bytes(original_dcp)
        evidence["cases"].append(
            run_case(
                "candidate-after",
                [PLAN, LAYOUT, INDEXER_TEST],
                expected_ids=baseline["test_ids"],
            )
        )
    finally:
        ATTENTION.write_bytes(original_attention)
        DCP.write_bytes(original_dcp)
        INDEXER_SOURCE.write_bytes(original_indexer)
        (RESULTS / "controls.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
    git("diff", "--exit-code", CANDIDATE, "--", "atom", "tests")


if __name__ == "__main__":
    main()
