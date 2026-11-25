"""Validate the source actually installed in a base or derived worker image."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


def main(phase):
    import torch
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_native as native

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one leased GPU must be visible")
    torch.cuda.set_device(0)
    assert "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    manifest = json.loads(Path("/bench/assets/manifest.json").read_text())
    source = Path(inspect.getfile(native)).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == manifest[f"{phase}_module_sha256"]
    output = Path("/bench/base-result" if phase == "base" else "/bench/result")
    output.mkdir(exist_ok=False)
    result = {"phase": phase, "module_sha256": digest, "records": [], "complete": False}

    def save():
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output / "result.json")

    save()
    expression = (
        "tp1_qkv_decode_tiles and bf16"
        if phase == "base"
        else "tp1_qkv_decode_tiles or (test_rocm_mxfp8_linear_matches_dequant_reference and 9856)"
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "-c",
        "/bench/assets/tests/pytest.ini",
        "--confcutdir=/bench/assets/tests",
        "/bench/assets/tests/test_rocm_mxfp8_linear.py",
        "-k",
        expression,
    ]
    if phase == "base":
        command.append("--maxfail=1")
    test = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
    )
    (output / "pytest.log").write_text(test.stdout)
    summaries = re.findall(
        r"(?m)^.*\b(?:passed|failed|skipped)\b.*\bin [\d.]+s.*$", test.stdout
    )
    result.update(
        pytest_command=command,
        pytest_rc=test.returncode,
        pytest_summary=summaries[-1] if summaries else None,
    )
    save()
    print(test.stdout, flush=True)
    if phase == "base":
        assert (
            test.returncode == 1
            and "assert (16, 16, 1024, 2, 2) == (64, 64, 512, 8, 2)" in test.stdout
        ), "Expected only the known untuned-selector regression"
    else:
        assert (
            test.returncode == 0
            and "passed" in test.stdout
            and "skipped" not in test.stdout
        )

    torch.manual_seed(20261125)
    n, k = 9856, 6144
    weights = [
        (
            torch.randn((n, k), device="cuda").mul_(0.25).to(torch.float8_e4m3fn),
            torch.randint(120, 126, (n, k // 32), device="cuda", dtype=torch.uint8),
        )
        for _ in range(8)
    ]
    saved_outputs = {}
    tensor_file = Path("/bench/logs/base_outputs.pt")
    if phase == "candidate":
        saved_outputs = torch.load(tensor_file, map_location="cpu", weights_only=True)
    for m in (16, 24, 32, 40, 48, 56, 64, 80):
        x = torch.randn((m, k), device="cuda").mul_(0.25).to(torch.float8_e4m3fn)
        sx = torch.randint(120, 126, (m, k // 32), device="cuda", dtype=torch.uint8)

        def call(index):
            w, sw = weights[index]
            return native._mxfp8_dot_scaled_gemm(x, sx, w, sw, torch.bfloat16)

        out = call(0).cpu()
        equal = None
        if phase == "base":
            saved_outputs[m] = out
        else:
            equal = torch.equal(out, saved_outputs[m])
        for index in range(8):
            call(index)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = [call(index) for index in range(8)]
        passes = []
        for _ in range(2):
            for _ in range(1000):
                graph.replay()
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(24)
            ]
            for start, end in events:
                start.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            passes.append(
                statistics.median(
                    start.elapsed_time(end) * 1000 / 8 for start, end in events
                )
            )
        graph_equal = torch.equal(graph_outputs[0].cpu(), out)
        assert graph_equal, "Graph replay differs from eager output"
        del graph, graph_outputs
        record = {
            "m": m,
            "selected_config": list(native._select_cfg(m, n, k)),
            "passes_us": passes,
            "repeat_delta_percent": abs(passes[0] - passes[1])
            / statistics.mean(passes)
            * 100,
            "bitwise_equal_to_base": equal,
            "graph_matches_eager": graph_equal,
        }
        result["records"].append(record)
        save()
        print(json.dumps(record), flush=True)
    if phase == "base":
        tensor_file.parent.mkdir(exist_ok=True)
        torch.save(saved_outputs, tensor_file)
    else:
        assert all(
            r["bitwise_equal_to_base"] for r in result["records"]
        ), "Output parity needs investigation before a model run"
    result.update(
        complete=True,
        baseline_outputs_sha256=hashlib.sha256(tensor_file.read_bytes()).hexdigest(),
    )
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["base", "candidate"])
    main(parser.parse_args().phase)
