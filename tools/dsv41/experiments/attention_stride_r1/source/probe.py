"""One attention weight load: numerical checks, complete windows and stage costs."""

import copy
import gc
import gzip
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from atom.models.deepseek_v41.owner_service.owner_cost_attention_chain import (
    forward_owner,
)
from binding import bind_forward
from candidate import attend_fp8_head_shards as candidate_adapter
from fixture import Fixture
from owner_window_cost_r4.device_clock import DeviceClock
from owner_window_cost_r4.fixture import make_config
from preweight import byte_equal
from preweight import run as preweight
from weights import Windows

OUTPUT = Path(os.environ["WINDOW_OUTPUT"])
RANK = int(os.environ["RANK"])


def save(name, value):
    (OUTPUT / f"rank{RANK}_{name}.json.gz").write_bytes(
        gzip.compress((json.dumps(value, sort_keys=True) + "\n").encode(), mtime=0)
    )


def progress(stage):
    print(f"ATTENTION_A0 rank={RANK} stage={stage}", flush=True)
    save("progress", {"stage": stage, "time": time.time()})


def capture(fn):
    for _ in range(3):
        fn()
    with graph_capture() as context:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            output = fn()
    return graph, output


def timing(graph, group, repeats=256):
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / repeats


def snapshot(values):
    return {key: value.clone() for key, value in values.items()}


def check(actual, expected, label):
    assert actual.keys() == expected.keys()
    wrong = [key for key in actual if not byte_equal(actual[key], expected[key])]
    if wrong:
        detail = {}
        for key in wrong:
            a, b = actual[key].float(), expected[key].float()
            detail[key] = float(
                torch.linalg.vector_norm(a - b)
                / torch.linalg.vector_norm(b).clamp_min(1e-30)
            )
        raise AssertionError((label, detail))
    for key in ("output", "attention", "ffn_input_0"):
        assert torch.isfinite(actual[key]).all().item(), (label, key, "nonfinite")
        assert torch.count_nonzero(actual[key]).item(), (label, key, "zero_output")
    return {"check": label, "all_output_bytes_equal": True, "outputs": list(actual)}


MEMO = (
    "selected",
    "candidates",
    "indptrs",
    "tiles",
    "candidate_blocks",
    "group_indices",
    "window_rows",
    "replay",
)


def exercise(fixture, group, measure):
    graphs = []
    outputs = {}
    results = {
        "fixture": fixture.describe(),
        "numerics": [],
        "windows": {},
        "stages": {},
    }
    try:
        # The seed graph is also the layer8 native timing arm. Its Python
        # metadata bindings must survive later captures so layer9 reads the
        # selection buffers this exact graph writes on every replay.
        fixture.restore_cache()
        a8, outputs["8_native"] = capture(lambda: fixture.run(8, "native"))
        graphs.append(a8)
        seed_bindings = {name: copy.copy(getattr(fixture.step, name)) for name in MEMO}
        b8, outputs["8_candidate"] = capture(lambda: fixture.run(8, "candidate"))
        graphs.append(b8)

        def seed9():
            fixture.restore_cache()
            a8.replay()
            for name, value in seed_bindings.items():
                setattr(fixture.step, name, copy.copy(value))

        seed9()
        a9, outputs["9_native"] = capture(lambda: fixture.run(9, "native"))
        graphs.append(a9)
        b9, outputs["9_candidate"] = capture(lambda: fixture.run(9, "candidate"))
        graphs.append(b9)
        arms = {8: (a8, b8), 9: (a9, b9)}
        previous = {}

        for iteration in range(3):
            fixture.update(iteration)
            input_before = {
                layer: state.residual.clone() for layer, state in fixture.inputs.items()
            }
            for layer in (8, 9):
                reset = fixture.restore_cache if layer == 8 else seed9
                reset()
                expected = snapshot(fixture.run(layer, "native"))
                if layer in previous:
                    assert not byte_equal(expected["output"], previous[layer])
                previous[layer] = expected["output"].clone()
                cache_expected = fixture.cache.backing.clone()
                reset()
                self_check = fixture.run(layer, "native")
                results["numerics"].append(
                    check(
                        self_check, expected, f"L{layer}_update{iteration}_native_self"
                    )
                )
                for variant, graph in zip(
                    ("native", "candidate"), arms[layer], strict=True
                ):
                    reset()
                    actual_eager = fixture.run(layer, variant)
                    results["numerics"].append(
                        check(
                            actual_eager,
                            expected,
                            f"L{layer}_update{iteration}_{variant}_eager",
                        )
                    )
                    assert byte_equal(fixture.cache.backing, cache_expected)
                    reset()
                    graph.replay()
                    torch.cuda.synchronize()
                    results["numerics"].append(
                        check(
                            outputs[f"{layer}_{variant}"],
                            expected,
                            f"L{layer}_update{iteration}_{variant}_graph",
                        )
                    )
                    assert byte_equal(fixture.cache.backing, cache_expected), (
                        layer,
                        iteration,
                        variant,
                        "state",
                    )
                    # A repeated tentative replay at the same position must
                    # be idempotent, including all PAGE/STATE writes.
                    graph.replay()
                    torch.cuda.synchronize()
                    check(outputs[f"{layer}_{variant}"], expected, "replay_idempotence")
                    assert byte_equal(fixture.cache.backing, cache_expected)
                del cache_expected, expected, self_check, actual_eager
            assert all(
                byte_equal(state.residual, input_before[layer])
                for layer, state in fixture.inputs.items()
            )
            save("numerics", results)

        if measure:
            for layer in (8, 9):
                if layer == 9:
                    seed9()
                rounds = []
                for _ in range(3):
                    row = [timing(arms[layer][i], group) for i in (0, 1, 1, 0)]
                    rounds.append({"order": "ABBA", "ms": row})
                results["windows"][str(layer)] = rounds
                save("windows", results)

            phases = {
                "prepare_qkv_index_cache": ("window_start", "attention_start"),
                "attention_four_h16": ("attention_start", "attention_end"),
                "projection_mhc": ("attention_end", "window_end"),
            }
            for layer in (8, 9):
                if layer == 9:
                    seed9()
                for phase, endpoints in phases.items():
                    clock = DeviceClock()

                    def mark(point, endpoints=endpoints, clock=clock):
                        if point in endpoints:
                            clock.record(endpoints.index(point))

                    observed = bind_forward(
                        forward_owner, candidate_adapter, boundary=mark
                    )
                    graph, observed_output = capture(
                        lambda layer=layer, observed=observed, mark=mark: fixture.run(
                            layer, "candidate", boundary=(observed, mark)
                        )
                    )
                    graphs.append(graph)
                    bracket = [
                        timing(arms[layer][1], group),
                        timing(graph, group),
                        timing(graph, group),
                        timing(arms[layer][1], group),
                    ]
                    intervals = []
                    for _ in range(16):
                        graph.replay()
                        torch.cuda.synchronize()
                        ticks = clock.read()
                        assert ticks[1] > ticks[0]
                        intervals.append((ticks[1] - ticks[0]) / clock.ticks_per_ms)
                    perturbation = (bracket[1] + bracket[2]) / (
                        bracket[0] + bracket[3]
                    ) - 1
                    results["stages"][f"{layer}_{phase}"] = {
                        "whole_bracket_ms": bracket,
                        "interval_ms": intervals,
                        "ticks_per_ms": clock.ticks_per_ms,
                        "whole_perturbation": perturbation,
                        "accepted_for_cost": abs(perturbation) <= 0.01,
                        "markers_per_graph": 2,
                    }
                    # Check the marked graph's real consumer output as well.
                    arms[layer][1].replay()
                    torch.cuda.synchronize()
                    check(
                        observed_output,
                        outputs[f"{layer}_candidate"],
                        "observer_output",
                    )
                    graph.reset()
                    graphs.remove(graph)
                    del observed_output, graph, clock, observed, mark
                    save("stages", results)
        return results
    finally:
        for graph in graphs:
            graph.reset()
        outputs.clear()
        graphs.clear()
        gc.collect()


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(20261030 + RANK)
    config, deployment = make_config()
    init_distributed_environment(
        world_size=4, rank=RANK, distributed_init_method="env://", local_rank=RANK
    )
    initialize_model_parallel(tensor_model_parallel_size=4)
    group = get_tp_group()
    fixture = windows = None
    report = {"rank": RANK, "started": time.time(), "model_accuracy_acceptance": False}
    try:
        report["deployment"] = deployment
        windows = Windows(config.hf_config)
        plan, index = windows.plan_load()
        save("load_plan", plan)
        progress("preweight_adapter")
        report["preweight"] = preweight()
        save("preweight", report["preweight"])
        progress("preweight_native_cache_layout")
        fixture = Fixture(config, windows, RANK, 384)
        save("preweight_cache", fixture.describe())
        progress("real_attention_weights_loading_once")
        report["weights"] = windows.load(plan, index)
        save("weights", report["weights"])
        report["cases"] = []
        for rows in (384, 221):
            progress(f"complete_attention_M{rows}")
            if fixture is None:
                fixture = Fixture(config, windows, RANK, rows)
            result = exercise(fixture, group, measure=rows == 384)
            fixture.close()
            fixture = None
            gc.collect()
            result["graphs_reset_before_cache_retirement"] = True
            report["cases"].append(result)
            save("result", report)
        progress("graphs_and_caches_released")
        windows = None
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)
        destroy_model_parallel()
        destroy_distributed_environment()
        report["native_communicator_destroy_returned"] = True
        report["elapsed_s"] = time.time() - report["started"]
        save("result", report)
        progress("complete")
    except BaseException:
        save("failure", {"traceback": traceback.format_exc(), "time": time.time()})
        raise
    finally:
        if fixture is not None:
            fixture.close()


if __name__ == "__main__":
    main()
