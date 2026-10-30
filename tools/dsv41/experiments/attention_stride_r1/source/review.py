"""Review only this new run, once after retirement and lossless export."""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = HERE / "REVIEW.json"
    assert not destination.exists(), "This new run is reviewed once"
    admission = read(HERE / "ADMISSION.json")
    source = read(HERE / "SOURCE.json.gz")
    assert admission["retired"] and admission["returncode"] == 0
    assert "No KFD PIDs currently running" in admission["final_kfd"]
    cards = [
        line.split(",")
        for line in admission["final_gpu"].splitlines()
        if line.startswith("card")
    ]
    assert len(cards) == 8 and all(row[1:3] == ["0", "0"] for row in cards)
    for path, digest in source["source_sha256"].items():
        assert sha(Path(path)) == digest, path
    for path, digest in admission["export_sha256"].items():
        assert sha(HERE / "output" / path) == digest, path
    ranks = [read(HERE / "output" / f"rank{rank}_result.json.gz") for rank in range(4)]
    for rank, result in enumerate(ranks):
        assert result["rank"] == rank and result["native_communicator_destroy_returned"]
        assert len(result["cases"]) == 2
        for case in result["cases"]:
            assert case["graphs_reset_before_cache_retirement"]
            assert len(case["numerics"]) == 30
            assert all(item["all_output_bytes_equal"] for item in case["numerics"])
    review = {
        "scope": "two complete attention windows; constructed inputs/history, real layer8/9 weights",
        "service_or_model_accuracy_acceptance": False,
        "ranks": [],
    }
    for rank, result in enumerate(ranks):
        case = result["cases"][0]
        row = {"rank": rank, "windows": {}, "costs": {}}
        for layer, rounds in case["windows"].items():
            native, candidate, speedups, drift = [], [], [], []
            for item in rounds:
                a1, b1, b2, a2 = item["ms"]
                native += [a1, a2]
                candidate += [b1, b2]
                speedups += [a1 / b1, a2 / b2]
                drift += [abs(a2 / a1 - 1), abs(b2 / b1 - 1)]
            a, b = statistics.mean(native), statistics.mean(candidate)
            row["windows"][layer] = {
                "native_ms": a,
                "candidate_ms": b,
                "saved_us": 1000 * (a - b),
                "speedup": a / b,
                "pair_speedups": speedups,
                "maximum_repeat_drift": max(drift),
                "qualifies_for_combination": min(speedups) > 1
                and (a - b) / a > max(drift),
            }
        for phase, cost in case["stages"].items():
            row["costs"][phase] = {
                "accepted": cost["accepted_for_cost"],
                "median_ms": statistics.median(cost["interval_ms"]),
                "whole_perturbation": cost["whole_perturbation"],
            }
        review["ranks"].append(row)
    review["local_gain_all_ranks_both_windows"] = all(
        value["qualifies_for_combination"]
        for rank in review["ranks"]
        for value in rank["windows"].values()
    )
    review["all_stage_costs_accepted"] = all(
        value["accepted"]
        for rank in review["ranks"]
        for value in rank["costs"].values()
    )
    review["next"] = (
        "Use accepted per-window stage costs to select H64 or preparation/projection; carry retained supply into a subsequent combination only after local gain. Do not sum stage times or infer whole-service savings."
    )
    destination.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
