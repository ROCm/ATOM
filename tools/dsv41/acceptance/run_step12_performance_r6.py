"""Retry only failedB2 and unstartedA2; deterministic smallest-owner warm case."""

import json
import os

import run_step12_performance as original
from step12_owner_warm_domain import CANDIDATE_WARM, check_domain


def main():
    if os.environ["DSV41_OWNER_SERVING"] == "1":
        original.WARM_CASES = CANDIDATE_WARM
        benchmark = original.benchmark
        phase = original.set_phase

        def reports(complete):
            return [
                check_domain(
                    json.loads(
                        (original.OUTPUT / f"owner_graph_rank{rank}.json").read_text()
                    ),
                    complete=complete,
                )
                for rank in range(4)
            ]

        def checked_benchmark(label, *args, **kwargs):
            result = benchmark(label, *args, **kwargs)
            if label == "warm_initial_four_owners":
                original.write_json(
                    original.OUTPUT / "initial_four_owner_graphs.json",
                    {"status": "initial_capacity16_all4_PASS", "ranks": reports(False)},
                )
            return result

        def checked_phase(value):
            if value == "sealed":
                original.write_json(
                    original.OUTPUT / "owner_warm_domain.json",
                    {
                        "status": "all4_required_full_graph_domain_before_formal_PASS",
                        "ranks": reports(True),
                    },
                )
            phase(value)

        original.benchmark = checked_benchmark
        original.set_phase = checked_phase
    original.main()


if __name__ == "__main__":
    main()
