"""CPU-only fidelity check using the actual EPLB planner, without importing torch."""

import ast
import collections
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import types


class Indices:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),)

    def dim(self):
        return 1

    def numel(self):
        return len(self.data)

    def tolist(self):
        return self.data.copy()


def main():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / ".github/scripts/eplb_replay_layer52.json").read_text(encoding="utf-8"))
    names = {"_LocalCopyAction", "_P2PAction", "_assign_sender_for_receiver",
             "_select_source_rank_for_receiver", "_plan_single_layer_migration"}
    source = ast.parse((root / "atom/model_ops/eplb.py").read_text(encoding="utf-8"))
    module = types.ModuleType("replay_planner_validation")
    sys.modules[module.__name__] = module
    module.__dict__["dataclass"] = dataclass
    extracted = ast.Module(body=ast.parse("from __future__ import annotations").body +
                           [node for node in source.body if getattr(node, "name", None) in names], type_ignores=[])
    exec(compile(extracted, "actual_eplb_planner", "exec"), module.__dict__)
    for case in data["cases"]:
        old, new = case["old_physical_to_logical"], case["new_physical_to_logical"]
        assert sorted(old) == sorted(new) == list(range(384))
        sends, recvs = collections.Counter(), collections.Counter()
        for rank in range(8):
            plan, local, send, recv = module._plan_single_layer_migration(
                old_p2l_layer=Indices(old), new_p2l_layer=Indices(new),
                num_local_physical_experts=48, num_gpu_per_node=8, rank=rank, world_size=8,
            )
            by_logical = {}
            for op, actions in (("isend", send), ("irecv", recv)):
                for action in actions:
                    for layout in case["weight_layout"]:
                        size = math.prod(layout["shape"][1:]) * layout["element_size"]
                        by_logical.setdefault(action.logical_expert_id, []).append(
                            {"op": op, "peer": action.peer_rank, "bytes": size})
                        key = (rank, action.peer_rank, action.logical_expert_id, size) if op == "isend" else (action.peer_rank, rank, action.logical_expert_id, size)
                        (sends if op == "isend" else recvs)[key] += 1
            ops = [op for batch in by_logical.values() for op in batch]
            expected = case["expected_ranks"][rank]
            assert expected["rank"] == rank
            assert len(ops) == expected["p2p_ops"]
            assert sum(op["bytes"] for op in ops) == expected["p2p_bytes_send_plus_receive"]
            digest = hashlib.sha256(json.dumps(ops, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert digest == expected["p2p_operations_sha256"], (case["rebalance"], rank)
            # Model each transfer with the expert ID and check every committed slot.
            weights = old[rank * 48:(rank + 1) * 48]
            buffer = [None] * 48
            for action in local:
                buffer[action.dst_slot] = weights[action.src_slot]
            for action in recv:
                buffer[action.local_slot] = action.logical_expert_id
            for src, dst in plan:
                assert buffer[src] is not None
                weights[dst] = buffer[src]
            assert weights == new[rank * 48:(rank + 1) * 48]
        assert sends == recvs
    print("Validated actual planner, all 24 rank/case operation fingerprints, peer matching and slot commits.")


if __name__ == "__main__":
    main()
