# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Exercise plugin weight reload against a real sleeping ATOM engine."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import ray
from atomrl.engine.inference.atom_ray_server import ATOMReplicaManager
from atomrl.plugin._megatron_hooks import update_weight_post_write
from transformers import AutoTokenizer


class _SingleGPUWorkerGroup:
    num_workers = 1

    def execute_all_sync(self, method: str):
        if method != "get_colocation_info":
            raise ValueError(f"unsupported worker method: {method}")
        return [
            {
                "node_id": ray.get_runtime_context().get_node_id(),
                "gpu_ids": ["0"],
            }
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    ray.init()
    manager = ATOMReplicaManager(
        actor_wg=_SingleGPUWorkerGroup(),
        model_name=args.model,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_num_seqs": 2,
            "max_model_len": 256,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.5,
        },
        max_concurrency=2,
    )
    try:
        manager.create()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt = tokenizer.encode("Hello", add_special_tokens=True)
        sampling = {
            "max_tokens": 2,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        ray.get(manager.servers[0].generate.remote(prompt, sampling))

        manager.sleep_all(level=2)
        update_weight_post_write(
            SimpleNamespace(_atomrl_atom_manager=manager),
            args.model,
            manager.servers,
        )

        output = ray.get(manager.servers[0].generate.remote(prompt, sampling))
        if not output["token_ids"]:
            raise RuntimeError(
                "generation after sleeping weight reload returned no tokens"
            )
        print("sleeping ATOM weight reload verification passed")
    finally:
        manager.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
