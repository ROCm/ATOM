"""Bind the measured HC/router partition to the effective C1 target route."""

import json
import logging
import os

from atom.models.qwen4_exp_mono.owned import binding as owned


def prepare_layers(layers, bound_weights, built, tp, device):
    prepared = owned.prepare_layers(layers, bound_weights, built, tp, device)
    if prepared is None or os.environ.get("QWEN_ROUTER_PARTITION") != "1":
        return prepared
    import torch

    import atom.models.qwen4_exp_mono.router.front as partition_front
    from atom.mono.runtime.compile import compile_kernels, launcher_resources

    kernel = partition_front.build(False)
    compile_kernels([(kernel, partition_front.ABI)])
    resources = launcher_resources(kernel)
    assert resources and all(
        not x["scratch"] and not x.get("vgpr_spill") and not x.get("sgpr_spill")
        for x in resources
    )
    assert tp == 1 and prepared.metadata["rows"] == 4
    prepared.chain.flags = torch.zeros(
        partition_front.FLAGS_BYTES // 4, device=device, dtype=torch.int32
    )
    prepared.chain.fronts[False] = kernel
    logging.getLogger("atom").info(
        "QWEN_ROUTER_PARTITION_BOUND %s",
        json.dumps(
            {
                "rows": 4,
                "k_splits": 2,
                "resources": resources,
                "hc_tasks": 160,
                "router_tasks": 66,
            }
        ),
    )
    return prepared
