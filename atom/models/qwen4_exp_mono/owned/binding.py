"""Formal C1/q4 binding; only prepare-time validation and no runtime probes."""

import hashlib
import json
import logging
import os
from pathlib import Path

from atom.models.qwen4_exp_mono.chain.moe_front_hc_integrated_v5 import (
    prepare_layers as prepare_v5,
)
from atom.models.qwen4_exp_mono.chain.moe_hc_fullrange_v7_adapter import (
    replace_hc_kernel_and_flags,
)
from atom.models.qwen4_exp_mono.owned.host import Chain


class PreparedOwned:
    def __init__(self, p, run):
        self.owner = p
        self.chain = Chain(p, run / "compiled_owned")
        self.metadata = {
            "variant": 2,
            "rows": 4,
            "slots": 41,
            "layers": 48,
            "resources": self.chain.resources,
            "sources": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in [
                    "front.py",
                    "device.py",
                    "native_g1.py",
                    "native_g2.py",
                    "host.py",
                ]
            },
        }

    def prepare_layer(self, lid, part, normed):
        self.chain.prepare(lid, part, normed, False)

    def forward_layer(self, lid, mixed, logits):
        return self.chain.experts(lid)


def prepare_layers(layers, bound_weights, built, tp, device):
    p = prepare_v5(layers, bound_weights, built, tp, device)
    if p is None:
        return None
    replace_hc_kernel_and_flags(p)
    owned = PreparedOwned(p, Path(os.environ["QWEN_MOE_HC_RUN"]))
    logging.getLogger("atom").info(
        "QWEN_EXPERT_OWNED_PREPARED %s", json.dumps(owned.metadata, sort_keys=True)
    )
    return owned
