"""Static q4 workspace for the measured attention HC/PTPC/BA launch."""

import json
import logging

import torch

import atom.models.qwen4_exp_mono.ba.kernel as ba_kernel
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.runtime.compile import compile_kernels, launcher_resources


class BaProjection:
    def __init__(self, device):
        self.launch = ba_kernel.build(False)
        compile_kernels([(self.launch, ba_kernel.ABI)])
        resources = launcher_resources(self.launch)
        assert resources and all(
            not r["scratch"] and not r.get("vgpr_spill") and not r.get("sgpr_spill")
            for r in resources
        )
        self.payload = torch.zeros(4 * 2560, device=device, dtype=torch.int32)
        self.output = torch.empty((4, 96), device=device, dtype=torch.bfloat16)
        self.partials = torch.zeros(5 * 4 * 96 * 2, device=device, dtype=torch.int32)
        self.epochs = torch.zeros(160, device=device, dtype=torch.int32)
        self.diag = torch.zeros(160 * DIAG_WORDS, device=device, dtype=torch.int32)
        logging.getLogger("atom").info(
            "QWEN_HC_BA_BOUND %s",
            json.dumps(
                {
                    "rows": 4,
                    "output": 96,
                    "parts": 5,
                    "resources": resources,
                    "isolated_payload": True,
                }
            ),
        )

    def run(self, values, weight):
        assert (
            weight.shape == (96, 2560)
            and weight.dtype == torch.bfloat16
            and weight.is_contiguous()
        )
        values = {
            **values,
            "payload": self.payload.data_ptr(),
            "ba_weight": weight.data_ptr(),
            "ba": self.output.data_ptr(),
            "partials": self.partials.data_ptr(),
            "epochs": self.epochs.data_ptr(),
            "diag": self.diag.data_ptr(),
        }
        self.launch(*ba_kernel.ABI.pack(values), stream=torch.cuda.current_stream())
