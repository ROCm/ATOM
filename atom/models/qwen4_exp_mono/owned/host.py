import torch

from atom.models.qwen4_exp_mono.owned import device, front
from atom.models.qwen4_exp_mono.owned.native_g2 import get_g2
from atom.models.qwen4_exp_mono.owned.resources import resources
from atom.mono.runtime.compile import compile_kernels, compile_only


class Chain:
    def __init__(self, p, out):
        self.p = p
        self.slot_map = torch.empty(41, device=p.mixed.device, dtype=torch.int32)
        self.payload = torch.zeros_like(p.payload)
        self.flags = torch.zeros_like(p.flags)
        self.diag = torch.zeros_like(p.diag)
        self.fronts = {}
        self.resources = {}
        for debug in (False, True):
            k = front.build(debug)
            compile_kernels([(k, front.ABI)])
            self.fronts[debug] = k
            self.resources["front_" + str(debug)] = resources(
                k, out / ("front_" + str(debug)), threads=512
            )
        self.g1 = device.build()
        compile_kernels([(self.g1, device.ABI)])
        self.resources["g1"] = resources(self.g1, out / "g1", threads=256)
        self.g2 = get_g2(
            16,
            128,
            128,
            False,
            8192,
            "atomic",
            640,
            "fp4",
            b_dtype="fp4",
            topk=11,
            SBM=16,
            persist=False,
            cu_num=0,
            out_dtype="bf16",
            g2_bf16_lds=True,
            g2_spart=402,
            enable_bias=False,
            g2_prefetch_ids=False,
            use_valid_token_count=False,
        )
        with compile_only():
            self.g2(*p.layers[0].g2.pack(torch.cuda.current_stream()))
        self.resources["g2"] = resources(self.g2, out / "g2", threads=256)

    def prepare(self, lid, part, normed, debug=False):
        p = self.p
        args = list(
            p.hc_layers[lid].pack(part.data_ptr(), normed.data_ptr(), normed.stride(0))
        )
        args[8] = self.payload.data_ptr()
        args[18:20] = self.flags.data_ptr(), self.diag.data_ptr()
        self.fronts[debug](*args, stream=torch.cuda.current_stream())

    def experts(self, lid):
        p = self.p
        b = p.front
        a = p.layers[lid].g1
        self.g1(
            p.mixed.data_ptr(),
            p.topk_ids.data_ptr(),
            p.topk_weights.data_ptr(),
            b.sorted_ids.data_ptr(),
            b.sorted_weights.data_ptr(),
            b.experts.data_ptr(),
            b.valid.data_ptr(),
            b.m_indices.data_ptr(),
            b.reverse.data_ptr(),
            b.out.data_ptr(),
            a.weight,
            a.weight_scale,
            a.intermediate,
            a.intermediate_scale,
            self.slot_map.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        args = list(p.layers[lid].g2.pack(torch.cuda.current_stream()))
        args[8] = self.slot_map.data_ptr()
        self.g2(*args)
        return b.out
