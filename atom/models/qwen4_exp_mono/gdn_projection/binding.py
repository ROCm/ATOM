"""Fixed q4 projection/conv binding; all allocations occur before graph capture."""

import torch

from atom.mono.runtime.compile import compile_kernels, launcher_resources

from .kernel import ABI, build


class ProjectionConv:
    def __init__(self, device, debug=False):
        self.kernel = build(debug)
        self.debug = debug
        self.z = torch.empty((4, 48, 128), device=device, dtype=torch.bfloat16)
        self.projected = torch.empty((4, 16384), device=device, dtype=torch.bfloat16)
        compile_kernels([(self.kernel, ABI)])
        self.resources = launcher_resources(self.kernel)
        assert self.resources and all(
            not x["scratch"] and not x.get("vgpr_spill") and not x.get("sgpr_spill")
            for x in self.resources
        )

    def __call__(self, x, scale, weight, impl, alpha, beta_input, conv, metadata, work):
        assert x.shape == (4, 2560) and x.dtype == torch.float8_e4m3fn
        assert weight["weight"].numel() == 16384 * 2560
        assert weight["weight"].is_contiguous() and weight["weight"].dtype == x.dtype
        assert scale.numel() == 4 and weight["weight_scale"].numel() == 16384
        assert scale.dtype == weight["weight_scale"].dtype == torch.float32
        assert impl.A_log.dtype == impl.dt_bias.dtype == torch.bfloat16
        assert metadata.num_spec_decodes == 1
        slots = metadata.spec_state_indices_tensor[:, 0]
        assert slots.numel() == 1 and slots.is_contiguous()
        if conv.shape[1] != 10240:
            conv = conv.transpose(-1, -2)
        assert conv.shape[1:] == (10240, 6) and conv.dtype == torch.bfloat16
        assert alpha.shape == beta_input.shape == (4, 48)
        q, k, v, g, beta = work
        values = {
            "x": x.data_ptr(),
            "weight": weight["weight"].data_ptr(),
            "x_scale": scale.data_ptr(),
            "w_scale": weight["weight_scale"].data_ptr(),
            "out": self.projected.data_ptr(),
            "conv_w": impl.conv1d.weight.data_ptr(),
            "history": conv.data_ptr(),
            "slots": slots.data_ptr(),
            "accepted": metadata.num_accepted_tokens.data_ptr(),
            "cs0": conv.stride(0),
            "cs1": conv.stride(1),
            "cs2": conv.stride(2),
            "alpha": alpha.data_ptr(),
            "beta_input": beta_input.data_ptr(),
            "abs0": alpha.stride(0),
            "bbs0": beta_input.stride(0),
            "a_log": impl.A_log.data_ptr(),
            "dt_bias": impl.dt_bias.data_ptr(),
            "q": q.data_ptr(),
            "k_out": k.data_ptr(),
            "v": v.data_ptr(),
            "g": g.data_ptr(),
            "beta": beta.data_ptr(),
            "z": self.z.data_ptr(),
        }
        self.kernel(*ABI.pack(values), stream=torch.cuda.current_stream())
        return self.z
