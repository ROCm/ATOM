"""MoE-only chunking for the isolated EP16 serving comparison."""

import json
import os

import torch


def benchmark_chunk_size():
    return int(os.environ.get("ATOM_BENCH_MOE_CHUNK_SIZE", "0"))


class MoriChunkedExperts(torch.nn.Module):
    def __init__(self, experts, chunk_size):
        super().__init__()
        assert experts.shared_experts is None
        assert not experts.prepare_finalize.supports_async()
        self.experts = experts
        self.chunk_size = chunk_size
        self._first_call = True

    def output_is_reduced(self):
        return self.experts.output_is_reduced()

    def forward(self, **kwargs):
        from atom.model_ops.fused_moe.megamoe_internode_experts import _group_rows
        from atom.utils.forward_context import get_forward_context

        x = kwargs["hidden_states"]
        group_rows, _ = _group_rows(x.shape[0])
        assert x.shape[0] <= group_rows
        cap = self.chunk_size
        if self._first_call:
            handles = self.experts.prepare_finalize._mori_ops
            print(
                "[MORI-CHUNK] "
                + json.dumps(
                    {
                        "rank": handles[0].config.rank,
                        "input_rows": x.shape[0],
                        "input_dtype": str(x.dtype),
                        "agreed_rows": group_rows,
                        "chunk": cap,
                        "send_capacity": handles[0].config.max_num_inp_token_per_rank,
                        "recv_capacity": handles[
                            0
                        ]._cpp_config.max_num_tokens_to_recv(),
                    }
                ),
                flush=True,
            )
            self._first_call = False
        if group_rows <= cap:
            # Return owned storage: the next collective reuses MORI's combine buffer.
            return self.experts(**kwargs).clone()
        assert not kwargs.get("inplace", False)
        context = get_forward_context().context
        across_dp = context.running_tokens_across_dp
        assert across_dp is not None
        output = None
        try:
            for start in range(0, group_rows, cap):
                chunk = dict(kwargs)
                end = min(start + cap, x.shape[0])
                begin = min(start, x.shape[0])
                for name in ("hidden_states", "topk_ids", "topk_weights", "x_scale"):
                    value = chunk.get(name)
                    if value is not None:
                        chunk[name] = value[begin:end]
                context.running_tokens_across_dp = [
                    max(0, min(n - start, cap)) for n in across_dp
                ]
                result = self.experts(**chunk)
                assert (
                    isinstance(result, torch.Tensor) and result.shape[0] == end - begin
                )
                # A routed FP4 input is packed along H; the output is unpacked BF16.
                if output is None:
                    output = torch.empty(
                        (x.shape[0], *result.shape[1:]),
                        dtype=result.dtype,
                        device=result.device,
                    )
                assert (
                    result.shape[1:] == output.shape[1:]
                    and result.dtype == output.dtype
                )
                output[begin:end].copy_(result)
        finally:
            context.running_tokens_across_dp = across_dp
        assert output is not None
        return output
