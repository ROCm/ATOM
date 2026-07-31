import pytest
import torch

from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="KDA PAD_SLOT_ID regression requires a ROCm GPU",
)
def test_kda_pad_slot_writes_zero_output():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_heads = head_dim = 1
    state_dim = 4

    q = torch.ones((1, 1, num_heads, state_dim), dtype=dtype, device=device)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    gate = torch.zeros_like(q)
    beta = torch.zeros((1, 1, num_heads), dtype=torch.float32, device=device)
    output = torch.full(
        (1, num_heads, state_dim),
        123.0,
        dtype=dtype,
        device=device,
    )
    state = torch.zeros(
        (1, num_heads, state_dim, state_dim),
        dtype=dtype,
        device=device,
    )

    fused_sigmoid_gating_delta_rule_update(
        A_log=torch.zeros(num_heads, dtype=torch.float32, device=device),
        a=gate,
        b=beta,
        dt_bias=torch.zeros(state_dim, dtype=torch.float32, device=device),
        q=q,
        k=k,
        v=v,
        o=output,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=device),
        ssm_state_indices=torch.tensor([-1], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.zeros_like(output))
