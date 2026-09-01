import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from aiter import QuantType, dtypes, get_hip_quant

from atom.model_ops.embed_head import replicated_embedding
from atom.model_ops.fused_mtp_prologue import (
    fused_mtp_embedding_dual_rmsnorm_fp8_quant,
)
from atom.model_ops.layernorm import fused_dual_rmsnorm_cat


@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_fused_mtp_prologue_matches_unfused_fp8_path(
    rows: int, token_dtype: torch.dtype
):
    torch.manual_seed(1234 + rows)
    hidden_size = 6144
    vocab_size = 32
    input_ids = torch.arange(rows, device="cuda", dtype=token_dtype) % vocab_size
    if rows > 1:
        input_ids[-1] = -1  # optimistic async-scheduling placeholder
    if rows > 2:
        input_ids[-2] = vocab_size  # positive out-of-range id

    embedding_weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    previous_hidden_states = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    enorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    actual_q, actual_scale = fused_mtp_embedding_dual_rmsnorm_fp8_quant(
        input_ids,
        embedding_weight,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        eps,
    )

    inputs_embeds = replicated_embedding(input_ids, embedding_weight)
    reference_input = fused_dual_rmsnorm_cat(
        inputs_embeds,
        enorm_weight,
        previous_hidden_states,
        hnorm_weight,
        eps,
    )
    reference_q, reference_scale = get_hip_quant(QuantType.per_Token)(
        reference_input, quant_dtype=dtypes.fp8
    )
    torch.cuda.synchronize()

    assert actual_q.dtype == dtypes.fp8
    assert actual_q.shape == (rows, 2 * hidden_size)
    assert torch.equal(actual_q.view(torch.uint8), reference_q.view(torch.uint8))
    assert torch.equal(actual_scale, reference_scale)
