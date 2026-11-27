"""`spec_verify_fused.spec_verify`'s next-token output is the gather it replaces:
``out[b, verdict[0, b]]``, the last token each request keeps (GPU only)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("spec_verify_fused is a GPU kernel", allow_module_level=True)

svf = pytest.importorskip("atom.model_ops.spec_verify_fused", exc_type=ImportError)


@pytest.mark.parametrize("bs", [1, 3, 8])
@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("agree", [0.0, 0.5, 1.0])
def test_next_ids_is_the_gather(bs, forced, agree):
    k, vocab = 7, 4096 * 3 + 17
    g = torch.Generator(device="cuda").manual_seed(bs * 31 + int(forced) * 7 + int(agree * 10))
    logits = torch.randn(bs * (k + 1), vocab, device="cuda", generator=g).to(torch.bfloat16)
    best = logits.float().argmax(-1).to(torch.int32)
    # Drafts: the argmax with probability `agree`, else another id.
    drafts = torch.where(
        torch.rand(bs * (k + 1), device="cuda", generator=g) < agree, best, (best + 1) % vocab
    )
    input_ids = torch.zeros(bs * (k + 1) + 1, dtype=torch.int32, device="cuda")
    target_idx = torch.tensor(
        [b * (k + 1) + p for b in range(bs) for p in range(k)], dtype=torch.int32, device="cuda"
    )
    input_ids[target_idx.long() + 1] = drafts[target_idx.long() + 1 - 1]
    bonus_idx = torch.tensor([b * (k + 1) + k for b in range(bs)], dtype=torch.int32, device="cuda")
    cu = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda") * k
    temps = torch.zeros(bs, dtype=torch.float32, device="cuda")
    cond = torch.full((k,), 0.6, dtype=torch.float32, device="cuda") if forced else None
    for _ in range(3):
        out, verdict, next_ids = svf.spec_verify(
            logits, temps, target_idx, bonus_idx, cu, input_ids, k, greedy_eps=1e-5, cond_rates=cond
        )
        ref = torch.gather(out.view(bs, k + 1), 1, verdict[0].view(-1, 1).long()).view(bs)
        assert torch.equal(next_ids, ref.to(torch.int32))
        assert bool((next_ids >= 0).all())
