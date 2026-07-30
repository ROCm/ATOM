import importlib.util
from pathlib import Path

import torch


# ``atom.model_ops.__init__`` imports the full attention stack, which is not
# configured in this focused CPU unit test.  Load the leaf sampler module so
# this test exercises only its dispatch decision.
_SPEC = importlib.util.spec_from_file_location(
    "atom_sampler_unit", Path(__file__).parents[1] / "atom/model_ops/sampler.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Sampler = _MODULE.Sampler


def test_greedy_without_filters_uses_argmax():
    """Temperature zero must not fall through to epsilon Gumbel sampling."""
    logits = torch.tensor([[0.1, 0.8, 0.3], [2.0, 1.0, 3.0]])
    temperatures = torch.full((2,), 1e-10)

    sampled = Sampler()(logits, temperatures, all_greedy=True)

    assert torch.equal(sampled, torch.tensor([1, 2], dtype=torch.int32))
