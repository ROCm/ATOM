# SPDX-License-Identifier: MIT
"""Unit test for the per-token fused AllReduce+RMSNorm+quant branch in RMSNorm.

Verifies branch SELECTION and output REPACKING without a GPU: the aiter comm
op and tp world size are monkeypatched, so we assert RMSNorm.forward routes a
per-token fused-quant config to tensor_model_parallel_fused_allreduce_rmsnorm_quant
and repacks its (fp8, residual, scale) into ((fp8, scale), residual).

This test needs the REAL atom.model_ops.layernorm source, not the lightweight
atom.* stubs that tests/conftest.py installs. Following the established pattern
in tests/test_mxfp4_moe_has_bias.py, setUpModule/tearDownModule snapshots,
wipes, and restores sys.modules["atom*"] so the effect is local to this file
and does not pollute other tests that rely on the stubs.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

_saved_atom_modules: dict[str, object] = {}


def setUpModule():
    global _saved_atom_modules
    _saved_atom_modules = {
        name: mod for name, mod in sys.modules.items() if name.startswith("atom")
    }
    for name in list(_saved_atom_modules):
        del sys.modules[name]


def tearDownModule():
    for name in [n for n in sys.modules if n.startswith("atom")]:
        del sys.modules[name]
    sys.modules.update(_saved_atom_modules)


def _install_fake_aiter(monkeypatch, captured):
    """Patch the symbols RMSNorm.forward uses for the per-token branch."""
    import atom.model_ops.layernorm as ln

    def fake_quant(input_, residual_inp_, weight_, eps, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        out_fp8 = torch.zeros_like(input_, dtype=torch.uint8)
        res_out = torch.zeros_like(residual_inp_)
        scale = torch.ones((input_.shape[0], 1), dtype=torch.float32)
        return out_fp8, res_out, scale

    monkeypatch.setattr(
        ln,
        "tensor_model_parallel_fused_allreduce_rmsnorm_quant",
        fake_quant,
        raising=True,
    )
    return ln


def _make_rmsnorm(ln, monkeypatch):
    """Build an RMSNorm whose state forces the per-token branch."""
    from aiter import QuantType

    monkeypatch.setattr(
        ln, "get_tensor_model_parallel_world_size", lambda: 8, raising=True
    )
    norm = ln.RMSNorm(dim=16, eps=1e-6, fused_allreduce=True, fused_quant=True)
    norm.quant_type = QuantType.per_Token
    norm.use_fused_quant = True
    norm.fused_allreduce = True
    norm.tp_size = 8
    return norm


def test_per_token_branch_selected_and_repacked(monkeypatch):
    captured = {}
    ln = _install_fake_aiter(monkeypatch, captured)
    norm = _make_rmsnorm(ln, monkeypatch)

    x = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    out, res_out = norm.forward(x, residual)

    assert captured.get("called") is True, "per-token comm op was not called"
    assert captured["kwargs"].get("quant_type") == "per_token"
    assert isinstance(out, tuple) and len(out) == 2
    fp8, scale = out
    assert scale.shape == (4, 1)
    assert res_out.shape == (4, 16)
