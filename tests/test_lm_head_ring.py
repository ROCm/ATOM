"""``ParallelLMHead`` routes small-row logits through aiter's ring GEMM when ``ATOM_LM_HEAD_RING`` is on.

The kernel is aiter's (bit-exactness and perf are tested there, on a gfx950); what is ATOM's, and tested here with a
stand-in module, is when the head takes it: only with the env on, a built shuffled copy, no caller ``out``, a
contiguous aligned bf16 ``[rows, hidden]`` with rows within the ring's range -- and never otherwise.
"""

import types

import pytest
import torch

embed_head = pytest.importorskip(
    "atom.model_ops.embed_head",
    reason="embed_head requires the Triton/AITER runtime",
    exc_type=ImportError,
)

N, K = 64, 32


class _Ring(types.SimpleNamespace):
    MAX_ROWS = 128

    def __init__(self, supported=True):
        super().__init__(calls=[], warmed=[], supported=supported)

    def ring_supported(self, m, n, k, device=None):
        return self.supported

    def shuffle_weight_ring(self, w):
        return w.clone()

    def warmup_ring(self, n, k, device, w, shuffled=True):
        self.warmed.append((n, k))

    def gemm_a16w16_ring(self, x, w):
        self.calls.append(tuple(x.shape))
        return x @ w.T


def _head(monkeypatch, ring, enabled=True, bias=None, dtype=torch.bfloat16):
    head = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.randn(N, K).to(dtype), requires_grad=False)
    head.bias = bias
    monkeypatch.setattr(embed_head, "_ring_module", lambda: ring)
    monkeypatch.setattr(embed_head.envs, "ATOM_LM_HEAD_RING", enabled, raising=False)
    monkeypatch.setattr(
        embed_head.tgemm, "mm", lambda x, w, b=None, out=None: x.to(w.dtype) @ w.T + 1.0
    )
    return head


def test_ring_takes_small_rows_after_load(monkeypatch):
    ring = _Ring()
    head = _head(monkeypatch, ring)
    head.process_weights_after_loading()
    assert ring.warmed == [(N, K)]
    x = torch.randn(7, K).bfloat16()
    assert torch.equal(head.local_logits(x), x @ head._ring_weight.T)
    assert ring.calls == [(7, K)]


@pytest.mark.parametrize("rows", [129, 1000])
def test_large_rows_keep_tgemm(monkeypatch, rows):
    ring = _Ring()
    head = _head(monkeypatch, ring)
    head.process_weights_after_loading()
    x = torch.randn(rows, K).bfloat16()
    assert torch.equal(head.local_logits(x), x @ head.weight.T + 1.0)
    assert ring.calls == []


def test_caller_storage_keeps_tgemm(monkeypatch):
    ring = _Ring()
    head = _head(monkeypatch, ring)
    head.process_weights_after_loading()
    x = torch.randn(8, K).bfloat16()
    head.local_logits(x, out=torch.empty(8, N, dtype=torch.bfloat16))
    assert ring.calls == []


def test_non_contiguous_or_wrong_dtype_keeps_tgemm(monkeypatch):
    ring = _Ring()
    head = _head(monkeypatch, ring)
    head.process_weights_after_loading()
    head.local_logits(torch.randn(K, 8).bfloat16().T)
    head.local_logits(torch.randn(8, K))
    assert ring.calls == []


@pytest.mark.parametrize(
    "enabled,ring,bias,dtype",
    [
        (False, _Ring(), None, torch.bfloat16),
        (True, None, None, torch.bfloat16),
        (True, _Ring(supported=False), None, torch.bfloat16),
        (True, _Ring(), torch.zeros(N), torch.bfloat16),
        (True, _Ring(), None, torch.float32),
    ],
)
def test_no_ring_weight_unless_everything_fits(monkeypatch, enabled, ring, bias, dtype):
    head = _head(monkeypatch, ring, enabled=enabled, bias=bias, dtype=dtype)
    head.process_weights_after_loading()
    assert head._ring_weight is None
    x = torch.randn(8, K).to(dtype)
    assert torch.equal(head.local_logits(x), x @ head.weight.T + 1.0)
