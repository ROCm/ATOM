"""Unit tests for the Kimi-K3 DSpark ``TARGET_VERIFY`` plugin helpers.

These exercise the pure-Python / CPU-tensor logic of the KDA (GDN) speculative
verify path (``kimi_k3_spec_verify``) without needing a GPU, a loaded model, or
an active CUDA graph -- mirroring the mocking style of
``test_sglang_gdn_forward_context``.
"""

from types import SimpleNamespace

import torch

from atom.plugin.sglang.kimi_k3_spec_verify import (
    _max_graph_bs,
    build_spec_gdn_metadata,
    build_spec_plan,
    is_target_verify,
    keepalive_if_capturing,
)


class _VerifyMode:
    @staticmethod
    def is_target_verify():
        return True


class _DecodeMode:
    @staticmethod
    def is_target_verify():
        return False


def test_is_target_verify_detects_mode():
    assert is_target_verify(SimpleNamespace(forward_mode=_VerifyMode())) is True
    assert is_target_verify(SimpleNamespace(forward_mode=_DecodeMode())) is False
    # Missing / malformed forward_mode is treated as "not verify", never raises.
    assert is_target_verify(SimpleNamespace(forward_mode=None)) is False
    assert is_target_verify(SimpleNamespace()) is False


def test_keepalive_is_noop_outside_cuda_graph_capture():
    a = torch.zeros(2)
    b = torch.ones(3)
    # Outside CUDA-graph capture (and on CPU-only CI) this is a pure pass-through.
    assert keepalive_if_capturing(a) is a
    x, y = keepalive_if_capturing(a, b)
    assert x is a and y is b


def test_max_graph_bs_prefers_linear_backend_bucket_count():
    backend = SimpleNamespace(state_indices_list=[object(), object(), object()])
    assert _max_graph_bs(backend, 1) == 3
    # Falls back to the live bs when no captured buckets exist.
    assert _max_graph_bs(SimpleNamespace(state_indices_list=None), 5) == 5
    assert _max_graph_bs(SimpleNamespace(), 7) == 7


def _make_verify_batch(bs: int, draft_token_num: int, committed: torch.Tensor):
    forward_batch = SimpleNamespace(
        forward_mode=_VerifyMode(),
        batch_size=bs,
        spec_info=SimpleNamespace(draft_token_num=draft_token_num),
    )
    linear_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=committed),
    )
    return forward_batch, linear_backend


def test_build_spec_plan_layout_and_caching():
    bs, draft_token_num = 2, 8
    # More slots than bs on purpose: only the first ``bs`` are committed.
    committed = torch.tensor([3, 5, 9, 9], dtype=torch.int32)
    forward_batch, linear_backend = _make_verify_batch(bs, draft_token_num, committed)

    plan = build_spec_plan(forward_batch, linear_backend)

    assert plan.bs == bs
    assert plan.draft_token_num == draft_token_num
    assert plan.num_spec == draft_token_num - 1
    assert torch.equal(plan.committed_indices, torch.tensor([3, 5], dtype=torch.int32))
    # spec_state_indices[i] = arange(i*T, (i+1)*T) -> flattened scratch rows.
    assert torch.equal(
        plan.spec_state_indices,
        torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        plan.spec_query_start_loc, torch.tensor([0, 8, 16], dtype=torch.int32)
    )
    assert torch.equal(plan.num_accepted_tokens, torch.ones(bs, dtype=torch.int32))
    assert bool(plan.spec_sequence_masks.all())
    assert plan.spec_sequence_masks.dtype == torch.bool

    # The plan is memoised on the forward batch (one build per verify step).
    assert forward_batch._atom_k3_spec_plan is plan
    assert build_spec_plan(forward_batch, linear_backend) is plan


def test_build_spec_gdn_metadata_marks_full_spec_batch():
    bs, draft_token_num = 3, 8
    committed = torch.arange(bs, dtype=torch.int32)
    forward_batch, linear_backend = _make_verify_batch(bs, draft_token_num, committed)

    plan = build_spec_plan(forward_batch, linear_backend)
    metadata = build_spec_gdn_metadata(plan)

    assert metadata.num_spec_decodes == bs
    assert metadata.num_spec_decode_tokens == bs * draft_token_num
    assert metadata.num_actual_tokens == bs * draft_token_num
    # A verify forward carries no decode / prefill work.
    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 0
    assert metadata.num_decode_tokens == 0
    assert metadata.num_prefill_tokens == 0
    # Spec fields are the plan tensors; non-spec fields are unused.
    assert metadata.spec_state_indices_tensor is plan.spec_state_indices
    assert metadata.spec_query_start_loc is plan.spec_query_start_loc
    assert metadata.non_spec_query_start_loc is None
    assert metadata.non_spec_state_indices_tensor is None


def test_target_verify_gate_is_kimi_k3_only(monkeypatch):
    """The shared GDN bridge must only enter K3 code for a K3 model."""
    from atom.plugin.sglang.attention_backend import attention_gdn

    def _set_config(hf_config):
        monkeypatch.setattr(
            attention_gdn,
            "get_current_atom_config",
            lambda: SimpleNamespace(hf_config=hf_config),
        )

    gate = attention_gdn.SGLangGDNForwardContext._is_kimi_k3_target_verify

    _set_config(SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]))
    assert gate() is True

    # Other GDN models share this bridge and must not reach K3 helpers.
    _set_config(SimpleNamespace(architectures=["Qwen3NextForCausalLM"]))
    assert gate() is False
    _set_config(None)
    assert gate() is False

    # No ATOM config set (e.g. dummy forward) must not raise.
    def _raise():
        raise AssertionError("Current atom config is not set")

    monkeypatch.setattr(attention_gdn, "get_current_atom_config", _raise)
    assert gate() is False
