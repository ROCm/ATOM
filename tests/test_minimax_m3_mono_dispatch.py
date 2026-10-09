# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments and which steps the MiniMax-M3 mono decode takes."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.minimax_m3.mono import dispatch, runner


class _Mode:
    def __init__(self, piecewise):
        self.piecewise = piecewise

    def requires_piecewise_compilation(self):
        return self.piecewise


def _atom_config(**overrides):
    fields = {
        "tensor_parallel_size": 4,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "pipeline_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "fp8",
        "kv_cache_block_size": 128,
        "max_model_len": 32768,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "speculative_config": None,
        "compilation_config": SimpleNamespace(cudagraph_mode=_Mode(False)),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(dispatch, "is_vllm", lambda: False)
    monkeypatch.setattr(dispatch, "is_sglang", lambda: False)


def test_supported_deployment_has_no_refusal(native):
    assert dispatch._config_refusal(_atom_config(), SimpleNamespace()) is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tensor_parallel_size": 8}, "TP 8"),
        ({"parallel_config": SimpleNamespace(data_parallel_size=2)}, "DP"),
        ({"pipeline_parallel_size": 2}, "PP"),
        ({"kv_cache_dtype": "bf16"}, "kv cache"),
        ({"index_cache_dtype": "bf16"}, "index cache"),
        ({"kv_cache_block_size": 16}, "block size"),
        ({"max_model_len": (1 << 20) + 1}, "max_model_len"),
        ({"enable_tbo_decode": True}, "TBO"),
        (
            {"compilation_config": SimpleNamespace(cudagraph_mode=_Mode(True))},
            "piecewise",
        ),
    ],
)
def test_each_unsupported_setting_is_refused(native, overrides, reason):
    assert reason in dispatch._config_refusal(
        _atom_config(**overrides), SimpleNamespace()
    )


def test_the_model_s_full_context_is_served(native):
    cfg = _atom_config(max_model_len=1 << 20)
    assert dispatch._config_refusal(cfg, SimpleNamespace()) is None


def test_index_cache_dtype_falls_back_to_kv_cache_dtype(native):
    cfg = _atom_config(index_cache_dtype=None)
    assert dispatch._config_refusal(cfg, SimpleNamespace()) is None


def test_use_index_cache_is_refused(native):
    why = dispatch._config_refusal(
        _atom_config(), SimpleNamespace(use_index_cache=True)
    )
    assert why == "use_index_cache"


def test_plugin_mode_is_refused(native, monkeypatch):
    monkeypatch.setattr(dispatch, "is_sglang", lambda: True)
    assert dispatch._config_refusal(_atom_config(), SimpleNamespace()) == "plugin mode"


# ---------------------------------------------------------------- per-step predicate


def _decode_md(rows=4, slots=None, **overrides):
    """Sparse decode metadata with ``rows`` table rows (a size-``rows`` graph)."""
    decode = {
        "max_query_len": 1,
        "block_table": torch.zeros(rows, 8, dtype=torch.int32),
        "seq_lens": torch.ones(rows, dtype=torch.int32),
    }
    decode.update(overrides)
    return SimpleNamespace(
        sparse_attention_metadata=SimpleNamespace(
            decode=SimpleNamespace(**decode),
            num_prefills=0,
            slot_mapping=(
                slots if slots is not None else torch.zeros(rows, dtype=torch.int64)
            ),
        )
    )


def _forward_context(md=None, is_prefill=False, ubatch_slices=None):
    return SimpleNamespace(
        context=SimpleNamespace(is_prefill=is_prefill),
        ubatch_slices=ubatch_slices,
        attn_metadata=md if md is not None else _decode_md(),
    )


def _mono(monkeypatch, fwd):
    """An enabled MonoDecode whose runner already exists (no GPU work)."""
    mono = dispatch.MonoDecode.__new__(dispatch.MonoDecode)
    mono._lm = SimpleNamespace(model=SimpleNamespace(aux_hidden_state_layers=()))
    mono._runner = object()
    mono._enabled = True
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return mono


def _step(n, pos_dtype=torch.int64):
    """(input_ids, positions) of an n-token decode step."""
    return torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=pos_dtype)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 12, 16])
def test_decode_steps_of_up_to_max_tokens_are_taken(monkeypatch, n):
    md = _forward_context(_decode_md(rows=16))
    assert _mono(monkeypatch, md).supports(*_step(n), None, None)


def test_more_than_max_tokens_take_the_original_path(monkeypatch):
    md = _forward_context(_decode_md(rows=32))
    assert not _mono(monkeypatch, md).supports(*_step(17), None, None)


def test_check_mode_takes_the_same_steps(monkeypatch):
    monkeypatch.setenv("ATOM_MONO_CHECK", "1")
    mono = _mono(monkeypatch, _forward_context())
    assert mono.supports(*_step(4), None, None)


@pytest.mark.parametrize(
    "fwd",
    [
        _forward_context(is_prefill=True),
        _forward_context(ubatch_slices=[0, 1]),
        _forward_context(_decode_md(rows=2, max_query_len=2)),
        _forward_context(_decode_md(block_table=torch.zeros(4, 8, dtype=torch.int64))),
        _forward_context(_decode_md(seq_lens=torch.ones(4, dtype=torch.int64))),
        _forward_context(SimpleNamespace(sparse_attention_metadata=None)),
        _forward_context(_decode_md(block_table=torch.zeros(2, 8, dtype=torch.int32))),
        _forward_context(_decode_md(seq_lens=torch.ones(2, dtype=torch.int32))),
        _forward_context(
            _decode_md(block_table=torch.zeros(4, 16, dtype=torch.int32)[:, :8])
        ),
        _forward_context(_decode_md(slots=torch.zeros(4, dtype=torch.int32))),
        _forward_context(_decode_md(slots=torch.zeros(2, dtype=torch.int64))),
    ],
    ids=[
        "prefill",
        "tbo",
        "q_not_dividing_tokens",
        "bt_int64",
        "seqlens_int64",
        "no_sparse_md",
        "fewer_table_rows",
        "fewer_seq_lens",
        "strided_table_rows",
        "slots_int32",
        "fewer_slots",
    ],
)
def test_other_steps_take_the_original_path(monkeypatch, fwd):
    assert not _mono(monkeypatch, fwd).supports(*_step(3), None, None)


def test_batches_and_foreign_inputs_take_the_original_path(monkeypatch):
    mono = _mono(monkeypatch, _forward_context())
    assert not mono.supports(*_step(1, torch.int32), None, None)
    assert not mono.supports(*_step(1), object(), None)
    assert not mono.supports(*_step(1), None, torch.zeros(1, 8))


def test_speculative_decoding_is_not_refused(native):
    cfg = _atom_config(speculative_config=object())
    assert dispatch._config_refusal(cfg, SimpleNamespace()) is None


@pytest.mark.parametrize("q, reqs", [(2, 1), (2, 2), (3, 1), (4, 1)])
def test_speculative_verify_steps_are_taken(monkeypatch, q, reqs):
    slots = torch.zeros(q * reqs, dtype=torch.int64)
    md = _forward_context(_decode_md(rows=reqs, slots=slots, max_query_len=q))
    assert _mono(monkeypatch, md).supports(*_step(q * reqs), None, None)


def test_verify_with_fewer_request_rows_takes_the_original_path(monkeypatch):
    slots = torch.zeros(4, dtype=torch.int64)
    md = _forward_context(_decode_md(rows=1, slots=slots, max_query_len=2))
    assert not _mono(monkeypatch, md).supports(*_step(4), None, None)


def test_eagle3_aux_layers_are_taken(monkeypatch):
    mono = _mono(monkeypatch, _forward_context())
    mono._lm.model.aux_hidden_state_layers = (2, 30, 57)
    assert mono.supports(*_step(1), None, None)


def test_token_rows_expand_a_verify_to_one_row_per_token():
    bt = torch.arange(12, dtype=torch.int32).view(3, 4)
    md = _decode_md(
        max_query_len=3,
        block_table=bt,
        seq_lens=torch.tensor([300, 129, 0], dtype=torch.int32),
    )
    rows_bt, rows_sl = runner.token_rows(_forward_context(md), 6)
    assert torch.equal(rows_bt, bt[[0, 0, 0, 1, 1, 1]])
    assert rows_sl.tolist() == [298, 299, 300, 127, 128, 129]
    assert rows_bt.is_contiguous() and rows_sl.dtype == torch.int32


def test_token_rows_pass_plain_decode_tables_through():
    md = _decode_md()
    decode = md.sparse_attention_metadata.decode
    rows_bt, rows_sl = runner.token_rows(_forward_context(md), 4)
    assert rows_bt is decode.block_table and rows_sl is decode.seq_lens


def test_disabled_mono_never_takes_a_step(monkeypatch):
    mono = _mono(monkeypatch, _forward_context())
    mono._enabled = False
    assert not mono.supports(*_step(1), None, None)
