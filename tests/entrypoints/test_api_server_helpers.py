# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for helpers in ``atom.entrypoints.openai.api_server`` that do
not require a GPU or a running engine.

The ``api_server`` module pulls in transformers + uvicorn + fastapi + an
engine-ready ``atom`` package at import time. The repo's ``tests/conftest.py``
already stubs several heavy imports; here we only test small pure-python
helpers, so if any transitive dependency is unavailable we skip the module
rather than block the rest of the suite.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest


def _install_api_server_stubs() -> list[str]:
    """Ensure attribute access ``atom.SamplingParams`` works under the stubbed
    ``atom`` package that ``tests/conftest.py`` installs, and stub any heavy
    transitive deps (``aiter``-backed engine core manager and its argparse
    helper) that ``api_server`` would otherwise drag in at import time.

    Stubs are only installed when the corresponding real module cannot be
    imported in this environment (e.g. Windows without ``aiter``). Any
    module we inject here is recorded and torn down in a module-level
    fixture so we don't leak stubs into tests that run later and expect
    the real implementation (notably ``tests/test_arg_utils_spec.py``).
    """
    import importlib

    from atom.sampling_params import SamplingParams  # real implementation

    atom_pkg = sys.modules.get("atom")
    if atom_pkg is not None and not hasattr(atom_pkg, "SamplingParams"):
        atom_pkg.SamplingParams = SamplingParams  # type: ignore[attr-defined]

    injected: list[str] = []

    def _try_import_else_stub(mod_name: str, attr_name: str, stub_cls) -> None:
        if mod_name in sys.modules:
            return
        try:
            importlib.import_module(mod_name)
        except Exception:
            stub = types.ModuleType(mod_name)
            setattr(stub, attr_name, stub_cls)
            sys.modules[mod_name] = stub
            injected.append(mod_name)

    class _StubCoreManager:  # noqa: D401 - placeholder
        def __init__(self, *a, **kw):
            pass

        def add_request(self, reqs):
            return None

    class _StubEngineArgs:  # noqa: D401 - placeholder
        @classmethod
        def add_cli_args(cls, parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine(self, tokenizer=None):
            return None

    _try_import_else_stub(
        "atom.model_engine.engine_core_mgr", "CoreManager", _StubCoreManager
    )
    _try_import_else_stub("atom.model_engine.arg_utils", "EngineArgs", _StubEngineArgs)
    return injected


try:
    _injected_modules = _install_api_server_stubs()
    import importlib

    api_server = importlib.import_module("atom.entrypoints.openai.api_server")
except Exception as exc:  # pragma: no cover - environment-dependent skip
    api_server = None  # type: ignore[assignment]
    _import_error = exc
    _injected_modules = []
else:
    _import_error = None
finally:
    # Remove any stubs we injected so tests collected *after* this module
    # (notably ``tests/test_arg_utils_spec.py``) can still import the real
    # ``atom.model_engine.arg_utils`` / ``engine_core_mgr``. ``api_server``
    # already bound the names it needed at module import time.
    for _mod_name in list(_injected_modules):
        sys.modules.pop(_mod_name, None)
    _injected_modules = []


pytestmark = pytest.mark.skipif(
    api_server is None,
    reason=f"api_server import unavailable: {_import_error!r}",
)


class TestCoerceN:
    """``_coerce_n`` normalizes the request ``n`` before engine fan-out."""

    def test_none_becomes_one(self):
        assert api_server._coerce_n(None, 0.8) == 1

    def test_zero_becomes_one(self):
        assert api_server._coerce_n(0, 0.8) == 1

    def test_negative_becomes_one(self):
        assert api_server._coerce_n(-2, 0.8) == 1

    def test_non_int_string_becomes_one(self):
        assert api_server._coerce_n("not-a-number", 0.8) == 1  # type: ignore[arg-type]

    def test_n_passes_through_when_temperature_positive(self):
        assert api_server._coerce_n(4, 0.7) == 4

    def test_n_collapses_to_one_under_greedy_sampling(self):
        # temperature==0 => greedy, so n>1 would produce identical siblings.
        assert api_server._coerce_n(4, 0.0) == 1

    def test_n_collapses_to_one_when_temperature_missing(self):
        assert api_server._coerce_n(4, None) == 1

    def test_n_one_with_greedy_stays_one(self):
        assert api_server._coerce_n(1, 0.0) == 1


class TestStreamDecodeDelta:
    def setup_method(self):
        api_server._stream_decode_states.clear()
        api_server._stream_queues.clear()

    def teardown_method(self):
        api_server._stream_decode_states.clear()
        api_server._stream_queues.clear()

    def test_withholds_replacement_char_until_sequence_resolves(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 4

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
                assert skip_special_tokens is True
                vocab = {1: "A", 2: "\ufffd", 3: "\u00e9"}
                return [vocab[token_id] for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                if tokens == ["A", "\ufffd", "\u00e9"]:
                    return "A\u00e9"
                if tokens == ["\ufffd", "\u00e9"]:
                    return "\u00e9"
                return "".join(tokens)

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())

        assert api_server._decode_stream_delta(("req", 0), [1]) == "A"
        assert api_server._decode_stream_delta(("req", 0), [2]) == ""
        assert api_server._decode_stream_delta(("req", 0), [3]) == "\u00e9"

    def test_keeps_decode_state_bucketed_by_request(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 3

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
                return [chr(ord("a") + token_id) for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())

        assert api_server._decode_stream_delta(("req-a", 0), [0]) == "a"
        assert api_server._decode_stream_delta(("req-a", 1), [1]) == "b"
        assert api_server._decode_stream_delta(("req-b", 0), [2]) == "c"

        assert set(api_server._stream_decode_states) == {"req-a", "req-b"}
        assert set(api_server._stream_decode_states["req-a"]) == {0, 1}

    def test_init_decode_state_from_prompt_tokens(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 5

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
                return [chr(ord("a") + token_id) for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())

        api_server._init_stream_decode_state("req", 0, [0, 1, 2])
        state = api_server._stream_decode_states["req"][0]
        assert state["tokens"] == ["a", "b", "c"]
        assert state["prefix_offset"] == 0
        assert state["read_offset"] == 3

        assert api_server._decode_stream_delta(("req", 0), [3]) == "d"
        assert state["tokens"] == ["d"]
        assert state["prefix_offset"] == 0
        assert state["read_offset"] == 1

    def test_long_decode_prunes_state_without_changing_output(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 8

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
                return [str(token_id) for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                return "|".join(tokens)

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())

        baseline_state = {
            "tokens": [],
            "prefix_offset": 0,
            "read_offset": 0,
        }
        expected_chunks = []
        for token_id in [idx % 8 for idx in range(100)]:
            new_tokens, decoded_text, prefix_offset, read_offset = (
                api_server._detokenize_incrementally(
                    tokenizer=api_server.tokenizer,
                    new_token_id=token_id,
                    prev_tokens=baseline_state["tokens"],
                    prefix_offset=baseline_state["prefix_offset"],
                    read_offset=baseline_state["read_offset"],
                    skip_special_tokens=True,
                )
            )
            baseline_state["tokens"].extend(new_tokens)
            baseline_state["prefix_offset"] = prefix_offset
            baseline_state["read_offset"] = read_offset
            expected_chunks.append(decoded_text)

        actual_chunks = [
            api_server._decode_stream_delta(("req", 0), [idx % 8]) for idx in range(100)
        ]
        state = api_server._stream_decode_states["req"][0]

        assert actual_chunks == expected_chunks
        assert len(state["tokens"]) <= 1
        assert state["prefix_offset"] == 0
        assert state["read_offset"] <= 1
        assert "token_ids" not in state
        assert "output_text" not in state

    def test_cleanup_removes_request_decode_bucket(self, monkeypatch):
        api_server._stream_decode_states["req-a"] = {0: {}, 1: {}}
        api_server._stream_decode_states["req-b"] = {0: {}}
        fake_engine = SimpleNamespace(
            io_processor=SimpleNamespace(requests={11: "seq-a", 12: "seq-b"})
        )
        monkeypatch.setattr(api_server, "engine", fake_engine)

        api_server.cleanup_streaming_request("req-a", 11)

        assert "req-a" not in api_server._stream_decode_states
        assert "req-b" in api_server._stream_decode_states
        assert 11 not in fake_engine.io_processor.requests
        assert 12 in fake_engine.io_processor.requests

    def test_send_stream_chunk_direct_decodes_on_event_loop(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 1

            def get_added_vocab(self):
                return {}

            def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
                return [chr(ord("a") + token_id) for token_id in token_ids]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)

        async def run_stream_callback():
            stream_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            api_server._stream_queues["req"] = stream_queue
            request_output = SimpleNamespace(
                output_tokens=[0],
                finished=False,
                finish_reason=None,
            )

            api_server._send_stream_chunk_direct(
                request_output,
                "req",
                stream_queue,
                loop,
            )
            assert "req" not in api_server._stream_decode_states
            return await asyncio.wait_for(stream_queue.get(), timeout=1.0)

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())

        chunk = asyncio.run(run_stream_callback())

        assert chunk["text"] == "a"
        assert api_server._stream_decode_states["req"][0]["tokens"] == ["a"]

    def test_direct_enqueue_skips_cleaned_request(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 1

            def get_added_vocab(self):
                return {}

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())
        stream_queue = asyncio.Queue()

        api_server._enqueue_decoded_stream_chunk_direct(
            "req", [0], False, None, None, stream_queue
        )

        assert "req" not in api_server._stream_decode_states
        assert stream_queue.empty()

    def test_tagged_enqueue_skips_cleaned_request(self, monkeypatch):
        class FakeTokenizer:
            is_fast = True

            def __len__(self):
                return 1

            def get_added_vocab(self):
                return {}

        monkeypatch.setattr(api_server, "tokenizer", FakeTokenizer())
        stream_queue = asyncio.Queue()

        api_server._enqueue_decoded_stream_chunk_tagged(
            "req", 1, [0], False, None, None, stream_queue
        )

        assert "req" not in api_server._stream_decode_states
        assert stream_queue.empty()


class TestBuildSamplingParams:
    """``_build_sampling_params`` threads ``n`` into SamplingParams."""

    def test_default_n_is_one(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
        )
        assert sp.n == 1

    def test_n_greater_than_one_propagates(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
            n=4,
        )
        assert sp.n == 4

    def test_invalid_n_rejected_by_sampling_params(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            api_server._build_sampling_params(
                temperature=0.8,
                max_tokens=16,
                stop_strings=None,
                ignore_eos=False,
                n=0,
            )
