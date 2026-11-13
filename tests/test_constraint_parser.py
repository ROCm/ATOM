# SPDX-License-Identifier: MIT
"""Public interface adoption and dependency boundaries."""

import subprocess
import sys

import pytest


def test_rules_do_not_import_gpu_dependencies():
    script = "import sys; from atom.constraint_parser import ChannelRule, parse_channel_rule; assert not any(n in sys.modules for n in ('torch', 'triton', 'aiter'))"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_registered_provider_traverses_complete_lifecycle(monkeypatch):
    from types import SimpleNamespace

    import torch

    from atom.model_engine import token_constraints as boundary

    events = []

    class Provider:
        backend = "third"
        mask_stage = 0
        commit_order = 20

        @staticmethod
        def enabled(batch):
            return getattr(batch, "third", None) is not None

        @staticmethod
        def masks_source(metadata):
            return True

        @staticmethod
        def prepare(runner, batch, metadata):
            events.append(("prepare", batch.third))

        @staticmethod
        def apply(step):
            events.append(("apply", step.batch.third))
            step.source[:, 1] = -float("inf")

        @staticmethod
        def commit(step, tokens, batch, deferred):
            assert tokens.tolist() == [0]
            events.append(("commit", batch.third))

        @staticmethod
        def release(runner, ids):
            assert ids == [71]
            events.append(("release", None))

    monkeypatch.setattr(boundary, "_PROVIDERS", [])
    boundary.register_constraint_provider(Provider)
    native = object()
    batch = SimpleNamespace(third=native, total_seqs_num=1)
    assert boundary.has_constraints(batch)
    assert boundary.needs_constraint_snapshot(batch, None)
    boundary.prepare_token_constraints(None, batch, None)
    step = boundary.TokenConstraints.apply(None, batch, torch.zeros(1, 3), None)
    assert torch.isneginf(step.logits[0, 1])
    step.commit(torch.tensor([0]), batch, False)
    boundary.release_token_constraints(None, [71])
    assert events == [
        ("prepare", native),
        ("apply", native),
        ("commit", native),
        ("release", None),
    ]


class _CharTokenizer:
    """Each name in `whole` is one id; every other character is its own id."""

    def __init__(self, whole):
        self.whole = sorted(whole, key=len, reverse=True)
        self.ids = {}

    def _id(self, piece):
        return self.ids.setdefault(piece, len(self.ids) + 1000)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        out, i = [], 0
        while i < len(text):
            for name in self.whole:
                if text.startswith(name, i):
                    out.append(self._id(name))
                    i += len(name)
                    break
            else:
                out.append(self._id(text[i]))
                i += 1
        return out


K3_CONTROLS = ("<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>")


def test_k3_channel_rule_compiles_to_the_forced_opening():
    from atom.entrypoints.openai import kimi_k3_tokens as k3
    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

    tokenizer = _CharTokenizer(K3_CONTROLS)
    channel = KimiK3Parser.required_call_channel(tokenizer)
    assert channel.after == tuple(tokenizer.encode(k3.RESPONSE_END))
    assert channel.end_ids == (tokenizer.ids["<|end_of_msg|>"],)
    start = tokenizer.encode(k3.TOOLS_START)
    assert channel.opens_at == len(start)
    assert channel.opening[: len(start)] == tuple(start)


@pytest.mark.parametrize("split", K3_CONTROLS)
def test_k3_channel_is_not_held_when_a_control_token_is_split(split, caplog):
    import logging

    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser

    tokenizer = _CharTokenizer(set(K3_CONTROLS) - {split})
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert KimiK3Parser.required_call_channel(tokenizer) is None
    (record,) = caplog.records
    message = record.getMessage()
    assert message.startswith("Kimi-K3 tool calls:")
    assert split in message and 'tool_choice="required"' in message
