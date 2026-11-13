# SPDX-License-Identifier: MIT
"""Public interface adoption and dependency boundaries."""

import subprocess
import sys
from typing import ClassVar

import pytest

from atom.constraint_parser import ConstraintParser


def test_rules_do_not_import_gpu_dependencies():
    script = "import sys; from atom.constraint_parser import ConstraintParser, ChannelRule; assert not any(n in sys.modules for n in ('torch', 'triton', 'aiter'))"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_registry_uses_public_parser(monkeypatch):
    from atom.constraint_parser import ChannelRule
    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
    from atom.entrypoints.openai.tool_parser.registry import required_call_channel

    calls = []

    def parse(cls, rule, tokenizer):
        assert isinstance(rule, ChannelRule)
        calls.append(tokenizer)
        return result

    result, tokenizer = object(), object()
    monkeypatch.setattr(ConstraintParser, "parse", classmethod(parse))
    assert required_call_channel(KimiK3Parser, tokenizer) is result
    assert KimiK3Parser.required_call_channel(tokenizer) is result
    assert calls == [tokenizer, tokenizer]


def test_registered_rule_parser():
    class Rule:
        pass

    class Parser(ConstraintParser):
        _adapters: ClassVar[dict] = dict(ConstraintParser._adapters)

    rule, context, native = Rule(), object(), object()

    def parse(actual, adapter):
        assert actual is rule and adapter is context
        return native

    Parser.register(Rule, parse)
    assert Parser.parse(rule, context) is native


def test_registered_provider_traverses_complete_lifecycle(monkeypatch):
    from types import SimpleNamespace

    import torch

    from atom.constraint_parser import ConstraintBinding
    from atom.model_engine import token_constraints as boundary

    events = []

    class Provider:
        backend = "third"
        mask_stage = 0
        commit_order = 20

        @staticmethod
        def hard_requests(binding, batch):
            return [0]

        @staticmethod
        def enabled(batch):
            return getattr(batch, "third", None) is not None

        @staticmethod
        def masks_source(metadata):
            return True

        @staticmethod
        def bind(batch):
            return (
                ConstraintBinding("third", batch.third)
                if Provider.enabled(batch)
                else None
            )

        @staticmethod
        def prepare(runner, batch, metadata, binding):
            events.append(("prepare", binding.value))

        @staticmethod
        def apply(step, binding):
            events.append(("apply", binding.value))
            step.source[:, 1] = -float("inf")

        @staticmethod
        def commit(step, tokens, batch, deferred, failure, binding):
            assert tokens.tolist() == [0]
            events.append(("commit", binding.value))

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
    step.commit(torch.tensor([0]), batch, False, None)
    boundary.release_token_constraints(None, [71])
    assert events == [
        ("prepare", native),
        ("apply", native),
        ("commit", native),
        ("release", None),
    ]


@pytest.mark.parametrize("spec", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_late_registered_source_mask_composes_with_channel(monkeypatch, spec, active):
    from types import SimpleNamespace

    import torch

    from atom.constraint_parser import ConstraintBinding
    from atom.model_engine import token_constraints as boundary
    from atom.model_ops.required_channel import constrain_logits_reference
    from atom.sampling_params import RequiredChannel

    class SourceMask:
        backend = "third-source"
        mask_stage = 0
        commit_order = 20

        @staticmethod
        def enabled(batch):
            return getattr(batch, "third", False)

        @staticmethod
        def masks_source(metadata):
            return True

        @staticmethod
        def hard_requests(binding, batch):
            return [0]

        @staticmethod
        def bind(batch):
            return ConstraintBinding("third-source", 7)

        @staticmethod
        def prepare(*args):
            pass

        @staticmethod
        def release(*args):
            pass

        @staticmethod
        def commit(*args):
            pass

        @staticmethod
        def apply(step, binding):
            kept = step.source[:, binding.value].clone()
            step.source.fill_(-float("inf"))
            step.source[:, binding.value] = kept

    monkeypatch.setattr(boundary, "_PROVIDERS", list(boundary._PROVIDERS))
    boundary.register_constraint_provider(SourceMask)
    channel = RequiredChannel(after=(1,), opening=(2,), opens_at=1, end_ids=(3,))
    states = torch.tensor([1 if active else -1])
    batch = SimpleNamespace(
        third=True,
        required_channel=channel,
        structured_requests=None,
        total_seqs_num=1,
    )
    metadata = (
        SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([1]),
            target_logits_indices=torch.tensor([0]),
            bonus_logits_indices=torch.tensor([1]),
            input_ids=torch.tensor([0, 0]),
        )
        if spec
        else None
    )

    def apply(runner, batch, bonus, metadata, target, preserve_masks):
        base = constrain_logits_reference(
            channel,
            states,
            bonus,
            preserve_masks=preserve_masks,
            target_logits=target,
            cu_num_draft_tokens=(
                None if metadata is None else metadata.cu_num_draft_tokens
            ),
            input_ids=None if metadata is None else metadata.input_ids,
            target_logits_indices=(
                None if metadata is None else metadata.target_logits_indices
            ),
        )
        return object(), base

    monkeypatch.setattr(boundary, "apply_required_channel", apply)
    step = boundary.TokenConstraints.apply(
        None, batch, torch.zeros(2 if spec else 1, 8), metadata
    )
    if active:
        assert step.failed(
            torch.tensor([[0, -1]]) if spec else torch.tensor([0])
        ).tolist() == [True]
    else:
        assert step.logits.argmax(-1).tolist() == [7]
        if spec:
            assert step.target.argmax(-1).tolist() == [7]
        assert not step.failed(
            torch.tensor([[7, -1]]) if spec else torch.tensor([7])
        ).any()


def test_target_only_failure_tracks_accepted_prefix():
    from types import SimpleNamespace

    import torch

    from atom.model_engine.token_constraints import TokenConstraints

    step = TokenConstraints()
    step.logits = torch.zeros(1, 8)
    step.invalid = None
    step.target_invalid = torch.tensor([False, True])
    step.metadata = SimpleNamespace(cu_num_draft_tokens=torch.tensor([2]))
    assert step.failed(torch.tensor([[7, -1, -1]])).tolist() == [False]
    assert step.failed(torch.tensor([[7, 3, -1]])).tolist() == [True]


@pytest.mark.parametrize("inplace", [False, True])
def test_provider_failure_reports_preserve_previous_failures(monkeypatch, inplace):
    from types import SimpleNamespace

    import torch

    from atom.constraint_parser import ConstraintBinding
    from atom.model_engine import token_constraints as boundary

    class First:
        backend = "first"
        mask_stage = 0
        commit_order = 0

        @staticmethod
        def enabled(batch):
            return True

        @staticmethod
        def bind(batch):
            return ConstraintBinding("first", None)

        @staticmethod
        def hard_requests(binding, batch):
            return []

        @staticmethod
        def apply(step, binding):
            step.invalid = torch.tensor([True, False])
            step.target_invalid = torch.tensor([False, True])

    class Second(First):
        backend = "second"

        @staticmethod
        def apply(step, binding):
            invalid = torch.tensor([False, True])
            target_invalid = torch.tensor([True, False])
            if inplace:
                step.invalid.copy_(invalid)
                step.target_invalid.copy_(target_invalid)
            else:
                step.invalid = invalid
                step.target_invalid = target_invalid

    monkeypatch.setattr(boundary, "_PROVIDERS", [])
    boundary.register_constraint_provider(First)
    boundary.register_constraint_provider(Second)
    batch = SimpleNamespace(total_seqs_num=2)
    metadata = SimpleNamespace(
        bonus_logits_indices=torch.tensor([0, 2]),
        target_logits_indices=torch.tensor([1, 3]),
        cu_num_draft_tokens=torch.tensor([1, 2]),
    )
    step = boundary.TokenConstraints.apply(None, batch, torch.zeros(4, 8), metadata)
    assert step.invalid.tolist() == [True, True]
    assert step.target_invalid.tolist() == [True, True]
    assert step.failed(torch.tensor([[3, -1], [4, -1]])).tolist() == [True, True]


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
