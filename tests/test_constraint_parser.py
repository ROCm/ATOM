# SPDX-License-Identifier: MIT
"""Public interface adoption and dependency boundaries."""
import subprocess
import sys
from atom.constraint_parser import ConstraintParser, GrammarRule


def test_rules_do_not_import_gpu_dependencies():
    script = "import sys; from atom.constraint_parser import ConstraintParser, ChannelRule, GrammarRule; assert not any(n in sys.modules for n in ('torch', 'triton', 'aiter'))"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_schema_adapter_receives_unchanged_rule():
    class Compiler:
        def compile_json_schema(self, *args):
            assert args == ('{}', 'format', 'end')
            return result
    result = object()
    assert ConstraintParser.parse(GrammarRule('{}', 'format', 'end'), Compiler()) is result


def test_registry_uses_public_parser(monkeypatch):
    from atom.entrypoints.openai.tool_parser.kimi_k3_tool_parser import KimiK3Parser
    from atom.entrypoints.openai.tool_parser.registry import required_call_channel
    rule = KimiK3Parser.channel_rule()
    calls = []
    def parse(cls, actual, tokenizer):
        assert actual == rule
        calls.append(tokenizer)
        return result
    result, tokenizer = object(), object()
    monkeypatch.setattr(KimiK3Parser, 'parse', classmethod(parse))
    assert required_call_channel(KimiK3Parser, tokenizer) is result
    assert KimiK3Parser.required_call_channel(tokenizer) is result
    assert calls == [tokenizer, tokenizer]
