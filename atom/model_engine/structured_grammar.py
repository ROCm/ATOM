# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strict JSON Schema compilation: unsupported constraints must never disappear."""

import json

import torch


class GrammarMatcher:
    def __init__(self, compiled):
        self.matcher = compiled.deep_copy()

    def copy(self):
        return GrammarMatcher(self.matcher)

    def accept_token(self, token):
        return self.matcher.consume_token(int(token))

    def is_terminated(self):
        return self.matcher.is_stopped()

    def fill_next_token_bitmask(self, mask):
        data = bytearray(self.matcher.compute_bitmask())
        # llguidance can include a padded trailing word. Only model-vocabulary
        # bits are consumed by the runner; backend-only padding has no logits.
        bits = torch.frombuffer(data, dtype=torch.int32)
        mask.copy_(bits[: mask.numel()].reshape_as(mask))
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())


def allocate_token_bitmask(batch_size, vocab_size):
    return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)


class GrammarCompiler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compile_json_schema(self, schema, tool_format=None):
        from llguidance import LLMatcher

        from .schema_normalization import normalize_schema

        value = normalize_schema(json.loads(schema))
        if isinstance(value, dict):
            # This is an API JSON Schema, not a backend configuration object.
            # In particular, clients must not enable lenient/coerce_one_of or
            # replace JSON punctuation through an x-guidance annotation.
            value.pop("x-guidance", None)
        grammar = LLMatcher.grammar_from_json_schema(
            value, overrides={"lenient": False, "coerce_one_of": False}
        )
        if tool_format:
            spec = json.loads(tool_format)
            prefix, section = spec["prefix"], spec["section"]
            # Only the response text is constrained by response_format. Normal
            # function calls use the parser's native envelope, not beta strict
            # argument schemas. Names and structural strings are quoted literals.
            q = json.dumps
            names = " | ".join(q(name) for name in spec["names"])
            root = "calls" if spec["required"] else "answer | calls"
            invoke_open = q("<" + prefix + 'invoke name="')
            name_end = q('">')
            value = (
                dict(
                    value, **{"x-guidance": {"lenient": False, "coerce_one_of": False}}
                )
                if isinstance(value, dict)
                else value
            )
            lark = (
                f"start: {root}\nanswer: %json {q(value)}\n"
                f"calls: {q('<' + prefix + section + '>')} invoke+ {q('</' + prefix + section + '>')}\n"
                f"invoke: {invoke_open} name {name_end} body\n"
                f"name: {names}\nbody[suffix={q('</' + prefix + 'invoke>')}]: /(.|\\n)*/\n%ignore /[ \\t\\r\\n]+/\n"
            )
            grammar = {"grammars": [{"lark_grammar": lark}]}
        matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise ValueError(matcher.get_error())
        warnings = matcher.get_grammar_warnings()
        if warnings:
            raise ValueError("; ".join(warnings))
        return matcher
