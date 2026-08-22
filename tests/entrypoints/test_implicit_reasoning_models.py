# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Models that begin inside the reasoning channel with no marker anywhere.

DeepSeek-R1 emits `</think>` but neither its prompt nor its output carries
`<think>`. Nothing in a single response says so — its first token is already
reasoning and reads like an answer — so the fact has to be known before the
response starts, and the chat template is what knows it.

vLLM expresses the same fact by registering `DeepSeekR1ReasoningParser`, whose
only job is to override the streaming branch so a stream with no start token
counts as reasoning until the end marker. Registering a class per family and
reading the template are two spellings of one decision; this is the second.
"""

from __future__ import annotations

import pytest

from atom.entrypoints.openai.reasoning import (
    ReasoningFilter,
    separate_reasoning,
    template_opens_reasoning_implicitly,
)

# Shapes taken from the real templates on this box, reduced to what decides.
R1 = "...{{ content.split('</think>')|last }}...<｜Assistant｜>"
QWEN = "...<think>\n{{ reasoning }}\n</think>...<|im_start|>assistant\n<think>\n"
MINIMAX = "...[e~[\\n]~b]ai\\n..."


class TestTheRule:
    def test_a_template_that_closes_what_it_never_opens(self):
        assert template_opens_reasoning_implicitly(R1)

    def test_a_template_that_opens_its_own_does_not_count(self):
        """Qwen mentions both, so its model emits the opener itself.

        Treating it as implicit would put the model's own `<think>` inside the
        reasoning text and start every plain answer in the wrong channel.
        """
        assert not template_opens_reasoning_implicitly(QWEN)

    @pytest.mark.parametrize("template", [MINIMAX, "", "plain assistant template"])
    def test_a_template_with_no_reasoning_channel_does_not_count(self, template):
        assert not template_opens_reasoning_implicitly(template)


class TestWhatItBuys:
    RAW = "Let me work it out.</think>\n\n2 + 2 = 4."

    def test_unseeded_and_unflagged_the_end_marker_is_just_text(self):
        """The default, unchanged: nothing opened a channel, so nothing closed one."""
        reasoning, content = separate_reasoning(self.RAW)
        assert reasoning is None and content == self.RAW

    def test_flagged_the_reasoning_is_recovered(self):
        reasoning, content = separate_reasoning(self.RAW, starts_thinking=True)
        assert reasoning == "Let me work it out."
        assert content == "\n\n2 + 2 = 4."

    def test_streaming_and_non_streaming_agree_once_flagged(self):
        """The flag is one value read by both paths, so they cannot diverge."""
        reasoning, content = separate_reasoning(self.RAW, starts_thinking=True)

        rf = ReasoningFilter(starts_thinking=True)
        segments = []
        for i in range(0, len(self.RAW), 4):
            segments += rf.process(self.RAW[i : i + 4])
        segments += rf.flush()
        streamed_r = "".join(s for f, s in segments if f == "reasoning_content")
        streamed_c = "".join(s for f, s in segments if f == "content")

        # Compared byte-for-byte. This used to `.strip()` the streamed side
        # to make the two agree, which is what a divergence looks like when a
        # test is written around it: `content` was `"2 + 2 = 4."` here and
        # `"\n\n2 + 2 = 4."` on the wire.
        assert (reasoning or "") == streamed_r
        assert content == streamed_c


class TestAskingForNoThinkingOnlyCountsWhereItCanBeHonoured:
    """`thinking: disabled` reaches the model through the chat template's own
    switch. A template with no such switch cannot carry it.

    DeepSeek-R1 is that model: it begins inside the reasoning channel with no
    marker, and `resolve_reasoning_toggle` answers `None` for it. Asking it not
    to think puts nothing in the prompt, so it reasons exactly as always --
    and believing the request anyway stopped the channel being separated, so
    the client got the chain of thought and a literal `</think>` inside
    `content`. Reasoning that was asked not to happen and happened anyway is
    still reasoning; `anthropic_drop_reasoning` exists to withhold it, and it
    can only withhold what was separated.
    """

    ANSWER = "The user wants the capital.</think>Paris."

    @staticmethod
    def _channel(toggle, thinking_off):
        import atom.entrypoints.openai.api_server as api
        from atom.entrypoints.openai.reasoning_dialects import resolve_dialect

        before = (
            api.reasoning_dialect,
            api.model_starts_in_reasoning,
            api.reasoning_toggle,
        )
        try:
            api.reasoning_dialect, _ = resolve_dialect("<think></think>")
            api.model_starts_in_reasoning = True
            api.reasoning_toggle = toggle
            return api.reasoning_channel(False, thinking_off=thinking_off)
        finally:
            (
                api.reasoning_dialect,
                api.model_starts_in_reasoning,
                api.reasoning_toggle,
            ) = before

    def test_a_model_with_no_switch_is_separated_anyway(self):
        channel = self._channel(None, thinking_off=True)
        assert channel.split(self.ANSWER) == ("The user wants the capital.", "Paris.")

    def test_a_model_with_a_switch_takes_the_request_at_its_word(self):
        channel = self._channel(("enable_thinking", False, True), thinking_off=True)
        assert channel.split(self.ANSWER) == (None, self.ANSWER)

    @pytest.mark.parametrize(
        "toggle", [None, ("enable_thinking", False, True)], ids=["no-switch", "switch"]
    )
    def test_an_unstated_request_always_separates(self, toggle):
        """Absent means unstated, and unstated leaves the model's own default
        alone -- at this layer as at the prompt layer."""
        channel = self._channel(toggle, thinking_off=False)
        assert channel.split(self.ANSWER) == ("The user wants the capital.", "Paris.")
