//! Kimi-K3's reasoning channel, as ATOM's Python server reads it
//! (`atom/entrypoints/openai/kimi_k3_tokens.py`, the channel dialect in
//! `reasoning_dialects.py`).
//!
//! A K3 reply is channels, each framed by literal markers:
//!
//! ```text
//! <|open|>think<|sep|> REASONING <|close|>think<|sep|>
//! <|open|>response<|sep|> ANSWER <|close|>response<|sep|>
//! <|open|>tools<|sep|> ... <|close|>tools<|sep|>
//! ```
//!
//! The think channel is the reasoning. The answer's framing is dropped from
//! the content; the tools section is left whole for the `kimi_k3` tool
//! parser. A think block the model never closed ends where the response or
//! tools channel opens. The prompt opens the first channel itself (think, or
//! response with thinking off), and the render feeds that opener first.

use crate::reasoning_parser::{ParseError, ParserResult, ReasoningParser};

const THINK_START: &str = "<|open|>think<|sep|>";
const THINK_END: &str = "<|close|>think<|sep|>";
const RESPONSE_START: &str = "<|open|>response<|sep|>";
const TOOLS_START: &str = "<|open|>tools<|sep|>";

/// What ends the think channel: its closer, or the next channel opening
/// when the model skipped the closer (`extra_end_markers`). Only the closer
/// is consumed; an opener is the answer's to read.
const THINK_ENDS: [&str; 3] = [THINK_END, RESPONSE_START, TOOLS_START];

/// Framing around the answer, dropped from it (`CHANNEL_FRAMING`), and the
/// think channel's opener, which enters reasoning.
const ANSWER_MARKERS: [&str; 6] = [
    THINK_START,
    RESPONSE_START,
    "<|close|>response<|sep|>",
    "<|open|>message<|sep|>",
    "<|close|>message<|sep|>",
    "<|end_of_msg|>",
];

/// The largest buffered text a reply may hold back, as the crate's parsers
/// bound theirs.
const MAX_BUFFER: usize = 65536;

#[derive(Default)]
pub(super) struct KimiK3Parser {
    /// Text not yet handed out: a marker may still be completing at its end.
    buffer: String,
    in_reasoning: bool,
}

impl KimiK3Parser {
    /// Hand out what `buffer` resolves; `flush` also hands out a held tail.
    fn drain(&mut self, flush: bool) -> ParserResult {
        let mut result = ParserResult::default();
        loop {
            let markers: &[&str] = if self.in_reasoning {
                &THINK_ENDS
            } else {
                &ANSWER_MARKERS
            };
            if let Some((at, marker)) = earliest(&self.buffer, markers) {
                let before = &self.buffer[..at];
                let consumed = if self.in_reasoning {
                    result.reasoning_text.push_str(before);
                    self.in_reasoning = false;
                    if marker == THINK_END {
                        at + marker.len()
                    } else {
                        at
                    }
                } else {
                    result.normal_text.push_str(before);
                    self.in_reasoning = marker == THINK_START;
                    at + marker.len()
                };
                self.buffer.drain(..consumed);
                continue;
            }
            let keep = if flush {
                0
            } else {
                held_tail(&self.buffer, markers)
            };
            let ready = self.buffer.len() - keep;
            let out = if self.in_reasoning {
                &mut result.reasoning_text
            } else {
                &mut result.normal_text
            };
            out.push_str(&self.buffer[..ready]);
            self.buffer.drain(..ready);
            return result;
        }
    }
}

/// The first of `markers` in `text`: its offset and which.
fn earliest<'m>(text: &str, markers: &[&'m str]) -> Option<(usize, &'m str)> {
    markers
        .iter()
        .filter_map(|m| text.find(m).map(|at| (at, *m)))
        .min_by_key(|(at, _)| *at)
}

/// The length of `text`'s longest suffix that may be the start of a marker.
fn held_tail(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|m| (1..m.len()).rev().map(move |n| &m[..n]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

impl ReasoningParser for KimiK3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.buffer.push_str(text);
        let result = self.drain(true);
        self.reset();
        Ok(ParserResult::new(
            result.normal_text.trim().to_string(),
            result.reasoning_text.trim().to_string(),
        ))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        if self.buffer.len() + text.len() > MAX_BUFFER {
            return Err(ParseError::BufferOverflow(self.buffer.len() + text.len()));
        }
        self.buffer.push_str(text);
        Ok(self.drain(false))
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.in_reasoning = false;
    }

    fn model_type(&self) -> &str {
        "kimi_k3"
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPLY: &str = "<|open|>think<|sep|>Let me think.<|close|>think<|sep|>\
        <|open|>response<|sep|>Hi there.<|close|>response<|sep|>";

    #[test]
    fn a_reply_splits_into_reasoning_and_the_unframed_answer() {
        let mut parser = KimiK3Parser::default();
        let result = parser.detect_and_parse_reasoning(REPLY).unwrap();
        assert_eq!(result.reasoning_text, "Let me think.");
        assert_eq!(result.normal_text, "Hi there.");

        // Streamed a few bytes at a time, from a prompt that opened think.
        let mut parser = KimiK3Parser::default();
        let (mut normal, mut reasoning) = (String::new(), String::new());
        let reply = &REPLY[THINK_START.len()..];
        let pieces = std::iter::once(THINK_START).chain(
            reply
                .as_bytes()
                .chunks(3)
                .map(|c| std::str::from_utf8(c).unwrap()),
        );
        for piece in pieces {
            let r = parser.parse_reasoning_streaming_incremental(piece).unwrap();
            normal.push_str(&r.normal_text);
            reasoning.push_str(&r.reasoning_text);
        }
        assert_eq!(
            (normal.as_str(), reasoning.as_str()),
            ("Hi there.", "Let me think.")
        );
    }

    #[test]
    fn an_unclosed_think_ends_at_the_tools_channel_which_stays_for_the_tool_parser() {
        let mut parser = KimiK3Parser::default();
        let tools = "<|open|>tools<|sep|><|open|>call tool=\"f\" index=\"0\"<|sep|>{}";
        let result = parser
            .detect_and_parse_reasoning(&format!("{THINK_START}plan{tools}"))
            .unwrap();
        assert_eq!(result.reasoning_text, "plan");
        assert_eq!(result.normal_text, tools);
    }
}
