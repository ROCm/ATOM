//! MiniMax-M3 tool-call parser.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::{Map, Value};

use crate::tool_parser::{
    errors::{ParserError, ParserResult},
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

pub const MINIMAX_NS: &str = "]<]minimax[>[";

pub struct MiniMaxParser {
    invoke: Regex,
    parameter: Regex,
    buffer: String,
    emitted_calls: usize,
}

impl MiniMaxParser {
    pub fn new() -> Self {
        Self {
            invoke: Regex::new(
                r#"(?s)<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>|<invoke\s+name="([^"]+)"\s*>(.*)$"#,
            )
            .expect("valid MiniMax invoke pattern"),
            // Rust's regex engine does not support backreferences. Capture
            // both tag names and verify their equality while parsing.
            parameter: Regex::new(r"(?s)<([\w-]+)>(.*?)</([\w-]+)>")
                .expect("valid MiniMax parameter pattern"),
            buffer: String::new(),
            emitted_calls: 0,
        }
    }

    fn parse_value(value: &str) -> Value {
        serde_json::from_str(value.trim()).unwrap_or_else(|_| Value::String(value.to_string()))
    }

    fn parse_calls(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !text.contains(MINIMAX_NS) {
            return Ok((text.to_string(), Vec::new()));
        }
        let clean = text.replace(MINIMAX_NS, "");
        let tool_start = clean.find("<tool_call>");
        let content = match tool_start {
            Some(index) => clean[..index].trim().to_string(),
            None => clean.trim().to_string(),
        };
        let mut calls = Vec::new();
        for invoke in self.invoke.captures_iter(&clean) {
            let name = invoke
                .get(1)
                .or_else(|| invoke.get(3))
                .map(|capture| capture.as_str().trim())
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let body = invoke
                .get(2)
                .or_else(|| invoke.get(4))
                .map(|capture| capture.as_str())
                .unwrap_or_default();
            let mut args = Map::new();
            for parameter in self.parameter.captures_iter(body) {
                let key = parameter.get(1).expect("parameter name").as_str().trim();
                let value = parameter.get(2).expect("parameter body").as_str();
                let closing_tag = parameter
                    .get(3)
                    .expect("parameter closing tag")
                    .as_str()
                    .trim();
                if !key.is_empty() && key == closing_tag {
                    args.insert(key.to_string(), Self::parse_value(value));
                }
            }
            calls.push(ToolCall {
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::to_string(&args)
                        .map_err(|error| ParserError::ParsingFailed(error.to_string()))?,
                },
            });
        }
        Ok((content, calls))
    }
}

impl Default for MiniMaxParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MiniMaxParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_calls(output)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let (normal_text, calls) = self.parse_calls(&self.buffer)?;
        let mut result = StreamingParseResult::default();
        if self.emitted_calls == 0 {
            result.normal_text = normal_text;
        }
        for (index, call) in calls.into_iter().enumerate().skip(self.emitted_calls) {
            result.calls.push(ToolCallItem {
                tool_index: index,
                name: Some(call.function.name),
                parameters: call.function.arguments,
            });
            self.emitted_calls += 1;
        }
        Ok(result)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(MINIMAX_NS)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.emitted_calls = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{MiniMaxParser, MINIMAX_NS};
    use crate::tool_parser::traits::ToolParser;

    #[tokio::test]
    async fn parses_minimax_m3_namespace_format() {
        let parser = MiniMaxParser::new();
        let output = format!(
            "answer{ns}<tool_call>{ns}<invoke name=\"weather\">{ns}<city>Paris{ns}</city>{ns}</invoke>{ns}</tool_call>",
            ns = MINIMAX_NS
        );
        let (content, calls) = parser.parse_complete(&output).await.unwrap();

        assert_eq!(content, "answer");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }
}
