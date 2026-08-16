//! DeepSeek-V4 DSML tool-call parser.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::{Map, Value};

use crate::tool_parser::{
    errors::{ParserError, ParserResult},
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const DSML: &str = "｜DSML｜";

pub struct DsmlParser {
    invoke: Regex,
    parameter: Regex,
    buffer: String,
    emitted_calls: usize,
}

impl DsmlParser {
    pub fn new() -> Self {
        Self {
            invoke: Regex::new(
                r#"(?s)<(?:｜DSML｜)?invoke\s+name="([^"]+)"\s*(?:/>|>(.*?)</(?:｜DSML｜)?invoke>)"#,
            )
            .expect("valid DSML invoke pattern"),
            parameter: Regex::new(
                r#"(?s)<(?:｜DSML｜)?parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</(?:｜DSML｜)?parameter>"#,
            )
            .expect("valid DSML parameter pattern"),
            buffer: String::new(),
            emitted_calls: 0,
        }
    }

    fn start_index(text: &str) -> Option<usize> {
        [
            format!("<{DSML}tool_call"),
            format!("<{DSML}invoke"),
            "<invoke name=".to_string(),
            "<tool_calls>".to_string(),
        ]
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()
    }

    fn decode_value(value: &str, string_attr: Option<&str>) -> Value {
        match string_attr {
            Some("true") => Value::String(value.to_string()),
            Some("false") => {
                serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
            }
            _ => serde_json::from_str(value.trim())
                .unwrap_or_else(|_| Value::String(value.trim().to_string())),
        }
    }

    fn unwrap_arguments(mut args: Map<String, Value>) -> Map<String, Value> {
        for _ in 0..4 {
            if args.len() != 1 {
                break;
            }
            let Some((key, value)) = args.iter().next().map(|(k, v)| (k.clone(), v.clone())) else {
                break;
            };
            if key != "arguments" && key != "input" {
                break;
            }
            let inner = match value {
                Value::Object(value) => value,
                Value::String(value) => match serde_json::from_str(&value) {
                    Ok(Value::Object(value)) => value,
                    _ => break,
                },
                _ => break,
            };
            args = inner;
        }
        args
    }

    fn parse_calls(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        let Some(start) = Self::start_index(text) else {
            return Ok((text.to_string(), Vec::new()));
        };
        let mut calls = Vec::new();
        for captures in self.invoke.captures_iter(&text[start..]) {
            let name = captures
                .get(1)
                .expect("invoke name capture")
                .as_str()
                .to_string();
            let body = captures.get(2).map_or("", |capture| capture.as_str());
            let mut args = Map::new();
            for parameter in self.parameter.captures_iter(body) {
                let key = parameter.get(1).expect("parameter name").as_str();
                let value = parameter.get(3).expect("parameter body").as_str();
                args.insert(
                    key.to_string(),
                    Self::decode_value(value, parameter.get(2).map(|capture| capture.as_str())),
                );
            }
            if args.is_empty() {
                if let Ok(Value::Object(value)) = serde_json::from_str(body.trim()) {
                    args = value;
                }
            }
            let arguments = serde_json::to_string(&Self::unwrap_arguments(args))
                .map_err(|error| ParserError::ParsingFailed(error.to_string()))?;
            calls.push(ToolCall {
                function: FunctionCall { name, arguments },
            });
        }
        Ok((text[..start].trim().to_string(), calls))
    }
}

impl Default for DsmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DsmlParser {
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
        Self::start_index(text).is_some() || text.contains(DSML)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.emitted_calls = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::DsmlParser;
    use crate::tool_parser::traits::ToolParser;

    #[tokio::test]
    async fn parses_dsml_invoke_and_json_arguments() {
        let parser = DsmlParser::new();
        let (content, calls) = parser
            .parse_complete(
                "before<｜DSML｜tool_calls><｜DSML｜invoke name=\"weather\"><｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>",
            )
            .await
            .unwrap();

        assert_eq!(content, "before");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
    }
}
