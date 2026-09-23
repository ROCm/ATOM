//! MiniMax-M3 tool-call parser.
//!
//! Wire format (ATOM MiniMax M3) is preserved:
//! `]<]minimax[>[` namespace tokens interleaved with
//! `<tool_call><invoke name="...">` + `<param>value</param>` tags.
//!
//! Streaming/schema behavior is adapted from SMG `minimax_m2.rs` ideas only
//! (coercion, tool-name checks, incremental args, reset/boundary handling).

use std::{collections::HashMap, fmt::Write as FmtWrite};

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::{Map, Value};

use crate::tool_parser::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

pub const MINIMAX_NS: &str = "]<]minimax[>[";

pub struct MiniMaxParser {
    invoke: Regex,
    /// Complete invoke only (streaming); incomplete trailing invoke is ignored.
    invoke_complete: Regex,
    invoke_name_attr: Regex,
    parameter: Regex,
    buffer: String,
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    streamed_args_for_tool: Vec<String>,
    current_function_name: String,
    current_parameters: HashMap<String, Value>,
    in_tool_call: bool,
    function_name_sent: bool,
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,
    invoke_end_token: &'static str,
}

impl MiniMaxParser {
    pub fn new() -> Self {
        Self {
            invoke: Regex::new(
                r#"(?s)<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>|<invoke\s+name="([^"]+)"\s*>(.*)$"#,
            )
            .expect("valid MiniMax invoke pattern"),
            invoke_complete: Regex::new(r#"(?s)<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>"#)
                .expect("valid MiniMax complete invoke pattern"),
            invoke_name_attr: Regex::new(r#"name="([^"]+)""#)
                .expect("valid MiniMax invoke name attribute pattern"),
            // Rust's regex engine does not support backreferences. Capture
            // both tag names and verify their equality while parsing.
            parameter: Regex::new(r"(?s)<([\w-]+)>(.*?)</([\w-]+)>")
                .expect("valid MiniMax parameter pattern"),
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            current_function_name: String::new(),
            current_parameters: HashMap::new(),
            in_tool_call: false,
            function_name_sent: false,
            tool_call_start_token: "<tool_call>",
            tool_call_end_token: "</tool_call>",
            invoke_end_token: "</invoke>",
        }
    }

    /// Infer a JSON value when no schema type is available.
    fn parse_value(text: &str) -> Value {
        let text = text.trim();
        match text {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }
        if let Ok(num) = text.parse::<i64>() {
            return Value::Number(num.into());
        }
        if let Ok(num) = text.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(num) {
                return Value::Number(n);
            }
        }
        serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
    }

    fn decode_xml_entities(text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
    }

    fn coerce_parameter(value_str: &str, declared_type: Option<&str>) -> Value {
        let decoded = Self::decode_xml_entities(value_str);
        helpers::coerce_by_schema_type(&decoded, declared_type).unwrap_or_else(|| {
            if decoded.starts_with('{') || decoded.starts_with('[') {
                serde_json::from_str::<Value>(&decoded)
                    .unwrap_or_else(|_| Self::parse_value(&decoded))
            } else {
                Self::parse_value(&decoded)
            }
        })
    }

    fn parse_parameters(
        &self,
        params_text: &str,
        param_types: &HashMap<String, String>,
    ) -> Map<String, Value> {
        let mut parameters = Map::new();
        for parameter in self.parameter.captures_iter(params_text) {
            let key = parameter.get(1).expect("parameter name").as_str().trim();
            let value = parameter.get(2).expect("parameter body").as_str();
            let closing_tag = parameter
                .get(3)
                .expect("parameter closing tag")
                .as_str()
                .trim();
            if key.is_empty() || key != closing_tag {
                continue;
            }
            // Skip structural tags that can appear if the body is still partial.
            if key == "invoke" || key == "tool_call" {
                continue;
            }
            parameters.insert(
                key.to_string(),
                Self::coerce_parameter(value, param_types.get(key).map(String::as_str)),
            );
        }
        parameters
    }

    fn strip_complete_namespace_tokens(buffer: &mut String) {
        while let Some(pos) = buffer.find(MINIMAX_NS) {
            buffer.replace_range(pos..pos + MINIMAX_NS.len(), "");
        }
    }

    fn parse_calls(&self, text: &str, tools: &[Tool]) -> ParserResult<(String, Vec<ToolCall>)> {
        if !text.contains(MINIMAX_NS) {
            return Ok((text.to_string(), Vec::new()));
        }
        let clean = text.replace(MINIMAX_NS, "");
        let tool_start = clean.find(self.tool_call_start_token);
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
            let param_types = helpers::param_types_for_function(tools, name);
            let args = self.parse_parameters(body, &param_types);
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

    /// Stream newly completed `<tag>…</tag>` parameters as JSON fragments.
    fn parse_and_stream_parameters(&mut self, text: &str, tools: &[Tool]) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

        let mut new_params = HashMap::new();
        for parameter in self.parameter.captures_iter(text) {
            let key = parameter.get(1).expect("parameter name").as_str().trim();
            let value = parameter.get(2).expect("parameter body").as_str();
            let closing_tag = parameter
                .get(3)
                .expect("parameter closing tag")
                .as_str()
                .trim();
            if key.is_empty() || key != closing_tag || key == "invoke" || key == "tool_call" {
                continue;
            }
            new_params.insert(
                key.to_string(),
                Self::coerce_parameter(value, param_types.get(key).map(String::as_str)),
            );
        }

        if new_params.is_empty() || new_params == self.current_parameters {
            return calls;
        }

        let tool_id = self.current_tool_id as usize;
        while self.streamed_args_for_tool.len() <= tool_id {
            self.streamed_args_for_tool.push(String::new());
        }

        if self.current_parameters.is_empty() {
            let mut json_fragment = String::with_capacity(256);
            json_fragment.push('{');
            let mut first = true;
            for (key, value) in &new_params {
                if !first {
                    json_fragment.push_str(", ");
                }
                let key_json = serde_json::to_string(key).unwrap_or_default();
                let value_json = serde_json::to_string(value).unwrap_or_default();
                let _ = write!(&mut json_fragment, "{key_json}: {value_json}");
                first = false;
            }
            calls.push(ToolCallItem {
                tool_index: tool_id,
                name: None,
                parameters: json_fragment.clone(),
            });
            self.streamed_args_for_tool[tool_id] = json_fragment;
        } else {
            let new_keys: Vec<_> = new_params
                .keys()
                .filter(|k| !self.current_parameters.contains_key(*k))
                .cloned()
                .collect();
            if !new_keys.is_empty() {
                let mut json_fragment = String::with_capacity(128);
                for key in &new_keys {
                    let value = &new_params[key];
                    let key_json = serde_json::to_string(key).unwrap_or_default();
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    let _ = write!(&mut json_fragment, ", {key_json}: {value_json}");
                }
                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: json_fragment.clone(),
                });
                self.streamed_args_for_tool[tool_id].push_str(&json_fragment);
            }
        }

        self.current_parameters = new_params;
        while self.prev_tool_call_arr.len() <= tool_id {
            self.prev_tool_call_arr.push(Value::Null);
        }
        self.prev_tool_call_arr[tool_id] = serde_json::json!({
            "name": self.current_function_name,
            "arguments": self.current_parameters,
        });

        calls
    }

    fn close_streamed_json_if_needed(&mut self, calls: &mut Vec<ToolCallItem>) {
        let tool_id = self.current_tool_id as usize;
        if tool_id >= self.streamed_args_for_tool.len() {
            return;
        }
        let current_streamed = &self.streamed_args_for_tool[tool_id];
        if current_streamed.is_empty() || current_streamed.ends_with('}') {
            return;
        }
        let open_braces = current_streamed.matches('{').count();
        let close_braces = current_streamed.matches('}').count();
        if open_braces > close_braces {
            calls.push(ToolCallItem {
                tool_index: tool_id,
                name: None,
                parameters: "}".to_string(),
            });
            self.streamed_args_for_tool[tool_id].push('}');
        }
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
        self.parse_calls(output, &[])
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_calls(output, tools)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        Self::strip_complete_namespace_tokens(&mut self.buffer);

        let mut normal_text = String::new();
        let mut calls = Vec::new();
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Hold back a trailing partial namespace token or partial `<tool_call>`.
                let hold = helpers::ends_with_partial_token(&self.buffer, MINIMAX_NS)
                    .or_else(|| {
                        helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                    })
                    .unwrap_or(0);
                if hold > 0 {
                    let end = self.buffer.len() - hold;
                    normal_text = self.buffer[..end].to_string();
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    normal_text.clone_from(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            if !self.in_tool_call {
                if let Some(start) = self.buffer.find(self.tool_call_start_token) {
                    normal_text = self.buffer[..start].to_string();
                    self.buffer =
                        self.buffer[start + self.tool_call_start_token.len()..].to_string();
                    self.in_tool_call = true;
                    self.function_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    continue;
                }
                break;
            }

            // Inside a `<tool_call>` wrapper.
            if !self.function_name_sent {
                if let Some(end_pos) = self.buffer.find(self.tool_call_end_token) {
                    let next_invoke = self.buffer.find("<invoke");
                    if next_invoke.is_none_or(|i| end_pos < i) {
                        self.buffer =
                            self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                        self.in_tool_call = false;
                        self.current_function_name.clear();
                        self.current_parameters.clear();
                        continue;
                    }
                }

                // Prefer a complete invoke match so we do not treat a partial
                // opening tag as done; fall back to name-attr scan only when
                // the opening `>` of `<invoke …>` is already present.
                let function_name =
                    if let Some(captures) = self.invoke_complete.captures(&self.buffer) {
                        captures
                            .get(1)
                            .map_or("", |m| m.as_str())
                            .trim()
                            .to_string()
                    } else if let Some(open_end) = self
                        .buffer
                        .find("<invoke")
                        .and_then(|start| self.buffer[start..].find('>').map(|rel| start + rel))
                    {
                        let open = &self.buffer[..=open_end];
                        self.invoke_name_attr
                            .captures(open)
                            .and_then(|c| c.get(1).map(|m| m.as_str().trim().to_string()))
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };

                if function_name.is_empty() {
                    break;
                }

                // Tool-name validation when a tool list is provided.
                if !tools.is_empty() && !tool_indices.contains_key(&function_name) {
                    if let Some(invoke_end) = self.buffer.find(self.invoke_end_token) {
                        self.buffer =
                            self.buffer[invoke_end + self.invoke_end_token.len()..].to_string();
                        self.current_function_name.clear();
                        self.current_parameters.clear();
                        continue;
                    }
                    break;
                }

                self.current_function_name.clone_from(&function_name);
                self.function_name_sent = true;

                if self.current_tool_id == -1 {
                    self.current_tool_id = 0;
                }
                helpers::ensure_capacity(
                    self.current_tool_id,
                    &mut self.prev_tool_call_arr,
                    &mut self.streamed_args_for_tool,
                );

                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(function_name),
                    parameters: String::new(),
                });

                if let Some(pos) = self.buffer.find('>') {
                    // Advance past the opening `<invoke …>` only.
                    self.buffer = self.buffer[pos + 1..].to_string();
                }
                continue;
            }

            let buffer_copy = self.buffer.clone();
            let parameter_calls = self.parse_and_stream_parameters(&buffer_copy, tools);
            calls.extend(parameter_calls);

            if let Some(invoke_end) = self.buffer.find(self.invoke_end_token) {
                self.close_streamed_json_if_needed(&mut calls);
                self.buffer = self.buffer[invoke_end + self.invoke_end_token.len()..].to_string();
                self.function_name_sent = false;
                self.current_function_name.clear();
                self.current_parameters.clear();
                self.current_tool_id += 1;
                continue;
            }
            break;
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(MINIMAX_NS)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.in_tool_call = false;
        self.function_name_sent = false;
    }
}

#[cfg(test)]
mod tests {
    use super::{MiniMaxParser, MINIMAX_NS};
    use crate::tool_parser::traits::ToolParser;
    use openai_protocol::common::{Function, Tool};
    use serde_json::json;

    fn m3(parts: &[&str]) -> String {
        parts.join(MINIMAX_NS)
    }

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "weather".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    }
                }),
                strict: None,
            },
        }]
    }

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

    #[tokio::test]
    async fn schema_keeps_numeric_looking_string() {
        let parser = MiniMaxParser::new();
        let tools = weather_tools();
        let output = m3(&[
            "",
            "<tool_call>",
            r#"<invoke name="weather">"#,
            "<city>42",
            "</city>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (_content, calls) = parser
            .parse_complete_with_tools(&output, &tools)
            .await
            .unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], json!("42"));
    }

    #[tokio::test]
    async fn schema_coerces_integer_parameter() {
        let parser = MiniMaxParser::new();
        let tools = weather_tools();
        let output = m3(&[
            "",
            "<tool_call>",
            r#"<invoke name="weather">"#,
            "<days>3",
            "</days>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (_content, calls) = parser
            .parse_complete_with_tools(&output, &tools)
            .await
            .unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["days"], json!(3));
    }

    #[tokio::test]
    async fn without_schema_infers_number() {
        let parser = MiniMaxParser::new();
        let output = m3(&[
            "",
            "<tool_call>",
            r#"<invoke name="weather">"#,
            "<days>3",
            "</days>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (_content, calls) = parser.parse_complete(&output).await.unwrap();
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["days"], json!(3));
    }

    #[tokio::test]
    async fn streaming_emits_name_then_incremental_parameters() {
        let mut parser = MiniMaxParser::new();
        let tools = weather_tools();
        let chunks = [
            format!("pre{MINIMAX_NS}"),
            "<tool_call>".to_string(),
            format!(r#"{MINIMAX_NS}<invoke name="weather">"#),
            format!("<city>Paris{MINIMAX_NS}</city>"),
            format!("<days>2{MINIMAX_NS}</days>"),
            format!("{MINIMAX_NS}</invoke>{MINIMAX_NS}</tool_call>"),
        ];

        let mut normal = String::new();
        let mut names = Vec::new();
        let mut args_json = String::new();
        for chunk in &chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            normal.push_str(&result.normal_text);
            for call in result.calls {
                if let Some(name) = call.name {
                    names.push(name);
                } else {
                    args_json.push_str(&call.parameters);
                }
            }
        }

        assert_eq!(normal, "pre");
        assert_eq!(names, vec!["weather"]);
        let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        assert_eq!(args["city"], json!("Paris"));
        assert_eq!(args["days"], json!(2));
    }

    #[tokio::test]
    async fn streaming_skips_unknown_tool_name() {
        let mut parser = MiniMaxParser::new();
        let tools = weather_tools();
        let output = m3(&[
            "",
            "<tool_call>",
            r#"<invoke name="not_a_tool">"#,
            "<city>X",
            "</city>",
            "</invoke>",
            r#"<invoke name="weather">"#,
            "<city>Paris",
            "</city>",
            "</invoke>",
            "</tool_call>",
        ]);
        let result = parser.parse_incremental(&output, &tools).await.unwrap();
        let names: Vec<_> = result
            .calls
            .iter()
            .filter_map(|c| c.name.as_deref())
            .collect();
        assert_eq!(names, vec!["weather"]);
    }

    #[tokio::test]
    async fn reset_clears_streaming_state() {
        let mut parser = MiniMaxParser::new();
        let tools = weather_tools();
        let _ = parser
            .parse_incremental(&format!("{MINIMAX_NS}<tool_call>"), &tools)
            .await
            .unwrap();
        assert!(parser.in_tool_call);
        parser.reset();
        assert!(!parser.in_tool_call);
        assert!(parser.buffer.is_empty());
        assert_eq!(parser.current_tool_id, -1);
        assert!(parser.prev_tool_call_arr.is_empty());
        assert!(parser.streamed_args_for_tool.is_empty());
    }

    #[tokio::test]
    async fn holds_partial_namespace_token_at_boundary() {
        let mut parser = MiniMaxParser::new();
        let first = parser
            .parse_incremental("hello ]<]minim", &[])
            .await
            .unwrap();
        assert_eq!(first.normal_text, "hello ");
        assert_eq!(parser.buffer, "]<]minim");

        // Complete the partial NS, then open/close an empty tool_call wrapper.
        let second = parser
            .parse_incremental("ax[>[<tool_call></tool_call>", &[])
            .await
            .unwrap();
        assert!(second.calls.is_empty());
        assert!(!parser.in_tool_call);
    }

    #[tokio::test]
    async fn parallel_invokes_in_one_tool_call() {
        let parser = MiniMaxParser::new();
        let output = m3(&[
            "",
            "<tool_call>",
            r#"<invoke name="weather">"#,
            "<city>Paris",
            "</city>",
            "</invoke>",
            r#"<invoke name="weather">"#,
            "<city>London",
            "</city>",
            "</invoke>",
            "</tool_call>",
        ]);
        let (_content, calls) = parser.parse_complete(&output).await.unwrap();
        assert_eq!(calls.len(), 2);
        assert!(calls[0].function.arguments.contains("Paris"));
        assert!(calls[1].function.arguments.contains("London"));
    }

    #[test]
    fn has_markers_requires_m3_namespace() {
        let parser = MiniMaxParser::new();
        assert!(!parser.has_tool_markers("<tool_call></tool_call>"));
        assert!(parser.has_tool_markers(&format!("{MINIMAX_NS}<tool_call>")));
    }
}
