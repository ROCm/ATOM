//! DeepSeek V4 / V4.1 DSML tool-call parser with dialect-specific markers.
//!
//! Adapted from `smg/crates/tool_parser/src/parsers/deepseek_dsml.rs` (V4 path).
//! ATOM keeps `DsmlParser` / `DsmlParser::new` as the public V4 constructor and
//! does not register DeepSeek V3 / V3.1 / V3.2 variants.
//!
//! ```text
//! <｜DSML｜tool_calls>
//! <｜DSML｜invoke name="func">
//! <｜DSML｜parameter name="key" string="true">value</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//!
//! Also supports direct JSON inside invoke blocks as a fallback format.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// DeepSeek end-of-sentence marker. Some engines emit this as raw text at the
/// end of a truncated turn; it must never bleed into tool-call argument bytes.
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";

/// Strip a trailing partial DSML closing tag from a string.
///
/// If the string ends with a prefix of `closing_tag` (e.g. `"Tokyo</｜DSML｜para"`
/// ends with a prefix of `"</｜DSML｜parameter>"`), that trailing portion is removed.
/// Unlike character-set stripping, this only removes text that actually starts
/// the specified closing tag, so legitimate value bytes are preserved.
fn strip_dsml_trailing(s: &str, closing_tag: &str) -> String {
    for (idx, _) in s.char_indices() {
        if closing_tag.starts_with(&s[idx..]) {
            return s[..idx].to_string();
        }
    }
    s.to_string()
}

pub struct DsmlParser {
    block_open: String,
    block_close: String,
    parameter_end_tag: String,
    invoke_end_tag: String,
    /// Regex for extracting full outer-block content
    tool_call_complete_regex: Regex,
    /// Regex for extracting complete invoke blocks (name + body)
    invoke_complete_regex: Regex,
    /// Regex for extracting complete parameter tags (name, string attr, value)
    parameter_complete_regex: Regex,
    /// Regex for matching partial parameter tag during streaming (no closing tag)
    partial_parameter_regex: Regex,
    /// Regex for matching invoke blocks (complete or partial, for streaming)
    invoke_regex: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,
    /// Stores complete tool call info for each tool being parsed
    prev_tool_call_arr: Vec<Value>,
    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,
    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,
    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,
}

impl DsmlParser {
    /// Create a DeepSeek V4 parser (outer block token `tool_calls`).
    pub fn new() -> Self {
        Self::for_dialect("｜DSML｜", "tool_calls")
    }

    /// The released V4.1 dialect has a space after the sentinel and calls.
    pub fn v41() -> Self {
        Self::for_dialect("｜DSML｜ ", "calls")
    }

    fn for_dialect(prefix: &str, section: &str) -> Self {
        let block_open = format!("<{prefix}{section}>");
        let block_close = format!("</{prefix}{section}>");
        let parameter_end_tag = format!("</{prefix}parameter>");
        let invoke_end_tag = format!("</{prefix}invoke>");
        let compile = |pattern: String| Regex::new(&pattern).expect("Valid DSML regex");
        let tool_call_complete_regex = compile(format!("(?s){block_open}(.*?){block_close}"));
        let invoke_complete_regex = compile(format!(
            r#"(?s)<{prefix}invoke\s+name="([^"]+)"\s*>(.*?){invoke_end_tag}"#
        ));
        let parameter_complete_regex = compile(format!(
            r#"(?s)<{prefix}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?){parameter_end_tag}"#
        ));
        let partial_parameter_regex = compile(format!(
            r#"(?s)<{prefix}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)$"#
        ));
        let invoke_regex = compile(format!(
            r#"(?s)<{prefix}invoke\s+name="([^"]*)"\s*>(.*?)({invoke_end_tag}|$)"#
        ));

        Self {
            block_open,
            block_close,
            parameter_end_tag,
            invoke_end_tag,
            tool_call_complete_regex,
            invoke_complete_regex,
            parameter_complete_regex,
            partial_parameter_regex,
            invoke_regex,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse DSML parameters from invoke content into a JSON string.
    ///
    /// Supports two formats:
    /// 1. Direct JSON: content starts with `{` — returned as-is
    /// 2. XML parameters: `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`
    ///
    /// When `allow_partial` is true (streaming), also matches open parameter tags
    /// and strips trailing DSML fragments.
    fn parse_parameters_from_dsml(&self, invoke_content: &str, allow_partial: bool) -> String {
        let trimmed = invoke_content.trim();

        // Direct JSON path
        if trimmed.starts_with('{') {
            if allow_partial {
                // `strip_dsml_trailing` handles partial `</｜DSML｜invoke>` prefixes
                // but can't match the EOS sentinel (different prefix). Strip it
                // unconditionally so a truncated turn doesn't leak EOS into args.
                return strip_dsml_trailing(trimmed, &self.invoke_end_tag).replace(EOS_TOKEN, "");
            } else if trimmed.ends_with('}') {
                return trimmed.to_string();
            }
        }

        // XML parameter path
        let mut params = serde_json::Map::new();

        for cap in self.parameter_complete_regex.captures_iter(invoke_content) {
            let name = cap.get(1).map_or("", |m| m.as_str());
            let is_string = cap.get(2).map_or("true", |m| m.as_str());
            // Strip any stray EOS marker — should never legitimately appear
            // inside a closed parameter, but defend against malformed output.
            let value = cap.get(3).map_or("", |m| m.as_str()).replace(EOS_TOKEN, "");

            let json_value = if is_string == "true" {
                Value::String(value.to_string())
            } else {
                serde_json::from_str(value.trim())
                    .unwrap_or_else(|_| Value::String(value.to_string()))
            };

            params.insert(name.to_string(), json_value);
        }

        // Partial parameter matching for streaming
        // Following SGLang: strip DSML fragments from remaining content BEFORE
        // running the partial regex, so the regex captures a clean value.
        if allow_partial {
            // Find where the last complete parameter match ended
            let last_match_end = self
                .parameter_complete_regex
                .find_iter(invoke_content)
                .last()
                .map(|m| m.end())
                .unwrap_or(0);

            let remaining = &invoke_content[last_match_end..];
            let cleaned = strip_dsml_trailing(remaining, &self.parameter_end_tag);

            if let Some(cap) = self.partial_parameter_regex.captures(&cleaned) {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let is_string = cap.get(2).map_or("true", |m| m.as_str());
                // Strip EOS without trimming string values — `strip_dsml_trailing` above only
                // handles `</｜DSML｜parameter>` prefixes, so a truncated turn
                // with `value<EOS>` would otherwise stream EOS as arg bytes.
                let value = cap.get(3).map_or("", |m| m.as_str()).replace(EOS_TOKEN, "");
                let value = value.as_str();

                // Only add if we have actual content and this param isn't already complete
                if is_string == "true" && !value.is_empty() && !params.contains_key(name) {
                    params.insert(name.to_string(), Value::String(value.to_string()));
                }
            }
        }

        serde_json::to_string(&Value::Object(params)).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse a single complete invoke block into a ToolCall
    fn parse_invoke(&self, name: &str, content: &str) -> ToolCall {
        let arguments = self.parse_parameters_from_dsml(content, false);

        ToolCall {
            function: FunctionCall {
                name: name.trim().to_string(),
                arguments,
            },
        }
    }
}

impl Default for DsmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DsmlParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let idx = text
            .find(self.block_open.as_str())
            .ok_or_else(|| ParserError::ParsingFailed("DSML marker not found".to_string()))?;
        let normal_text = text[..idx].trim_end().to_string();

        let mut tools = Vec::new();

        for fc_cap in self.tool_call_complete_regex.captures_iter(text) {
            let fc_content = fc_cap.get(1).map_or("", |m| m.as_str());

            for inv_cap in self.invoke_complete_regex.captures_iter(fc_content) {
                let func_name = inv_cap.get(1).map_or("", |m| m.as_str());
                let invoke_content = inv_cap.get(2).map_or("", |m| m.as_str());

                tools.push(self.parse_invoke(func_name, invoke_content));
            }
        }

        if tools.is_empty() {
            return Ok((normal_text, vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        // Check for DSML markers or partial DSML prefixes.
        //
        // `<｜DSML｜` is a single BPE token in DeepSeek's tokenizer (id 128793),
        // so real streams deliver it atomically. We flag the stream as DSML as
        // soon as the sentinel appears anywhere in the buffer — we don't wait
        // for a complete outer `<｜DSML｜tool_calls>` opener, because live
        // backends chunk the opener into per-token pieces after the sentinel
        // (e.g. `<｜DSML｜` + `tool` + `_c` + `all` + `s` + `>`). Without this,
        // chunk 2 of a live stream would flush the buffer on the passthrough
        // path and lose the sentinel, turning every subsequent chunk into plain
        // text.
        let has_dsml = current_text.contains("<｜DSML｜");
        let has_partial_prefix = current_text.char_indices().any(|(i, _)| {
            [
                "<｜DSML｜",
                "</｜DSML｜",
                self.block_close.as_str(),
                self.parameter_end_tag.as_str(),
                self.invoke_end_tag.as_str(),
            ]
            .iter()
            .any(|tag| tag.starts_with(&current_text[i..]))
        });

        if !has_dsml && !has_partial_prefix {
            let mut normal_text = std::mem::take(&mut self.buffer);
            for end_token in [
                self.block_close.as_str(),
                &self.invoke_end_tag,
                &self.parameter_end_tag,
                EOS_TOKEN,
            ] {
                normal_text = normal_text.replace(end_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // If we have partial prefix but no actual DSML content, buffer and wait
        if !has_dsml && has_partial_prefix {
            return Ok(StreamingParseResult::default());
        }

        let mut normal_text = String::new();
        if self.current_tool_id < 0 {
            if let Some(start) = self.buffer.find("<｜DSML｜") {
                normal_text.push_str(&self.buffer[..start]);
                self.buffer.drain(..start);
            }
        }
        let tool_indices = helpers::get_tool_indices(tools);
        let mut all_calls: Vec<ToolCallItem> = Vec::new();

        // Process invoke blocks in a loop (handles multiple complete invokes in buffer)
        loop {
            let buf_snapshot = self.buffer.clone();
            let invoke_match = self.invoke_regex.captures(&buf_snapshot);

            let captures = match invoke_match {
                Some(c) => c,
                None => break,
            };

            let func_name = captures
                .get(1)
                .map_or(String::new(), |m| m.as_str().trim().to_string());
            let invoke_content = captures
                .get(2)
                .map_or(String::new(), |m| m.as_str().to_string());
            let is_complete = captures
                .get(3)
                .is_some_and(|m| m.as_str().contains(self.invoke_end_tag.as_str()));
            let match_end = captures.get(0).map(|m| m.end());
            drop(captures);

            // Skip if tool name is absent or not in the provided tools list.
            // Empty names reach this branch because `invoke_regex` allows
            // `name=""` (quantifier `*` not `+`); the loosened regex + this
            // guard together ensure a malformed `name=""` block is advanced
            // past instead of trapping the buffer forever.
            let name_invalid =
                func_name.is_empty() || !tool_indices.contains_key(func_name.as_str());
            if name_invalid {
                tracing::debug!("Invalid tool name '{}' - skipping", func_name);
                if is_complete {
                    // Complete invalid invoke — advance buffer past it and try next
                    if let Some(end) = match_end {
                        self.buffer = self.buffer[end..].to_string();
                    }
                    continue;
                } else {
                    // Incomplete invalid invoke — reset state and wait for more data
                    // Return any calls already collected from previous complete invokes
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut self.current_tool_name_sent,
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult {
                        normal_text,
                        calls: all_calls,
                    });
                }
            }

            // Initialize state on first tool
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            // Emit tool name if not sent
            if !self.current_tool_name_sent && !func_name.is_empty() {
                all_calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(func_name.to_string()),
                    parameters: String::new(),
                });
                self.current_tool_name_sent = true;

                let tool_id = self.current_tool_id as usize;
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": {},
                });
            }

            // Parse current arguments (partial or complete)
            let current_args = self.parse_parameters_from_dsml(&invoke_content, !is_complete);
            let tool_id = self.current_tool_id as usize;

            // Compute diff against what we've already sent
            let sent_len = self
                .streamed_args_for_tool
                .get(tool_id)
                .map(|s| s.len())
                .unwrap_or(0);

            let prev_args = if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id]
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            };

            let argument_diff = if is_complete {
                if sent_len < current_args.len() {
                    Some(current_args[sent_len..].to_string())
                } else {
                    Some(String::new())
                }
            } else if let Some(prev) = &prev_args {
                if current_args == *prev {
                    None
                } else {
                    let prefix = helpers::find_common_prefix(prev, &current_args);
                    if prefix.len() > sent_len {
                        Some(prefix[sent_len..].to_string())
                    } else {
                        None
                    }
                }
            } else {
                None
            };

            if let Some(diff) = argument_diff {
                if !diff.is_empty() {
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }
                    all_calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: diff,
                    });
                }
            }

            // Update prev state
            if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": current_args,
                });
            }

            // If invoke is complete, advance to next tool
            if is_complete {
                if let Some(end) = match_end {
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    self.buffer.clear();
                }
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                continue;
            } else {
                break;
            }
        }

        if let Some(end) = self.buffer.find(self.block_close.as_str()) {
            let tail = end + self.block_close.len();
            normal_text.push_str(&self.buffer[tail..]);
            self.buffer.clear();
        }
        Ok(StreamingParseResult {
            normal_text,
            calls: all_calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.block_open.as_str())
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json;

    use super::DsmlParser;
    use crate::tool_parser::traits::ToolParser;

    fn create_test_tools() -> Vec<Tool> {
        vec![
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "search".to_string(),
                    description: Some("Search".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": { "query": {"type": "string"} }
                    }),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Weather".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "location": {"type": "string"}
                        }
                    }),
                    strict: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "process".to_string(),
                    description: Some("Process".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "count": {"type": "number"},
                            "enabled": {"type": "boolean"}
                        }
                    }),
                    strict: None,
                },
            },
        ]
    }

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

    #[tokio::test]
    async fn parse_complete_mixed_types_and_direct_json() {
        let parser = DsmlParser::new();

        let mixed = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"process\">\n",
            "<｜DSML｜parameter name=\"text\" string=\"true\">hello</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"count\" string=\"false\">42</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"enabled\" string=\"false\">true</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        );
        let (_normal_text, tools) = parser.parse_complete(mixed).await.unwrap();
        let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert_eq!(args["text"], "hello");
        assert_eq!(args["count"], 42);
        assert_eq!(args["enabled"], true);

        let json_input = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "{\"location\": \"Beijing\"}\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        );
        let (_normal_text, tools) = parser.parse_complete(json_input).await.unwrap();
        let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Beijing");
    }

    #[test]
    fn v4_format_detection_ignores_v32_block() {
        let parser = DsmlParser::new();
        assert!(parser.has_tool_markers("<｜DSML｜tool_calls>"));
        assert!(!parser.has_tool_markers("<｜DSML｜function_calls>"));
        assert!(!parser.has_tool_markers("plain text"));
    }

    #[tokio::test]
    async fn v32_payload_passthrough() {
        let parser = DsmlParser::new();
        let v32_input = concat!(
            "<｜DSML｜function_calls>\n",
            "<｜DSML｜invoke name=\"search\">\n",
            "<｜DSML｜parameter name=\"query\" string=\"true\">test</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜function_calls>",
        );
        let (normal_text, tools) = parser.parse_complete(v32_input).await.unwrap();
        assert!(tools.is_empty());
        assert_eq!(normal_text, v32_input);
    }

    #[tokio::test]
    async fn streaming_single_tool_emits_name_then_parameter_deltas() {
        let tools = create_test_tools();
        let mut parser = DsmlParser::new();
        let chunks = [
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">",
            "Beijing",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        ];

        let mut found_name = false;
        let mut collected_args = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            for call in result.calls {
                if let Some(name) = call.name {
                    assert_eq!(name, "get_weather");
                    found_name = true;
                }
                collected_args.push_str(&call.parameters);
            }
        }
        assert!(found_name);
        assert!(collected_args.contains("Beijing"));
    }

    #[tokio::test]
    async fn streaming_strips_eos_from_partial_parameter() {
        let tools = create_test_tools();
        let mut parser = DsmlParser::new();
        let chunks = [
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing",
            "<｜end▁of▁sentence｜>",
        ];

        let mut collected_args = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            for call in result.calls {
                collected_args.push_str(&call.parameters);
            }
        }
        assert!(
            !collected_args.contains("<｜end▁of▁sentence｜>"),
            "EOS must not leak into streamed argument bytes, got: {collected_args:?}"
        );
    }

    #[tokio::test]
    async fn streaming_malformed_empty_name_does_not_trap_buffer() {
        let tools = create_test_tools();
        let mut parser = DsmlParser::new();
        let chunks = [
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"\">junk</｜DSML｜invoke>\n",
            "<｜DSML｜invoke name=\"search\">\n",
            "<｜DSML｜parameter name=\"query\" string=\"true\">bar</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        ];

        let mut tool_names: Vec<String> = Vec::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            for call in result.calls {
                if let Some(name) = call.name {
                    tool_names.push(name);
                }
            }
        }
        assert_eq!(tool_names, vec!["search"]);
    }

    #[tokio::test]
    async fn streaming_bpe_chunked_opener() {
        let tools = create_test_tools();
        let mut parser = DsmlParser::new();
        let chunks = [
            "\n\n",
            "<｜DSML｜",
            "tool",
            "_c",
            "all",
            "s",
            ">\n",
            "<｜DSML｜",
            "inv",
            "oke",
            " name",
            "=\"",
            "get",
            "_",
            "weather",
            "\">\n",
            "<｜DSML｜",
            "parameter",
            " name",
            "=\"",
            "location",
            "\"",
            " string",
            "=\"",
            "true",
            "\">",
            "Tokyo",
            "</｜DSML｜",
            "parameter",
            ">\n",
            "</｜DSML｜",
            "inv",
            "oke",
            ">\n",
            "</｜DSML｜",
            "tool",
            "_c",
            "all",
            "s",
            ">",
        ];

        let mut tool_names: Vec<String> = Vec::new();
        let mut collected_args = String::new();
        let mut normal_text = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            normal_text.push_str(&result.normal_text);
            for call in result.calls {
                if let Some(name) = call.name {
                    tool_names.push(name);
                }
                collected_args.push_str(&call.parameters);
            }
        }

        assert_eq!(tool_names, vec!["get_weather"]);
        assert!(collected_args.contains("Tokyo"));
        assert!(!normal_text.contains("<｜DSML｜"));
    }

    #[tokio::test]
    async fn reset_clears_streaming_state() {
        let tools = create_test_tools();
        let mut parser = DsmlParser::new();
        let _ = parser
            .parse_incremental(
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">",
                &tools,
            )
            .await
            .unwrap();
        parser.reset();

        let chunks = [
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_weather\">\n",
            "<｜DSML｜parameter name=\"location\" string=\"true\">Oslo</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>",
        ];
        let mut tool_names = Vec::new();
        let mut collected_args = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            for call in result.calls {
                if let Some(name) = call.name {
                    tool_names.push(name);
                }
                collected_args.push_str(&call.parameters);
            }
        }
        assert_eq!(tool_names, vec!["get_weather"]);
        assert!(collected_args.contains("Oslo"));
    }
    #[tokio::test]
    async fn v41_shared_fixtures_at_every_utf8_boundary() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/entrypoints/fixtures/deepseek_v41_dsml.json"
        ))
        .unwrap();
        for fixture in fixtures.as_array().unwrap() {
            let text = fixture["text"].as_str().unwrap();
            let expected = fixture["calls"].as_array().unwrap();
            let tools: Vec<Tool> = expected
                .iter()
                .map(|call| Tool {
                    tool_type: "function".to_owned(),
                    function: Function {
                        name: call["name"].as_str().unwrap().to_owned(),
                        description: None,
                        parameters: json!({"type": "object"}),
                        strict: None,
                    },
                })
                .collect();
            let (content, calls) = DsmlParser::v41().parse_complete(text).await.unwrap();
            assert_eq!(content, fixture["content"].as_str().unwrap());
            assert_eq!(calls.len(), expected.len());
            for (actual, expected) in calls.iter().zip(expected) {
                assert_eq!(actual.function.name, expected["name"].as_str().unwrap());
                let args: serde_json::Value =
                    serde_json::from_str(&actual.function.arguments).unwrap();
                assert_eq!(args, expected["arguments"]);
            }
            let mut partitions: Vec<Vec<&str>> = text
                .char_indices()
                .map(|(i, _)| vec![&text[..i], &text[i..]])
                .collect();
            partitions.push(
                text.char_indices()
                    .map(|(i, c)| &text[i..i + c.len_utf8()])
                    .collect(),
            );
            for chunks in partitions {
                let mut parser = DsmlParser::v41();
                let mut content = String::new();
                let mut names: Vec<String> = Vec::new();
                let mut arguments: Vec<String> = Vec::new();
                for chunk in chunks {
                    let parsed = parser.parse_incremental(chunk, &tools).await.unwrap();
                    content.push_str(&parsed.normal_text);
                    for call in parsed.calls {
                        if let Some(name) = call.name {
                            names.push(name);
                            arguments.push(String::new());
                        }
                        arguments[call.tool_index].push_str(&call.parameters);
                    }
                }
                assert_eq!(content, fixture["content"].as_str().unwrap());
                assert_eq!(names.len(), expected.len());
                for (i, call) in expected.iter().enumerate() {
                    assert_eq!(names[i], call["name"].as_str().unwrap());
                    let actual: serde_json::Value = serde_json::from_str(&arguments[i]).unwrap();
                    assert_eq!(actual, call["arguments"]);
                }
            }
        }
    }
}
