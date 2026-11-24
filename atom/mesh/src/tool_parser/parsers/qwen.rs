//! Qwen tool-call parsers for XML and JSON-in-tag formats.

mod coder {
    use std::sync::LazyLock;

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

    /// Qwen Coder format parser for tool calls
    ///
    /// Handles the Qwen Coder specific XML format:
    /// `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`
    ///
    /// Features:
    /// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    /// - XML-style function declaration: `<function=name>`
    /// - XML-style parameters: `<parameter=key>value</parameter>`
    ///
    /// Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8?chat_template=default
    pub struct QwenCoderParser {
        /// Regex for extracting tool calls in parse_complete
        extractor: Regex,

        /// Buffer for accumulating incomplete patterns across chunks
        buffer: String,

        /// Stores complete tool call info (name and arguments) for each tool being parsed
        prev_tool_call_arr: Vec<Value>,

        /// Index of currently streaming tool call (-1 means no active tool)
        current_tool_id: i32,

        /// Flag for whether current tool's name has been sent to client
        current_tool_name_sent: bool,

        /// Tracks raw JSON string content streamed to client for each tool's arguments
        streamed_args_for_tool: Vec<String>,

        /// Token configuration
        tool_call_start_token: &'static str,
        tool_call_end_token: &'static str,

        /// XML format streaming state
        in_tool_call: bool,
        /// Bare `<function=...>` calls are calls (`read_as_atom`).
        bare_calls: bool,
        /// The open call is a bare `<function=...>`, outside a `<tool_call>` wrapper.
        bare_call: bool,
        current_function_name: String,
        current_parameters: serde_json::Map<String, Value>,

        /// Precompiled regex patterns for XML format parsing
        xml_function_pattern: Regex,
        xml_param_pattern: Regex,
    }

    /// A call ATOM's HTTP parser reads outside a `<tool_call>` wrapper too.
    const BARE_CALL_START: &str = "<function=";

    /// Wrapped calls, and bare ones, as ATOM's HTTP parser reads them; a bare one's last
    /// parameter may be cut off in its name, which is then no parameter.
    static WRAPPED_OR_BARE_CALLS: LazyLock<Regex> = LazyLock::new(|| {
        let pattern = concat!(
            r"(?s)<tool_call>\s*(.*?)\s*</tool_call>",
            r"|(<function=[^<>]*>(?:\s*<parameter=[^>]*>.*?</parameter>)*",
            r"(?:\s*<parameter=[^<>]*)?\s*</function>)",
        );
        Regex::new(pattern).expect("Valid regex pattern")
    });

    /// Decode HTML entities in a string (equivalent to Python's html.unescape)
    ///
    /// Handles common HTML entities like &amp; &lt; &gt; &quot; &#39; and numeric entities
    #[cfg(test)]
    fn html_unescape(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '&' {
                let mut entity = String::new();
                let mut consumed_semicolon = false;
                while let Some(&next) = chars.peek() {
                    if next == ';' {
                        chars.next();
                        consumed_semicolon = true;
                        break;
                    }
                    if next.is_alphanumeric() || next == '#' {
                        entity.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                let decoded = match entity.as_str() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "quot" => "\"",
                    "apos" => "'",
                    "nbsp" => "\u{00A0}",
                    s if s.starts_with('#') => {
                        let num_str = &s[1..];
                        let code_point = if num_str.starts_with('x') || num_str.starts_with('X') {
                            u32::from_str_radix(&num_str[1..], 16).ok()
                        } else {
                            num_str.parse::<u32>().ok()
                        };
                        if let Some(cp) = code_point {
                            if let Some(ch) = char::from_u32(cp) {
                                result.push(ch);
                                continue;
                            }
                        }
                        // Invalid numeric entity, reconstruct original
                        result.push('&');
                        result.push_str(&entity);
                        if consumed_semicolon {
                            result.push(';');
                        }
                        continue;
                    }
                    _ => {
                        // Unknown entity, reconstruct original
                        result.push('&');
                        result.push_str(&entity);
                        if consumed_semicolon {
                            result.push(';');
                        }
                        continue;
                    }
                };
                result.push_str(decoded);
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Parse a raw parameter value, similar to Python's _safe_val
    ///
    /// 1. Decode HTML entities
    /// 2. Try to parse as JSON (numbers, booleans, null, objects, arrays)
    /// 3. Fall back to string if JSON parsing fails
    fn safe_val(raw: &str) -> Value {
        let trimmed = raw.trim();

        // Try JSON parsing first
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            return v;
        }

        // Handle Python-style literals (True, False, None)
        match trimmed {
            "True" => return Value::Bool(true),
            "False" => return Value::Bool(false),
            "None" => return Value::Null,
            _ => {}
        }

        // Fall back to string
        Value::String(trimmed.to_string())
    }

    fn coerce_value(raw: &str, declared_type: Option<&str>) -> Value {
        helpers::coerce_by_schema_type(raw.trim(), declared_type).unwrap_or_else(|| safe_val(raw))
    }

    impl QwenCoderParser {
        /// Create a new Qwen Coder parser
        pub fn new() -> Self {
            // Support XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
            let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>";
            let extractor = Regex::new(pattern).expect("Valid regex pattern");

            // Precompile XML format regex patterns for performance
            let xml_function_pattern =
                Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
            let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
                .expect("Valid XML parameter pattern");

            Self {
                extractor,
                buffer: String::new(),
                prev_tool_call_arr: Vec::new(),
                current_tool_id: -1,
                current_tool_name_sent: false,
                streamed_args_for_tool: Vec::new(),
                tool_call_start_token: "<tool_call>",
                tool_call_end_token: "</tool_call>",
                in_tool_call: false,
                bare_calls: false,
                bare_call: false,
                current_function_name: String::new(),
                current_parameters: serde_json::Map::new(),
                xml_function_pattern,
                xml_param_pattern,
            }
        }

        /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
        fn parse_xml_format(
            &self,
            content: &str,
            tools: &[Tool],
        ) -> ParserResult<Option<ToolCall>> {
            let function_captures = self
                .xml_function_pattern
                .captures(content)
                .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;

            let function_name = function_captures
                .get(1)
                .ok_or_else(|| {
                    ParserError::ParsingFailed("Function name capture failed".to_string())
                })?
                .as_str()
                .trim()
                .to_string();

            if function_name.is_empty() {
                return Ok(None);
            }

            let param_types = helpers::param_types_for_function(tools, &function_name);
            let mut parameters = serde_json::Map::new();

            for cap in self.xml_param_pattern.captures_iter(content) {
                if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                    let key = key_match.as_str().trim().to_string();
                    let value = value_match.as_str();
                    let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                    parameters.insert(key, json_value);
                }
            }

            let arguments = serde_json::to_string(&parameters)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: function_name,
                    arguments,
                },
            }))
        }

        /// Parse and stream complete parameters from buffer
        /// Returns tool call items to emit (similar to Python's _parse_and_stream_parameters)
        fn parse_and_stream_parameters(&mut self, tools: &[Tool]) -> Vec<ToolCallItem> {
            let mut calls: Vec<ToolCallItem> = vec![];
            let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

            // Find all complete parameter patterns in the open call, not in calls after it
            let mut new_params = serde_json::Map::new();
            let end = self.buffer.find(self.call_end_token());
            let call = &self.buffer[..end.unwrap_or(self.buffer.len())];
            for cap in self.xml_param_pattern.captures_iter(call) {
                if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                    let key = key_match.as_str().trim().to_string();
                    let value = value_match.as_str();
                    let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                    new_params.insert(key, json_value);
                }
            }

            // Calculate parameter diff and stream updates
            if new_params != self.current_parameters {
                let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];

                if self.current_parameters.is_empty() {
                    // First parameter(s) - build JSON fragment (without closing brace)
                    let mut items = Vec::new();
                    for (key, value) in &new_params {
                        let key_json =
                            serde_json::to_string(key).unwrap_or_else(|_| format!("\"{}\"", key));
                        let value_json = serde_json::to_string(value).unwrap_or_default();
                        items.push(format!("{}: {}", key_json, value_json));
                    }
                    let json_fragment = format!("{{{}", items.join(", "));

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    *current_args = json_fragment;
                } else {
                    // Additional parameters - add them incrementally
                    let new_keys: Vec<_> = new_params
                        .keys()
                        .filter(|k| !self.current_parameters.contains_key(*k))
                        .collect();

                    if !new_keys.is_empty() {
                        let mut continuation_parts = Vec::new();
                        for key in new_keys {
                            if let Some(value) = new_params.get(key) {
                                let key_json = serde_json::to_string(key)
                                    .unwrap_or_else(|_| format!("\"{}\"", key));
                                let value_json = serde_json::to_string(value).unwrap_or_default();
                                continuation_parts.push(format!("{}: {}", key_json, value_json));
                            }
                        }

                        let json_fragment = format!(", {}", continuation_parts.join(", "));

                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: None,
                            parameters: json_fragment.clone(),
                        });
                        current_args.push_str(&json_fragment);
                    }
                }

                // Update current state
                self.current_parameters = new_params.clone();
                if let Some(tool_obj) =
                    self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
                {
                    tool_obj.insert("arguments".to_string(), Value::Object(new_params));
                }
            }

            calls
        }

        fn parse_complete_inner(
            &self,
            text: &str,
            tools: &[Tool],
        ) -> ParserResult<(String, Vec<ToolCall>)> {
            if !self.has_tool_markers(text) {
                return Ok((text.to_string(), vec![]));
            }

            let tool_indices = helpers::get_tool_indices(tools);
            let (mut outside, mut end, mut parsed) = (String::new(), 0, Vec::new());
            for captures in self.extractor.captures_iter(text) {
                let Some(content) = captures.get(1).or_else(|| captures.get(2)) else {
                    continue;
                };
                match self.parse_xml_format(content.as_str().trim(), tools) {
                    // A bare call names a declared tool, as when streamed; else it is text.
                    Ok(Some(tool))
                        if captures.get(1).is_some()
                            || tool_indices.contains_key(&tool.function.name) =>
                    {
                        let call = captures.get(0).unwrap();
                        outside.push_str(&text[end..call.start()]);
                        end = call.end();
                        parsed.push(tool);
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!("Failed to parse XML tool call: {:?}", error);
                    }
                }
            }
            if parsed.is_empty() {
                return Ok((text.to_string(), vec![]));
            }
            let normal_text = match self.bare_calls {
                // As ATOM reads it, a reply's text is all of it outside its calls (a
                // wrapped one's from its wrapper); a `<function=` prose quotes stays.
                true => outside + &text[end..],
                false => text[..text.find(self.tool_call_start_token).unwrap_or(0)].to_string(),
            };
            Ok((normal_text, parsed))
        }

        /// Where the first call in `text` starts, and whether it is a bare `<function=...>`.
        fn call_start(&self, text: &str) -> Option<(usize, bool)> {
            let wrapped = text.find(self.tool_call_start_token);
            let bare = self
                .bare_calls
                .then(|| text.find(BARE_CALL_START))
                .flatten();
            match (wrapped, bare) {
                (Some(w), Some(b)) if b < w => Some((b, true)),
                (Some(w), _) => Some((w, false)),
                (None, bare) => bare.map(|b| (b, true)),
            }
        }

        /// The tag that ends the open call: a bare call's `</function>`, else its wrapper's.
        fn call_end_token(&self) -> &'static str {
            match self.bare_call {
                true => "</function>",
                false => self.tool_call_end_token,
            }
        }

        /// Reset streaming state for next tool call
        fn reset_streaming_state(&mut self) {
            self.in_tool_call = false;
            self.bare_call = false;
            self.current_tool_name_sent = false;
            self.current_function_name.clear();
            self.current_parameters.clear();
        }
    }

    impl Default for QwenCoderParser {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl ToolParser for QwenCoderParser {
        async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
            self.parse_complete_inner(text, &[])
        }

        async fn parse_complete_with_tools(
            &self,
            text: &str,
            tools: &[Tool],
        ) -> ParserResult<(String, Vec<ToolCall>)> {
            self.parse_complete_inner(text, tools)
        }

        async fn parse_incremental(
            &mut self,
            chunk: &str,
            tools: &[Tool],
        ) -> ParserResult<StreamingParseResult> {
            self.buffer.push_str(chunk);

            let mut normal_text = String::new();
            let mut calls: Vec<ToolCallItem> = vec![];

            // Build tool indices for validation
            let tool_indices = helpers::get_tool_indices(tools);

            loop {
                // If we're not in a tool call and don't see a start token, return normal text
                if !self.in_tool_call && self.call_start(&self.buffer).is_none() {
                    // Check for partial start token
                    let partial = |token| helpers::ends_with_partial_token(&self.buffer, token);
                    if partial(self.tool_call_start_token).is_none()
                        && !(self.bare_calls && partial(BARE_CALL_START).is_some())
                    {
                        normal_text.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                    break;
                }

                // Look for tool call start
                if !self.in_tool_call {
                    if let Some((s, bare)) = self.call_start(&self.buffer) {
                        // A bare call keeps its `<function=`, read as the name next.
                        let opener = if bare {
                            0
                        } else {
                            self.tool_call_start_token.len()
                        };
                        // As ATOM reads it, a `<tool_call>` no `<function=` follows is text.
                        let rest = self.buffer[s + opener..].trim_start();
                        if self.bare_calls && !bare && !rest.starts_with(BARE_CALL_START) {
                            let quoted = match BARE_CALL_START.starts_with(rest) {
                                true => s,
                                false => s + opener,
                            };
                            normal_text.push_str(&self.buffer[..quoted]);
                            self.buffer.drain(..quoted);
                            if quoted == s {
                                break;
                            }
                            continue;
                        }
                        normal_text.push_str(&self.buffer[..s]);
                        self.buffer = self.buffer[s + opener..].to_string();
                        self.in_tool_call = true;
                        self.bare_call = bare;
                        self.current_tool_name_sent = false;
                        self.current_function_name.clear();
                        self.current_parameters.clear();
                        continue;
                    } else {
                        break;
                    }
                }

                // We're in a tool call, try to parse function name if not sent yet
                if !self.current_tool_name_sent {
                    // As unary reads it, a bare header's name holds no `<`: up to one, it is prose.
                    if self.bare_call {
                        let name = &self.buffer[BARE_CALL_START.len()..];
                        let ends = name.find(['<', '>']);
                        if let Some(cut) = ends.filter(|&at| name[at..].starts_with('<')) {
                            let cut = BARE_CALL_START.len() + cut;
                            normal_text.push_str(&self.buffer[..cut]);
                            self.buffer.drain(..cut);
                            self.reset_streaming_state();
                            continue;
                        }
                    }
                    if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                        let declared = captures.get(0).unwrap().end();
                        // A bare one is a call only when it names a tool and call syntax
                        // follows; else it is prose, and so is its `<function=...>`.
                        if self.bare_call {
                            let named = (captures.get(1)).is_some_and(|name| {
                                tool_indices.contains_key(name.as_str().trim())
                            });
                            let rest = self.buffer[declared..].trim_start();
                            let call = ["<parameter=", "</function>"];
                            if !(named && call.iter().any(|token| rest.starts_with(token))) {
                                if named && call.iter().any(|token| token.starts_with(rest)) {
                                    break;
                                }
                                normal_text.push_str(&self.buffer[..declared]);
                                self.buffer.drain(..declared);
                                self.reset_streaming_state();
                                continue;
                            }
                        }
                        if let Some(name_match) = captures.get(1) {
                            let function_name = name_match.as_str().trim().to_string();

                            // Validate function name
                            if tool_indices.contains_key(&function_name) {
                                self.current_function_name = function_name.clone();
                                self.current_tool_name_sent = true;

                                // Initialize tool call tracking
                                if self.current_tool_id == -1 {
                                    self.current_tool_id = 0;
                                }

                                // Ensure tracking arrays are large enough
                                helpers::ensure_capacity(
                                    self.current_tool_id,
                                    &mut self.prev_tool_call_arr,
                                    &mut self.streamed_args_for_tool,
                                );

                                // Store tool call info
                                self.prev_tool_call_arr[self.current_tool_id as usize] = serde_json::json!({
                                    "name": function_name,
                                    "arguments": {}
                                });

                                // Send tool name
                                calls.push(ToolCallItem {
                                    tool_index: self.current_tool_id as usize,
                                    name: Some(function_name),
                                    parameters: String::new(),
                                });

                                // Remove processed function declaration from buffer
                                self.buffer =
                                    self.buffer[captures.get(0).unwrap().end()..].to_string();
                                continue;
                            } else {
                                // Invalid function name, reset state
                                tracing::warn!("Invalid function name: {}", function_name);
                                self.reset_streaming_state();
                                normal_text.push_str(&self.buffer);
                                self.buffer.clear();
                                break;
                            }
                        }
                    } else {
                        // Function name not complete yet, wait for more text
                        break;
                    }
                }

                // Parse parameters (only complete ones)
                if self.current_tool_name_sent {
                    let param_calls = self.parse_and_stream_parameters(tools);
                    calls.extend(param_calls);

                    // Check if tool call is complete
                    let end_token = self.call_end_token();
                    if let Some(end_pos) = self.buffer.find(end_token) {
                        // Close JSON object if we have parameters
                        let current_args =
                            &self.streamed_args_for_tool[self.current_tool_id as usize];
                        // A call with no parameters has `{}` for arguments.
                        if current_args.is_empty() {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: "{}".to_string(),
                            });
                            self.streamed_args_for_tool[self.current_tool_id as usize]
                                .push_str("{}");
                        } else {
                            // Count braces to check if JSON is complete
                            let open_braces = current_args.matches('{').count();
                            let close_braces = current_args.matches('}').count();
                            if open_braces > close_braces {
                                calls.push(ToolCallItem {
                                    tool_index: self.current_tool_id as usize,
                                    name: None,
                                    parameters: "}".to_string(),
                                });
                                self.streamed_args_for_tool[self.current_tool_id as usize]
                                    .push('}');
                            }
                        }

                        // Complete the tool call
                        self.buffer = self.buffer[end_pos + end_token.len()..].to_string();
                        self.reset_streaming_state();
                        self.current_tool_id += 1;
                        continue;
                    } else {
                        // Tool call not complete yet, wait for more text
                        break;
                    }
                }

                break;
            }

            Ok(StreamingParseResult { normal_text, calls })
        }

        fn has_tool_markers(&self, text: &str) -> bool {
            self.call_start(text).is_some()
        }

        fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
            helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
        }

        /// As `QwenXmlParser._is_truncated_call` reads an open `<function=NAME>` at end
        /// of input: a call when what follows the declared name is a parameter or the
        /// function's close, whole or cut off; then its open tags are closed.
        fn end_of_input_closers(&self, tools: &[Tool]) -> Option<String> {
            if !self.in_tool_call {
                return None;
            }
            let rest = match self.current_tool_name_sent {
                true => self.buffer.trim_start(),
                // A bare call is held at its declared name until what follows the name arrives.
                false => {
                    let declared = self.xml_function_pattern.captures(&self.buffer)?;
                    let tool_indices = helpers::get_tool_indices(tools);
                    if !(self.bare_call && tool_indices.contains_key(declared[1].trim())) {
                        return None;
                    }
                    self.buffer[declared.get(0)?.end()..].trim_start()
                }
            };
            let followers = ["<parameter=", "</function>"];
            let continues = |token: &str| rest.starts_with(token) || token.starts_with(rest);
            if !followers.into_iter().any(continues) {
                return None;
            }
            // Text after the call's `</function>` is not closed into its wrapper.
            let after = rest
                .split_once("</function>")
                .map(|(_, after)| after.trim());
            if after.is_some_and(|after| !after.is_empty()) {
                return None;
            }
            let mut closers = String::new();
            // Outside a value, a tag cut off is completed, so a bare call rereads whole.
            if rest.rfind("<parameter=") <= rest.rfind("</parameter>") {
                let cut = |tag: &'static str| {
                    helpers::ends_with_partial_token(rest, tag).map(|arrived| &tag[arrived..])
                };
                let tags = ["<parameter=", "</function>", "</parameter>"];
                closers.extend(tags.into_iter().find_map(cut));
            }
            let rest = format!("{rest}{closers}");
            // A value is closed; a parameter's cut-off name is no parameter, as on HTTP.
            if let Some(open) = rest.rfind("<parameter=") {
                if Some(open) > rest.rfind("</parameter>") && rest[open..].contains('>') {
                    closers.push_str("</parameter>");
                }
            }
            if !rest.contains("</function>") {
                closers.push_str("</function>");
            }
            if !self.bare_call {
                closers.push_str(self.tool_call_end_token);
            }
            Some(closers)
        }

        fn end_of_input_text(&mut self) -> String {
            if self.in_tool_call && self.current_tool_name_sent {
                return String::new();
            }
            let opener = match self.in_tool_call && !self.bare_call {
                true => self.tool_call_start_token,
                false => "",
            };
            let text = format!("{opener}{}", self.buffer);
            self.buffer.clear();
            self.reset_streaming_state();
            text
        }

        /// ATOM's HTTP parser reads a bare `<function=...>` call as a call; SGLang's and
        /// vLLM's return it as text.
        fn read_as_atom(&mut self) {
            self.bare_calls = true;
            self.extractor = WRAPPED_OR_BARE_CALLS.clone();
        }

        fn reset(&mut self) {
            helpers::reset_parser_state(
                &mut self.buffer,
                &mut self.prev_tool_call_arr,
                &mut self.current_tool_id,
                &mut self.current_tool_name_sent,
                &mut self.streamed_args_for_tool,
            );
            self.reset_streaming_state();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Closing a call cut off at the end of a long reply reads only the open call the
        /// parser still holds: the prose before it is released, not kept.
        #[tokio::test]
        async fn end_of_input_closers_read_only_the_open_call() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let mut parser = QwenCoderParser::new();
            let prose = "word ".repeat(200_000);
            parser.parse_incremental(&prose, &tools).await.unwrap();
            let call = "<tool_call><function=get_weather><parameter=city>Paris";
            parser.parse_incremental(call, &tools).await.unwrap();
            assert!(
                parser.buffer.len() < 64,
                "{} bytes held",
                parser.buffer.len()
            );
            let closers = parser.end_of_input_closers(&tools);
            assert_eq!(
                closers.as_deref(),
                Some("</parameter></function></tool_call>")
            );
        }

        /// A bare `<function=...>` call streams as one, the text before it as text; a
        /// bare name that prose follows stays text.
        #[tokio::test]
        async fn a_bare_function_call_streams_as_a_call() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let reply = "Sure. <function=get_weather><parameter=city>Paris</parameter></function>";
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let (mut text, mut name, mut args) = (String::new(), String::new(), String::new());
            for piece in reply.as_bytes().chunks(5) {
                let piece = std::str::from_utf8(piece).unwrap();
                let result = parser.parse_incremental(piece, &tools).await.unwrap();
                text.push_str(&result.normal_text);
                for call in result.calls {
                    name.push_str(call.name.as_deref().unwrap_or_default());
                    args.push_str(&call.parameters);
                }
            }
            assert_eq!((text.as_str(), name.as_str()), ("Sure. ", "get_weather"));
            assert_eq!(args, r#"{"city": "Paris"}"#);

            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let prose = "Write <function=get_weather> then its parameters.";
            let result = parser.parse_incremental(prose, &tools).await.unwrap();
            assert_eq!(result.normal_text, prose);
            assert!(result.calls.is_empty());
        }

        /// A parser not reading as ATOM returns a bare call as text, unary and streamed,
        /// as SGLang's and vLLM's do.
        #[tokio::test]
        async fn a_bare_function_call_is_text_unless_read_as_atom() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let reply = "Sure. <function=get_weather><parameter=city>Paris</parameter></function>";
            let mut parser = QwenCoderParser::new();
            let (text, calls) = parser
                .parse_complete_with_tools(reply, &tools)
                .await
                .unwrap();
            assert_eq!((text.as_str(), calls.len()), (reply, 0));
            let mut streamed = String::new();
            for piece in reply.as_bytes().chunks(5) {
                let piece = std::str::from_utf8(piece).unwrap();
                let result = parser.parse_incremental(piece, &tools).await.unwrap();
                assert!(result.calls.is_empty());
                streamed.push_str(&result.normal_text);
            }
            assert_eq!(streamed, reply);
            assert!(!parser.has_tool_markers(reply));
        }

        /// A bare call's reply keeps its prose up to the call, a `<tool_call>` it quotes too.
        #[tokio::test]
        async fn a_quoted_wrapper_before_a_bare_call_stays_text() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let prose = "Use <tool_call> as the wrapper. ";
            let reply = format!(
                "{prose}<function=get_weather><parameter=city>Paris</parameter></function>"
            );
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let (text, calls) = parser
                .parse_complete_with_tools(&reply, &[tool])
                .await
                .unwrap();
            assert_eq!((text.as_str(), calls.len()), (prose, 1));
            for size in 1..=9 {
                let (streamed, calls) = stream(&reply, size).await;
                assert_eq!((streamed.as_str(), calls), (prose, 1), "{size}-byte pieces");
            }
        }

        /// What `reply` streams in `size`-byte pieces, then at end of input, read as ATOM
        /// with get_weather declared: (text, calls named).
        async fn stream(reply: &str, size: usize) -> (String, usize) {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let mut read = Vec::new();
            for piece in reply.as_bytes().chunks(size) {
                let piece = std::str::from_utf8(piece).unwrap();
                read.push(parser.parse_incremental(piece, &tools).await.unwrap());
            }
            if let Some(closers) = parser.end_of_input_closers(&tools) {
                read.push(parser.parse_incremental(&closers, &tools).await.unwrap());
            }
            let calls = read.iter().flat_map(|result| &result.calls);
            let calls = calls.filter(|call| call.name.is_some()).count();
            let mut text: String = read.iter().map(|result| &*result.normal_text).collect();
            text.push_str(&parser.end_of_input_text());
            (text, calls)
        }

        /// A bare call naming no declared tool is text, unary as streamed.
        #[tokio::test]
        async fn a_bare_call_naming_no_declared_tool_is_text() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let reply = "The format is <function=YOUR FUNCTION NAME></function>.";
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let unary = parser.parse_complete_with_tools(reply, &[tool]).await;
            let (text, calls) = unary.unwrap();
            assert_eq!((text.as_str(), calls.len()), (reply, 0));
            for size in 1..=9 {
                assert_eq!(stream(reply, size).await, (reply.to_string(), 0));
            }
        }

        /// A bare header whose name runs into a `<` is text, and a call after it is read,
        /// unary as streamed.
        #[tokio::test]
        async fn a_call_after_an_unclosed_bare_header_is_read() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let prose = "Example <function=NAME ... ";
            let reply = format!("{prose}<function=get_weather></function>");
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let unary = parser.parse_complete_with_tools(&reply, &[tool]).await;
            let (text, calls) = unary.unwrap();
            assert_eq!((text.as_str(), calls.len()), (prose, 1));
            for size in 1..=9 {
                let (streamed, calls) = stream(&reply, size).await;
                assert_eq!((streamed.as_str(), calls), (prose, 1), "{size}-byte pieces");
            }
        }

        /// A reply's text is all of it outside its calls, unary as streamed.
        #[tokio::test]
        async fn text_between_and_after_calls_is_text() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let call = "<function=get_weather><parameter=city>Paris</parameter></function>";
            let reply = format!("Before {call} Between <tool_call>{call}</tool_call> After");
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let unary = parser.parse_complete_with_tools(&reply, &[tool]).await;
            let (text, calls) = unary.unwrap();
            let outside = "Before  Between  After";
            assert_eq!((text.as_str(), calls.len()), (outside, 2));
            for size in 1..=9 {
                assert_eq!(stream(&reply, size).await, (outside.to_string(), 2));
            }
        }

        /// Text a parser holds at end of input as a call's possible start, and that never
        /// became one, is text.
        #[tokio::test]
        async fn a_held_opener_no_call_followed_is_text_at_end_of_input() {
            for reply in [
                "The prefix is <fun",
                "The prefix is <tool_c",
                "Cut: <function=get_wea",
                "Cut: <tool_call><function=get_wea",
                "Quote: <tool_call>\n",
            ] {
                for size in 1..=5 {
                    let read = stream(reply, size).await;
                    assert_eq!(read, (reply.to_string(), 0), "{size}-byte pieces");
                }
            }
        }

        /// Each call in one delta streams its own arguments, `{}` when it has none.
        #[tokio::test]
        async fn calls_in_one_delta_stream_their_own_arguments() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let call = |city: &str| match city {
                "" => "<function=get_weather></function>".to_string(),
                _ => format!("<function=get_weather><parameter=city>{city}</parameter></function>"),
            };
            for (a, b, args) in [
                (
                    "Paris",
                    "London",
                    [r#"{"city": "Paris"}"#, r#"{"city": "London"}"#],
                ),
                ("", "", ["{}", "{}"]),
            ] {
                let bare = call(a) + &call(b);
                let wrapped = format!(
                    "<tool_call>{}</tool_call><tool_call>{}</tool_call>",
                    call(a),
                    call(b)
                );
                for reply in [bare, wrapped] {
                    let mut parser = QwenCoderParser::new();
                    parser.read_as_atom();
                    let result = parser.parse_incremental(&reply, &tools).await.unwrap();
                    let mut streamed = [String::new(), String::new()];
                    for call in result.calls {
                        streamed[call.tool_index].push_str(&call.parameters);
                    }
                    assert_eq!(streamed, args, "{reply}");
                }
            }
        }

        /// A cut-off wrapper is not closed past text that follows the call's `</function>`.
        #[tokio::test]
        async fn text_after_a_cut_off_wrapper_s_function_is_not_closed_into_the_call() {
            let tool: Tool = serde_json::from_value(serde_json::json!({
                "type": "function", "function": {"name": "get_weather", "parameters": {}},
            }))
            .unwrap();
            let tools = [tool];
            let reply = "Before <tool_call><function=get_weather><parameter=city>Paris</parameter></function> After";
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            parser.parse_incremental(reply, &tools).await.unwrap();
            assert_eq!(parser.end_of_input_closers(&tools), None);
            let mut parser = QwenCoderParser::new();
            parser.read_as_atom();
            let (text, calls) = parser
                .parse_complete_with_tools(reply, &tools)
                .await
                .unwrap();
            assert!(text.ends_with(" After") && calls.len() == 1, "{text}");
        }

        #[test]
        fn test_html_unescape_basic() {
            assert_eq!(html_unescape("&amp;"), "&");
            assert_eq!(html_unescape("&lt;"), "<");
            assert_eq!(html_unescape("&gt;"), ">");
            assert_eq!(html_unescape("&quot;"), "\"");
            assert_eq!(html_unescape("&apos;"), "'");
        }

        #[test]
        fn test_html_unescape_numeric() {
            assert_eq!(html_unescape("&#60;"), "<");
            assert_eq!(html_unescape("&#x3C;"), "<");
            assert_eq!(html_unescape("&#x3c;"), "<");
        }

        #[test]
        fn test_html_unescape_mixed() {
            assert_eq!(
                html_unescape("Hello &amp; World &lt;tag&gt;"),
                "Hello & World <tag>"
            );
        }

        #[test]
        fn test_html_unescape_unknown() {
            // Unknown entities with semicolon should be preserved as-is
            assert_eq!(html_unescape("&unknown;"), "&unknown;");
            // Unterminated entities should NOT have semicolon added
            assert_eq!(html_unescape("&foo bar"), "&foo bar");
            assert_eq!(html_unescape("&"), "&");
            assert_eq!(html_unescape("& "), "& ");
        }

        #[test]
        fn test_safe_val_json() {
            assert_eq!(safe_val("42"), Value::Number(42.into()));
            assert_eq!(safe_val("1.5"), serde_json::json!(1.5));
            assert_eq!(safe_val("true"), Value::Bool(true));
            assert_eq!(safe_val("false"), Value::Bool(false));
            assert_eq!(safe_val("null"), Value::Null);
            assert_eq!(
                safe_val(r#"{"key": "value"}"#),
                serde_json::json!({"key": "value"})
            );
            assert_eq!(safe_val(r#"[1, 2, 3]"#), serde_json::json!([1, 2, 3]));
        }

        #[test]
        fn test_safe_val_python_literals() {
            assert_eq!(safe_val("True"), Value::Bool(true));
            assert_eq!(safe_val("False"), Value::Bool(false));
            assert_eq!(safe_val("None"), Value::Null);
        }

        #[test]
        fn test_safe_val_string_fallback() {
            assert_eq!(
                safe_val("hello world"),
                Value::String("hello world".to_string())
            );
            assert_eq!(safe_val("  spaces  "), Value::String("spaces".to_string()));
        }

        #[test]
        fn test_safe_val_preserves_html_entities() {
            assert_eq!(
                safe_val("&lt;div&gt;"),
                Value::String("&lt;div&gt;".to_string())
            );
            assert_eq!(
                safe_val("Tom &amp; Jerry"),
                Value::String("Tom &amp; Jerry".to_string())
            );
        }
    }
}

mod json {
    use async_trait::async_trait;
    use openai_protocol::common::Tool;
    use regex::Regex;
    use serde_json::Value;

    use crate::tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        partial_json::PartialJson,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall},
    };

    /// Qwen format parser for tool calls
    ///
    /// Handles the Qwen 2.5/3 specific format:
    /// `<tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>`
    ///
    /// Features:
    /// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    /// - Each individual call is separated by `\n`
    /// - Function Call Object: JSON object with "name" and "arguments" fields
    ///
    /// Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    pub struct QwenParser {
        /// Parser for handling incomplete JSON during streaming
        partial_json: PartialJson,

        /// Regex for extracting tool calls in parse_complete
        extractor: Regex,

        /// Buffer for accumulating incomplete patterns across chunks
        buffer: String,

        /// Stores complete tool call info (name and arguments) for each tool being parsed
        prev_tool_call_arr: Vec<Value>,

        /// Index of currently streaming tool call (-1 means no active tool)
        current_tool_id: i32,

        /// Flag for whether current tool's name has been sent to client
        current_tool_name_sent: bool,

        /// Tracks raw JSON string content streamed to client for each tool's arguments
        streamed_args_for_tool: Vec<String>,

        /// Buffer for normal text that might precede partial end tokens
        normal_text_buffer: String,

        /// Token configuration
        /// Start/end tokens for each individual tool call (not the entire sequence)
        individual_tool_start_token: &'static str,
        individual_tool_end_token: &'static str,
        tool_call_separator: &'static str,
    }

    impl QwenParser {
        /// Create a new Qwen parser
        pub fn new() -> Self {
            // Use (?s) flag for DOTALL mode to handle newlines
            let pattern = r"(?s)<tool_call>\n(.*?)\n</tool_call>";
            let extractor = Regex::new(pattern).expect("Valid regex pattern");

            Self {
                partial_json: PartialJson::default(),
                extractor,
                buffer: String::new(),
                prev_tool_call_arr: Vec::new(),
                current_tool_id: -1,
                current_tool_name_sent: false,
                streamed_args_for_tool: Vec::new(),
                normal_text_buffer: String::new(),
                individual_tool_start_token: "<tool_call>\n",
                individual_tool_end_token: "\n</tool_call>",
                tool_call_separator: "\n",
            }
        }

        /// Parse a single JSON object into a ToolCall
        fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
            let name = obj.get("name").and_then(|v| v.as_str());

            if let Some(name) = name {
                // Get arguments - Qwen uses "arguments" key
                let empty_obj = Value::Object(serde_json::Map::new());
                let args = obj.get("arguments").unwrap_or(&empty_obj);

                // Convert arguments to JSON string
                let arguments = serde_json::to_string(args)
                    .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

                Ok(Some(ToolCall {
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                }))
            } else {
                Ok(None)
            }
        }
    }

    impl Default for QwenParser {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl ToolParser for QwenParser {
        async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
            // Check if text contains Qwen format
            if !self.has_tool_markers(text) {
                return Ok((text.to_string(), vec![]));
            }

            // Find where the first tool call begins
            let idx = text.find("<tool_call>").unwrap(); // Safe because has_tool_markers checked
            let normal_text = text[..idx].to_string();

            // Extract tool calls
            let mut tools = Vec::new();
            for captures in self.extractor.captures_iter(text) {
                if let Some(json_str) = captures.get(1) {
                    let parsed = serde_json::from_str::<Value>(json_str.as_str().trim())
                        .map_err(|e| ParserError::ParsingFailed(e.to_string()))
                        .and_then(|v| self.parse_single_object(&v));

                    match parsed {
                        Ok(Some(tool)) => tools.push(tool),
                        Ok(None) => continue,
                        Err(e) => {
                            tracing::debug!("Failed to parse tool call: {:?}", e);
                            continue;
                        }
                    }
                }
            }

            // If no tools were successfully parsed despite having markers, return entire text as fallback
            if tools.is_empty() {
                return Ok((text.to_string(), vec![]));
            }

            Ok((normal_text, tools))
        }

        async fn parse_incremental(
            &mut self,
            chunk: &str,
            tools: &[Tool],
        ) -> ParserResult<StreamingParseResult> {
            // Append new text to buffer
            self.buffer.push_str(chunk);
            let current_text = &self.buffer.clone();

            // Check if current_text has tool_call
            let has_tool_start = self.has_tool_markers(current_text)
                || (self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator));

            if !has_tool_start {
                // Only clear buffer if we're sure no tool call is starting
                if helpers::ends_with_partial_token(&self.buffer, self.individual_tool_start_token)
                    .is_none()
                {
                    let normal_text = self.buffer.clone();
                    self.buffer.clear();

                    return Ok(StreamingParseResult {
                        normal_text,
                        calls: vec![],
                    });
                } else {
                    // Might be partial individual_tool_start_token, keep buffering
                    return Ok(StreamingParseResult::default());
                }
            }

            // Build tool indices
            let tool_indices = helpers::get_tool_indices(tools);

            // Determine start index for JSON parsing
            let start_idx = if let Some(pos) = current_text.find(self.individual_tool_start_token) {
                pos + self.individual_tool_start_token.len()
            } else if self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator)
            {
                self.tool_call_separator.len()
            } else {
                0
            };

            let mut result = helpers::handle_json_tool_streaming(
                current_text,
                start_idx,
                &mut self.partial_json,
                &tool_indices,
                &mut self.buffer,
                &mut self.current_tool_id,
                &mut self.current_tool_name_sent,
                &mut self.streamed_args_for_tool,
                &mut self.prev_tool_call_arr,
            )?;

            // Qwen-specific: Handle partial end tokens in normal text
            // After tool calls complete, normal text might contain partial "</tool_call>" tags
            if !result.normal_text.is_empty() {
                self.normal_text_buffer.push_str(&result.normal_text);

                // Check if buffer contains complete end token (without leading newline)
                let end_token_without_newline = &self.individual_tool_end_token[1..]; // "</tool_call>"
                if self.normal_text_buffer.contains(end_token_without_newline) {
                    // Complete end token found - clean it and return
                    let cleaned_text = self
                        .normal_text_buffer
                        .replace(end_token_without_newline, "");
                    self.normal_text_buffer.clear();
                    result.normal_text = cleaned_text;
                } else {
                    // Check if buffer might contain partial end token at the end
                    if let Some(partial_match_len) = helpers::ends_with_partial_token(
                        &self.normal_text_buffer,
                        end_token_without_newline,
                    ) {
                        // Keep potential partial match in buffer, return the rest
                        let split_point = self.normal_text_buffer.len() - partial_match_len;
                        result.normal_text = self.normal_text_buffer[..split_point].to_string();
                        self.normal_text_buffer =
                            self.normal_text_buffer[split_point..].to_string();
                    } else {
                        // No partial match, return all buffered text
                        result.normal_text = self.normal_text_buffer.clone();
                        self.normal_text_buffer.clear();
                    }
                }
            }

            Ok(result)
        }

        fn has_tool_markers(&self, text: &str) -> bool {
            text.contains("<tool_call>")
        }

        fn get_unstreamed_tool_args(&self) -> Option<Vec<crate::tool_parser::types::ToolCallItem>> {
            helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
        }

        fn reset(&mut self) {
            helpers::reset_parser_state(
                &mut self.buffer,
                &mut self.prev_tool_call_arr,
                &mut self.current_tool_id,
                &mut self.current_tool_name_sent,
                &mut self.streamed_args_for_tool,
            );
        }
    }
}

pub use coder::QwenCoderParser;
pub use json::QwenParser;
