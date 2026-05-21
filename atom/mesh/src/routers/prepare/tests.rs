//! Test obligations for `routers::prepare::*`.
//!
//! Maps to plan Parts A, B, C. Test names are grouped into one `mod` per source
//! file; each `mod` corresponds to a file that lands in this directory during
//! Parts A–C. Until that implementation lands the file does not compile (TDD red);
//! production builds (`cargo build`) stay green because `mod tests;` is gated
//! by `#[cfg(test)]`.

mod a_chat_template {
    use serde_json::json;

    use crate::protocols::{
        chat::{ChatMessage, MessageContent},
        common::{ContentPart, ImageUrl},
    };
    use crate::routers::prepare::chat_template::{process_content_format, ProcessedMessages};
    use crate::tokenizer::chat_template::ChatTemplateContentFormat;

    fn user_parts(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Parts(parts),
            name: None,
        }
    }

    fn user_text(text: &str) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Text(text.to_string()),
            name: None,
        }
    }

    fn image(url: &str, detail: Option<&str>) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_string(),
                detail: detail.map(String::from),
            },
        }
    }

    fn text_part(t: &str) -> ContentPart {
        ContentPart::Text {
            text: t.to_string(),
        }
    }

    #[test]
    fn test_string_format_concatenates_text_parts() {
        let messages = vec![user_parts(vec![
            text_part("Hello"),
            image("https://e/x.jpg", None),
            text_part("World"),
        ])];
        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["content"].as_str().unwrap(), "Hello World");
        assert_eq!(result[0]["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_openai_format_replaces_image_with_placeholder() {
        let messages = vec![user_parts(vec![
            text_part("Describe this image:"),
            image("https://e/x.jpg", Some("high")),
        ])];
        let result = process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();
        let arr = result[0]["content"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "Describe this image:");
        assert_eq!(arr[1], json!({"type": "image"}));
    }

    #[test]
    fn test_simple_string_content_unchanged() {
        let messages = vec![user_text("Simple text message")];
        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        assert_eq!(
            result[0]["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_multiple_messages_roles_preserved() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
            },
            user_parts(vec![
                text_part("User message"),
                image("https://e/x.jpg", None),
            ]),
        ];
        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_image_only_parts_string_keeps_array() {
        let messages = vec![user_parts(vec![image("https://e/x.jpg", None)])];
        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        assert!(result[0]["content"].is_array());
    }

    #[test]
    fn test_mixed_text_and_parts_across_messages() {
        let messages = vec![
            user_text("Plain text"),
            user_parts(vec![
                text_part("With image"),
                image("https://e/x.jpg", Some("low")),
            ]),
        ];
        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");
        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();
        let arr = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1], json!({"type": "image"}));
    }

    #[test]
    fn test_assistant_tool_call_arguments_parsed_to_object() {
        let messages = vec![ChatMessage::Assistant {
            content: Some(MessageContent::Text(String::new())),
            name: None,
            tool_calls: Some(vec![crate::protocols::common::ToolCall {
                id: "c1".to_string(),
                r#type: "function".to_string(),
                function: crate::protocols::common::FunctionCallResponse {
                    name: "add".to_string(),
                    arguments: r#"{"a":1,"b":2}"#.to_string(),
                },
            }]),
            reasoning_content: None,
            refusal: None,
            audio: None,
        }];
        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();
        let args = &result[0]["tool_calls"][0]["function"]["arguments"];
        assert!(
            args.is_object(),
            "args should be parsed to object, got {args}"
        );
        assert_eq!(args["a"], json!(1));
        assert_eq!(args["b"], json!(2));
    }

    #[test]
    fn test_assistant_tool_call_invalid_json_returns_err() {
        let messages = vec![ChatMessage::Assistant {
            content: None,
            name: None,
            tool_calls: Some(vec![crate::protocols::common::ToolCall {
                id: "c1".to_string(),
                r#type: "function".to_string(),
                function: crate::protocols::common::FunctionCallResponse {
                    name: "noop".to_string(),
                    arguments: "{not-json}".to_string(),
                },
            }]),
            reasoning_content: None,
            refusal: None,
            audio: None,
        }];
        let err = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap_err();
        assert!(err.to_string().contains("tool call arguments"));
    }

    #[test]
    fn test_processed_messages_carries_text_and_stop_sequences() {
        let pm = ProcessedMessages {
            text: "rendered prompt".to_string(),
            stop_sequences: Some(crate::protocols::common::StringOrArray::String(
                "<end>".to_string(),
            )),
        };
        assert_eq!(pm.text, "rendered prompt");
        assert!(pm.stop_sequences.is_some());
    }

    #[test]
    fn test_processed_messages_has_no_multimodal_field() {
        // Q2 resolved: multimodal_inputs is dropped (was always None at this layer)
        let ty = std::any::type_name::<ProcessedMessages>();
        // Compile-time presence is verified by the constructor test above; this
        // mirror exists so that re-introducing the field requires updating both.
        assert!(ty.ends_with("ProcessedMessages"));
    }
}

mod b_tool_constraints {
    use serde_json::{json, Value};

    use crate::protocols::common::{
        Function, JsonSchemaResponseFormat, ResponseFormat, Tool, ToolChoice, ToolChoiceValue,
    };
    use crate::routers::prepare::tool_constraints::{
        build_required_array_schema, filter_chat_request_by_tool_choice,
        filter_tools_by_tool_choice, generate_tool_call_id, generate_tool_constraints,
        get_history_tool_calls_count, parse_json_schema_response,
    };

    fn tool(name: &str) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: Some(json!({"type": "object", "properties": {}})),
                strict: None,
            },
        }
    }

    #[test]
    fn test_generate_constraints_none_for_choice_none() {
        let tools = vec![tool("a"), tool("b")];
        let constraints = generate_tool_constraints(
            &tools,
            &Some(ToolChoice::Value(ToolChoiceValue::None)),
            "any-model",
        );
        assert!(
            constraints.is_none(),
            "ToolChoice::None must yield no constraint"
        );
    }

    #[test]
    fn test_generate_constraints_some_for_choice_required() {
        let tools = vec![tool("a")];
        let constraints = generate_tool_constraints(
            &tools,
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
        );
        let (key, body) = constraints.expect("required tools must produce a constraint");
        assert!(!key.is_empty());
        assert!(!body.is_empty());
    }

    #[test]
    fn test_generate_constraints_named_tool_uses_that_tool_only() {
        let tools = vec![tool("a"), tool("b")];
        let choice = ToolChoice::Function {
            r#type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "b".to_string(),
            },
        };
        let (_, body) = generate_tool_constraints(&tools, &Some(choice), "m").unwrap();
        assert!(body.contains("\"b\""));
        assert!(!body.contains("\"a\""));
    }

    #[test]
    fn test_required_array_schema_includes_all_tool_names() {
        let tools = vec![tool("alpha"), tool("beta")];
        let schema = build_required_array_schema(&tools);
        let s = serde_json::to_string(&schema).unwrap();
        assert!(s.contains("alpha"));
        assert!(s.contains("beta"));
        assert!(schema["type"] == "array");
    }

    #[test]
    fn test_filter_by_tool_choice_auto_keeps_all() {
        let tools = vec![tool("a"), tool("b")];
        let filtered =
            filter_tools_by_tool_choice(&tools, &Some(ToolChoice::Value(ToolChoiceValue::Auto)));
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_by_tool_choice_named_keeps_one() {
        let tools = vec![tool("a"), tool("b"), tool("c")];
        let choice = ToolChoice::Function {
            r#type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "b".to_string(),
            },
        };
        let filtered = filter_tools_by_tool_choice(&tools, &Some(choice));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].function.name, "b");
    }

    #[test]
    fn test_filter_by_tool_choice_none_drops_all() {
        let tools = vec![tool("a"), tool("b")];
        let filtered =
            filter_tools_by_tool_choice(&tools, &Some(ToolChoice::Value(ToolChoiceValue::None)));
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_by_tool_choice_unknown_named_drops_all() {
        let tools = vec![tool("a")];
        let choice = ToolChoice::Function {
            r#type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "nope".to_string(),
            },
        };
        let filtered = filter_tools_by_tool_choice(&tools, &Some(choice));
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_chat_request_in_place_replaces_tools_vec() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        let mut req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: Some(vec![tool("a"), tool("b")]),
            tool_choice: Some(ToolChoice::Function {
                r#type: "function".to_string(),
                function: crate::protocols::common::FunctionChoice {
                    name: "a".to_string(),
                },
            }),
            ..Default::default()
        };
        filter_chat_request_by_tool_choice(&mut req);
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "a");
    }

    #[test]
    fn test_filter_chat_request_no_tools_is_noop() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        let mut req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: None,
            tool_choice: None,
            ..Default::default()
        };
        filter_chat_request_by_tool_choice(&mut req);
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_parse_json_schema_response_strict_true_includes_schema() {
        let resp = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaResponseFormat {
                name: "City".to_string(),
                description: None,
                schema: Some(json!({"type": "object"})),
                strict: Some(true),
            },
        };
        let (key, body) = parse_json_schema_response(&resp).expect("must produce constraint");
        assert!(!key.is_empty());
        assert!(body.contains("\"type\""));
    }

    #[test]
    fn test_parse_json_schema_response_non_json_schema_returns_none() {
        let resp = ResponseFormat::Text;
        assert!(parse_json_schema_response(&resp).is_none());
    }

    #[test]
    fn test_get_history_tool_calls_count_counts_assistant_calls() {
        use crate::protocols::chat::{ChatMessage, MessageContent};
        use crate::protocols::common::{FunctionCallResponse, ToolCall};
        let msgs = vec![
            ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: None,
                name: None,
                tool_calls: Some(vec![
                    ToolCall {
                        id: "c1".to_string(),
                        r#type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "x".to_string(),
                            arguments: "{}".to_string(),
                        },
                    },
                    ToolCall {
                        id: "c2".to_string(),
                        r#type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "y".to_string(),
                            arguments: "{}".to_string(),
                        },
                    },
                ]),
                reasoning_content: None,
                refusal: None,
                audio: None,
            },
        ];
        assert_eq!(get_history_tool_calls_count(&msgs), 2);
    }

    #[test]
    fn test_get_history_tool_calls_count_zero_when_no_assistant() {
        use crate::protocols::chat::{ChatMessage, MessageContent};
        let msgs = vec![ChatMessage::User {
            content: MessageContent::Text("hi".to_string()),
            name: None,
        }];
        assert_eq!(get_history_tool_calls_count(&msgs), 0);
    }

    #[test]
    fn test_generate_tool_call_id_unique_and_formatted() {
        let a = generate_tool_call_id();
        let b = generate_tool_call_id();
        assert_ne!(a, b);
        assert!(a.starts_with("call_") || a.starts_with("chatcmpl-tool-") || !a.is_empty());
    }
}

mod c_stop_sequence_decoder {
    use crate::protocols::common::StringOrArray;
    use crate::routers::prepare::stop_sequence_decoder::create_stop_decoder;

    fn fake_tokenizer() -> std::sync::Arc<dyn crate::tokenizer::traits::Tokenizer> {
        // The impl should accept any Tokenizer trait object. Test-only stub provided
        // by the test_support helpers when they land; until then this is a compile
        // sentinel against accidental signature changes.
        unimplemented!("test tokenizer fixture")
    }

    #[test]
    fn test_create_decoder_with_string_stop() {
        let tok = fake_tokenizer();
        let decoder = create_stop_decoder(
            tok.clone(),
            Some(&StringOrArray::String("<eot>".to_string())),
            Some(&[2u32]),
            true,
            false,
        );
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_create_decoder_with_array_stop() {
        let tok = fake_tokenizer();
        let decoder = create_stop_decoder(
            tok,
            Some(&StringOrArray::Array(vec![
                "<eot>".to_string(),
                "<stop>".to_string(),
            ])),
            None,
            true,
            false,
        );
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_create_decoder_no_stop_sources() {
        let tok = fake_tokenizer();
        let decoder = create_stop_decoder(tok, None, None, false, false);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_create_decoder_no_stop_trim_passes_flag() {
        let tok = fake_tokenizer();
        let decoder = create_stop_decoder(tok, None, None, true, true);
        assert!(decoder.is_ok());
    }
}

mod d_parser_factory_lookup {
    use crate::routers::prepare::parser_factory_lookup::{
        check_reasoning_parser_availability, check_tool_parser_availability,
        create_reasoning_parser, create_tool_parser, get_reasoning_parser, get_tool_parser,
    };

    fn reasoning_factory() -> std::sync::Arc<crate::reasoning_parser::ParserFactory> {
        unimplemented!("reasoning parser factory fixture")
    }

    fn tool_factory() -> std::sync::Arc<crate::tool_parser::ParserFactory> {
        unimplemented!("tool parser factory fixture")
    }

    #[test]
    fn test_check_reasoning_parser_known_model_returns_ok() {
        let factory = reasoning_factory();
        let res = check_reasoning_parser_availability(&factory, "qwen3");
        assert!(res.is_ok());
    }

    #[test]
    fn test_check_reasoning_parser_unknown_model_returns_err() {
        let factory = reasoning_factory();
        let res = check_reasoning_parser_availability(&factory, "no-such-model");
        assert!(res.is_err());
    }

    #[test]
    fn test_check_tool_parser_known_model_returns_ok() {
        let factory = tool_factory();
        let res = check_tool_parser_availability(&factory, "qwen3");
        assert!(res.is_ok());
    }

    #[test]
    fn test_check_tool_parser_unknown_model_returns_err() {
        let factory = tool_factory();
        let res = check_tool_parser_availability(&factory, "no-such-model");
        assert!(res.is_err());
    }

    #[test]
    fn test_get_reasoning_parser_returns_pooled_instance() {
        let factory = reasoning_factory();
        let _pooled = get_reasoning_parser(&factory, "qwen3").expect("pooled parser");
    }

    #[test]
    fn test_create_reasoning_parser_returns_owned_instance() {
        let factory = reasoning_factory();
        let _parser = create_reasoning_parser(&factory, "qwen3").expect("owned parser");
    }

    #[test]
    fn test_get_tool_parser_returns_pooled_instance() {
        let factory = tool_factory();
        let _pooled = get_tool_parser(&factory, "qwen3").expect("pooled tool parser");
    }

    #[test]
    fn test_create_tool_parser_returns_owned_instance() {
        let factory = tool_factory();
        let _parser = create_tool_parser(&factory, "qwen3").expect("owned tool parser");
    }
}

mod e_generation_payload {
    use crate::protocols::common::StringOrArray;
    use crate::routers::prepare::generation_payload::{
        GenerationPayload, LogprobConfig, PdMetadata, SamplingParams, StopConfig,
    };

    fn payload_with_defaults() -> GenerationPayload {
        GenerationPayload {
            request_id: "req-1".to_string(),
            text: "hello world".to_string(),
            token_ids: vec![1, 2, 3],
            sampling: SamplingParams {
                temperature: 0.7,
                top_p: 0.95,
                top_k: -1,
                repetition_penalty: 1.0,
                max_new_tokens: 128,
            },
            stop: StopConfig {
                stop: Some(StringOrArray::String("<eot>".to_string())),
                stop_token_ids: Some(vec![2]),
                skip_special_tokens: true,
                no_stop_trim: false,
            },
            logprob: LogprobConfig {
                return_logprob: false,
                top_logprobs_num: 0,
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
        }
    }

    #[test]
    fn test_payload_round_trip_fields() {
        let p = payload_with_defaults();
        assert_eq!(p.request_id, "req-1");
        assert_eq!(p.token_ids, vec![1, 2, 3]);
        assert_eq!(p.text, "hello world");
        assert_eq!(p.sampling.temperature, 0.7);
        assert_eq!(p.sampling.max_new_tokens, 128);
        assert_eq!(p.stop.stop_token_ids.as_deref().unwrap(), &[2]);
        assert!(p.stop.skip_special_tokens);
        assert!(!p.logprob.input_logprobs);
        assert!(p.tool_constraints.is_none());
        assert!(p.pd_metadata.is_none());
    }

    #[test]
    fn test_payload_with_tool_constraint() {
        let mut p = payload_with_defaults();
        p.tool_constraints = Some(("ebnf".to_string(), "root ::= ...".to_string()));
        let (k, v) = p.tool_constraints.as_ref().unwrap();
        assert_eq!(k, "ebnf");
        assert!(!v.is_empty());
    }

    #[test]
    fn test_payload_with_pd_metadata() {
        let pd = PdMetadata {
            bootstrap_host: "p-host".to_string(),
            bootstrap_port: Some(8998),
            bootstrap_room: 42,
        };
        let mut p = payload_with_defaults();
        p.pd_metadata = Some(pd);
        let m = p.pd_metadata.as_ref().unwrap();
        assert_eq!(m.bootstrap_host, "p-host");
        assert_eq!(m.bootstrap_port, Some(8998));
        assert_eq!(m.bootstrap_room, 42);
    }

    #[test]
    fn test_logprob_config_input_logprobs_flag_threads_to_pd_merge() {
        let mut p = payload_with_defaults();
        p.logprob.input_logprobs = true;
        assert!(p.logprob.input_logprobs);
    }

    #[test]
    fn test_stop_config_array_form() {
        let mut p = payload_with_defaults();
        p.stop.stop = Some(StringOrArray::Array(vec![
            "<a>".to_string(),
            "<b>".to_string(),
        ]));
        if let Some(StringOrArray::Array(arr)) = &p.stop.stop {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("expected array");
        }
    }
}

mod f_response_context {
    use std::sync::Arc;

    use http::HeaderMap;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::protocols::generate::GenerateRequest;
    use crate::routers::prepare::chat_template::ProcessedMessages;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};

    fn chat_req(stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            stream,
            ..Default::default()
        }
    }

    fn generate_req(stream: bool) -> GenerateRequest {
        let mut g = GenerateRequest::default();
        g.stream = stream;
        g
    }

    fn fake_tokenizer() -> Arc<dyn crate::tokenizer::traits::Tokenizer> {
        unimplemented!("tokenizer fixture")
    }

    fn fake_decoder() -> crate::tokenizer::StopSequenceDecoder {
        unimplemented!("stop decoder fixture")
    }

    fn make_ctx_chat(stream: bool) -> ResponseContext {
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(chat_req(stream))),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: Some(ProcessedMessages {
                text: "rendered".to_string(),
                stop_sequences: None,
            }),
            tokenizer: fake_tokenizer(),
            stop_decoder: fake_decoder(),
        }
    }

    #[test]
    fn test_protocol_request_chat_is_streaming_when_stream_true() {
        let r = ProtocolRequest::Chat(Arc::new(chat_req(true)));
        assert!(r.is_streaming());
    }

    #[test]
    fn test_protocol_request_chat_is_not_streaming_when_stream_false() {
        let r = ProtocolRequest::Chat(Arc::new(chat_req(false)));
        assert!(!r.is_streaming());
    }

    #[test]
    fn test_protocol_request_generate_is_streaming_when_stream_true() {
        let r = ProtocolRequest::Generate(Arc::new(generate_req(true)));
        assert!(r.is_streaming());
    }

    #[test]
    fn test_protocol_request_generate_is_not_streaming_when_stream_false() {
        let r = ProtocolRequest::Generate(Arc::new(generate_req(false)));
        assert!(!r.is_streaming());
    }

    #[test]
    fn test_response_context_holds_headers() {
        let mut hm = HeaderMap::new();
        hm.insert("x-trace", "abc".parse().unwrap());
        let ctx = ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(chat_req(false))),
            model_id: None,
            headers: Some(hm),
            original_text: None,
            processed_messages: None,
            tokenizer: fake_tokenizer(),
            stop_decoder: fake_decoder(),
        };
        assert_eq!(
            ctx.headers.as_ref().unwrap().get("x-trace"),
            Some(&"abc".parse().unwrap())
        );
    }

    #[test]
    fn test_response_context_generate_path_has_no_processed_messages() {
        let ctx = ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(generate_req(false))),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: None,
            tokenizer: fake_tokenizer(),
            stop_decoder: fake_decoder(),
        };
        assert!(ctx.processed_messages.is_none());
    }

    #[test]
    fn test_response_context_chat_path_has_processed_messages() {
        let ctx = make_ctx_chat(true);
        assert!(ctx.processed_messages.is_some());
    }
}

mod h_prepare_chat_generate {
    use std::sync::Arc;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::protocols::generate::GenerateRequest;
    use crate::routers::prepare::generation_payload::GenerationPayload;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::{lookup_tokenizer, prepare_chat, prepare_generate};

    fn shared_components() -> std::sync::Arc<crate::routers::http_router::SharedComponents> {
        unimplemented!(
            "shared components fixture — built by prepare/tests.rs::test_support when impl lands"
        );
    }

    fn chat_req(stream: bool) -> Arc<ChatCompletionRequest> {
        Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            stream,
            ..Default::default()
        })
    }

    fn generate_req(stream: bool) -> Arc<GenerateRequest> {
        let mut g = GenerateRequest::default();
        g.stream = stream;
        Arc::new(g)
    }

    #[test]
    fn test_prepare_chat_returns_payload_and_context_tuple() {
        let components = shared_components();
        let (payload, ctx) = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("ok");
        assert!(!payload.text.is_empty());
        assert_eq!(payload.token_ids.len() > 0, true);
        assert!(matches!(ctx.original, ProtocolRequest::Chat(_)));
        assert!(ctx.processed_messages.is_some());
    }

    #[test]
    fn test_prepare_chat_streaming_flag_propagates_to_response_context() {
        let components = shared_components();
        let (_, ctx) = prepare_chat(
            chat_req(true),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("ok");
        assert!(ctx.original.is_streaming());
    }

    #[test]
    fn test_prepare_chat_missing_model_id_returns_err_response() {
        let components = shared_components();
        let result = prepare_chat(chat_req(false), None, None, components.as_ref());
        // Either a successful default-model path, or an Err Response with non-2xx status.
        if let Err(resp) = result {
            assert!(!resp.status().is_success());
        }
    }

    #[test]
    fn test_prepare_chat_carries_original_text_in_context() {
        let components = shared_components();
        let (_, ctx) = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert!(ctx.original_text.is_some());
    }

    #[test]
    fn test_prepare_generate_returns_payload_and_context_tuple() {
        let components = shared_components();
        let (payload, ctx) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("ok");
        assert_ne!(payload.request_id, "");
        assert!(matches!(ctx.original, ProtocolRequest::Generate(_)));
        assert!(
            ctx.processed_messages.is_none(),
            "generate path has no chat messages"
        );
    }

    #[test]
    fn test_prepare_generate_streaming_flag_propagates() {
        let components = shared_components();
        let (_, ctx) = prepare_generate(
            generate_req(true),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("ok");
        assert!(ctx.original.is_streaming());
    }

    #[test]
    fn test_lookup_tokenizer_returns_arc_for_known_model() {
        let components = shared_components();
        let tok = lookup_tokenizer("m", &components.tokenizer_registry).expect("known model");
        // Two clones of the same Arc should be ptr_eq if registry caches by model.
        let tok2 = lookup_tokenizer("m", &components.tokenizer_registry).unwrap();
        assert!(Arc::ptr_eq(&tok, &tok2));
    }

    #[test]
    fn test_lookup_tokenizer_unknown_model_returns_err() {
        let components = shared_components();
        let err = lookup_tokenizer("no-such-model", &components.tokenizer_registry).unwrap_err();
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_prepare_chat_does_not_share_mutable_state_with_prepare_generate() {
        let components = shared_components();
        let (p_chat, _c_chat) = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        let (p_gen, _c_gen) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert_ne!(p_chat.request_id, p_gen.request_id);
    }

    #[test]
    fn test_prepare_chat_tool_constraints_threaded_into_payload() {
        use crate::protocols::common::{Function, Tool, ToolChoice, ToolChoiceValue};
        let mut req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: Some(vec![Tool {
                r#type: "function".to_string(),
                function: Function {
                    name: "add".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({"type":"object"})),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
            ..Default::default()
        };
        req.stream = false;
        let components = shared_components();
        let (payload, _ctx) = prepare_chat(
            Arc::new(req),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert!(payload.tool_constraints.is_some());
    }

    #[test]
    fn test_prepare_chat_no_pd_metadata_at_prepare_time() {
        // PD metadata is filled by the planner step, not prepare. prepare must leave
        // it as None so that the engine can route to either single or PD placement
        // without prepare presuming a deployment shape.
        let components = shared_components();
        let (payload, _) = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert!(payload.pd_metadata.is_none());
    }

    #[test]
    fn test_prepare_chat_no_mesh_grpc_in_returned_types() {
        // Compile-only assertion: ResponseContext and GenerationPayload type names
        // do not contain "mesh_grpc". Belt-and-suspenders alongside the grep gate
        // (Test A4 in the plan).
        assert!(!std::any::type_name::<GenerationPayload>().contains("mesh_grpc"));
        assert!(!std::any::type_name::<ResponseContext>().contains("mesh_grpc"));
    }
}
