mod a_chat_template {
    use serde_json::json;

    use crate::protocols::{
        chat::{ChatMessage, MessageContent},
        common::{ContentPart, ImageUrl},
    };
    use crate::routers::prepare::chat_template::{
        process_content_format, process_tool_call_arguments, ProcessedMessages,
    };
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
        let mut messages = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": "{\"a\":1,\"b\":2}"
                }
            }]
        })];
        process_tool_call_arguments(&mut messages).unwrap();
        let args = &messages[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "args should be parsed to object");
        assert_eq!(args["a"], json!(1));
        assert_eq!(args["b"], json!(2));
    }

    #[test]
    fn test_assistant_tool_call_invalid_json_returns_err() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "noop",
                    "arguments": "{not-json}"
                }
            }]
        })];
        let err = process_tool_call_arguments(&mut messages).unwrap_err();
        assert!(err.contains("tool call arguments"));
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
        let ty = std::any::type_name::<ProcessedMessages>();
        assert!(ty.ends_with("ProcessedMessages"));
    }
}

mod b_tool_constraints {
    use serde_json::json;

    use crate::protocols::common::{
        Function, JsonSchemaFormat, ResponseFormat, Tool, ToolChoice, ToolChoiceValue,
    };
    use crate::routers::prepare::tool_constraints::{
        filter_chat_request_by_tool_choice, filter_tools_by_tool_choice, generate_tool_call_id,
        generate_tool_constraints, get_history_tool_calls_count, parse_json_schema_response,
    };

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: json!({"type": "object", "properties": {}}),
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
        )
        .unwrap();
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
        )
        .unwrap();
        let (key, body) = constraints.expect("required tools must produce a constraint");
        assert!(!key.is_empty());
        assert!(!body.is_empty());
    }

    #[test]
    fn test_generate_constraints_named_tool_uses_that_tool_only() {
        let tools = vec![tool("a"), tool("b")];
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "b".to_string(),
            },
        };
        let (_, body) = generate_tool_constraints(&tools, &Some(choice), "m")
            .unwrap()
            .unwrap();
        assert!(!body.is_empty());
    }

    #[test]
    fn test_required_array_schema_includes_all_tool_names() {
        let tools = vec![tool("alpha"), tool("beta")];
        let (_, body) = generate_tool_constraints(
            &tools,
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
        )
        .unwrap()
        .unwrap();
        assert!(body.contains("alpha"));
        assert!(body.contains("beta"));
        assert!(body.contains("\"type\":\"array\""));
    }

    #[test]
    fn test_filter_by_tool_choice_auto_keeps_all() {
        let tools = vec![tool("a"), tool("b")];
        let filtered =
            filter_tools_by_tool_choice(&tools, &Some(ToolChoice::Value(ToolChoiceValue::Auto)));
        assert!(filtered.is_none());
    }

    #[test]
    fn test_filter_by_tool_choice_named_keeps_one() {
        let tools = vec![tool("a"), tool("b"), tool("c")];
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "b".to_string(),
            },
        };
        let filtered = filter_tools_by_tool_choice(&tools, &Some(choice)).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].function.name, "b");
    }

    #[test]
    fn test_filter_by_tool_choice_none_drops_all() {
        let tools = vec![tool("a"), tool("b")];
        let filtered =
            filter_tools_by_tool_choice(&tools, &Some(ToolChoice::Value(ToolChoiceValue::None)));
        assert!(filtered.is_none());
    }

    #[test]
    fn test_filter_by_tool_choice_unknown_named_drops_all() {
        let tools = vec![tool("a")];
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "nope".to_string(),
            },
        };
        let filtered = filter_tools_by_tool_choice(&tools, &Some(choice)).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_chat_request_in_place_replaces_tools_vec() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: Some(vec![tool("a"), tool("b")]),
            tool_choice: Some(ToolChoice::Function {
                tool_type: "function".to_string(),
                function: crate::protocols::common::FunctionChoice {
                    name: "a".to_string(),
                },
            }),
            ..Default::default()
        };
        let filtered = filter_chat_request_by_tool_choice(&req);
        let tools = filtered.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "a");
    }

    #[test]
    fn test_filter_chat_request_no_tools_is_noop() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: None,
            tool_choice: None,
            ..Default::default()
        };
        let filtered = filter_chat_request_by_tool_choice(&req);
        assert!(filtered.tools.is_none());
    }

    #[test]
    fn test_parse_json_schema_response_strict_true_includes_schema() {
        let _resp = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: "City".to_string(),
                schema: json!({"type": "object"}),
                strict: Some(true),
            },
        };
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "add".to_string(),
            },
        };
        let (calls, remaining) = parse_json_schema_response("{\"a\":1}", &Some(choice), "m", 0);
        let calls = calls.expect("must produce tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "add");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_json_schema_response_non_json_schema_returns_none() {
        let (calls, remaining) = parse_json_schema_response("hello", &None, "m", 0);
        assert!(calls.is_none());
        assert_eq!(remaining, "hello");
    }

    #[test]
    fn test_get_history_tool_calls_count_counts_assistant_calls() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        use crate::protocols::common::{FunctionCallResponse, ToolCall};
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![
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
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: "x".to_string(),
                                arguments: Some("{}".to_string()),
                            },
                        },
                        ToolCall {
                            id: "c2".to_string(),
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: "y".to_string(),
                                arguments: Some("{}".to_string()),
                            },
                        },
                    ]),
                    reasoning_content: None,
                },
            ],
            ..Default::default()
        };
        assert_eq!(get_history_tool_calls_count(&req), 2);
    }

    #[test]
    fn test_get_history_tool_calls_count_zero_when_no_assistant() {
        use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            ..Default::default()
        };
        assert_eq!(get_history_tool_calls_count(&req), 0);
    }

    #[test]
    fn test_generate_tool_call_id_unique_and_formatted() {
        let a = generate_tool_call_id("gpt-4", "add", 0, 0);
        let b = generate_tool_call_id("gpt-4", "add", 0, 0);
        assert_ne!(a, b);
        assert!(a.starts_with("call_"));
    }

    #[test]
    fn test_generate_constraints_function_empty_tools_returns_none() {
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "missing".to_string(),
            },
        };
        let out = generate_tool_constraints(&[], &Some(choice), "m").unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_generate_constraints_none_tool_choice_returns_none() {
        let out = generate_tool_constraints(&[tool("a")], &None, "m").unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_generate_constraints_allowed_required_with_empty_returns_none() {
        let choice = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "required".to_string(),
            tools: vec![],
        };
        let out = generate_tool_constraints(&[], &Some(choice), "m").unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_generate_constraints_allowed_required_returns_array_schema() {
        let choice = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "required".to_string(),
            tools: vec![],
        };
        let (key, body) = generate_tool_constraints(&[tool("a")], &Some(choice), "m")
            .unwrap()
            .unwrap();
        assert_eq!(key, "json_schema");
        assert!(body.contains("\"type\":\"array\""));
    }

    #[test]
    fn test_generate_constraints_allowed_auto_mode_returns_none() {
        let choice = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "auto".to_string(),
            tools: vec![],
        };
        let out = generate_tool_constraints(&[tool("a")], &Some(choice), "m").unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_required_array_schema_consolidates_defs() {
        let mut t1 = tool("a");
        t1.function.parameters = json!({
            "type": "object",
            "$defs": {"X": {"type": "string"}},
        });
        let mut t2 = tool("b");
        t2.function.parameters = json!({
            "type": "object",
            "$defs": {"X": {"type": "string"}},
        });
        let (_, body) = generate_tool_constraints(
            &[t1, t2],
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
        )
        .unwrap()
        .unwrap();
        assert!(body.contains("\"$defs\""));
        assert!(body.contains("\"X\""));
    }

    #[test]
    fn test_required_array_schema_conflicting_defs_returns_err() {
        let mut t1 = tool("a");
        t1.function.parameters = json!({
            "type": "object",
            "$defs": {"X": {"type": "string"}},
        });
        let mut t2 = tool("b");
        t2.function.parameters = json!({
            "type": "object",
            "$defs": {"X": {"type": "number"}},
        });
        let err = generate_tool_constraints(
            &[t1, t2],
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
        )
        .unwrap_err();
        assert!(err.contains("conflicting"));
    }

    #[test]
    fn test_filter_tools_allowed_keeps_only_listed_ones() {
        use crate::protocols::common::ToolReference;
        let tools = vec![tool("a"), tool("b"), tool("c")];
        let choice = ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "auto".to_string(),
            tools: vec![ToolReference::Function {
                name: "b".to_string(),
            }],
        };
        let filtered = filter_tools_by_tool_choice(&tools, &Some(choice)).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].function.name, "b");
    }

    #[test]
    fn test_parse_json_schema_response_required_parses_array() {
        let json = r#"[{"name":"foo","parameters":{"x":1}}]"#;
        let (calls, remaining) = parse_json_schema_response(
            json,
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
            0,
        );
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "foo");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_json_schema_response_required_invalid_returns_none_with_text() {
        let (calls, remaining) = parse_json_schema_response(
            "not-json",
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
            0,
        );
        assert!(calls.is_none());
        assert_eq!(remaining, "not-json");
    }

    #[test]
    fn test_parse_json_schema_response_function_invalid_returns_none_with_text() {
        let choice = ToolChoice::Function {
            tool_type: "function".to_string(),
            function: crate::protocols::common::FunctionChoice {
                name: "foo".to_string(),
            },
        };
        let (calls, remaining) = parse_json_schema_response("not-json", &Some(choice), "m", 0);
        assert!(calls.is_none());
        assert_eq!(remaining, "not-json");
    }

    #[test]
    fn test_parse_json_schema_response_required_skips_invalid_items() {
        let json = r#"[{"name":"a","parameters":{}}, "not-object", {"missing":"name"}]"#;
        let (calls, _) = parse_json_schema_response(
            json,
            &Some(ToolChoice::Value(ToolChoiceValue::Required)),
            "m",
            0,
        );
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "a");
    }

    #[test]
    fn test_generate_tool_call_id_kimi_uses_global_index_format() {
        let id = generate_tool_call_id("Kimi-K2", "add", 0, 3);
        assert_eq!(id, "functions.add:3");
    }

    #[test]
    fn test_generate_tool_call_id_kimi_case_insensitive() {
        let id = generate_tool_call_id("model-KIMI-1b", "foo", 2, 0);
        assert_eq!(id, "functions.foo:2");
    }
}

mod c_stop_decoder_builder {
    use crate::protocols::common::StringOrArray;
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::test_mocks;

    #[test]
    fn test_create_decoder_with_string_stop() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(
            &tok,
            Some(&StringOrArray::String("<eot>".to_string())),
            Some(&vec![2u32]),
            true,
            false,
        );
    }

    #[test]
    fn test_create_decoder_with_array_stop() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(
            &tok,
            Some(&StringOrArray::Array(vec![
                "<eot>".to_string(),
                "<stop>".to_string(),
            ])),
            None,
            true,
            false,
        );
    }

    #[test]
    fn test_create_decoder_no_stop_sources() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(&tok, None, None, false, false);
    }

    #[test]
    fn test_create_decoder_no_stop_trim_passes_flag() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(&tok, None, None, true, true);
    }

    #[test]
    fn test_create_decoder_string_stop_with_no_stop_trim_visible() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(
            &tok,
            Some(&StringOrArray::String("<eot>".to_string())),
            Some(&vec![2u32]),
            true,
            true,
        );
    }

    #[test]
    fn test_create_decoder_array_stop_with_no_stop_trim_visible() {
        let tok = test_mocks::tokenizer();
        let _decoder = create_stop_decoder(
            &tok,
            Some(&StringOrArray::Array(vec![
                "<a>".to_string(),
                "<b>".to_string(),
            ])),
            Some(&vec![1u32, 2u32]),
            false,
            true,
        );
    }
}

mod d_parser_factory_lookup {
    use crate::routers::prepare::parser_factory_lookup::{
        check_reasoning_parser_availability, check_tool_parser_availability,
        create_reasoning_parser, create_tool_parser, get_reasoning_parser, get_tool_parser,
    };
    use crate::routers::test_mocks;

    #[test]
    fn test_check_reasoning_parser_known_model_returns_ok() {
        let factory = test_mocks::reasoning_parser_factory();
        assert!(check_reasoning_parser_availability(&factory, None, "qwen3"));
    }

    #[test]
    fn test_check_reasoning_parser_unknown_model_returns_err() {
        let factory = test_mocks::reasoning_parser_factory();
        assert!(!check_reasoning_parser_availability(
            &factory,
            None,
            "no-such-model"
        ));
    }

    #[test]
    fn test_check_tool_parser_known_model_returns_ok() {
        let factory = test_mocks::tool_parser_factory();
        assert!(check_tool_parser_availability(&factory, None, "qwen3"));
    }

    #[test]
    fn test_check_tool_parser_unknown_model_returns_err() {
        let factory = test_mocks::tool_parser_factory();
        assert!(!check_tool_parser_availability(
            &factory,
            None,
            "no-such-model"
        ));
    }

    #[test]
    fn test_get_reasoning_parser_returns_pooled_instance() {
        let factory = test_mocks::reasoning_parser_factory();
        let _pooled = get_reasoning_parser(&factory, None, "qwen3");
    }

    #[test]
    fn test_create_reasoning_parser_returns_owned_instance() {
        let factory = test_mocks::reasoning_parser_factory();
        let _parser = create_reasoning_parser(&factory, None, "qwen3").expect("owned parser");
    }

    #[test]
    fn test_get_tool_parser_returns_pooled_instance() {
        let factory = test_mocks::tool_parser_factory();
        let _pooled = get_tool_parser(&factory, None, "qwen3");
    }

    #[test]
    fn test_create_tool_parser_returns_owned_instance() {
        let factory = test_mocks::tool_parser_factory();
        let _parser = create_tool_parser(&factory, None, "qwen3").expect("owned tool parser");
    }

    #[test]
    fn test_check_reasoning_parser_with_configured_known_returns_true() {
        let factory = test_mocks::reasoning_parser_factory();
        assert!(check_reasoning_parser_availability(
            &factory,
            Some("qwen3"),
            "unrelated-model"
        ));
    }

    #[test]
    fn test_check_reasoning_parser_with_configured_unknown_returns_false() {
        let factory = test_mocks::reasoning_parser_factory();
        assert!(!check_reasoning_parser_availability(
            &factory,
            Some("no-such-parser"),
            "qwen3"
        ));
    }

    #[test]
    fn test_check_tool_parser_with_configured_known_returns_true() {
        let factory = test_mocks::tool_parser_factory();
        assert!(check_tool_parser_availability(
            &factory,
            Some("qwen"),
            "unrelated"
        ));
    }

    #[test]
    fn test_check_tool_parser_with_configured_unknown_returns_false() {
        let factory = test_mocks::tool_parser_factory();
        assert!(!check_tool_parser_availability(
            &factory,
            Some("no-such-parser"),
            "qwen3"
        ));
    }

    #[test]
    fn test_get_reasoning_parser_with_configured_known_returns_pooled() {
        let factory = test_mocks::reasoning_parser_factory();
        let _pooled = get_reasoning_parser(&factory, Some("qwen3"), "unrelated");
    }

    #[test]
    fn test_get_reasoning_parser_with_configured_unknown_falls_back_to_model() {
        let factory = test_mocks::reasoning_parser_factory();
        let _pooled = get_reasoning_parser(&factory, Some("no-such"), "qwen3");
    }

    #[test]
    fn test_create_reasoning_parser_with_configured_known_returns_owned() {
        let factory = test_mocks::reasoning_parser_factory();
        let _p = create_reasoning_parser(&factory, Some("qwen3"), "unrelated").expect("owned");
    }

    #[test]
    fn test_create_reasoning_parser_with_configured_unknown_falls_back_to_model() {
        let factory = test_mocks::reasoning_parser_factory();
        let _p = create_reasoning_parser(&factory, Some("no-such"), "qwen3").expect("fallback");
    }

    #[test]
    fn test_create_reasoning_parser_unknown_model_returns_none() {
        let factory = test_mocks::reasoning_parser_factory();
        assert!(create_reasoning_parser(&factory, None, "no-such-model").is_none());
    }

    #[test]
    fn test_get_tool_parser_with_configured_known_returns_pooled() {
        let factory = test_mocks::tool_parser_factory();
        let _pooled = get_tool_parser(&factory, Some("qwen3"), "unrelated");
    }

    #[test]
    fn test_get_tool_parser_with_configured_unknown_falls_back_to_model() {
        let factory = test_mocks::tool_parser_factory();
        let _pooled = get_tool_parser(&factory, Some("no-such"), "qwen3");
    }

    #[test]
    fn test_create_tool_parser_with_configured_known_returns_owned() {
        let factory = test_mocks::tool_parser_factory();
        let _p = create_tool_parser(&factory, Some("qwen3"), "unrelated").expect("owned");
    }

    #[test]
    fn test_create_tool_parser_with_configured_unknown_falls_back_to_model() {
        let factory = test_mocks::tool_parser_factory();
        let _p = create_tool_parser(&factory, Some("no-such"), "qwen3").expect("fallback");
    }

    #[test]
    fn test_create_tool_parser_unknown_model_falls_back_to_default() {
        let factory = test_mocks::tool_parser_factory();
        let _p = create_tool_parser(&factory, None, "no-such-model");
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
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                max_new_tokens: Some(128),
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
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
                logprob_start_len: -1,
                token_ids_logprob: Vec::new(),
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[test]
    fn test_payload_round_trip_fields() {
        let p = payload_with_defaults();
        assert_eq!(p.request_id, "req-1");
        assert_eq!(p.token_ids, vec![1, 2, 3]);
        assert_eq!(p.text, "hello world");
        assert_eq!(p.sampling.temperature, 0.7);
        assert_eq!(p.sampling.max_new_tokens, Some(128));
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
            bootstrap_port: 8998,
            bootstrap_room: 42,
        };
        let mut p = payload_with_defaults();
        p.pd_metadata = Some(pd);
        let m = p.pd_metadata.as_ref().unwrap();
        assert_eq!(m.bootstrap_host, "p-host");
        assert_eq!(m.bootstrap_port, 8998);
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
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::test_mocks;

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
        super::minimal_generate_request_with_stream(stream)
    }

    fn make_ctx_chat(stream: bool) -> ResponseContext {
        let tok = test_mocks::tokenizer();
        let decoder = create_stop_decoder(&tok, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(chat_req(stream))),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: Some(ProcessedMessages {
                text: "rendered".to_string(),
                stop_sequences: None,
            }),
            tokenizer: tok,
            stop_decoder: decoder,
            request_id: "req-1".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
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
        let tok = test_mocks::tokenizer();
        let decoder = create_stop_decoder(&tok, None, None, true, false);
        let mut hm = HeaderMap::new();
        hm.insert("x-trace", "abc".parse().unwrap());
        let ctx = ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(chat_req(false))),
            model_id: None,
            headers: Some(hm),
            original_text: None,
            processed_messages: None,
            tokenizer: tok,
            stop_decoder: decoder,
            request_id: "req-2".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        };
        assert_eq!(ctx.headers.as_ref().unwrap().get("x-trace").unwrap(), "abc");
    }

    #[test]
    fn test_response_context_generate_path_has_no_processed_messages() {
        let tok = test_mocks::tokenizer();
        let decoder = create_stop_decoder(&tok, None, None, true, false);
        let ctx = ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(generate_req(false))),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: None,
            tokenizer: tok,
            stop_decoder: decoder,
            request_id: "req-3".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        };
        assert!(ctx.processed_messages.is_none());
    }

    #[test]
    fn test_response_context_chat_path_has_processed_messages() {
        let ctx = make_ctx_chat(true);
        assert!(ctx.processed_messages.is_some());
    }
}

fn minimal_generate_request_with_stream(
    stream: bool,
) -> crate::protocols::generate::GenerateRequest {
    crate::protocols::generate::GenerateRequest {
        text: Some("hi".to_string()),
        model: None,
        input_ids: None,
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: None,
        return_logprob: None,
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream,
        log_metrics: false,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    }
}

mod h_prepare_chat_generate {
    use std::sync::Arc;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::protocols::generate::GenerateRequest;
    use crate::routers::prepare::generation_payload::GenerationPayload;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::{lookup_tokenizer, prepare_chat, prepare_generate};
    use crate::routers::test_mocks;

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
        Arc::new(super::minimal_generate_request_with_stream(stream))
    }

    #[test]
    fn test_prepare_chat_returns_err_without_hf_tokenizer() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let components = test_mocks::app_context_with_tokenizer_registry(registry);
        let result = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        );
        let err = result.err().expect("MockTokenizer is not HF");
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_prepare_chat_with_hf_tokenizer_returns_payload() {
        let components = test_mocks::app_context_with_hf_tokenizer("m");
        let (payload, ctx) = prepare_chat(
            chat_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("HF tokenizer must succeed");
        assert!(payload.request_id.starts_with("chatcmpl-"));
        assert!(matches!(ctx.original, ProtocolRequest::Chat(_)));
        let pm = ctx.processed_messages.as_ref().expect("processed messages");
        assert!(
            pm.text.contains("user:") && pm.text.contains("hi"),
            "rendered chat should reflect the chat template, got: {}",
            pm.text
        );
        assert_eq!(payload.text, pm.text);
    }

    #[test]
    fn test_prepare_chat_streaming_flag_threads_through() {
        let components = test_mocks::app_context_with_hf_tokenizer("m");
        let (payload, ctx) = prepare_chat(
            chat_req(true),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("HF tokenizer must succeed");
        assert!(payload.stream);
        assert!(ctx.original.is_streaming());
    }

    #[test]
    fn test_prepare_chat_empty_messages_still_renders() {
        let components = test_mocks::app_context_with_hf_tokenizer("m");
        let req = Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            ..Default::default()
        });
        let (payload, _) = prepare_chat(req, None, Some("m".to_string()), components.as_ref())
            .expect("empty messages renders to empty string under our template");
        assert_eq!(payload.text, "");
    }

    #[test]
    fn test_prepare_chat_with_tools_threads_constraints_and_disables_skip_special() {
        use crate::protocols::common::{
            Function as ToolFunction, Tool, ToolChoice, ToolChoiceValue,
        };
        let components = test_mocks::app_context_with_hf_tokenizer("m");
        let req = Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: "ping".to_string(),
                    description: None,
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
            skip_special_tokens: true,
            ..Default::default()
        });
        let (payload, _) = prepare_chat(req, None, Some("m".to_string()), components.as_ref())
            .expect("tools-present path must succeed");
        assert!(
            payload.tool_constraints.is_some(),
            "Required choice must emit a constraint"
        );
        assert!(
            !payload.stop.skip_special_tokens,
            "tools+non-None choice must disable skip_special_tokens"
        );
    }

    #[test]
    fn test_prepare_chat_tool_choice_none_keeps_skip_special() {
        use crate::protocols::common::{
            Function as ToolFunction, Tool, ToolChoice, ToolChoiceValue,
        };
        let components = test_mocks::app_context_with_hf_tokenizer("m");
        let req = Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: "ping".to_string(),
                    description: None,
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::None)),
            skip_special_tokens: true,
            ..Default::default()
        });
        let (payload, _) = prepare_chat(req, None, Some("m".to_string()), components.as_ref())
            .expect("None choice path must succeed");
        assert!(payload.stop.skip_special_tokens);
    }

    #[test]
    fn test_prepare_chat_missing_model_id_returns_err_response() {
        let components = test_mocks::app_context();
        let result = prepare_chat(chat_req(false), None, None, components.as_ref());
        let err = result.err().expect("model_id missing must be Err");
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_prepare_chat_unknown_model_returns_err_response() {
        let components = test_mocks::app_context();
        let result = prepare_chat(
            chat_req(false),
            None,
            Some("nope".to_string()),
            components.as_ref(),
        );
        let err = result.err().expect("unknown model must be Err");
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_prepare_generate_returns_payload_and_context_tuple() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let components = test_mocks::app_context_with_tokenizer_registry(registry);
        let (payload, ctx) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .expect("ok");
        assert!(!payload.request_id.is_empty());
        assert!(matches!(ctx.original, ProtocolRequest::Generate(_)));
        assert!(
            ctx.processed_messages.is_none(),
            "generate path has no chat messages"
        );
    }

    #[test]
    fn test_prepare_generate_streaming_flag_propagates() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let components = test_mocks::app_context_with_tokenizer_registry(registry);
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
    fn test_prepare_generate_missing_model_id_returns_err() {
        let components = test_mocks::app_context();
        let result = prepare_generate(generate_req(false), None, None, components.as_ref());
        let err = result.err().expect("model_id missing must be Err");
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_lookup_tokenizer_returns_arc_for_known_model() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let tok = lookup_tokenizer("m", &registry).expect("known model");
        let tok2 = lookup_tokenizer("m", &registry).unwrap();
        assert!(Arc::ptr_eq(&tok, &tok2));
    }

    #[test]
    fn test_lookup_tokenizer_unknown_model_returns_err() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let err = lookup_tokenizer("no-such-model", &registry)
            .err()
            .expect("unknown model must be Err");
        assert!(!err.status().is_success());
    }

    #[test]
    fn test_prepare_generate_request_ids_unique_per_call() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let components = test_mocks::app_context_with_tokenizer_registry(registry);
        let (p1, _) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        let (p2, _) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert_ne!(p1.request_id, p2.request_id);
    }

    #[test]
    fn test_prepare_generate_no_pd_metadata_at_prepare_time() {
        let registry = test_mocks::tokenizer_registry_with("m");
        let components = test_mocks::app_context_with_tokenizer_registry(registry);
        let (payload, _) = prepare_generate(
            generate_req(false),
            None,
            Some("m".to_string()),
            components.as_ref(),
        )
        .unwrap();
        assert!(payload.pd_metadata.is_none());
    }

    #[test]
    fn test_prepare_no_mesh_grpc_in_returned_types() {
        assert!(!std::any::type_name::<GenerationPayload>().contains("mesh_grpc"));
        assert!(!std::any::type_name::<ResponseContext>().contains("mesh_grpc"));
    }
}
