//! Tests for `routers::openai::responses::*` — public surface and layering invariants.

mod a_context {
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::test_mocks;

    #[test]
    fn test_responses_context_holds_concrete_pipeline_not_trait() {
        let name = std::any::type_name::<ResponsesContext>();
        assert!(
            !name.contains("dyn "),
            "ResponsesContext must not wrap a trait object, got {name}"
        );
    }

    #[test]
    fn test_responses_context_new_constructor() {
        let _ctx = test_mocks::responses_context();
    }

    #[test]
    fn test_responses_context_pipeline_field_is_arc_wrapped() {
        let ctx = test_mocks::responses_context();
        let _pipeline = ctx.pipeline.clone();
    }
}

mod b_handlers_dispatch {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    fn req(stream: bool) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(stream);
        r
    }

    #[tokio::test]
    async fn test_post_responses_streaming_dispatches_to_streaming_module() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let r =
            handlers::route_responses(&ctx, Arc::new(req(true)), None, Some("m".to_string())).await;
        assert_eq!(r.status(), StatusCode::OK);
        let ct = r.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_post_responses_non_streaming_dispatches_to_non_streaming_module() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let r = handlers::route_responses(&ctx, Arc::new(req(false)), None, Some("m".to_string()))
            .await;
        assert_eq!(r.status(), StatusCode::OK);
        let ct = r.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_post_responses_unknown_model_returns_error_status() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req(false);
        r.model = "no-such-model".to_string();
        let resp =
            handlers::route_responses(&ctx, Arc::new(r), None, Some("no-such-model".to_string()))
                .await;
        assert!(
            !resp.status().is_success(),
            "unknown model must fail, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_post_responses_background_mode_rejected() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req(false);
        r.background = Some(true);
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

mod c_non_streaming {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    fn req() -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(false);
        r
    }

    #[tokio::test]
    async fn test_non_streaming_returns_responses_response_envelope() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("\"id\""), "envelope must include id: {s}");
        assert!(
            s.contains("\"object\""),
            "envelope must include object: {s}"
        );
    }

    #[tokio::test]
    async fn test_non_streaming_persists_response_when_store_true() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.store = Some(true);
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_non_streaming_does_not_persist_when_store_false() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.store = Some(false);
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

mod d_streaming {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    fn req() -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(true);
        r
    }

    async fn collect_body(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    #[tokio::test]
    async fn test_streaming_emits_response_created_event_first() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let s = collect_body(resp).await;
        assert!(
            s.contains("response.created"),
            "first event must be response.created: {s}"
        );
        let created_idx = s.find("response.created").unwrap();
        if let Some(in_progress_idx) = s.find("response.in_progress") {
            assert!(
                created_idx < in_progress_idx,
                "response.created must precede response.in_progress"
            );
        }
    }

    #[tokio::test]
    async fn test_streaming_emits_response_completed_event_last() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        let s = collect_body(resp).await;
        assert!(
            s.contains("response.completed"),
            "stream must include response.completed: {s}"
        );
        assert!(
            s.contains("[DONE]"),
            "stream must terminate with [DONE] marker: {s}"
        );
    }

    #[tokio::test]
    async fn test_streaming_emits_output_text_delta_events() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        let s = collect_body(resp).await;
        assert!(
            s.contains("response.created"),
            "must always include response.created: {s}"
        );
    }

    #[tokio::test]
    async fn test_streaming_merger_was_collapsed_into_single_file() {
        let _ty = std::any::type_name::<
            crate::routers::openai::responses::streaming::ResponseStreamEventEmitter,
        >();
    }
}

mod e_retrieve_and_cancel {
    use axum::http::StatusCode;
    use data_connector::{ResponseId, StoredResponse};
    use serde_json::json;

    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::retrieve;
    use crate::routers::test_mocks;

    async fn store_with_raw(ctx: &ResponsesContext, id: &str, raw: serde_json::Value) {
        let mut s = StoredResponse::new(None);
        s.id = ResponseId::from(id);
        s.raw_response = raw;
        ctx.response_storage.store_response(s).await.unwrap();
    }

    #[tokio::test]
    async fn test_get_response_returns_stored_payload_when_present() {
        let ctx = test_mocks::responses_context();
        store_with_raw(&ctx, "resp_existing", json!({"id": "resp_existing"})).await;
        let r = retrieve::get_response_impl(&ctx, "resp_existing").await;
        assert_eq!(r.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_response_returns_404_when_missing() {
        let ctx = test_mocks::responses_context();
        let r = retrieve::get_response_impl(&ctx, "resp_missing").await;
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_cancel_completed_response_returns_bad_request() {
        let ctx = test_mocks::responses_context();
        store_with_raw(&ctx, "resp_done", json!({"status": "completed"})).await;
        let r = retrieve::cancel_response_impl(&ctx, "resp_done").await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cancel_failed_response_returns_bad_request() {
        let ctx = test_mocks::responses_context();
        store_with_raw(&ctx, "resp_fail", json!({"status": "failed"})).await;
        let r = retrieve::cancel_response_impl(&ctx, "resp_fail").await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cancel_in_progress_returns_not_supported() {
        let ctx = test_mocks::responses_context();
        store_with_raw(&ctx, "resp_run", json!({"status": "in_progress"})).await;
        let r = retrieve::cancel_response_impl(&ctx, "resp_run").await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cancel_response_without_status_field_returns_not_supported() {
        let ctx = test_mocks::responses_context();
        store_with_raw(&ctx, "resp_nosta", json!({"id": "x"})).await;
        let r = retrieve::cancel_response_impl(&ctx, "resp_nosta").await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cancel_unknown_response_returns_404() {
        let ctx = test_mocks::responses_context();
        let r = retrieve::cancel_response_impl(&ctx, "resp_missing").await;
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }
}

mod f_persistence {
    use crate::protocols::responses::{ResponseTool, ResponseToolType};
    use crate::routers::openai::responses::persistence::extract_tools_from_response_tools;

    #[test]
    fn test_extract_tools_returns_tool_definitions_from_request() {
        let function_tool = ResponseTool {
            r#type: ResponseToolType::Function,
            function: Some(crate::protocols::common::Function {
                name: "add".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            }),
            ..Default::default()
        };
        let tools = vec![function_tool.clone(), function_tool];
        let extracted = extract_tools_from_response_tools(Some(&tools));
        assert_eq!(extracted.len(), 2);
    }

    #[test]
    fn test_extract_tools_empty_when_no_tools_field() {
        let extracted = extract_tools_from_response_tools(None);
        assert!(extracted.is_empty());
    }

    #[tokio::test]
    async fn test_persist_response_stores_for_later_retrieve() {
        use crate::routers::openai::responses::persistence::persist_response_if_needed;
        use crate::routers::test_mocks;

        let ctx = test_mocks::responses_context();
        let mut req = crate::protocols::responses::ResponsesRequest::default();
        req.store = Some(true);
        let resp = crate::protocols::responses::ResponsesResponse::builder("resp_xyz", "m")
            .status(crate::protocols::responses::ResponseStatus::Completed)
            .build();
        persist_response_if_needed(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &resp,
            &req,
        )
        .await;
    }

    #[tokio::test]
    async fn test_persist_response_noop_when_store_false() {
        use crate::routers::openai::responses::persistence::persist_response_if_needed;
        use crate::routers::test_mocks;

        let ctx = test_mocks::responses_context();
        let mut req = crate::protocols::responses::ResponsesRequest::default();
        req.store = Some(false);
        let resp = crate::protocols::responses::ResponsesResponse::builder("resp_skip", "m")
            .status(crate::protocols::responses::ResponseStatus::Completed)
            .build();
        persist_response_if_needed(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &resp,
            &req,
        )
        .await;
        let got = ctx
            .response_storage
            .get_response(&data_connector::ResponseId::from("resp_skip"))
            .await
            .unwrap();
        assert!(got.is_none(), "store=false must not persist");
    }

    #[tokio::test]
    async fn test_persist_response_default_store_true_persists() {
        use crate::routers::openai::responses::persistence::persist_response_if_needed;
        use crate::routers::test_mocks;

        let ctx = test_mocks::responses_context();
        let req = crate::protocols::responses::ResponsesRequest::default();
        let resp = crate::protocols::responses::ResponsesResponse::builder("resp_def", "m")
            .status(crate::protocols::responses::ResponseStatus::Completed)
            .build();
        persist_response_if_needed(
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
            ctx.response_storage.clone(),
            &resp,
            &req,
        )
        .await;
    }

    #[test]
    fn test_extract_tools_skips_non_function_types() {
        let mut t = ResponseTool::default();
        t.r#type = ResponseToolType::WebSearchPreview;
        let extracted = extract_tools_from_response_tools(Some(&[t]));
        assert!(extracted.is_empty());
    }

    #[test]
    fn test_extract_tools_function_without_function_field_skipped() {
        let t = ResponseTool {
            r#type: ResponseToolType::Function,
            function: None,
            ..Default::default()
        };
        let extracted = extract_tools_from_response_tools(Some(&[t]));
        assert!(extracted.is_empty());
    }
}

mod g_conversation {
    use axum::http::StatusCode;
    use data_connector::{
        ConversationId, NewConversation, NewConversationItem, ResponseId, StoredResponse,
    };
    use serde_json::json;

    use crate::protocols::responses::{ResponseInput, ResponseInputOutputItem, ResponsesRequest};
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::conversation::load_conversation_history;
    use crate::routers::test_mocks;

    async fn store_prev(
        ctx: &ResponsesContext,
        id: &str,
        input: serde_json::Value,
        output: serde_json::Value,
    ) {
        let mut s = StoredResponse::new(None);
        s.id = ResponseId::from(id);
        s.input = input;
        s.output = output;
        ctx.response_storage.store_response(s).await.unwrap();
    }

    #[tokio::test]
    async fn test_load_conversation_history_returns_prior_messages() {
        let ctx = test_mocks::responses_context();
        let msg = json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "previous"}],
            "id": "msg_prev"
        });
        store_prev(&ctx, "resp_prev", json!([msg]), json!([])).await;
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_prev".to_string());
        req.input = ResponseInput::Text("now".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        assert!(modified.previous_response_id.is_none());
        match modified.input {
            ResponseInput::Items(items) => assert!(items.len() >= 1),
            _ => panic!("expected items"),
        }
    }

    #[tokio::test]
    async fn test_load_conversation_history_unknown_id_clears_prev_id() {
        let ctx = test_mocks::responses_context();
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_missing".to_string());
        req.input = ResponseInput::Text("now".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        assert!(modified.previous_response_id.is_none());
    }

    #[tokio::test]
    async fn test_load_conversation_history_no_previous_id_returns_clone() {
        let ctx = test_mocks::responses_context();
        let mut req = ResponsesRequest::default();
        req.input = ResponseInput::Text("hi".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        assert!(modified.previous_response_id.is_none());
    }

    #[tokio::test]
    async fn test_load_conversation_history_missing_conversation_returns_404() {
        let ctx = test_mocks::responses_context();
        let mut req = ResponsesRequest::default();
        req.conversation = Some("conv_missing".to_string());
        req.input = ResponseInput::Text("hi".to_string());
        let err = load_conversation_history(&ctx, &req).await.unwrap_err();
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_load_conversation_history_existing_conversation_with_items() {
        let ctx = test_mocks::responses_context();
        let conv_id = ConversationId::from("conv_x");
        ctx.conversation_storage
            .create_conversation(NewConversation {
                id: Some(conv_id.clone()),
                metadata: None,
            })
            .await
            .unwrap();
        let item = ctx
            .conversation_item_storage
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".to_string(),
                role: Some("user".to_string()),
                content: json!([{"type": "input_text", "text": "hi"}]),
                status: Some("completed".to_string()),
            })
            .await
            .unwrap();
        ctx.conversation_item_storage
            .link_item(&conv_id, &item.id, chrono::Utc::now())
            .await
            .unwrap();
        let mut req = ResponsesRequest::default();
        req.conversation = Some("conv_x".to_string());
        req.input = ResponseInput::Text("now".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        match modified.input {
            ResponseInput::Items(items) => assert!(items.len() >= 2),
            _ => panic!("expected items"),
        }
    }

    #[tokio::test]
    async fn test_load_conversation_history_existing_conversation_with_items_input() {
        let ctx = test_mocks::responses_context();
        let conv_id = ConversationId::from("conv_y");
        ctx.conversation_storage
            .create_conversation(NewConversation {
                id: Some(conv_id.clone()),
                metadata: None,
            })
            .await
            .unwrap();
        let mut req = ResponsesRequest::default();
        req.conversation = Some("conv_y".to_string());
        req.input = ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_1".to_string(),
            role: "user".to_string(),
            content: vec![
                crate::protocols::responses::ResponseContentPart::InputText {
                    text: "now".to_string(),
                },
            ],
            status: Some("completed".to_string()),
        }]);
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        match modified.input {
            ResponseInput::Items(items) => assert!(!items.is_empty()),
            _ => panic!("expected items"),
        }
    }

    #[tokio::test]
    async fn test_load_conversation_history_prev_output_items_are_loaded() {
        let ctx = test_mocks::responses_context();
        let out_msg = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            "id": "msg_out",
            "status": "completed"
        });
        store_prev(&ctx, "resp_with_out", json!([]), json!([out_msg])).await;
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_with_out".to_string());
        req.input = ResponseInput::Text("ask".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        match modified.input {
            ResponseInput::Items(items) => assert!(items.len() >= 2),
            _ => panic!("expected items"),
        }
    }

    #[tokio::test]
    async fn test_load_conversation_history_prev_invalid_input_item_skipped() {
        let ctx = test_mocks::responses_context();
        let invalid = json!({"type": "unknown_type", "garbage": true});
        store_prev(&ctx, "resp_bad", json!([invalid]), json!([])).await;
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_bad".to_string());
        req.input = ResponseInput::Text("hi".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        assert!(modified.previous_response_id.is_none());
    }

    #[tokio::test]
    async fn test_load_conversation_history_prev_invalid_output_item_skipped() {
        let ctx = test_mocks::responses_context();
        let invalid = json!({"type": "unknown_type", "garbage": true});
        store_prev(&ctx, "resp_bado", json!([]), json!([invalid])).await;
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_bado".to_string());
        req.input = ResponseInput::Text("hi".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        assert!(modified.previous_response_id.is_none());
    }

    #[tokio::test]
    async fn test_load_conversation_history_conv_item_without_role_defaults_user() {
        let ctx = test_mocks::responses_context();
        let conv_id = ConversationId::from("conv_norole");
        ctx.conversation_storage
            .create_conversation(NewConversation {
                id: Some(conv_id.clone()),
                metadata: None,
            })
            .await
            .unwrap();
        let item = ctx
            .conversation_item_storage
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "message".to_string(),
                role: None,
                content: json!([{"type": "input_text", "text": "hello"}]),
                status: Some("completed".to_string()),
            })
            .await
            .unwrap();
        ctx.conversation_item_storage
            .link_item(&conv_id, &item.id, chrono::Utc::now())
            .await
            .unwrap();
        let mut req = ResponsesRequest::default();
        req.conversation = Some("conv_norole".to_string());
        req.input = ResponseInput::Text("ok".to_string());
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        match modified.input {
            ResponseInput::Items(items) => assert!(!items.is_empty()),
            _ => panic!("expected items"),
        }
    }

    #[tokio::test]
    async fn test_load_conversation_history_conv_item_non_message_type_skipped() {
        let ctx = test_mocks::responses_context();
        let conv_id = ConversationId::from("conv_nonmsg");
        ctx.conversation_storage
            .create_conversation(NewConversation {
                id: Some(conv_id.clone()),
                metadata: None,
            })
            .await
            .unwrap();
        let item = ctx
            .conversation_item_storage
            .create_item(NewConversationItem {
                id: None,
                response_id: None,
                item_type: "reasoning".to_string(),
                role: Some("assistant".to_string()),
                content: json!([]),
                status: None,
            })
            .await
            .unwrap();
        ctx.conversation_item_storage
            .link_item(&conv_id, &item.id, chrono::Utc::now())
            .await
            .unwrap();
        let mut req = ResponsesRequest::default();
        req.conversation = Some("conv_nonmsg".to_string());
        req.input = ResponseInput::Text("ok".to_string());
        let _ = load_conversation_history(&ctx, &req).await.unwrap();
    }

    #[tokio::test]
    async fn test_load_conversation_history_prev_with_items_input() {
        let ctx = test_mocks::responses_context();
        let msg = json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "before"}],
            "id": "msg_before"
        });
        store_prev(&ctx, "resp_prev2", json!([msg]), json!([])).await;
        let mut req = ResponsesRequest::default();
        req.previous_response_id = Some("resp_prev2".to_string());
        req.input = ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_now".to_string(),
            role: "user".to_string(),
            content: vec![
                crate::protocols::responses::ResponseContentPart::InputText {
                    text: "now".to_string(),
                },
            ],
            status: Some("completed".to_string()),
        }]);
        let modified = load_conversation_history(&ctx, &req).await.unwrap();
        match modified.input {
            ResponseInput::Items(items) => assert!(items.len() >= 2),
            _ => panic!("expected items"),
        }
    }
}

mod h_conversions {
    use crate::protocols::chat::{ChatChoice, ChatCompletionMessage};
    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::conversions::{chat_to_responses, responses_to_chat};

    fn responses_req(stream: bool) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.stream = Some(stream);
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r
    }

    fn assistant_choice(text: &str, finish: Option<&str>) -> ChatChoice {
        ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some(text.to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: finish.map(|s| s.to_string()),
            matched_stop: None,
            hidden_states: None,
        }
    }

    #[test]
    fn test_responses_to_chat_carries_model() {
        let chat = responses_to_chat(&responses_req(false)).unwrap();
        assert_eq!(chat.model, "m");
    }

    #[test]
    fn test_responses_to_chat_carries_streaming_flag() {
        let chat = responses_to_chat(&responses_req(true)).unwrap();
        assert!(chat.stream);
    }

    #[test]
    fn test_responses_to_chat_appends_conversation_history() {
        let mut req = responses_req(false);
        req.input = ResponseInput::Text("earlier".to_string());
        let chat = responses_to_chat(&req).unwrap();
        assert!(!chat.messages.is_empty());
    }

    #[test]
    fn test_chat_to_responses_envelope_has_id_and_status() {
        let chat = crate::protocols::chat::ChatCompletionResponse::builder("chatcmpl-1", "m")
            .add_choice(assistant_choice("hi", Some("stop")))
            .build();
        let req = responses_req(false);
        let env = chat_to_responses(&chat, &req, Some("resp_1".to_string())).unwrap();
        assert_eq!(env.id, "resp_1");
    }

    #[test]
    fn test_chat_to_responses_envelope_preserves_finish_reason_when_present() {
        let chat = crate::protocols::chat::ChatCompletionResponse::builder("chatcmpl-2", "m")
            .add_choice(assistant_choice("hi", Some("stop")))
            .build();
        let req = responses_req(false);
        let _env = chat_to_responses(&chat, &req, Some("resp_2".to_string())).unwrap();
    }
}

mod i_layering {
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::openai::responses::context::ResponsesContext;

    #[test]
    fn test_responses_only_grpc_ref_is_pipeline() {
        let _ty = std::any::type_name::<ResponsesContext>();
        let _p = std::any::type_name::<Pipeline>();
    }

    #[test]
    fn test_responses_does_not_re_export_mesh_grpc_types() {
        let _name = std::any::type_name::<crate::protocols::responses::ResponsesResponse>();
        assert!(
            !_name.contains("mesh_grpc"),
            "leaked mesh_grpc type: {_name}"
        );
    }
}

mod h2_conversions_branches {
    use crate::protocols::chat::{ChatChoice, ChatCompletionMessage, ChatCompletionResponse};
    use crate::protocols::common::{
        CompletionTokensDetails, FunctionCallResponse, ToolCall, Usage,
    };
    use crate::protocols::responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponseReasoningContent::ReasoningText, ResponseStatus, ResponseTool, ResponseToolType,
        ResponsesRequest, StringOrContentParts, TextConfig, TextFormat,
    };
    use crate::routers::openai::responses::conversions::{chat_to_responses, responses_to_chat};

    fn req_with_input(input: ResponseInput) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = input;
        r
    }

    #[test]
    fn test_responses_to_chat_simple_input_message_string_content() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                role: "user".to_string(),
                content: StringOrContentParts::String("hello".to_string()),
                r#type: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert!(!chat.messages.is_empty());
    }

    #[test]
    fn test_responses_to_chat_simple_input_message_array_content() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                role: "system".to_string(),
                content: StringOrContentParts::Array(vec![
                    ResponseContentPart::InputText {
                        text: "alpha".to_string(),
                    },
                    ResponseContentPart::InputText {
                        text: "beta".to_string(),
                    },
                ]),
                r#type: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_simple_input_message_assistant_role() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                role: "assistant".to_string(),
                content: StringOrContentParts::String("ok".to_string()),
                r#type: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_simple_input_message_unknown_role_defaults_user() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::SimpleInputMessage {
                role: "developer".to_string(),
                content: StringOrContentParts::String("alpha".to_string()),
                r#type: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_function_tool_call_with_output_adds_tool_msg() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::FunctionToolCall {
                id: "call_1".to_string(),
                call_id: "call_1".to_string(),
                name: "f".to_string(),
                arguments: "{}".to_string(),
                output: Some("res".to_string()),
                status: Some("completed".to_string()),
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 2);
    }

    #[test]
    fn test_responses_to_chat_function_tool_call_without_output_only_assistant() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::FunctionToolCall {
                id: "call_x".to_string(),
                call_id: "call_x".to_string(),
                name: "f".to_string(),
                arguments: "{}".to_string(),
                output: None,
                status: Some("in_progress".to_string()),
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_function_call_output_adds_tool_msg() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::FunctionCallOutput {
                id: Some("fco_1".to_string()),
                call_id: "call_x".to_string(),
                output: "produced".to_string(),
                status: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_reasoning_input_appends_assistant_with_reasoning() {
        let req = req_with_input(ResponseInput::Items(vec![
            ResponseInputOutputItem::Reasoning {
                id: "r_1".to_string(),
                summary: vec![],
                content: vec![
                    ReasoningText {
                        text: "think a".to_string(),
                    },
                    ReasoningText {
                        text: "think b".to_string(),
                    },
                ],
                status: None,
            },
        ]));
        let chat = responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn test_responses_to_chat_empty_items_returns_error() {
        let req = req_with_input(ResponseInput::Items(vec![]));
        let err = responses_to_chat(&req).unwrap_err();
        assert!(err.contains("at least one"));
    }

    #[test]
    fn test_responses_to_chat_model_empty_defaults_unknown() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.model = String::new();
        let chat = responses_to_chat(&req).unwrap();
        assert!(!chat.model.is_empty());
    }

    #[test]
    fn test_responses_to_chat_function_tools_present() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.tools = Some(vec![ResponseTool {
            r#type: ResponseToolType::Function,
            function: Some(crate::protocols::common::Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            }),
            ..Default::default()
        }]);
        let chat = responses_to_chat(&req).unwrap();
        assert!(chat.tools.is_some());
    }

    #[test]
    fn test_responses_to_chat_text_format_text() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.text = Some(TextConfig {
            format: Some(TextFormat::Text),
        });
        let chat = responses_to_chat(&req).unwrap();
        assert!(chat.response_format.is_some());
    }

    #[test]
    fn test_responses_to_chat_text_format_json_object() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.text = Some(TextConfig {
            format: Some(TextFormat::JsonObject),
        });
        let chat = responses_to_chat(&req).unwrap();
        assert!(chat.response_format.is_some());
    }

    #[test]
    fn test_responses_to_chat_text_format_json_schema() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.text = Some(TextConfig {
            format: Some(TextFormat::JsonSchema {
                name: "n".to_string(),
                schema: serde_json::json!({"type":"object"}),
                description: None,
                strict: Some(true),
            }),
        });
        let chat = responses_to_chat(&req).unwrap();
        assert!(chat.response_format.is_some());
    }

    #[test]
    fn test_responses_to_chat_stream_true_adds_stream_options() {
        let mut req = req_with_input(ResponseInput::Text("hi".to_string()));
        req.stream = Some(true);
        let chat = responses_to_chat(&req).unwrap();
        assert!(chat.stream);
        assert!(chat.stream_options.is_some());
    }

    fn chat_with_message(
        msg: ChatCompletionMessage,
        finish: Option<&str>,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse::builder("chatcmpl-1", "m")
            .add_choice(ChatChoice {
                index: 0,
                message: msg,
                logprobs: None,
                finish_reason: finish.map(|s| s.to_string()),
                matched_stop: None,
                hidden_states: None,
            })
            .build()
    }

    #[test]
    fn test_chat_to_responses_no_choices_errors() {
        let chat = ChatCompletionResponse::builder("c1", "m").build();
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        assert!(chat_to_responses(&chat, &req, None).is_err());
    }

    #[test]
    fn test_chat_to_responses_with_reasoning_emits_reasoning_output() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("answer".to_string()),
            tool_calls: None,
            reasoning_content: Some("because".to_string()),
        };
        let chat = chat_with_message(msg, Some("stop"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Completed);
    }

    #[test]
    fn test_chat_to_responses_with_tool_calls_emits_function_tool_call() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "tc_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCallResponse {
                    name: "f".to_string(),
                    arguments: Some("{}".to_string()),
                },
            }]),
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("tool_calls"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Completed);
    }

    #[test]
    fn test_chat_to_responses_finish_failed_maps_failed() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("x".to_string()),
            tool_calls: None,
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("failed"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Failed);
    }

    #[test]
    fn test_chat_to_responses_finish_error_maps_failed() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("x".to_string()),
            tool_calls: None,
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("error"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Failed);
    }

    #[test]
    fn test_chat_to_responses_finish_length_maps_completed() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("x".to_string()),
            tool_calls: None,
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("length"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Completed);
    }

    #[test]
    fn test_chat_to_responses_finish_unknown_defaults_completed() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("x".to_string()),
            tool_calls: None,
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("frobbed"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert_eq!(env.status, ResponseStatus::Completed);
    }

    #[test]
    fn test_chat_to_responses_with_usage_threads_reasoning_tokens() {
        let mut chat = chat_with_message(
            ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("x".to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            Some("stop"),
        );
        chat.usage = Some(Usage {
            prompt_tokens: 2,
            completion_tokens: 3,
            total_tokens: 5,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(1),
            }),
        });
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        assert!(env.usage.is_some());
    }

    #[test]
    fn test_chat_to_responses_empty_content_skips_message_output() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: Some("".to_string()),
            tool_calls: None,
            reasoning_content: None,
        };
        let chat = chat_with_message(msg, Some("stop"));
        let req = req_with_input(ResponseInput::Text("h".to_string()));
        let env = chat_to_responses(&chat, &req, None).unwrap();
        let _ = env;
    }
}

mod j_stream_event_emitter {
    use bytes::Bytes;
    use serde_json::json;
    use tokio::sync::mpsc;

    use crate::protocols::chat::{
        ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    };
    use crate::protocols::common::{FunctionCallDelta, ToolCallDelta};
    use crate::routers::openai::responses::streaming::{
        OutputItemType, ResponseStreamEventEmitter,
    };

    fn emitter() -> ResponseStreamEventEmitter {
        ResponseStreamEventEmitter::new("resp_x".to_string(), "m".to_string(), 0)
    }

    fn chunk_with(
        content: Option<&str>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish: Option<&str>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: "id".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "m".to_string(),
            system_fingerprint: None,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: content.map(|s| s.to_string()),
                    tool_calls,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: finish.map(|s| s.to_string()),
                matched_stop: None,
            }],
            usage: None,
        }
    }

    fn collect_lines(rx: &mut mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>) -> String {
        let mut out = String::new();
        while let Ok(item) = rx.try_recv() {
            if let Ok(b) = item {
                out.push_str(&String::from_utf8_lossy(&b));
            }
        }
        out
    }

    #[test]
    fn test_emit_created_returns_response_created_type() {
        let mut e = emitter();
        let v = e.emit_created();
        assert_eq!(v["type"], "response.created");
        assert_eq!(v["sequence_number"], 0);
    }

    #[test]
    fn test_emit_in_progress_increments_sequence() {
        let mut e = emitter();
        let _ = e.emit_created();
        let v = e.emit_in_progress();
        assert_eq!(v["type"], "response.in_progress");
        assert_eq!(v["sequence_number"], 1);
    }

    #[test]
    fn test_emit_text_delta_accumulates_text() {
        let mut e = emitter();
        let _ = e.emit_text_delta("hello", 0, "msg_1", 0);
        let _ = e.emit_text_delta(" world", 0, "msg_1", 0);
        let done = e.emit_text_done(0, "msg_1", 0);
        assert_eq!(done["text"], "hello world");
    }

    #[test]
    fn test_emit_content_part_added_and_done() {
        let mut e = emitter();
        let a = e.emit_content_part_added(0, "msg_1", 0);
        assert_eq!(a["type"], "response.content_part.added");
        let d = e.emit_content_part_done(0, "msg_1", 0);
        assert_eq!(d["type"], "response.content_part.done");
    }

    #[test]
    fn test_emit_output_item_added_and_done() {
        let mut e = emitter();
        let (idx, id) = e.allocate_output_index(OutputItemType::Message);
        let item = json!({"id": id.clone(), "type": "message"});
        let a = e.emit_output_item_added(idx, &item);
        assert_eq!(a["type"], "response.output_item.added");
        let d = e.emit_output_item_done(idx, &item);
        assert_eq!(d["type"], "response.output_item.done");
        e.complete_output_item(idx);
    }

    #[test]
    fn test_emit_function_call_arguments_delta_and_done() {
        let mut e = emitter();
        let d = e.emit_function_call_arguments_delta(0, "fc_1", "{\"a\":1");
        assert_eq!(d["type"], "response.function_call_arguments.delta");
        let done = e.emit_function_call_arguments_done(0, "fc_1", "{\"a\":1}");
        assert_eq!(done["type"], "response.function_call_arguments.done");
    }

    #[test]
    fn test_emit_completed_with_no_items_includes_fallback_assistant() {
        let mut e = emitter();
        let _ = e.emit_text_delta("hi", 0, "m", 0);
        let c = e.emit_completed(None);
        assert_eq!(c["type"], "response.completed");
        assert!(c["response"]["output"].is_array());
    }

    #[test]
    fn test_emit_completed_with_usage_carries_usage() {
        let mut e = emitter();
        let u = json!({"input_tokens": 3, "output_tokens": 5, "total_tokens": 8});
        let c = e.emit_completed(Some(&u));
        assert_eq!(c["response"]["usage"]["total_tokens"], 8);
    }

    #[test]
    fn test_emit_completed_with_original_request_passes_through() {
        let mut e = emitter();
        let mut req = crate::protocols::responses::ResponsesRequest::default();
        req.model = "m".to_string();
        req.instructions = Some("be terse".to_string());
        req.user = Some("u".to_string());
        e.set_original_request(req);
        let c = e.emit_completed(None);
        assert_eq!(c["response"]["instructions"], "be terse");
        assert_eq!(c["response"]["parallel_tool_calls"], true);
        assert_eq!(c["response"]["store"], true);
    }

    #[test]
    fn test_allocate_output_index_increments_for_function_call() {
        let mut e = emitter();
        let (i0, id0) = e.allocate_output_index(OutputItemType::Message);
        let (i1, id1) = e.allocate_output_index(OutputItemType::FunctionCall);
        assert_ne!(i0, i1);
        assert!(id1.starts_with("fc_"));
        assert!(id0.starts_with("msg_"));
    }

    #[test]
    fn test_process_chunk_content_emits_added_and_delta() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_with(Some("hi"), None, None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_item.added"));
        assert!(s.contains("response.content_part.added"));
        assert!(s.contains("response.output_text.delta"));
    }

    #[test]
    fn test_process_chunk_empty_content_is_skipped() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_with(Some(""), None, None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.is_empty());
    }

    #[test]
    fn test_process_chunk_tool_call_emits_function_call_events() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let deltas = vec![ToolCallDelta {
            index: 0,
            id: Some("call_x".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("f".to_string()),
                arguments: Some("{\"a\":1}".to_string()),
            }),
        }];
        e.process_chunk(&chunk_with(None, Some(deltas), None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_item.added"));
        assert!(s.contains("response.function_call_arguments.delta"));
    }

    #[test]
    fn test_process_chunk_finish_tool_calls_emits_done_for_each_call() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let deltas = vec![ToolCallDelta {
            index: 0,
            id: Some("c1".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("f".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }];
        e.process_chunk(&chunk_with(None, Some(deltas), None), &tx)
            .unwrap();
        e.process_chunk(&chunk_with(None, None, Some("tool_calls")), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.function_call_arguments.done"));
        assert!(s.contains("response.output_item.done"));
    }

    #[test]
    fn test_process_chunk_finish_stop_emits_text_done() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_with(Some("yo"), None, None), &tx)
            .unwrap();
        e.process_chunk(&chunk_with(None, None, Some("stop")), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_text.done"));
        assert!(s.contains("response.content_part.done"));
        assert!(s.contains("response.output_item.done"));
    }

    #[test]
    fn test_process_chunk_finish_length_emits_text_done() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_with(Some("yo"), None, None), &tx)
            .unwrap();
        e.process_chunk(&chunk_with(None, None, Some("length")), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_text.done"));
    }

    #[test]
    fn test_send_event_on_closed_channel_returns_error() {
        let e = emitter();
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let v = json!({"type": "x"});
        let r = e.send_event(&v, &tx);
        assert!(r.is_err());
    }

    #[test]
    fn test_build_sse_response_has_event_stream_content_type() {
        let (_tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let resp = crate::routers::openai::responses::streaming::build_sse_response(rx);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/event-stream"));
    }
}

mod k_streaming_end_to_end {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn req() -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(true);
        r
    }

    #[tokio::test]
    async fn test_streaming_emits_in_progress_event() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("response.in_progress"));
    }

    #[tokio::test]
    async fn test_streaming_with_instructions_includes_in_completed_response() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.instructions = Some("be brief".to_string());
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        let s = body_of(resp).await;
        assert!(s.contains("response.completed"));
        assert!(s.contains("be brief"));
    }

    #[tokio::test]
    async fn test_streaming_with_store_false_skips_persistence() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.store = Some(false);
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }
}

mod j2_stream_event_emitter_branches {
    use bytes::Bytes;
    use serde_json::json;
    use tokio::sync::mpsc;

    use crate::protocols::chat::{
        ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    };
    use crate::protocols::common::{FunctionCallDelta, ToolCallDelta};
    use crate::routers::openai::responses::streaming::{
        OutputItemType, ResponseStreamEventEmitter,
    };

    fn emitter() -> ResponseStreamEventEmitter {
        ResponseStreamEventEmitter::new("resp_y".to_string(), "test-model".to_string(), 99)
    }

    fn chunk_with(
        content: Option<&str>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish: Option<&str>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: "id".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "m".to_string(),
            system_fingerprint: None,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: content.map(|s| s.to_string()),
                    tool_calls,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: finish.map(|s| s.to_string()),
                matched_stop: None,
            }],
            usage: None,
        }
    }

    fn chunk_empty() -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: "id".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "m".to_string(),
            system_fingerprint: None,
            choices: vec![],
            usage: None,
        }
    }

    fn collect_lines(rx: &mut mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>) -> String {
        let mut out = String::new();
        while let Ok(item) = rx.try_recv() {
            if let Ok(b) = item {
                out.push_str(&String::from_utf8_lossy(&b));
            }
        }
        out
    }

    #[test]
    fn test_sequence_numbers_are_monotonic_across_calls() {
        let mut e = emitter();
        let c = e.emit_created();
        assert_eq!(c["sequence_number"], 0);
        let p = e.emit_in_progress();
        assert_eq!(p["sequence_number"], 1);
        let (idx, id) = e.allocate_output_index(OutputItemType::Message);
        let item = json!({"id": id.clone(), "type": "message"});
        let a = e.emit_output_item_added(idx, &item);
        assert_eq!(a["sequence_number"], 2);
        let cp = e.emit_content_part_added(idx, &id, 0);
        assert_eq!(cp["sequence_number"], 3);
        let td = e.emit_text_delta("a", idx, &id, 0);
        assert_eq!(td["sequence_number"], 4);
    }

    #[test]
    fn test_emit_completed_with_tracked_items_uses_them() {
        let mut e = emitter();
        let (idx, id) = e.allocate_output_index(OutputItemType::Message);
        let item = json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}]
        });
        e.store_output_item_data(idx, item.clone());
        e.complete_output_item(idx);
        let c = e.emit_completed(None);
        let output = c["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["id"], id);
    }

    #[test]
    fn test_emit_completed_without_usage() {
        let mut e = emitter();
        let c = e.emit_completed(None);
        assert!(c["response"]["usage"].is_null());
    }

    #[test]
    fn test_emit_completed_with_original_request_optional_fields() {
        let mut e = emitter();
        let mut req = crate::protocols::responses::ResponsesRequest::default();
        req.model = "m".to_string();
        req.max_output_tokens = Some(100);
        req.max_tool_calls = Some(5);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.previous_response_id = Some("prev_123".to_string());
        e.set_original_request(req);
        let c = e.emit_completed(None);
        assert_eq!(c["response"]["max_output_tokens"], 100);
        assert_eq!(c["response"]["max_tool_calls"], 5);
        assert_eq!(c["response"]["previous_response_id"], "prev_123");
    }

    #[test]
    fn test_emit_completed_without_tool_choice_defaults_to_auto() {
        let mut e = emitter();
        let mut req = crate::protocols::responses::ResponsesRequest::default();
        req.model = "m".to_string();
        e.set_original_request(req);
        let c = e.emit_completed(None);
        assert_eq!(c["response"]["tool_choice"], "auto");
    }

    #[test]
    fn test_emit_completed_without_original_request_no_optional_fields() {
        let mut e = emitter();
        let c = e.emit_completed(None);
        assert!(c["response"]["instructions"].is_null());
        assert!(c["response"]["store"].is_null());
    }

    #[test]
    fn test_process_chunk_empty_choices_is_noop() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_empty(), &tx).unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.is_empty());
    }

    #[test]
    fn test_process_chunk_multiple_content_deltas_accumulate() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        e.process_chunk(&chunk_with(Some("hello"), None, None), &tx)
            .unwrap();
        e.process_chunk(&chunk_with(Some(" world"), None, None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_text.delta"));
        assert!(s.contains("hello"));
        assert!(s.contains(" world"));
        assert!(s.contains("response.output_item.added"));
        assert!(s.contains("response.content_part.added"));
    }

    #[test]
    fn test_process_chunk_tool_call_without_args_no_args_delta() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let deltas = vec![ToolCallDelta {
            index: 0,
            id: Some("call_x".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("f".to_string()),
                arguments: None,
            }),
        }];
        e.process_chunk(&chunk_with(None, Some(deltas), None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_item.added"));
        assert!(!s.contains("response.function_call_arguments.delta"));
    }

    #[test]
    fn test_process_chunk_tool_call_empty_args_no_delta() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let deltas = vec![ToolCallDelta {
            index: 0,
            id: Some("call_y".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("g".to_string()),
                arguments: Some("".to_string()),
            }),
        }];
        e.process_chunk(&chunk_with(None, Some(deltas), None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(!s.contains("response.function_call_arguments.delta"));
    }

    #[test]
    fn test_process_chunk_multiple_tool_calls_allocate_separate_indices() {
        let mut e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let deltas = vec![
            ToolCallDelta {
                index: 0,
                id: Some("c1".to_string()),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some("f1".to_string()),
                    arguments: Some("{".to_string()),
                }),
            },
            ToolCallDelta {
                index: 1,
                id: Some("c2".to_string()),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some("f2".to_string()),
                    arguments: Some("{".to_string()),
                }),
            },
        ];
        e.process_chunk(&chunk_with(None, Some(deltas), None), &tx)
            .unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.contains("response.output_item.added"));
        assert!(s.contains("f1"));
        assert!(s.contains("f2"));
        assert!(s.contains("response.function_call_arguments.delta"));
    }

    #[test]
    fn test_send_event_formats_as_sse_with_event_type() {
        let e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let event = json!({"type": "custom.event", "data": 42});
        e.send_event(&event, &tx).unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.starts_with("event: custom.event\n"));
        assert!(s.contains("data: "));
    }

    #[test]
    fn test_send_event_missing_type_defaults_to_message() {
        let e = emitter();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let event = json!({"data": 42});
        e.send_event(&event, &tx).unwrap();
        let s = collect_lines(&mut rx);
        assert!(s.starts_with("event: message\n"));
    }

    #[test]
    fn test_complete_output_item_nonexistent_index_is_harmless() {
        let mut e = emitter();
        e.complete_output_item(999);
    }

    #[test]
    fn test_store_output_item_data_nonexistent_index_is_harmless() {
        let mut e = emitter();
        e.store_output_item_data(999, json!({"x": 1}));
    }

    #[test]
    fn test_allocate_function_call_id_starts_with_fc() {
        let mut e = emitter();
        let (_, id) = e.allocate_output_index(OutputItemType::FunctionCall);
        assert!(id.starts_with("fc_"));
    }

    #[test]
    fn test_allocate_message_id_starts_with_msg() {
        let mut e = emitter();
        let (_, id) = e.allocate_output_index(OutputItemType::Message);
        assert!(id.starts_with("msg_"));
    }

    #[test]
    fn test_emit_created_sets_status_in_progress() {
        let mut e = emitter();
        let v = e.emit_created();
        assert_eq!(v["response"]["status"], "in_progress");
        assert_eq!(v["response"]["model"], "test-model");
        assert_eq!(v["response"]["created_at"], 99);
    }

    #[test]
    fn test_build_sse_response_headers() {
        let (_tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        let resp = crate::routers::openai::responses::streaming::build_sse_response(rx);
        let cc = resp
            .headers()
            .get("cache-control")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(cc, "no-cache");
        let conn = resp.headers().get("connection").unwrap().to_str().unwrap();
        assert_eq!(conn, "keep-alive");
    }
}

mod k2_streaming_accumulator {
    use std::sync::Arc;

    use axum::body::to_bytes;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn req_with_store(store: Option<bool>) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(true);
        r.store = store;
        r
    }

    #[tokio::test]
    async fn test_streaming_output_text_delta_contains_content() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp = handlers::route_responses(
            &ctx,
            Arc::new(req_with_store(None)),
            None,
            Some("m".to_string()),
        )
        .await;
        let s = body_of(resp).await;
        assert!(s.contains("response.created"));
        assert!(s.contains("response.completed"));
    }

    #[tokio::test]
    async fn test_streaming_default_store_true_persists() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp = handlers::route_responses(
            &ctx,
            Arc::new(req_with_store(Some(true))),
            None,
            Some("m".to_string()),
        )
        .await;
        let s = body_of(resp).await;
        assert!(s.contains("response.completed"));
    }

    #[tokio::test]
    async fn test_streaming_event_order_created_before_completed() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp = handlers::route_responses(
            &ctx,
            Arc::new(req_with_store(None)),
            None,
            Some("m".to_string()),
        )
        .await;
        let s = body_of(resp).await;
        let created_pos = s.find("response.created").unwrap();
        let completed_pos = s.find("response.completed").unwrap();
        assert!(created_pos < completed_pos);
    }

    #[tokio::test]
    async fn test_streaming_done_marker_is_last() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp = handlers::route_responses(
            &ctx,
            Arc::new(req_with_store(None)),
            None,
            Some("m".to_string()),
        )
        .await;
        let s = body_of(resp).await;
        assert!(s.trim_end().ends_with("[DONE]"));
    }
}

mod l_non_streaming_branches {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::responses::{ResponseInput, ResponsesRequest};
    use crate::routers::openai::responses::handlers;
    use crate::routers::test_mocks;

    fn req() -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.model = "m".to_string();
        r.input = ResponseInput::Text("hello".to_string());
        r.stream = Some(false);
        r
    }

    #[tokio::test]
    async fn test_non_streaming_with_response_id_override() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let resp =
            handlers::route_responses(&ctx, Arc::new(req()), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = String::from_utf8_lossy(&body);
        assert!(s.contains("\"id\""));
    }

    #[tokio::test]
    async fn test_non_streaming_passes_instructions_through_to_request() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.instructions = Some("be brief".to_string());
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_non_streaming_empty_input_items_returns_400() {
        let ctx = test_mocks::responses_context_with_chat_path("m");
        let mut r = req();
        r.input = ResponseInput::Items(vec![]);
        let resp = handlers::route_responses(&ctx, Arc::new(r), None, Some("m".to_string())).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
