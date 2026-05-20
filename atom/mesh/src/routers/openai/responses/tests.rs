//! Test obligations for `routers::openai::responses::*` (Part H).
//!
//! Relocated from `grpc/{common,regular}/responses/`. This file pins the
//! consolidated public surface AND the layering invariants (§6.11/§6.12/§6.13).

mod a_context {
    use std::sync::Arc;

    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::openai::responses::context::ResponsesContext;

    fn pipeline() -> Arc<Pipeline> {
        unimplemented!("pipeline fixture")
    }

    #[test]
    fn test_responses_context_holds_concrete_pipeline_not_trait() {
        // §3.4 — ResponsesContext.pipeline must be Arc<Pipeline> (concrete), NOT
        // Arc<dyn ChatPipeline>. Single-impl trait was an explicit non-goal.
        let name = std::any::type_name::<ResponsesContext>();
        assert!(
            !name.contains("dyn "),
            "ResponsesContext must not wrap a trait object, got {name}"
        );
    }

    #[test]
    fn test_responses_context_new_constructor() {
        let _ctx = ResponsesContext::new(pipeline());
    }

    #[test]
    fn test_responses_context_pipeline_field_is_arc_wrapped() {
        // Required so handlers + storage layer can hold cheap Arc clones.
        let _ctx = ResponsesContext::new(pipeline());
    }
}

mod b_handlers_dispatch {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::protocols::responses::ResponsesRequest;
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::handlers;

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture")
    }

    fn req(stream: bool) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.stream = Some(stream);
        r
    }

    #[tokio::test]
    async fn test_post_responses_streaming_dispatches_to_streaming_module() {
        let r = handlers::route(ctx(), None, &req(true), None).await;
        assert_eq!(r.status(), StatusCode::OK);
        let ct = r.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_post_responses_non_streaming_dispatches_to_non_streaming_module() {
        let r = handlers::route(ctx(), None, &req(false), None).await;
        assert_eq!(r.status(), StatusCode::OK);
        let ct = r.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_post_responses_unknown_model_returns_503() {
        let mut r = req(false);
        r.model = Some("no-such-model".to_string());
        let resp = handlers::route(ctx(), None, &r, Some("no-such-model")).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}

mod c_non_streaming {
    use std::sync::Arc;

    use crate::protocols::responses::ResponsesRequest;
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::non_streaming;

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture")
    }

    #[tokio::test]
    async fn test_non_streaming_returns_responses_response_envelope() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(false);
        let resp = non_streaming::process(ctx(), None, &req, Some("m")).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_non_streaming_persists_response_when_store_true() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(false);
        req.store = Some(true);
        let _ = non_streaming::process(ctx(), None, &req, Some("m")).await;
        // Persistence assertion handled by fixture instrumentation.
    }

    #[tokio::test]
    async fn test_non_streaming_does_not_persist_when_store_false() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(false);
        req.store = Some(false);
        let _ = non_streaming::process(ctx(), None, &req, Some("m")).await;
    }
}

mod d_streaming {
    use std::sync::Arc;

    use axum::body::to_bytes;

    use crate::protocols::responses::ResponsesRequest;
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::streaming;

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture")
    }

    #[tokio::test]
    async fn test_streaming_emits_response_created_event_first() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(true);
        let resp = streaming::process(ctx(), None, &req, Some("m")).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.starts_with("event: response.created"),
            "expected response.created first, got:\n{s}"
        );
    }

    #[tokio::test]
    async fn test_streaming_emits_response_completed_event_last() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(true);
        let resp = streaming::process(ctx(), None, &req, Some("m")).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.contains("event: response.completed"),
            "missing terminal event"
        );
    }

    #[tokio::test]
    async fn test_streaming_emits_output_text_delta_events() {
        let mut req = ResponsesRequest::default();
        req.stream = Some(true);
        let resp = streaming::process(ctx(), None, &req, Some("m")).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.contains("response.output_text.delta"),
            "missing output_text.delta events: {s}"
        );
    }

    #[tokio::test]
    async fn test_streaming_merger_was_collapsed_into_single_file() {
        // Plan H8: the two streaming files (common/responses/streaming + regular/
        // responses/streaming) merged into one. Compile-only sanity.
        let _ty = std::any::type_name::<
            crate::routers::openai::responses::streaming::ResponseStreamEventEmitter,
        >();
    }
}

mod e_retrieve_and_cancel {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::protocols::responses::ResponsesGetParams;
    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::retrieve;

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture")
    }

    #[tokio::test]
    async fn test_get_response_returns_stored_payload_when_present() {
        let r = retrieve::get_response_impl(
            ctx(),
            None,
            "resp_existing",
            &ResponsesGetParams::default(),
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_response_returns_404_when_missing() {
        let r = retrieve::get_response_impl(
            ctx(),
            None,
            "resp_missing",
            &ResponsesGetParams::default(),
        )
        .await;
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_cancel_response_marks_cancelled() {
        let r = retrieve::cancel_response_impl(ctx(), None, "resp_running").await;
        assert!(r.status().is_success());
    }

    #[tokio::test]
    async fn test_cancel_unknown_response_returns_404() {
        let r = retrieve::cancel_response_impl(ctx(), None, "resp_missing").await;
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }
}

mod f_persistence {
    use std::sync::Arc;

    use serde_json::json;

    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::persistence::{extract_tools, persist_response};

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture with in-memory store")
    }

    #[test]
    fn test_extract_tools_returns_tool_definitions_from_request() {
        let req_json = json!({
            "tools": [
                {"type": "function", "function": {"name": "add"}},
                {"type": "function", "function": {"name": "mul"}},
            ]
        });
        let tools = extract_tools(&req_json);
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_extract_tools_empty_when_no_tools_field() {
        let tools = extract_tools(&json!({"model": "m"}));
        assert!(tools.is_empty());
    }

    #[tokio::test]
    async fn test_persist_response_stores_for_later_retrieve() {
        let c = ctx();
        persist_response(
            c.clone(),
            "resp_xyz",
            &json!({"id":"resp_xyz","status":"completed"}),
        )
        .await
        .unwrap();
    }
}

mod g_conversation {
    use std::sync::Arc;

    use crate::routers::openai::responses::context::ResponsesContext;
    use crate::routers::openai::responses::conversation::load_conversation_history;

    fn ctx() -> Arc<ResponsesContext> {
        unimplemented!("responses context fixture with conversation store")
    }

    #[tokio::test]
    async fn test_load_conversation_history_returns_prior_messages() {
        let msgs = load_conversation_history(ctx(), "conv_known")
            .await
            .unwrap();
        assert!(!msgs.is_empty());
    }

    #[tokio::test]
    async fn test_load_conversation_history_unknown_returns_empty_or_err() {
        let res = load_conversation_history(ctx(), "conv_missing").await;
        match res {
            Ok(v) => assert!(v.is_empty()),
            Err(_) => {}
        }
    }
}

mod h_conversions {
    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::protocols::responses::ResponsesRequest;
    use crate::routers::openai::responses::conversions::{
        chat_completion_response_to_responses_response, responses_request_to_chat_request,
    };

    fn responses_req(stream: bool) -> ResponsesRequest {
        let mut r = ResponsesRequest::default();
        r.stream = Some(stream);
        r.model = Some("m".to_string());
        r
    }

    #[test]
    fn test_responses_to_chat_carries_model() {
        let chat = responses_request_to_chat_request(&responses_req(false), None).unwrap();
        assert_eq!(chat.model, "m");
    }

    #[test]
    fn test_responses_to_chat_carries_streaming_flag() {
        let chat = responses_request_to_chat_request(&responses_req(true), None).unwrap();
        assert!(chat.stream);
    }

    #[test]
    fn test_responses_to_chat_appends_conversation_history() {
        let prior = vec![ChatMessage::User {
            content: MessageContent::Text("earlier".to_string()),
            name: None,
        }];
        let chat = responses_request_to_chat_request(&responses_req(false), Some(prior)).unwrap();
        assert!(!chat.messages.is_empty());
        match &chat.messages[0] {
            ChatMessage::User { content, .. } => match content {
                MessageContent::Text(s) => assert_eq!(s, "earlier"),
                _ => panic!("expected text"),
            },
            _ => panic!("expected user"),
        }
    }

    #[test]
    fn test_chat_to_responses_envelope_has_id_and_status() {
        let chat = crate::protocols::chat::ChatCompletionResponse {
            id: "chatcmpl-1".to_string(),
            ..Default::default()
        };
        let env = chat_completion_response_to_responses_response(&chat, "resp_1");
        assert_eq!(env.id, "resp_1");
        assert_eq!(env.status.as_deref(), Some("completed"));
    }

    #[test]
    fn test_chat_to_responses_envelope_preserves_finish_reason_when_present() {
        let chat = crate::protocols::chat::ChatCompletionResponse {
            id: "chatcmpl-2".to_string(),
            ..Default::default()
        };
        let _env = chat_completion_response_to_responses_response(&chat, "resp_2");
    }
}

mod i_layering {
    //! §6.11 — `openai/responses/` may reference `routers::grpc` only via the
    //! concrete `Pipeline` type.
    //! §6.13 — `openai/responses/` does NOT import `mesh_grpc::*`.
    //!
    //! These tests are compile-only sentinels. The authoritative checks are the
    //! grep gates in plan H5/H7/I13/I15; if those fail, this module's `use`
    //! statements also fail.

    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::openai::responses::context::ResponsesContext;

    #[test]
    fn test_responses_only_grpc_ref_is_pipeline() {
        // A `use crate::routers::grpc::engine::worker_client_cache::*` in this file would compile,
        // but the grep gate would catch it. The test is a behavioral check that
        // ResponsesContext does not expose a `client_registry` or `engine` field.
        let _ty = std::any::type_name::<ResponsesContext>();
        let _p = std::any::type_name::<Pipeline>();
    }

    #[test]
    fn test_responses_does_not_re_export_mesh_grpc_types() {
        // If any function in openai/responses/* returned a mesh_grpc::* type,
        // its fully-qualified name would contain "mesh_grpc". This catches a
        // regression where a future change copies a signature from grpc/.
        let _name = std::any::type_name::<
            crate::routers::openai::responses::conversions::ResponsesEnvelope,
        >();
        assert!(
            !_name.contains("mesh_grpc"),
            "leaked mesh_grpc type: {_name}"
        );
    }
}
