// Integration test for Responses API

use axum::http::StatusCode;
use mesh::{
    config::RouterConfig,
    protocols::{
        common::{GenerationRequest, ToolChoice, ToolChoiceValue, UsageInfo},
        responses::{
            ReasoningEffort, ResponseInput, ResponseReasoningParam, ResponseTool, ResponseToolType,
            ResponsesRequest, ServiceTier, Truncation,
        },
    },
    routers::{conversations, RouterFactory},
};

#[tokio::test]
async fn test_conversations_crud_basic() {
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create
    let create_body = serde_json::json!({ "metadata": { "project": "alpha" } });
    let create_resp =
        conversations::create_conversation(&ctx.conversation_storage, create_body.clone()).await;
    assert_eq!(create_resp.status(), StatusCode::OK);
    let create_bytes = axum::body::to_bytes(create_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let create_json: serde_json::Value = serde_json::from_slice(&create_bytes).unwrap();
    let conv_id = create_json["id"].as_str().expect("id missing");
    assert!(conv_id.starts_with("conv_"));
    assert_eq!(create_json["object"], "conversation");

    // Get
    let get_resp = conversations::get_conversation(&ctx.conversation_storage, conv_id).await;
    assert_eq!(get_resp.status(), StatusCode::OK);
    let get_bytes = axum::body::to_bytes(get_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let get_json: serde_json::Value = serde_json::from_slice(&get_bytes).unwrap();
    assert_eq!(get_json["metadata"]["project"], serde_json::json!("alpha"));

    // Update (merge)
    let update_body = serde_json::json!({ "metadata": { "owner": "alice" } });
    let upd_resp =
        conversations::update_conversation(&ctx.conversation_storage, conv_id, update_body.clone())
            .await;
    assert_eq!(upd_resp.status(), StatusCode::OK);
    let upd_bytes = axum::body::to_bytes(upd_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let upd_json: serde_json::Value = serde_json::from_slice(&upd_bytes).unwrap();
    assert_eq!(upd_json["metadata"]["project"], serde_json::json!("alpha"));
    assert_eq!(upd_json["metadata"]["owner"], serde_json::json!("alice"));

    // Delete
    let del_resp = conversations::delete_conversation(&ctx.conversation_storage, conv_id).await;
    assert_eq!(del_resp.status(), StatusCode::OK);
    let del_bytes = axum::body::to_bytes(del_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let del_json: serde_json::Value = serde_json::from_slice(&del_bytes).unwrap();
    assert_eq!(del_json["deleted"], serde_json::json!(true));

    // Get again -> 404
    let not_found = conversations::get_conversation(&ctx.conversation_storage, conv_id).await;
    assert_eq!(not_found.status(), StatusCode::NOT_FOUND);
}

#[test]
fn test_responses_request_creation() {
    let request = ResponsesRequest {
        background: Some(false),
        include: None,
        input: ResponseInput::Text("Hello, world!".to_string()),
        instructions: Some("Be helpful".to_string()),
        max_output_tokens: Some(100),
        max_tool_calls: None,
        metadata: None,
        model: "test-model".to_string(),
        parallel_tool_calls: Some(true),
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::Medium),
            summary: None,
        }),
        service_tier: Some(ServiceTier::Auto),
        store: Some(true),
        stream: Some(false),
        temperature: Some(0.7),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
        tools: Some(vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }]),
        top_logprobs: Some(5),
        top_p: Some(0.9),
        truncation: Some(Truncation::Disabled),
        text: None,
        user: Some("test-user".to_string()),
        request_id: Some("resp_test123".to_string()),
        priority: 0,
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        stop: None,
        top_k: -1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    assert!(!request.is_stream());
    assert_eq!(request.get_model(), Some("test-model"));
    let routing_text = request.extract_text_for_routing();
    assert_eq!(routing_text, "Hello, world!");
}

#[test]
fn test_responses_request_sglang_extensions() {
    // Test that SGLang-specific sampling parameters are present and serializable
    let request = ResponsesRequest {
        background: Some(false),
        include: None,
        input: ResponseInput::Text("Test".to_string()),
        instructions: None,
        max_output_tokens: Some(50),
        max_tool_calls: None,
        metadata: None,
        model: "test-model".to_string(),
        parallel_tool_calls: Some(true),
        previous_response_id: None,
        reasoning: None,
        service_tier: Some(ServiceTier::Auto),
        store: Some(true),
        stream: Some(false),
        temperature: Some(0.8),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Auto)),
        tools: Some(vec![]),
        top_logprobs: Some(0),
        top_p: Some(0.95),
        truncation: Some(Truncation::Auto),
        text: None,
        user: None,
        request_id: Some("resp_test456".to_string()),
        priority: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.2),
        stop: None,
        // SGLang-specific extensions:
        top_k: 10,
        min_p: 0.05,
        repetition_penalty: 1.1,
        conversation: None,
    };

    // Verify SGLang extensions are present
    assert_eq!(request.top_k, 10);
    assert_eq!(request.min_p, 0.05);
    assert_eq!(request.repetition_penalty, 1.1);

    // Verify serialization works with SGLang extensions
    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(parsed.top_k, 10);
    assert_eq!(parsed.min_p, 0.05);
    assert_eq!(parsed.repetition_penalty, 1.1);
}

#[test]
fn test_usage_conversion() {
    // Construct UsageInfo directly with cached token details
    let usage_info = UsageInfo {
        prompt_tokens: 15,
        completion_tokens: 25,
        total_tokens: 40,
        reasoning_tokens: Some(8),
        prompt_tokens_details: Some(mesh::protocols::common::PromptTokenUsageInfo {
            cached_tokens: 3,
        }),
    };
    let response_usage = usage_info.to_response_usage();

    assert_eq!(response_usage.input_tokens, 15);
    assert_eq!(response_usage.output_tokens, 25);
    assert_eq!(response_usage.total_tokens, 40);

    // Check details are converted correctly
    assert!(response_usage.input_tokens_details.is_some());
    assert_eq!(
        response_usage
            .input_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens,
        3
    );

    assert!(response_usage.output_tokens_details.is_some());
    assert_eq!(
        response_usage
            .output_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        8
    );

    let back_to_usage = response_usage.to_usage_info();
    assert_eq!(back_to_usage.prompt_tokens, 15);
    assert_eq!(back_to_usage.completion_tokens, 25);
    assert_eq!(back_to_usage.reasoning_tokens, Some(8));
}

#[test]
fn test_reasoning_param_default() {
    let param = ResponseReasoningParam {
        effort: Some(ReasoningEffort::Medium),
        summary: None,
    };

    let json = serde_json::to_string(&param).unwrap();
    let parsed: ResponseReasoningParam = serde_json::from_str(&json).unwrap();

    assert!(matches!(parsed.effort, Some(ReasoningEffort::Medium)));
}

#[test]
fn test_json_serialization() {
    let request = ResponsesRequest {
        background: Some(true),
        include: None,
        input: ResponseInput::Text("Test input".to_string()),
        instructions: Some("Test instructions".to_string()),
        max_output_tokens: Some(200),
        max_tool_calls: Some(5),
        metadata: None,
        model: "gpt-4".to_string(),
        parallel_tool_calls: Some(false),
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::High),
            summary: None,
        }),
        service_tier: Some(ServiceTier::Priority),
        store: Some(false),
        stream: Some(true),
        temperature: Some(0.9),
        tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
        tools: Some(vec![ResponseTool {
            r#type: ResponseToolType::CodeInterpreter,
            ..Default::default()
        }]),
        top_logprobs: Some(10),
        top_p: Some(0.8),
        truncation: Some(Truncation::Auto),
        text: None,
        user: Some("test_user".to_string()),
        request_id: Some("resp_comprehensive_test".to_string()),
        priority: 1,
        frequency_penalty: Some(0.3),
        presence_penalty: Some(0.4),
        stop: None,
        top_k: 50,
        min_p: 0.1,
        repetition_penalty: 1.2,
        conversation: None,
    };

    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(
        parsed.request_id,
        Some("resp_comprehensive_test".to_string())
    );
    assert_eq!(parsed.model, "gpt-4");
    assert_eq!(parsed.background, Some(true));
    assert_eq!(parsed.stream, Some(true));
    assert_eq!(parsed.tools.as_ref().map(|t| t.len()), Some(1));
}

#[tokio::test]
async fn test_conversation_items_create_and_get() {
    // Test creating items and getting a specific item
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create conversation
    let create_conv = serde_json::json!({});
    let conv_resp =
        conversations::create_conversation(&ctx.conversation_storage, create_conv).await;
    assert_eq!(conv_resp.status(), StatusCode::OK);
    let conv_bytes = axum::body::to_bytes(conv_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_json: serde_json::Value = serde_json::from_slice(&conv_bytes).unwrap();
    let conv_id = conv_json["id"].as_str().unwrap();

    // Create items
    let create_items = serde_json::json!({
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}]
            }
        ]
    });

    let items_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        create_items,
    )
    .await;
    assert_eq!(items_resp.status(), StatusCode::OK);
    let items_bytes = axum::body::to_bytes(items_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let items_json: serde_json::Value = serde_json::from_slice(&items_bytes).unwrap();

    // Verify response structure
    assert_eq!(items_json["object"], "list");
    assert!(items_json["data"].is_array());

    // Get first item
    let item_id = items_json["data"][0]["id"].as_str().unwrap();
    let get_resp = conversations::get_conversation_item(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        item_id,
        None,
    )
    .await;
    assert_eq!(get_resp.status(), StatusCode::OK);
    let get_bytes = axum::body::to_bytes(get_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let get_json: serde_json::Value = serde_json::from_slice(&get_bytes).unwrap();

    // Verify item structure
    assert_eq!(get_json["id"], item_id);
    assert_eq!(get_json["type"], "message");
    assert_eq!(get_json["role"], "user");
}

#[tokio::test]
async fn test_conversation_items_delete() {
    // Test deleting an item from a conversation
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create conversation
    let create_conv = serde_json::json!({});
    let conv_resp =
        conversations::create_conversation(&ctx.conversation_storage, create_conv).await;
    let conv_bytes = axum::body::to_bytes(conv_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_json: serde_json::Value = serde_json::from_slice(&conv_bytes).unwrap();
    let conv_id = conv_json["id"].as_str().unwrap();

    // Create item
    let create_items = serde_json::json!({
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Test"}]
            }
        ]
    });

    let items_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        create_items,
    )
    .await;
    let items_bytes = axum::body::to_bytes(items_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let items_json: serde_json::Value = serde_json::from_slice(&items_bytes).unwrap();
    let item_id = items_json["data"][0]["id"].as_str().unwrap();

    // List items (should have 1)
    let list_resp = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        None,
        None,
        None,
    )
    .await;
    let list_bytes = axum::body::to_bytes(list_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_json: serde_json::Value = serde_json::from_slice(&list_bytes).unwrap();
    assert_eq!(list_json["data"].as_array().unwrap().len(), 1);

    // Delete item
    let del_resp = conversations::delete_conversation_item(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        item_id,
    )
    .await;
    assert_eq!(del_resp.status(), StatusCode::OK);

    // List items again (should have 0)
    let list_resp2 = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        None,
        None,
        None,
    )
    .await;
    let list_bytes2 = axum::body::to_bytes(list_resp2.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_json2: serde_json::Value = serde_json::from_slice(&list_bytes2).unwrap();
    assert_eq!(list_json2["data"].as_array().unwrap().len(), 0);

    // Item should NOT be gettable from this conversation after deletion (link removed)
    let get_resp = conversations::get_conversation_item(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        item_id,
        None,
    )
    .await;
    assert_eq!(get_resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_conversation_items_max_limit() {
    // Test that creating > 20 items returns error
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create conversation
    let create_conv = serde_json::json!({});
    let conv_resp =
        conversations::create_conversation(&ctx.conversation_storage, create_conv).await;
    let conv_bytes = axum::body::to_bytes(conv_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_json: serde_json::Value = serde_json::from_slice(&conv_bytes).unwrap();
    let conv_id = conv_json["id"].as_str().unwrap();

    // Try to create 21 items (over limit)
    let mut items = Vec::new();
    for i in 0..21 {
        items.push(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": format!("Message {}", i)}]
        }));
    }
    let create_items = serde_json::json!({ "items": items });

    let items_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        create_items,
    )
    .await;
    assert_eq!(items_resp.status(), StatusCode::BAD_REQUEST);

    let items_bytes = axum::body::to_bytes(items_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let items_text = String::from_utf8_lossy(&items_bytes);
    assert!(items_text.contains("Cannot add more than 20 items"));
}

#[tokio::test]
async fn test_conversation_items_unsupported_type() {
    // Test that unsupported item types return error
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create conversation
    let create_conv = serde_json::json!({});
    let conv_resp =
        conversations::create_conversation(&ctx.conversation_storage, create_conv).await;
    let conv_bytes = axum::body::to_bytes(conv_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_json: serde_json::Value = serde_json::from_slice(&conv_bytes).unwrap();
    let conv_id = conv_json["id"].as_str().unwrap();

    // Try to create item with completely unsupported type
    let create_items = serde_json::json!({
        "items": [
            {
                "type": "totally_invalid_type",
                "content": []
            }
        ]
    });

    let items_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_id,
        create_items,
    )
    .await;
    assert_eq!(items_resp.status(), StatusCode::BAD_REQUEST);

    let items_bytes = axum::body::to_bytes(items_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let items_text = String::from_utf8_lossy(&items_bytes);
    assert!(items_text.contains("Unsupported item type"));
}

#[tokio::test]
async fn test_conversation_items_multi_conversation_sharing() {
    // Test that items can be shared across conversations via soft delete
    let router_cfg = RouterConfig::builder()
        .regular_mode(vec!["http://localhost".to_string()])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(8 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .log_level("warn")
        .max_concurrent_requests(8)
        .queue_timeout_secs(5)
        .build_unchecked();

    let ctx = crate::common::create_test_context(router_cfg).await;
    let _router = RouterFactory::create_router(&ctx).await.expect("router");

    // Create two conversations
    let conv_a_resp =
        conversations::create_conversation(&ctx.conversation_storage, serde_json::json!({})).await;
    let conv_a_bytes = axum::body::to_bytes(conv_a_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_a_json: serde_json::Value = serde_json::from_slice(&conv_a_bytes).unwrap();
    let conv_a_id = conv_a_json["id"].as_str().unwrap();

    let conv_b_resp =
        conversations::create_conversation(&ctx.conversation_storage, serde_json::json!({})).await;
    let conv_b_bytes = axum::body::to_bytes(conv_b_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let conv_b_json: serde_json::Value = serde_json::from_slice(&conv_b_bytes).unwrap();
    let conv_b_id = conv_b_json["id"].as_str().unwrap();

    // Create item in conversation A
    let create_items = serde_json::json!({
        "items": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Shared message"}]
            }
        ]
    });

    let items_a_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_a_id,
        create_items,
    )
    .await;
    let items_a_bytes = axum::body::to_bytes(items_a_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let items_a_json: serde_json::Value = serde_json::from_slice(&items_a_bytes).unwrap();
    let item_id = items_a_json["data"][0]["id"].as_str().unwrap();

    // Reference the same item in conversation B
    let reference_items = serde_json::json!({
        "items": [
            {
                "type": "item_reference",
                "id": item_id
            }
        ]
    });

    let items_b_resp = conversations::create_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_b_id,
        reference_items,
    )
    .await;
    assert_eq!(items_b_resp.status(), StatusCode::OK);

    // Verify item appears in both conversations
    let list_a = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_a_id,
        None,
        None,
        None,
    )
    .await;
    let list_a_bytes = axum::body::to_bytes(list_a.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_a_json: serde_json::Value = serde_json::from_slice(&list_a_bytes).unwrap();
    assert_eq!(list_a_json["data"].as_array().unwrap().len(), 1);

    let list_b = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_b_id,
        None,
        None,
        None,
    )
    .await;
    let list_b_bytes = axum::body::to_bytes(list_b.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_b_json: serde_json::Value = serde_json::from_slice(&list_b_bytes).unwrap();
    assert_eq!(list_b_json["data"].as_array().unwrap().len(), 1);

    // Delete from conversation A
    conversations::delete_conversation_item(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_a_id,
        item_id,
    )
    .await;

    // Should be removed from A
    let list_a2 = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_a_id,
        None,
        None,
        None,
    )
    .await;
    let list_a2_bytes = axum::body::to_bytes(list_a2.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_a2_json: serde_json::Value = serde_json::from_slice(&list_a2_bytes).unwrap();
    assert_eq!(list_a2_json["data"].as_array().unwrap().len(), 0);

    // Should still exist in B (soft delete)
    let list_b2 = conversations::list_conversation_items(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_b_id,
        None,
        None,
        None,
    )
    .await;
    let list_b2_bytes = axum::body::to_bytes(list_b2.into_body(), usize::MAX)
        .await
        .unwrap();
    let list_b2_json: serde_json::Value = serde_json::from_slice(&list_b2_bytes).unwrap();
    assert_eq!(list_b2_json["data"].as_array().unwrap().len(), 1);

    // Item should still be directly gettable
    let get_resp = conversations::get_conversation_item(
        &ctx.conversation_storage,
        &ctx.conversation_item_storage,
        conv_b_id,
        item_id,
        None,
    )
    .await;
    assert_eq!(get_resp.status(), StatusCode::OK);
}
