//! Streaming infrastructure for /v1/responses endpoint
//!
//! Contains both the OpenAI-compatible event emitter / SSE response builder and
//! the chat→responses stream conversion that drives them.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{self, StatusCode},
    response::Response,
};
use bytes::Bytes;
use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{context::ResponsesContext, persistence::persist_response_if_needed};
use crate::protocols::{
    chat::{ChatCompletionRequest, ChatCompletionStreamResponse},
    common::{Usage, UsageInfo},
    event_types::{
        ContentPartEvent, FunctionCallEvent, OutputItemEvent, OutputTextEvent, ResponseEvent,
    },
    responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};

pub(crate) enum OutputItemType {
    Message,
    FunctionCall,
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
    item_data: Option<serde_json::Value>,
}

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.content_part.added
/// - response.output_text.delta (multiple)
/// - response.output_text.done
/// - response.content_part.done
/// - response.output_item.done
/// - response.completed
pub(crate) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    pub response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    accumulated_text: String,
    has_emitted_created: bool,
    has_emitted_in_progress: bool,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>,
    current_item_id: Option<String>,
    original_request: Option<ResponsesRequest>,
    // Maps tool_call delta index → (output_index, item_id, name, accumulated_args).
    tool_call_items: Vec<(usize, String, String, String)>,
}

impl ResponseStreamEventEmitter {
    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
        let message_id = format!("msg_{}", Uuid::new_v4());

        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            message_id,
            accumulated_text: String::new(),
            has_emitted_created: false,
            has_emitted_in_progress: false,
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
            original_request: None,
            tool_call_items: Vec::new(),
        }
    }

    pub fn set_original_request(&mut self, request: ResponsesRequest) {
        self.original_request = Some(request);
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    pub fn emit_created(&mut self) -> serde_json::Value {
        self.has_emitted_created = true;
        json!({
            "type": ResponseEvent::CREATED,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": []
            }
        })
    }

    pub fn emit_in_progress(&mut self) -> serde_json::Value {
        self.has_emitted_in_progress = true;
        json!({
            "type": ResponseEvent::IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    pub fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": ContentPartEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": ""
            }
        })
    }

    pub fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": OutputTextEvent::DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta
        })
    }

    pub fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": OutputTextEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    pub fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": ContentPartEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": self.accumulated_text.clone()
            }
        })
    }

    pub fn emit_completed(&mut self, usage: Option<&serde_json::Value>) -> serde_json::Value {
        let output: Vec<serde_json::Value> = self
            .output_items
            .iter()
            .filter_map(|item| {
                if item.status == ItemStatus::Completed {
                    item.item_data.clone()
                } else {
                    None
                }
            })
            .collect();

        // Fall back to a generic assistant message when no items were tracked.
        let output = if output.is_empty() {
            vec![json!({
                "id": self.message_id.clone(),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": self.accumulated_text.clone()
                }]
            })]
        } else {
            output
        };

        let mut response_obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output
        });

        if let Some(usage_val) = usage {
            response_obj["usage"] = usage_val.clone();
        }

        if let Some(ref req) = self.original_request {
            Self::add_optional_field(&mut response_obj, "instructions", &req.instructions);
            Self::add_optional_field(
                &mut response_obj,
                "max_output_tokens",
                &req.max_output_tokens,
            );
            Self::add_optional_field(&mut response_obj, "max_tool_calls", &req.max_tool_calls);
            Self::add_optional_field(
                &mut response_obj,
                "previous_response_id",
                &req.previous_response_id,
            );
            Self::add_optional_field(&mut response_obj, "reasoning", &req.reasoning);
            Self::add_optional_field(&mut response_obj, "temperature", &req.temperature);
            Self::add_optional_field(&mut response_obj, "top_p", &req.top_p);
            Self::add_optional_field(&mut response_obj, "truncation", &req.truncation);
            Self::add_optional_field(&mut response_obj, "user", &req.user);

            response_obj["parallel_tool_calls"] = json!(req.parallel_tool_calls.unwrap_or(true));
            response_obj["store"] = json!(req.store.unwrap_or(true));
            response_obj["tools"] = json!(req.tools.as_ref().unwrap_or(&vec![]));
            response_obj["metadata"] = json!(req.metadata.as_ref().unwrap_or(&Default::default()));

            if let Some(ref tc) = req.tool_choice {
                response_obj["tool_choice"] = json!(tc);
            } else {
                response_obj["tool_choice"] = json!("auto");
            }
        }

        json!({
            "type": ResponseEvent::COMPLETED,
            "sequence_number": self.next_sequence(),
            "response": response_obj
        })
    }

    fn add_optional_field<T: serde::Serialize>(
        obj: &mut serde_json::Value,
        key: &str,
        value: &Option<T>,
    ) {
        if let Some(val) = value {
            obj[key] = json!(val);
        }
    }

    pub fn emit_function_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub fn emit_function_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": OutputItemEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    pub fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        // emit_completed later replays stored item_data for tracked items.
        self.store_output_item_data(output_index, item.clone());

        json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::new_v4().to_string().replace("-", ""))
    }

    pub fn allocate_output_index(&mut self, item_type: OutputItemType) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id_prefix = match &item_type {
            OutputItemType::FunctionCall => "fc",
            OutputItemType::Message => "msg",
        };

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
            item_data: None,
        });

        (index, id)
    }

    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    pub fn store_output_item_data(&mut self, output_index: usize, item_data: serde_json::Value) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.item_data = Some(item_data);
        }
    }

    pub fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    if self.current_item_id.is_none() {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::Message);

                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx)?;
                        self.has_emitted_output_item_added = true;

                        // Store for subsequent events
                        self.current_item_id = Some(item_id);
                        self.current_message_output_index = Some(output_index);
                    }

                    let output_index = self.current_message_output_index.unwrap();
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0; // Single content part for now

                    // Emit content_part.added before first delta
                    if !self.has_emitted_content_part_added {
                        let event =
                            self.emit_content_part_added(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        self.has_emitted_content_part_added = true;
                    }

                    let event =
                        self.emit_text_delta(content, output_index, &item_id, content_index);
                    self.send_event(&event, tx)?;
                }
            }

            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let tc_index = delta.index as usize;

                    while self.tool_call_items.len() <= tc_index {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::FunctionCall);
                        self.tool_call_items.push((
                            output_index,
                            item_id,
                            String::new(),
                            String::new(),
                        ));
                    }

                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            self.tool_call_items[tc_index].2.push_str(name);
                        }
                    }

                    // First delta for the tool call carries its id; emit output_item.added once.
                    if let Some(delta_id) = &delta.id {
                        let output_index = self.tool_call_items[tc_index].0;
                        let item_id = self.tool_call_items[tc_index].1.clone();
                        let tc_name = self.tool_call_items[tc_index].2.clone();
                        let item = json!({
                            "id": item_id,
                            "type": "function_call",
                            "call_id": delta_id,
                            "name": tc_name,
                            "arguments": "",
                            "status": "in_progress"
                        });
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx)?;
                    }

                    if let Some(function) = &delta.function {
                        if let Some(args) = &function.arguments {
                            if !args.is_empty() {
                                self.tool_call_items[tc_index].3.push_str(args);
                                let output_index = self.tool_call_items[tc_index].0;
                                let item_id = self.tool_call_items[tc_index].1.clone();
                                let event = self.emit_function_call_arguments_delta(
                                    output_index,
                                    &item_id,
                                    args,
                                );
                                self.send_event(&event, tx)?;
                            }
                        }
                    }
                }
            }

            if let Some(reason) = &choice.finish_reason {
                if reason == "tool_calls" {
                    let tool_calls: Vec<_> = self.tool_call_items.clone();
                    for (output_index, item_id, tc_name, accumulated_args) in &tool_calls {
                        let event = self.emit_function_call_arguments_done(
                            *output_index,
                            item_id,
                            accumulated_args,
                        );
                        self.send_event(&event, tx)?;

                        let item = json!({
                            "id": item_id,
                            "type": "function_call",
                            "call_id": item_id,
                            "name": tc_name,
                            "arguments": accumulated_args,
                            "status": "completed"
                        });
                        let event = self.emit_output_item_done(*output_index, &item);
                        self.send_event(&event, tx)?;
                        self.complete_output_item(*output_index);
                    }
                }

                if reason == "stop" || reason == "length" {
                    let output_index = self.current_message_output_index.unwrap();
                    // Clone to release the borrow on `self` before calling &mut self methods.
                    let item_id = self.current_item_id.clone().unwrap();
                    let content_index = 0;

                    if self.has_emitted_content_part_added {
                        let event = self.emit_text_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        let event =
                            self.emit_content_part_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                    }

                    if self.has_emitted_output_item_added {
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": self.accumulated_text.clone()
                            }]
                        });
                        let event = self.emit_output_item_done(output_index, &item);
                        self.send_event(&event, tx)?;
                    }

                    self.complete_output_item(output_index);
                }
            }
        }

        Ok(())
    }

    pub fn send_event(
        &self,
        event: &serde_json::Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let event_json = serde_json::to_string(event)
            .map_err(|e| format!("Failed to serialize event: {}", e))?;

        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        let sse_message = format!("event: {}\ndata: {}\n\n", event_type, event_json);

        if tx.send(Ok(Bytes::from(sse_message))).is_err() {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }
}

pub(crate) fn build_sse_response(
    rx: mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>,
) -> Response {
    let stream = UnboundedReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Get chat streaming response
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request.clone(),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await;

    // Extract body from chat response
    let (_parts, body) = chat_response.into_parts();

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();

    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            tx.clone(),
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            let error_event = json!({
                "error": {
                    "message": e,
                    "type": "stream_error"
                }
            });
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_event))));
        }

        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    build_sse_response(rx)
}

async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    let mut accumulator = StreamingResponseAccumulator::new(&original_request);

    let response_id = format!("resp_{}", Uuid::new_v4());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {}", e))?;

        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        // SSE wire format: each chunk is "data: <json>\n\n" or "data: <json>".
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();

            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                    accumulator.process_chunk(&chat_chunk);
                    event_emitter.process_chunk(&chat_chunk, &tx)?;
                }
                Err(_) => {
                    // Not a chat chunk (error event, keep-alive, …): pass through unchanged.
                    debug!("Non-chunk SSE event, passing through: {}", event);
                    if tx.send(Ok(Bytes::from(format!("{}\n\n", event)))).is_err() {
                        return Err("Client disconnected".to_string());
                    }
                }
            }
        }
    }

    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    let completed_event = event_emitter.emit_completed(usage_json.as_ref());
    event_emitter.send_event(&completed_event, &tx)?;

    let final_response = accumulator.finalize();
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
    )
    .await;

    Ok(())
}

struct StreamingResponseAccumulator {
    response_id: String,
    model: String,
    created_at: i64,

    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,

    finish_reason: Option<String>,
    usage: Option<Usage>,

    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(original_request: &ResponsesRequest) -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            created_at: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if self.response_id.is_empty() {
            self.response_id = chunk.id.clone();
            self.model = chunk.model.clone();
            self.created_at = chunk.created as i64;
        }

        // Responses API does not support n>1; only the first choice is processed.
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                self.content_buffer.push_str(content);
            }

            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    // delta.index is a u32 here (not Option<u32> as in some chat APIs).
                    let index = delta.index as usize;

                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: String::new(),
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    // Update the tool call at this index
                    if let ResponseOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            id.push_str(delta_id);
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            });
        }

        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", self.response_id),
                summary: vec![],
                content: vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
                status: Some("completed".to_string()),
            });
        }

        output.extend(self.tool_calls);

        let status = match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::Completed,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        };

        let usage = self.usage.as_ref().map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
    }
}
