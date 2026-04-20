//! Streaming infrastructure for /v1/responses endpoint

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::protocols::{
    chat::ChatCompletionStreamResponse,
    event_types::{
        ContentPartEvent, FunctionCallEvent, OutputItemEvent, OutputTextEvent, ResponseEvent,
    },
    responses::ResponsesRequest,
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
    // Output item tracking
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>, // Tracks output_index of current message
    current_item_id: Option<String>,             // Tracks item_id of current item
    original_request: Option<ResponsesRequest>,
    // Tool call tracking: maps tool_call delta index → (output_index, item_id, accumulated_args)
    tool_call_items: Vec<(usize, String, String, String)>, // (output_index, item_id, name, accumulated_args)
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

    /// Set the original request for including all fields in response.completed
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
        // Build output array from tracked items
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

        // If no items were tracked (legacy path), fall back to generic message
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

        // Build base response object
        let mut response_obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output
        });

        // Add usage if provided
        if let Some(usage_val) = usage {
            response_obj["usage"] = usage_val.clone();
        }

        // Add all original request fields if available
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

            // tool_choice: serialize if present, otherwise use "auto"
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

    /// Helper to add optional fields to JSON object
    fn add_optional_field<T: serde::Serialize>(
        obj: &mut serde_json::Value,
        key: &str,
        value: &Option<T>,
    ) {
        if let Some(val) = value {
            obj[key] = json!(val);
        }
    }

    // ========================================================================
    // Function Call Event Emission Methods
    // ========================================================================

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

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Emit response.output_item.added event
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

    /// Emit response.output_item.done event
    pub fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        // Store the item data for later use in emit_completed
        self.store_output_item_data(output_index, item.clone());

        json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::new_v4().to_string().replace("-", ""))
    }

    /// Allocate next output index and track item
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

    /// Mark output item as completed and store its data
    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Store output item data when emitting output_item.done
    pub fn store_output_item_data(&mut self, output_index: usize, item_data: serde_json::Value) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.item_data = Some(item_data);
        }
    }

    /// Process a chunk and emit appropriate events
    pub fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::Message);

                        // Build message item structure
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        // Emit output_item.added
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

                    // Emit text delta
                    let event =
                        self.emit_text_delta(content, output_index, &item_id, content_index);
                    self.send_event(&event, tx)?;
                }
            }

            // Process tool call deltas
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let tc_index = delta.index as usize;

                    // Ensure we have a tracked item for this tool call index
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

                    // Accumulate name from first delta
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            self.tool_call_items[tc_index].2.push_str(name);
                        }
                    }

                    // First delta for this tool call: emit output_item.added
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

                    // Emit arguments delta
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

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                if reason == "tool_calls" {
                    // Emit done events for all tracked tool calls
                    let tool_calls: Vec<_> = self.tool_call_items.clone();
                    for (output_index, item_id, tc_name, accumulated_args) in &tool_calls {
                        // Emit function_call_arguments.done
                        let event = self.emit_function_call_arguments_done(
                            *output_index,
                            item_id,
                            accumulated_args,
                        );
                        self.send_event(&event, tx)?;

                        // Emit output_item.done with complete function call item
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
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0;

                    // Emit closing events
                    if self.has_emitted_content_part_added {
                        let event = self.emit_text_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        let event =
                            self.emit_content_part_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                    }

                    if self.has_emitted_output_item_added {
                        // Build complete message item for output_item.done
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

                    // Mark item as completed
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

        // Extract event type from the JSON for SSE event field
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        // Format as SSE with event: field
        let sse_message = format!("event: {}\ndata: {}\n\n", event_type, event_json);

        if tx.send(Ok(Bytes::from(sse_message))).is_err() {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }
}

/// Build a Server-Sent Events (SSE) response
///
/// Creates a Response with proper SSE headers and streaming body.
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
