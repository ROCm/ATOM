//! Validated routing views; callers retain raw request bodies.
use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{ContentPart, GenerationRequest, InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    validated::Normalizable,
};
use serde::{Serialize, Serializer};
use validator::Validate;

#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    pub route: &'static str,
    pub model: Option<String>,
    pub text: String,
    pub batch_size: Option<usize>,
    pub stream: bool,
    pub return_logprob: bool,
    /// See [`ParsedInference::media_count`]; set by `ParsedInference` only.
    pub media_count: usize,
}

impl InferenceMetadata {
    pub fn carries_media(&self) -> bool {
        self.media_count > 0
    }

    /// Execution needs only protocol facts; routing text can be released after placement.
    pub fn execution_metadata(&self) -> Self {
        Self {
            route: self.route,
            model: self.model.clone(),
            text: String::new(),
            batch_size: self.batch_size,
            stream: self.stream,
            return_logprob: self.return_logprob,
            media_count: self.media_count,
        }
    }
}

pub trait InferenceRequest {
    fn metadata(&self) -> InferenceMetadata;
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let mut metadata = self.metadata();
        if !include_text {
            metadata.text.clear();
        }
        metadata
    }
}
impl InferenceRequest for ChatCompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        InferenceMetadata {
            route: "/v1/chat/completions",
            model: Some(self.model.clone()),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size: self.n.filter(|n| *n > 1).map(|n| n as usize),
            stream: self.is_stream(),
            return_logprob: self.logprobs,
            media_count: 0,
        }
    }
}
/// The routing copy of a chat with media, which token routing templates.
pub(crate) struct RoutingChat {
    /// The chat's messages with digests in place of their media; the template
    /// takes the rest of the chat from the client's request.
    pub messages: Vec<ChatMessage>,
    /// For ATOM backends: one digest of the chat up to and including its last
    /// media part, with whether each image's `detail` is `"low"`, the one value
    /// that changes its pixels; it goes before the routing text and before the
    /// templated prompt. ATOM keys every KV block of a media prompt by its
    /// media's pixels and token positions, so two prompts that differ only in an
    /// image, in whether its `detail` is low or in the text before an image share
    /// no block, however long their shared text; a later turn after the same
    /// media keeps the digest. DeepSeek-V4.1 also keys every block by each
    /// prompt token's type, so its prompts that differ in length after the last
    /// media share no block though the digest is one; nor does the digest see a
    /// URL whose image changes between requests. `None` for vLLM and SGLang,
    /// whose prefix caches key a block by the media it covers, as the in-place
    /// digests do.
    pub front: Option<String>,
}

/// A message's content; `None` for a function message and an assistant without one.
fn content(message: &ChatMessage) -> Option<&MessageContent> {
    match message {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. } => Some(content),
        ChatMessage::Assistant { content, .. } => content.as_ref(),
        ChatMessage::Function { .. } => None,
    }
}

/// A routing copy of `request`'s messages with each image and video replaced in
/// place by a text digest of its URL. Routing text and template tokens
/// otherwise drop media or render it as an identity-free placeholder, so
/// requests differing only in media would share a prefix the engine's
/// media-keyed prefix cache never shares.
fn with_media_digests(request: &ChatCompletionRequest) -> Vec<ChatMessage> {
    request.messages.iter().map(digested).collect()
}

/// A copy of `message` with digests in place of its media, which it never copies.
fn digested(message: &ChatMessage) -> ChatMessage {
    let digest = |content: &MessageContent| match content {
        MessageContent::Parts(parts) => MessageContent::Parts(
            parts
                .iter()
                .map(|part| match media_url(part) {
                    Some((kind, url)) => ContentPart::Text {
                        text: media_digest(kind, url),
                    },
                    None => part.clone(),
                })
                .collect(),
        ),
        MessageContent::Text(_) => content.clone(),
    };
    match message {
        ChatMessage::System { content, name } => ChatMessage::System {
            content: digest(content),
            name: name.clone(),
        },
        ChatMessage::User { content, name } => ChatMessage::User {
            content: digest(content),
            name: name.clone(),
        },
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => ChatMessage::Tool {
            content: digest(content),
            tool_call_id: tool_call_id.clone(),
        },
        ChatMessage::Developer {
            content,
            tools,
            name,
        } => ChatMessage::Developer {
            content: digest(content),
            tools: tools.clone(),
            name: name.clone(),
        },
        ChatMessage::Assistant {
            content,
            name,
            tool_calls,
            reasoning_content,
        } => ChatMessage::Assistant {
            content: content.as_ref().map(digest),
            name: name.clone(),
            tool_calls: tool_calls.clone(),
            reasoning_content: reasoning_content.clone(),
        },
        ChatMessage::Function { .. } => message.clone(),
    }
}

/// A message's content parts; none for text content or no content.
fn parts(message: &ChatMessage) -> &[ContentPart] {
    match content(message) {
        Some(MessageContent::Parts(parts)) => parts,
        _ => &[],
    }
}

/// The kind and URL of each image and video part of a chat's messages.
fn media_parts(request: &ChatCompletionRequest) -> impl Iterator<Item = (&'static str, &str)> {
    request
        .messages
        .iter()
        .flat_map(parts)
        .filter_map(media_url)
}

/// A media part's kind and URL; `None` for text.
fn media_url(part: &ContentPart) -> Option<(&'static str, &str)> {
    match part {
        ContentPart::ImageUrl { image_url } => Some(("image", &image_url.url)),
        ContentPart::VideoUrl { video_url } => Some(("video", &video_url.url)),
        ContentPart::Text { .. } => None,
    }
}

/// `messages` as sent, a chat body's JSON, with the digests
/// [`with_media_digests`] puts in place of their media: for the encoders that
/// read the body as sent (`chat_render`).
pub(crate) fn digest_sent_media(messages: &mut serde_json::Value) {
    let parts = messages
        .as_array_mut()
        .into_iter()
        .flatten()
        .filter_map(|message| message.get_mut("content")?.as_array_mut())
        .flatten();
    for part in parts {
        let kind = match part.get("type").and_then(serde_json::Value::as_str) {
            Some("image_url") => "image",
            Some("video_url") => "video",
            _ => continue,
        };
        let url = part.pointer(&format!("/{kind}_url/url"));
        let Some(url) = url.and_then(serde_json::Value::as_str) else {
            continue;
        };
        *part = serde_json::json!({"type": "text", "text": media_digest(kind, url)});
    }
}

/// See [`RoutingChat::front`]: one digest of `digested`, `request`'s messages
/// with their media digested in place, up to and including the last media part,
/// and of whether each image's `detail` is `"low"`, which the in-place digests
/// leave out.
fn front_digest(request: &ChatCompletionRequest, digested: &[ChatMessage]) -> String {
    let (message, part) = request
        .messages
        .iter()
        .enumerate()
        .flat_map(|(m, message)| {
            let media = parts(message).iter().enumerate();
            media.filter_map(move |(p, part)| media_url(part).map(|_| (m, p)))
        })
        .last()
        .expect("a front digest is for a chat with media");
    let mut front = serde_json::to_value(&digested[..=message]).expect("messages serialize");
    front[message]["content"]
        .as_array_mut()
        .expect("media is a content part")
        .truncate(part + 1);
    // Only `detail: "low"` changes an image's pixels; any other value leaves them.
    let low_details: Vec<bool> = request
        .messages
        .iter()
        .flat_map(parts)
        .filter_map(|part| match part {
            ContentPart::ImageUrl { image_url } => Some(image_url.detail.as_deref() == Some("low")),
            _ => None,
        })
        .collect();
    let bytes = serde_json::to_vec(&(front, low_details)).expect("JSON values serialize");
    format!(
        "\u{1}media:{:016x}\u{1}",
        xxhash_rust::xxh3::xxh3_64(&bytes)
    )
}

/// A digest marks its media's place in the routing text. Unlike a template's
/// placeholder it differs by URL; client text that spells the same bytes
/// would match it, which only merges routing keys, never request content.
fn media_digest(kind: &str, url: &str) -> String {
    format!(
        "\u{1}{kind}:{:016x}\u{1}",
        xxhash_rust::xxh3::xxh3_64(url.as_bytes())
    )
}

impl InferenceRequest for CompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let batch_size = match &self.prompt {
            StringOrArray::Array(values) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/v1/completions",
            model: Some(self.model.clone()),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size,
            stream: self.is_stream(),
            return_logprob: self.logprobs.is_some(),
            media_count: 0,
        }
    }
}
impl InferenceRequest for GenerateRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let batch_size = match &self.input_ids {
            Some(InputIds::Batch(values)) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/generate",
            model: self.model.clone(),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size,
            stream: self.is_stream(),
            return_logprob: self.return_logprob.unwrap_or(false),
            media_count: 0,
        }
    }
}

pub enum ParsedInference {
    Chat(ChatCompletionRequest),
    Completion(CompletionRequest),
    Generate(GenerateRequest),
    Messages(serde_json::Value),
    Responses(serde_json::Value),
}
impl ParsedInference {
    /// Every ingress parses here, so an empty model is rejected for all routers.
    pub fn parse(path: &str, bytes: &[u8]) -> Result<Self, String> {
        let parsed = Self::parse_body(path, bytes)?;
        // Reads the model only, so the media are left uncounted.
        if parsed
            .metadata_of(0, false)
            .model
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err("model is required".into());
        }
        Ok(parsed)
    }

    fn parse_body(path: &str, bytes: &[u8]) -> Result<Self, String> {
        use crate::routers::ingress::{Api, EndpointSpec};
        let endpoint = EndpointSpec::find(path).ok_or("unsupported inference API")?;
        match endpoint.api {
            Api::Chat => {
                let mut request: ChatCompletionRequest =
                    serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                request.normalize();
                request.validate().map_err(|e| e.to_string())?;
                Ok(Self::Chat(request))
            }
            Api::Completions => serde_json::from_slice(bytes)
                .map(Self::Completion)
                .map_err(|e| e.to_string()),
            Api::Generate => serde_json::from_slice(bytes)
                .map(Self::Generate)
                .map_err(|e| e.to_string()),
            Api::Messages | Api::Responses => {
                let body: serde_json::Value =
                    serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                Self::validate_api_body(&body, endpoint.api == Api::Messages)?;
                Ok(if endpoint.api == Api::Messages {
                    Self::Messages(body)
                } else {
                    Self::Responses(body)
                })
            }
        }
    }
    pub fn requires_state_domain(&self) -> bool {
        let Self::Responses(body) = self else {
            return false;
        };
        ["previous_response_id", "conversation"]
            .iter()
            .any(|key| body.get(*key).is_some_and(|v| !v.is_null()))
            || body
                .get("input")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| items.iter().any(Self::is_item_reference))
    }

    /// `item_reference` items may omit `type`; an untyped item with an `id` and
    /// no message `role` refers to stored state.
    fn is_item_reference(item: &serde_json::Value) -> bool {
        use serde_json::Value;
        match item.get("type").and_then(Value::as_str) {
            Some(kind) => kind == "item_reference",
            None => {
                item.get("id").is_some_and(Self::is_nonempty_string) && item.get("role").is_none()
            }
        }
    }

    /// Validate only what routing reads: the body shape, `model`, `stream` and
    /// the Responses state keys. Everything else is the backend's to accept or
    /// reject, since backends extend these APIs (optional `max_tokens`,
    /// `top_k: -1`, other roles and sampling ranges).
    fn validate_api_body(body: &serde_json::Value, messages: bool) -> Result<(), String> {
        use serde_json::Value;
        if !body.is_object() {
            return Err("request must be a JSON object".into());
        }
        if !body.get("model").is_some_and(Self::is_nonempty_string) {
            return Err("model must be a nonempty string".into());
        }
        Self::validate_optional_field(body, "stream", Value::is_boolean, "a boolean")?;
        if messages {
            Ok(())
        } else {
            Self::validate_state_keys(body)
        }
    }

    /// The keys `requires_state_domain` reads to keep stateful requests on
    /// their owner.
    fn validate_state_keys(body: &serde_json::Value) -> Result<(), String> {
        use serde_json::Value;
        Self::validate_optional_field(
            body,
            "input",
            |v| {
                v.is_string()
                    || v.as_array()
                        .is_some_and(|items| items.iter().all(Value::is_object))
            },
            "text or an array of input items",
        )?;
        Self::validate_optional_field(
            body,
            "previous_response_id",
            Self::is_nonempty_string,
            "a nonempty string",
        )?;
        Self::validate_optional_field(
            body,
            "conversation",
            |v| Self::is_nonempty_string(v) || v.get("id").is_some_and(Self::is_nonempty_string),
            "a conversation ID or an object containing id",
        )?;
        if ["previous_response_id", "conversation"]
            .iter()
            .all(|key| body.get(*key).is_some_and(|v| !v.is_null()))
        {
            return Err("conversation and previous_response_id are mutually exclusive".into());
        }
        Ok(())
    }

    fn validate_optional_field(
        body: &serde_json::Value,
        key: &str,
        valid: impl FnOnce(&serde_json::Value) -> bool,
        expected: &str,
    ) -> Result<(), String> {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            if !valid(value) {
                return Err(format!("{key} must be {expected}"));
            }
        }
        Ok(())
    }

    fn is_nonempty_string(value: &serde_json::Value) -> bool {
        value.as_str().is_some_and(|s| !s.trim().is_empty())
    }

    pub fn input_tokens(&self) -> Result<Option<Vec<u32>>, &'static str> {
        match self {
            Self::Generate(GenerateRequest {
                input_ids: Some(InputIds::Single(ids)),
                ..
            }) => ids
                .iter()
                .map(|&id| u32::try_from(id))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
                .map_err(|_| "token routing requires nonnegative input_ids"),
            Self::Generate(GenerateRequest {
                input_ids: Some(InputIds::Batch(_)),
                ..
            }) => Err("token routing requires a single input_ids array"),
            _ => Ok(None),
        }
    }

    /// Images, video, audio or embeddings, which the request's tokens do not show but an
    /// engine keys KV blocks by (ATOM every block of the prompt, by the media and
    /// the text up to the last, DeepSeek-V4.1 also by the prompt's length; vLLM and
    /// SGLang the blocks from the media part on): a chat's image and video
    /// parts; a generate request's image, video and audio items (arrays,
    /// nested as a batch's are, count the non-null items they hold, so `[]`
    /// and `[[],[]]` count none), and its `input_embeds` as one.
    pub(crate) fn media_count(&self) -> usize {
        fn items(data: &serde_json::Value) -> usize {
            match data {
                serde_json::Value::Array(values) => values.iter().map(items).sum(),
                serde_json::Value::Null => 0,
                _ => 1,
            }
        }
        match self {
            Self::Chat(chat) => media_parts(chat).count(),
            Self::Generate(generate) => {
                let items = |data: &Option<serde_json::Value>| data.as_ref().map_or(0, items);
                items(&generate.image_data)
                    + items(&generate.video_data)
                    + items(&generate.audio_data)
                    + usize::from(generate.input_embeds.is_some())
            }
            Self::Completion(_) | Self::Messages(_) | Self::Responses(_) => 0,
        }
    }

    /// Routing metadata with its text and, for a chat with media, its
    /// [`RoutingChat`], built once for both. `media_count` is this request's
    /// [`Self::media_count`], counted once by the caller. With `with_front`
    /// (ATOM backends), the routing text starts with [`RoutingChat::front`].
    pub(crate) fn routing_metadata(
        &self,
        media_count: usize,
        with_front: bool,
    ) -> (InferenceMetadata, Option<RoutingChat>) {
        let Some(chat) = self.chat().filter(|_| media_count > 0) else {
            return (self.metadata_of(media_count, true), None);
        };
        let mut metadata = self.metadata_of(media_count, false);
        // The routing text reads only the messages.
        let messages_only = ChatCompletionRequest {
            messages: with_media_digests(chat),
            ..Default::default()
        };
        let front = with_front.then(|| front_digest(chat, &messages_only.messages));
        metadata.text =
            front.clone().unwrap_or_default() + &messages_only.extract_text_for_routing();
        let routed = RoutingChat {
            messages: messages_only.messages,
            front,
        };
        (metadata, Some(routed))
    }

    pub fn chat(&self) -> Option<&ChatCompletionRequest> {
        if let Self::Chat(chat) = self {
            Some(chat)
        } else {
            None
        }
    }
}

// Keep stable context before input without cloning the request.
struct RoutingView<'a> {
    body: &'a serde_json::Value,
    keys: &'static [&'static str],
}

impl Serialize for RoutingView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(
            self.keys
                .iter()
                .filter_map(|key| self.body.get(*key).map(|value| (*key, value))),
        )
    }
}

impl InferenceRequest for ParsedInference {
    /// The routing metadata vLLM and SGLang backends get.
    fn metadata(&self) -> InferenceMetadata {
        self.routing_metadata(self.media_count(), false).0
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        if include_text {
            self.metadata()
        } else {
            self.metadata_of(self.media_count(), false)
        }
    }
}

impl ParsedInference {
    fn metadata_of(&self, media_count: usize, include_text: bool) -> InferenceMetadata {
        let mut metadata = match self {
            Self::Chat(r) => r.metadata_with_text(include_text),
            Self::Completion(r) => r.metadata_with_text(include_text),
            Self::Generate(r) => r.metadata_with_text(include_text),
            Self::Messages(body) | Self::Responses(body) => {
                let messages = matches!(self, Self::Messages(_));
                let keys: &[&str] = if messages {
                    &["system", "tools", "tool_choice", "messages"]
                } else {
                    &[
                        "instructions",
                        "tools",
                        "tool_choice",
                        "input",
                        "previous_response_id",
                        "conversation",
                    ]
                };
                let text = if include_text {
                    let view = RoutingView { body, keys };
                    serde_json::to_string(&view).expect("JSON values are serializable")
                } else {
                    String::new()
                };
                InferenceMetadata {
                    route: if messages {
                        "/v1/messages"
                    } else {
                        "/v1/responses"
                    },
                    model: body["model"].as_str().map(str::to_owned),
                    text,
                    batch_size: None,
                    stream: body["stream"].as_bool().unwrap_or(false),
                    return_logprob: false,
                    media_count: 0,
                }
            }
        };
        metadata.media_count = media_count;
        metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_facts_follow_typed_null_and_batch_semantics_without_retaining_text() {
        for (route, body, batch, stream, logprobs) in [
            (
                "/v1/chat/completions",
                r#"{"model":"m","messages":[{"role":"user","content":"large prompt"}],"n":4,"stream":true,"logprobs":true}"#,
                Some(4),
                true,
                true,
            ),
            (
                "/v1/completions",
                r#"{"model":"m","prompt":["a","b"],"logprobs":0}"#,
                Some(2),
                false,
                true,
            ),
            (
                "/v1/completions",
                r#"{"model":"m","prompt":[],"logprobs":null}"#,
                None,
                false,
                false,
            ),
            (
                "/generate",
                r#"{"model":"m","input_ids":[[1,2],[3,4]],"return_logprob":null}"#,
                Some(2),
                false,
                false,
            ),
        ] {
            let parsed = ParsedInference::parse(route, body.as_bytes()).unwrap();
            let metadata = parsed.metadata();
            assert_eq!(
                (
                    metadata.batch_size,
                    metadata.stream,
                    metadata.return_logprob
                ),
                (batch, stream, logprobs)
            );
            let execution = metadata.execution_metadata();
            assert!(execution.text.is_empty());
            assert_eq!(execution.model.as_deref(), Some("m"));
            assert_eq!(
                (
                    execution.route,
                    execution.batch_size,
                    execution.stream,
                    execution.return_logprob
                ),
                (route, batch, stream, logprobs)
            );
        }
    }

    fn chat_routing_text(body: &str) -> String {
        ParsedInference::parse("/v1/chat/completions", body.as_bytes())
            .unwrap()
            .metadata()
            .text
    }

    fn image_chat(url: &str) -> String {
        format!(
            r#"{{"model":"m","messages":[{{"role":"system","content":"sys"}},{{"role":"user","content":[{{"type":"text","text":"describe"}},{{"type":"image_url","image_url":{{"url":"{url}"}}}},{{"type":"text","text":"briefly"}}]}}]}}"#
        )
    }

    #[test]
    fn text_only_chat_routing_text_is_unchanged() {
        let body = r#"{"model":"m","messages":[{"role":"system","content":"sys"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},{"role":"assistant","content":"c","reasoning_content":"r"}]}"#;
        let request: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(chat_routing_text(body), request.extract_text_for_routing());
    }

    #[test]
    fn chat_routing_text_distinguishes_media_in_place() {
        let first = chat_routing_text(&image_chat("data:image/png;base64,AAAA"));
        let second = chat_routing_text(&image_chat("data:image/png;base64,BBBB"));
        assert_eq!(
            first,
            chat_routing_text(&image_chat("data:image/png;base64,AAAA"))
        );
        assert_ne!(first, second);
        assert!(!first.contains("AAAA"));
        let shared = first
            .chars()
            .zip(second.chars())
            .take_while(|(a, b)| a == b)
            .count();
        let before_digest = "sys describe \u{1}image:".len();
        assert!(first.starts_with("sys describe \u{1}image:"));
        assert!((before_digest..before_digest + 16).contains(&shared));
        assert!(first.ends_with("\u{1} briefly"));
    }

    #[test]
    fn video_digest_differs_from_image_digest_of_the_same_url() {
        let image = image_chat("https://example.com/m");
        let video = image.replace(
            r#""type":"image_url","image_url""#,
            r#""type":"video_url","video_url""#,
        );
        assert_ne!(chat_routing_text(&image), chat_routing_text(&video));
    }

    /// The digested copy is the chat's messages with each media part's text
    /// digest in its place, every other field of every role kept.
    #[test]
    fn media_digests_keep_every_message_field_but_the_media() {
        let image = |url: &str| serde_json::json!({"type": "image_url", "image_url": {"url": url, "detail": "high"}});
        let messages = serde_json::json!([
            {"role": "system", "content": "sys", "name": "s"},
            {"role": "developer", "name": "d", "content": [{"type": "text", "text": "dev"}],
             "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
            {"role": "user", "name": "u", "content": [{"type": "text", "text": "look"}, image("https://x/a.png")]},
            {"role": "assistant", "name": "a", "reasoning_content": "r",
             "content": [{"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}],
             "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": [image("https://x/b.png")]},
            {"role": "function", "name": "f", "content": "out"},
        ]);
        let mut expected = messages.clone();
        for message in expected.as_array_mut().unwrap() {
            let Some(parts) = message["content"].as_array_mut() else {
                continue;
            };
            for part in parts {
                for kind in ["image", "video"] {
                    if let Some(url) = part[format!("{kind}_url")]["url"].as_str() {
                        let text = media_digest(kind, url);
                        *part = serde_json::json!({"type": "text", "text": text});
                    }
                }
            }
        }
        let chat = |messages| -> ChatCompletionRequest {
            serde_json::from_value(serde_json::json!({"model": "m", "messages": messages})).unwrap()
        };
        let digested = with_media_digests(&chat(messages));
        assert_eq!(
            serde_json::to_value(digested).unwrap(),
            serde_json::to_value(chat(expected).messages).unwrap()
        );
    }

    #[test]
    fn messages_and_responses_leave_backend_fields_to_the_backend() {
        for (route, body) in [
            // No max_tokens, a system role, top_k -1 and backend sampling ranges.
            (
                "/v1/messages",
                r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"tool","content":42}],"temperature":1.5,"top_k":-1,"top_p":2}"#,
            ),
            ("/v1/messages", r#"{"model":"m"}"#),
            (
                "/v1/responses",
                r#"{"model":"m","input":[{"role":"bad","content":42}],"temperature":3,"max_output_tokens":0,"background":"yes"}"#,
            ),
        ] {
            assert!(
                ParsedInference::parse(route, body.as_bytes()).is_ok(),
                "{body}"
            );
        }
        for (route, body) in [
            ("/v1/messages", r#"[]"#),
            ("/v1/messages", r#"{"model":" "}"#),
            ("/v1/messages", r#"{"model":"m","stream":"yes"}"#),
            ("/v1/responses", r#"{"model":"m","input":[42]}"#),
            ("/v1/responses", r#"{"model":"m","previous_response_id":7}"#),
            (
                "/v1/responses",
                r#"{"model":"m","previous_response_id":"r","conversation":"c"}"#,
            ),
        ] {
            assert!(
                ParsedInference::parse(route, body.as_bytes()).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn media_is_told_from_text() {
        let chat = |role: &str, part: &str| {
            let body =
                format!(r#"{{"model":"m","messages":[{{"role":"{role}","content":[{part}]}}]}}"#);
            ("/v1/chat/completions", body)
        };
        let text = r#"{"type":"text","text":"hi"}"#;
        let image = r#"{"type":"image_url","image_url":{"url":"https://x/cat.png"}}"#;
        let generate = |media: &str| ("/generate", format!(r#"{{"input_ids":[1,2]{media}}}"#));
        for ((route, body), media) in [
            (chat("user", text), false),
            (chat("user", image), true),
            (chat("system", image), true),
            (generate(""), false),
            (generate(r#","image_data":"https://x/cat.png""#), true),
            (generate(r#","video_data":"https://x/cat.mp4""#), true),
            (generate(r#","audio_data":"https://x/cat.wav""#), true),
            (generate(r#","input_embeds":[[0.1,0.2]]"#), true),
        ] {
            let parsed = ParsedInference::parse(route, body.as_bytes()).unwrap();
            assert_eq!(parsed.media_count() > 0, media, "{body}");
            assert_eq!(parsed.metadata_with_text(false).carries_media(), media);
        }
    }

    #[test]
    fn media_count_counts_items() {
        let image = r#"{"type":"image_url","image_url":{"url":"https://x/cat.png"}}"#;
        let chat = format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":[{image},{{"type":"text","text":"hi"}},{image}]}}]}}"#
        );
        let generate = |media: &str| format!(r#"{{"input_ids":[1,2]{media}}}"#);
        for ((route, body), count) in [
            (("/v1/chat/completions", chat), 2),
            (
                ("/generate", generate(r#","image_data":["a.png","b.png"]"#)),
                2,
            ),
            (
                (
                    "/generate",
                    generate(r#","image_data":"a.png","audio_data":"a.wav""#),
                ),
                2,
            ),
            (
                ("/generate", generate(r#","input_embeds":[[0.1],[0.2]]"#)),
                1,
            ),
        ] {
            let metadata = ParsedInference::parse(route, body.as_bytes())
                .unwrap()
                .metadata_with_text(false);
            assert_eq!(metadata.media_count, count, "{body}");
        }
    }
}
