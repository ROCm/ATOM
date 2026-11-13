use serde_json::Value;

use crate::observability::ttft::SseFrames;

/// Best-effort usage observation; never retains an unbounded response or changes
/// bytes. Missing/oversized usage is left unknown rather than inferred.
#[derive(Default)]
pub(crate) struct UsageObserver {
    frames: SseFrames,
    json: Vec<u8>,
    disabled: bool,
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Usage {
    pub prompt: u64,
    pub completion: u64,
    /// Prompt tokens the engine reported as served from its prefix cache.
    pub cached: Option<u64>,
}

impl UsageObserver {
    pub fn feed(&mut self, bytes: &[u8], streaming: bool) {
        if self.disabled {
            return;
        }
        if streaming {
            self.frames.append(bytes);
            while let Some(frame) = self.frames.next_frame() {
                let Some(data) = crate::observability::ttft::sse_data(frame) else {
                    continue;
                };
                if let Some(usage) = Self::parse(data.as_bytes(), self.usage) {
                    self.usage = Some(usage);
                }
            }
            if self.frames.exceeded_limit() {
                self.disabled = true;
                self.frames = SseFrames::default();
            }
        } else if bytes.len() <= SseFrames::MAX_FRAME_BYTES.saturating_sub(self.json.len()) {
            self.json.extend_from_slice(bytes);
        } else {
            self.disabled = true;
            self.json.clear();
        }
    }

    fn parse(bytes: &[u8], previous: Option<Usage>) -> Option<Usage> {
        let payload: Value = serde_json::from_slice(bytes).ok()?;
        let usage = payload
            .get("usage")
            .or_else(|| payload.get("meta_info"))
            .or_else(|| payload["response"].get("usage"))
            .or_else(|| payload["message"].get("usage"))?;
        let prompt = usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                // Anthropic reports cached prompt tokens separately from input_tokens;
                // Responses' cached tokens are already included in input_tokens.
                let input = usage.get("input_tokens")?.as_u64()?;
                let cached = ["cache_read_input_tokens", "cache_creation_input_tokens"]
                    .iter()
                    .filter_map(|key| usage.get(*key).and_then(Value::as_u64))
                    .sum::<u64>();
                Some(input + cached)
            });
        let completion = usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64);
        let cached = usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| usage.pointer("/input_tokens_details/cached_tokens"))
            .or_else(|| usage.get("cache_read_input_tokens"))
            .and_then(Value::as_u64);
        // Messages publishes input usage at message_start and cumulative output at message_delta.
        Some(Usage {
            prompt: prompt.or_else(|| previous.map(|p| p.prompt))?,
            completion: completion.or_else(|| previous.map(|p| p.completion))?,
            cached: cached.or_else(|| previous.and_then(|p| p.cached)),
        })
    }

    pub fn usage(&self, streaming: bool) -> Option<Usage> {
        if streaming {
            self.usage
        } else if !self.disabled {
            Self::parse(&self.json, None)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u64, completion: u64, cached: Option<u64>) -> Option<Usage> {
        Some(Usage {
            prompt,
            completion,
            cached,
        })
    }

    #[test]
    fn usage_survives_arbitrary_frame_boundaries_and_buffer_is_bounded() {
        let mut observer = UsageObserver::default();
        let bytes = b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":8}}}\r\n\r\ndata: [DONE]\n\n";
        for byte in bytes {
            observer.feed(&[*byte], true);
        }
        assert_eq!(observer.usage, usage(12, 3, Some(8)));
        observer.feed(&vec![b'x'; SseFrames::MAX_FRAME_BYTES + 1], true);
        assert!(observer.disabled);
        let mut observer = UsageObserver::default();
        observer.feed(&vec![b'x'; SseFrames::MAX_FRAME_BYTES + 1], false);
        assert!(observer.disabled);
        assert!(observer.json.is_empty());
    }

    #[test]
    fn messages_merge_cumulative_usage_and_responses_read_nested_usage() {
        let messages = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",",
            "\"message\":{\"usage\":{\"input_tokens\":4,\"cache_read_input_tokens\":8,\"output_tokens\":0}}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",",
            "\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",",
            "\"usage\":{\"output_tokens\":5}}\n\n",
        );
        let responses = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":12,\"input_tokens_details\":{\"cached_tokens\":8},\"output_tokens\":5}}}\n\n";
        for bytes in [messages.as_bytes(), responses.as_bytes()] {
            for size in [1, 3, 17, 1024] {
                let mut observer = UsageObserver::default();
                for chunk in bytes.chunks(size) {
                    observer.feed(chunk, true);
                }
                assert_eq!(observer.usage(true), usage(12, 5, Some(8)));
            }
        }
        let mut observer = UsageObserver::default();
        observer.feed(br#"{"usage":{"input_tokens":12,"output_tokens":5}}"#, false);
        assert_eq!(observer.usage(false), usage(12, 5, None));
    }

    #[test]
    fn anthropic_cached_prompt_tokens_count_as_input() {
        let mut observer = UsageObserver::default();
        observer.feed(
            br#"{"usage":{"input_tokens":2,"cache_read_input_tokens":8,"cache_creation_input_tokens":4,"output_tokens":5}}"#,
            false,
        );
        assert_eq!(observer.usage(false), usage(14, 5, Some(8)));
        // Responses already include cached tokens in input_tokens.
        let mut observer = UsageObserver::default();
        observer.feed(
            br#"{"usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":8},"output_tokens":5}}"#,
            false,
        );
        assert_eq!(observer.usage(false), usage(12, 5, Some(8)));
    }
}
