use serde_json::Value;

use crate::observability::ttft::SseFrames;

/// Best-effort usage observation; never retains an unbounded response or changes
/// bytes. Missing/oversized usage is left unknown rather than inferred.
#[derive(Default)]
pub(super) struct UsageObserver {
    frames: SseFrames,
    json: Vec<u8>,
    disabled: bool,
    usage: Option<(u64, u64)>,
}

impl UsageObserver {
    pub fn feed(&mut self, bytes: &[u8], streaming: bool) {
        if self.disabled {
            return;
        }
        if streaming {
            self.frames.append(bytes);
            while let Some(frame) = self.frames.next_frame() {
                let Ok(frame) = std::str::from_utf8(frame) else {
                    continue;
                };
                let data = frame
                    .lines()
                    .filter_map(|line| {
                        line.strip_prefix("data:")
                            .map(|line| line.trim_start_matches(' '))
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if let Some(usage) = Self::parse(data.as_bytes()) {
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

    fn parse(bytes: &[u8]) -> Option<(u64, u64)> {
        let payload: Value = serde_json::from_slice(bytes).ok()?;
        let usage = payload.get("usage").or_else(|| payload.get("meta_info"))?;
        Some((
            usage.get("prompt_tokens")?.as_u64()?,
            usage.get("completion_tokens")?.as_u64()?,
        ))
    }

    pub fn record(&self, streaming: bool) {
        let usage = if streaming {
            self.usage
        } else if !self.disabled {
            Self::parse(&self.json)
        } else {
            None
        };
        if let Some((prompt, completion)) = usage {
            metrics::counter!("mesh_ext_proc_tokens_total", "kind" => "prompt").increment(prompt);
            metrics::counter!("mesh_ext_proc_tokens_total", "kind" => "completion")
                .increment(completion);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_survives_arbitrary_frame_boundaries_and_buffer_is_bounded() {
        let mut observer = UsageObserver::default();
        let bytes = b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\r\n\r\ndata: [DONE]\n\n";
        for byte in bytes {
            observer.feed(&[*byte], true);
        }
        assert_eq!(observer.usage, Some((12, 3)));
        observer.feed(&vec![b'x'; SseFrames::MAX_FRAME_BYTES + 1], true);
        assert!(observer.disabled);
        let mut observer = UsageObserver::default();
        observer.feed(&vec![b'x'; SseFrames::MAX_FRAME_BYTES + 1], false);
        assert!(observer.disabled);
        assert!(observer.json.is_empty());
    }
}
