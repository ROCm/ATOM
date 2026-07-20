//! Request events for observability and monitoring.

use tracing::debug;

pub trait Event {
    fn emit(&self);
}

/// Event emitted when a prefill-decode request pair is sent.
#[derive(Debug, Clone, Copy)]
pub struct RequestPDSentEvent<'a> {
    pub prefill_url: &'a str,
    pub decode_url: &'a str,
}

impl Event for RequestPDSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        debug!(
            prefill_url = %self.prefill_url,
            decode_url = %self.decode_url,
            "Sending concurrent requests"
        );
    }
}

/// Event emitted when a request is sent to a worker.
#[derive(Debug, Clone, Copy)]
pub struct RequestSentEvent<'a> {
    pub url: &'a str,
}

impl Event for RequestSentEvent<'_> {
    #[inline]
    fn emit(&self) {
        debug!(url = %self.url, "Sending request");
    }
}

/// Event emitted when concurrent requests are received.
#[derive(Debug, Clone, Copy)]
pub struct RequestReceivedEvent;

impl Event for RequestReceivedEvent {
    #[inline]
    fn emit(&self) {
        debug!("Received concurrent requests");
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn test_event_sizes() {
        assert_eq!(size_of::<RequestReceivedEvent>(), 0);
        assert_eq!(size_of::<RequestSentEvent>(), 16);
        assert_eq!(size_of::<RequestPDSentEvent>(), 32);
    }

    #[test]
    fn test_request_pd_sent_event_debug() {
        let event = RequestPDSentEvent {
            prefill_url: "http://p:8000",
            decode_url: "http://d:8000",
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("http://p:8000"));
        assert!(debug.contains("http://d:8000"));
    }

    #[test]
    fn test_request_sent_event_debug() {
        let event = RequestSentEvent {
            url: "http://worker:8000",
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("http://worker:8000"));
    }

    #[test]
    fn test_request_received_event_debug() {
        let event = RequestReceivedEvent;
        let debug = format!("{:?}", event);
        assert!(debug.contains("RequestReceivedEvent"));
    }

    #[test]
    fn test_event_emit_does_not_panic() {
        RequestPDSentEvent {
            prefill_url: "http://p:8000",
            decode_url: "http://d:8000",
        }
        .emit();

        RequestSentEvent {
            url: "http://w:8000",
        }
        .emit();

        RequestReceivedEvent.emit();
    }

    #[test]
    fn test_request_pd_sent_event_clone() {
        let event = RequestPDSentEvent {
            prefill_url: "http://p:8000",
            decode_url: "http://d:8000",
        };
        let cloned = event;
        assert_eq!(cloned.prefill_url, "http://p:8000");
        assert_eq!(cloned.decode_url, "http://d:8000");
    }
}
