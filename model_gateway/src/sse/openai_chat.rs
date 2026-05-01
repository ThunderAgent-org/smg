//! OpenAI Chat Completions (`/v1/chat/completions`) SSE parser.
//!
//! Stream shape:
//! ```text
//! data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
//!
//! data: {"id":"...","choices":[{"delta":{"content":" world"}}]}
//!
//! data: {"id":"...","choices":[],"usage":{"total_tokens":12,"prompt_tokens":3,"completion_tokens":9}}
//!
//! data: [DONE]
//!
//! ```
//!
//! - Token delta: each `data:` event with non-empty `delta.content` → +1 (Python heuristic)
//! - Usage extraction: the chunk where `choices == []` AND `usage` is present
//! - Strip target: that exact usage chunk's bytes (when client didn't ask for usage)

use serde_json::Value;

use crate::sse::extractor::{extract_data_payload, EventOutcome, ParsedUsage};

#[derive(Debug, Default)]
pub struct OpenAiChatState {
    /// Whether this stream's usage has already been emitted (idempotency).
    usage_emitted: bool,
}

pub(crate) fn process_event(
    state: &mut OpenAiChatState,
    event_bytes: &[u8],
    strip_usage_chunk: bool,
) -> EventOutcome {
    let mut outcome = EventOutcome::default();
    let mut event_has_usage = false;

    // Parse event into lines; identify usage-bearing chunk; emit token delta;
    // potentially strip the entire event from `forward`.
    for line in event_bytes.split(|b| *b == b'\n') {
        let Some(payload) = extract_data_payload(line) else {
            continue;
        };
        if payload == b"[DONE]" || payload.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_slice::<Value>(payload) else {
            continue;
        };

        // Token delta: count if delta.content is non-empty.
        if has_content_delta(&val) {
            outcome.token_delta = outcome.token_delta.saturating_add(1);
        }

        // Usage extraction: chunk with `usage` field (canonical shape: choices=[] + usage).
        if !state.usage_emitted {
            if let Some(parsed) = parse_usage(&val) {
                state.usage_emitted = true;
                outcome.usage = Some(parsed);
                event_has_usage = true;
            }
        }
    }

    // Forward decision: if this event carried the usage chunk AND we're in strip
    // mode, drop the entire event from the forwarded byte stream. Otherwise keep.
    if !(event_has_usage && strip_usage_chunk) {
        outcome.forward.extend_from_slice(event_bytes);
    }

    outcome
}

fn has_content_delta(val: &Value) -> bool {
    val.get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}

fn parse_usage(val: &Value) -> Option<ParsedUsage> {
    let usage = val.get("usage")?.as_object()?;
    let total = usage.get("total_tokens").and_then(Value::as_u64)?;
    Some(ParsedUsage {
        total_tokens: total,
        prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        completion_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
        cached_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|p| p.get("cached_tokens"))
            .and_then(Value::as_u64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::extractor::{SseExtractor, SseProtocol};

    fn build_event(json: &str) -> Vec<u8> {
        format!("data: {json}\n\n").into_bytes()
    }

    fn full_stream(strip: bool) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&build_event(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#));
        s.extend_from_slice(&build_event(r#"{"choices":[{"delta":{"content":" world"}}]}"#));
        s.extend_from_slice(&build_event(
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ));
        s.extend_from_slice(&build_event(
            r#"{"choices":[],"usage":{"total_tokens":12,"prompt_tokens":3,"completion_tokens":9}}"#,
        ));
        s.extend_from_slice(b"data: [DONE]\n\n");
        let _ = strip;
        s
    }

    #[test]
    fn full_stream_oneshot_extracts_usage() {
        let stream = full_stream(false);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        let parsed = e.feed(&stream);
        assert_eq!(
            parsed.usage,
            Some(ParsedUsage {
                total_tokens: 12,
                prompt_tokens: Some(3),
                completion_tokens: Some(9),
                cached_tokens: None,
            })
        );
    }

    #[test]
    fn token_delta_counts_content_events() {
        let stream = full_stream(false);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        let parsed = e.feed(&stream);
        // 2 events with content delta ("Hello", " world"); finish_reason and usage events have no content.
        assert_eq!(parsed.token_delta, 2);
    }

    #[test]
    fn partial_chunks_across_event_boundary() {
        let stream = full_stream(false);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        // Split midway through the second event.
        let mid = stream.len() / 2;
        let p1 = e.feed(&stream[..mid]);
        let p2 = e.feed(&stream[mid..]);
        let combined_forward = [p1.forward, p2.forward].concat();
        assert_eq!(
            combined_forward.len(),
            stream.len(),
            "forward bytes must equal input bytes when strip=false"
        );
        assert!(p1.usage.is_some() || p2.usage.is_some());
    }

    #[test]
    fn partial_chunks_split_inside_json() {
        let stream = full_stream(false);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        // Feed byte-by-byte to stress-test buffer.
        let mut got_usage = false;
        for b in &stream {
            let parsed = e.feed(&[*b]);
            if parsed.usage.is_some() {
                got_usage = true;
            }
        }
        assert!(got_usage, "byte-by-byte feed must still yield usage");
    }

    #[test]
    fn no_usage_chunk_falls_through() {
        let mut s = Vec::new();
        s.extend_from_slice(&build_event(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#));
        s.extend_from_slice(b"data: [DONE]\n\n");
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        let parsed = e.feed(&s);
        assert_eq!(parsed.usage, None);
        let flushed = e.flush();
        assert_eq!(flushed.usage, None);
    }

    #[test]
    fn strip_usage_chunk_when_enabled() {
        let stream = full_stream(true);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, true);
        let parsed = e.feed(&stream);
        assert_eq!(parsed.usage.as_ref().map(|u| u.total_tokens), Some(12));
        // The forward must NOT contain the usage chunk bytes.
        let forward_str = String::from_utf8_lossy(&parsed.forward);
        assert!(
            !forward_str.contains(r#""total_tokens":12"#),
            "stripped output must not contain usage payload, got: {forward_str}"
        );
        assert!(forward_str.contains("Hello"), "content delta still present");
        assert!(forward_str.contains("[DONE]"), "[DONE] still present");
    }

    #[test]
    fn keep_usage_chunk_when_disabled() {
        let stream = full_stream(false);
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        let parsed = e.feed(&stream);
        let forward_str = String::from_utf8_lossy(&parsed.forward);
        assert!(forward_str.contains(r#""total_tokens":12"#));
    }
}
