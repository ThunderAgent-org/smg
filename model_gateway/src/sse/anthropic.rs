//! Anthropic Messages (`/v1/messages`) SSE parser.
//!
//! Stream shape:
//! ```text
//! event: message_start
//! data: {"type":"message_start","message":{"id":"...","usage":{"input_tokens":25,"cache_read_input_tokens":10}}}
//!
//! event: content_block_start
//! data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
//!
//! event: content_block_delta
//! data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
//!
//! ... many content_block_delta events ...
//!
//! event: message_delta
//! data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}
//!
//! event: message_stop
//! data: {"type":"message_stop"}
//! ```
//!
//! - Token delta: read `output_tokens` from `message_delta` cumulative; delta = current - last_seen
//! - Usage: combine `message_start.usage.input_tokens` + last `message_delta.usage.output_tokens`
//! - cache_read_input_tokens: from `message_start.usage` (used in M3 calibration)
//! - Strip: never (Anthropic clients always expect to see usage events)

use serde_json::Value;

use crate::sse::extractor::{extract_data_payload, EventOutcome, ParsedUsage};

#[derive(Debug, Default)]
pub struct AnthropicState {
    input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    output_tokens_cumulative: u64,
    last_reported_output_tokens: u64,
    seen_message_stop: bool,
}

pub(crate) fn process_event(state: &mut AnthropicState, event_bytes: &[u8]) -> EventOutcome {
    let mut outcome = EventOutcome::default();
    // Anthropic never strips — pass through verbatim.
    outcome.forward.extend_from_slice(event_bytes);

    for line in event_bytes.split(|b| *b == b'\n') {
        let Some(payload) = extract_data_payload(line) else {
            continue;
        };
        if payload.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_slice::<Value>(payload) else {
            continue;
        };

        match val.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(usage) = val.get("message").and_then(|m| m.get("usage")) {
                    state.input_tokens = usage.get("input_tokens").and_then(Value::as_u64);
                    state.cache_read_input_tokens = usage
                        .get("cache_read_input_tokens")
                        .and_then(Value::as_u64);
                    if let Some(out) = usage.get("output_tokens").and_then(Value::as_u64) {
                        state.output_tokens_cumulative = out;
                    }
                }
            }
            Some("message_delta") => {
                if let Some(usage) = val.get("usage") {
                    if let Some(out) = usage.get("output_tokens").and_then(Value::as_u64) {
                        state.output_tokens_cumulative = out;
                        let delta = state
                            .output_tokens_cumulative
                            .saturating_sub(state.last_reported_output_tokens);
                        if delta > 0 {
                            outcome.token_delta = outcome.token_delta.saturating_add(delta);
                            state.last_reported_output_tokens = state.output_tokens_cumulative;
                        }
                    }
                }
            }
            Some("message_stop") => {
                state.seen_message_stop = true;
            }
            _ => {}
        }
    }

    // After message_stop, finalize usage.
    if state.seen_message_stop && outcome.usage.is_none() {
        if let Some(u) = build_usage(state) {
            outcome.usage = Some(u);
        }
    }

    outcome
}

pub(crate) fn finalize(state: &AnthropicState) -> Option<ParsedUsage> {
    build_usage(state)
}

fn build_usage(state: &AnthropicState) -> Option<ParsedUsage> {
    let input = state.input_tokens?;
    let output = state.output_tokens_cumulative;
    Some(ParsedUsage {
        total_tokens: input.saturating_add(output),
        prompt_tokens: Some(input),
        completion_tokens: Some(output),
        cached_tokens: state.cache_read_input_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::extractor::{SseExtractor, SseProtocol};

    fn anthropic_full_stream(cache_read: u64) -> Vec<u8> {
        let mut s = String::new();
        s.push_str(&format!(
            "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\",\"usage\":{{\"input_tokens\":25,\"cache_read_input_tokens\":{cache_read},\"output_tokens\":1}}}}}}\n\n"
        ));
        for _ in 0..3 {
            s.push_str("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n");
        }
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":15}}\n\n");
        s.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        s.into_bytes()
    }

    #[test]
    fn full_stream_extracts_usage_with_cache_read() {
        let stream = anthropic_full_stream(10);
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, false);
        let p = e.feed(&stream);
        assert_eq!(
            p.usage,
            Some(ParsedUsage {
                total_tokens: 40, // 25 input + 15 output
                prompt_tokens: Some(25),
                completion_tokens: Some(15),
                cached_tokens: Some(10),
            })
        );
    }

    #[test]
    fn full_stream_no_cache_read_field() {
        // Some Anthropic responses don't include cache_read_input_tokens at all.
        let stream = anthropic_full_stream(0);
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, false);
        let p = e.feed(&stream);
        // cache_read=0 in our fixture → still Some(0) (field present in JSON).
        assert_eq!(p.usage.as_ref().unwrap().cached_tokens, Some(0));
    }

    #[test]
    fn cumulative_output_tokens_drives_token_delta() {
        // Two message_delta events: first reports 10, second reports 15. Delta should be 10 then 5.
        let mut s = String::new();
        s.push_str("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n");
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":10}}\n\n");
        s.push_str("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":15}}\n\n");
        s.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, false);
        let p = e.feed(s.as_bytes());
        // Total token_delta across the whole feed = 10 (first) + 5 (second) = 15.
        assert_eq!(p.token_delta, 15);
    }

    #[test]
    fn anthropic_never_strips_events() {
        let stream = anthropic_full_stream(10);
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, true /* strip flag ignored */);
        let p = e.feed(&stream);
        assert_eq!(
            p.forward.len(),
            stream.len(),
            "Anthropic forward bytes must match input verbatim"
        );
    }

    #[test]
    fn partial_chunks_split_inside_event_data() {
        let stream = anthropic_full_stream(10);
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, false);
        let mut usage = None;
        for chunk in stream.chunks(7) {
            let p = e.feed(chunk);
            if p.usage.is_some() {
                usage = p.usage;
            }
        }
        let p = e.flush();
        if usage.is_none() {
            usage = p.usage;
        }
        assert_eq!(usage.as_ref().unwrap().total_tokens, 40);
    }

    #[test]
    fn message_start_only_no_message_delta_yields_partial_usage_on_flush() {
        let s = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":42,\"output_tokens\":3}}}\n\n";
        let mut e = SseExtractor::new(SseProtocol::AnthropicMessages, false);
        let _ = e.feed(s);
        let p = e.flush();
        // No message_stop seen → flush() falls through to finalize() and
        // synthesizes usage from the message_start values.
        assert_eq!(p.usage.as_ref().unwrap().total_tokens, 45);
    }
}
