//! OpenAI Responses (`/v1/responses`) SSE parser.
//!
//! Stream shape:
//! ```text
//! event: response.created
//! data: {"type":"response.created","response":{"id":"resp_..."}}
//!
//! event: response.output_text.delta
//! data: {"type":"response.output_text.delta","delta":"text"}
//!
//! ... many response.output_text.delta events ...
//!
//! event: response.completed
//! data: {"type":"response.completed","response":{...,"usage":{"input_tokens":10,"output_tokens":50,"total_tokens":60}}}
//! ```
//!
//! - Token delta: each `response.output_text.delta` event → +1 (heuristic)
//! - Usage extraction: from `response.completed` event payload's `response.usage` field
//! - Strip: never (Responses clients always expect to see response.completed)

use serde_json::Value;

use crate::sse::extractor::{extract_data_payload, EventOutcome, ParsedUsage};

#[derive(Debug, Default)]
pub struct ResponsesState {
    seen_completed: bool,
    final_usage: Option<ParsedUsage>,
}

pub(crate) fn process_event(state: &mut ResponsesState, event_bytes: &[u8]) -> EventOutcome {
    let mut outcome = EventOutcome::default();
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
        let event_type = val.get("type").and_then(Value::as_str).unwrap_or("");

        if event_type == "response.output_text.delta" {
            outcome.token_delta = outcome.token_delta.saturating_add(1);
        }

        if event_type == "response.completed" && !state.seen_completed {
            state.seen_completed = true;
            if let Some(usage) = val
                .get("response")
                .and_then(|r| r.get("usage"))
                .and_then(Value::as_object)
            {
                let total = usage
                    .get("total_tokens")
                    .and_then(Value::as_u64)
                    .or_else(|| {
                        let p = usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                        let o = usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                        Some(p.saturating_add(o))
                    })
                    .unwrap_or(0);
                let parsed = ParsedUsage {
                    total_tokens: total,
                    prompt_tokens: usage.get("input_tokens").and_then(Value::as_u64),
                    completion_tokens: usage.get("output_tokens").and_then(Value::as_u64),
                    cached_tokens: usage
                        .get("input_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(Value::as_u64),
                };
                state.final_usage = Some(parsed.clone());
                outcome.usage = Some(parsed);
            }
        }
    }

    outcome
}

pub(crate) fn finalize(state: &ResponsesState) -> Option<ParsedUsage> {
    state.final_usage.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::extractor::{SseExtractor, SseProtocol};

    fn responses_full_stream() -> Vec<u8> {
        let mut s = String::new();
        s.push_str("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n");
        for _ in 0..5 {
            s.push_str("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n");
        }
        s.push_str("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":50,\"total_tokens\":60}}}\n\n");
        s.into_bytes()
    }

    #[test]
    fn full_stream_extracts_usage() {
        let stream = responses_full_stream();
        let mut e = SseExtractor::new(SseProtocol::OpenAiResponses, false);
        let p = e.feed(&stream);
        assert_eq!(
            p.usage,
            Some(ParsedUsage {
                total_tokens: 60,
                prompt_tokens: Some(10),
                completion_tokens: Some(50),
                cached_tokens: None,
            })
        );
    }

    #[test]
    fn token_delta_counts_output_text_delta_events() {
        let stream = responses_full_stream();
        let mut e = SseExtractor::new(SseProtocol::OpenAiResponses, false);
        let p = e.feed(&stream);
        assert_eq!(p.token_delta, 5);
    }

    #[test]
    fn no_completed_event_yields_no_usage() {
        let mut s = String::new();
        s.push_str("event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n");
        s.push_str("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}\n\n");
        let mut e = SseExtractor::new(SseProtocol::OpenAiResponses, false);
        let p = e.feed(s.as_bytes());
        let f = e.flush();
        assert_eq!(p.usage, None);
        assert_eq!(f.usage, None);
    }

    #[test]
    fn forward_passes_through_unchanged() {
        let stream = responses_full_stream();
        let mut e = SseExtractor::new(SseProtocol::OpenAiResponses, false);
        let p = e.feed(&stream);
        assert_eq!(p.forward.len(), stream.len());
    }

    #[test]
    fn handles_partial_chunks() {
        let stream = responses_full_stream();
        let mut e = SseExtractor::new(SseProtocol::OpenAiResponses, false);
        let mut usage = None;
        for chunk in stream.chunks(11) {
            let p = e.feed(chunk);
            if p.usage.is_some() {
                usage = p.usage;
            }
        }
        assert_eq!(usage.as_ref().unwrap().total_tokens, 60);
    }
}
