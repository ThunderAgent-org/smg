//! Server-Sent Events (SSE) parsing for Thunder streaming usage extraction.
//!
//! Three protocols supported:
//! - OpenAI Chat Completions (`/v1/chat/completions`)
//! - Anthropic Messages (`/v1/messages`)
//! - OpenAI Responses (`/v1/responses`)
//!
//! Each parser:
//! 1. Buffers incoming bytes across chunk boundaries
//! 2. Splits on SSE event delimiter `\n\n`
//! 3. Per-line `data:` extraction + JSON parse
//! 4. Protocol-specific usage extraction
//! 5. Optional usage-chunk stripping (OpenAI Chat only)
//! 6. Token-delta emission for incremental tracking
//!
//! Spec: `docs/superpowers/specs/2026-05-01-thunder-phase7-production-design.md` §3.2

pub mod anthropic;
pub mod extractor;
pub mod openai_chat;
pub mod responses;

pub use extractor::{ParsedChunk, ParsedUsage, SseExtractor, SseProtocol};

/// Token-progress emission threshold (events for OpenAI Chat / Responses; ignored for
/// Anthropic which uses message_delta cumulative readings).
pub const INCREMENTAL_TOKEN_INTERVAL: u64 = 20;

/// Detect protocol from upstream endpoint URL.
pub fn detect_protocol(endpoint: &str) -> SseProtocol {
    if endpoint.contains("/v1/messages") {
        SseProtocol::AnthropicMessages
    } else if endpoint.contains("/v1/responses") {
        SseProtocol::OpenAiResponses
    } else {
        SseProtocol::OpenAiChat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_protocol_anthropic() {
        assert_eq!(detect_protocol("/v1/messages"), SseProtocol::AnthropicMessages);
        assert_eq!(
            detect_protocol("http://b1:8000/v1/messages"),
            SseProtocol::AnthropicMessages
        );
    }

    #[test]
    fn detect_protocol_responses() {
        assert_eq!(detect_protocol("/v1/responses"), SseProtocol::OpenAiResponses);
    }

    #[test]
    fn detect_protocol_default_openai_chat() {
        assert_eq!(detect_protocol("/v1/chat/completions"), SseProtocol::OpenAiChat);
        assert_eq!(detect_protocol("/anything-else"), SseProtocol::OpenAiChat);
    }
}
