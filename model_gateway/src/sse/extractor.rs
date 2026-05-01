//! Generic SSE byte-buffer state machine. Handles cross-chunk boundary buffering,
//! event splitting on `\n\n`, line splitting on `\n`. Per-protocol parsers
//! (`openai_chat.rs`, `anthropic.rs`, `responses.rs`) layer on top via dispatch.

use crate::sse::{anthropic, openai_chat, responses};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseProtocol {
    OpenAiChat,
    AnthropicMessages,
    OpenAiResponses,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedUsage {
    pub total_tokens: u64,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Anthropic: `cache_read_input_tokens` (already counted toward input_tokens
    /// but should be excluded from prefill ratio in M3 calibration).
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Default)]
pub struct ParsedChunk {
    /// Bytes to forward to the client. Already stripped of the usage chunk if
    /// `strip_usage_chunk=true` (OpenAI Chat only).
    pub forward: Vec<u8>,
    /// `Some(...)` once per stream when usage is finalized; `None` on every other feed call.
    pub usage: Option<ParsedUsage>,
    /// Incremental token estimate for this feed (added to the running total
    /// since last `feed`). Routers emit `StreamingProgressEvent` when this
    /// crosses `INCREMENTAL_TOKEN_INTERVAL`.
    pub token_delta: u64,
}

/// Per-protocol parsing state.
#[derive(Debug)]
pub(crate) enum ProtocolState {
    OpenAiChat(openai_chat::OpenAiChatState),
    AnthropicMessages(anthropic::AnthropicState),
    OpenAiResponses(responses::ResponsesState),
}

impl ProtocolState {
    fn new(protocol: SseProtocol) -> Self {
        match protocol {
            SseProtocol::OpenAiChat => Self::OpenAiChat(openai_chat::OpenAiChatState::default()),
            SseProtocol::AnthropicMessages => {
                Self::AnthropicMessages(anthropic::AnthropicState::default())
            }
            SseProtocol::OpenAiResponses => {
                Self::OpenAiResponses(responses::ResponsesState::default())
            }
        }
    }
}

/// SSE byte-buffer state machine.
#[derive(Debug)]
pub struct SseExtractor {
    pub(crate) buffer: Vec<u8>,
    pub(crate) state: ProtocolState,
    pub(crate) strip_usage_chunk: bool,
    pub(crate) usage_extracted: bool,
}

impl SseExtractor {
    pub fn new(protocol: SseProtocol, strip_usage_chunk: bool) -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            state: ProtocolState::new(protocol),
            strip_usage_chunk,
            usage_extracted: false,
        }
    }

    /// Feed the next chunk of bytes from the upstream stream. Returns
    /// (filtered_bytes_for_client, optional_usage, token_delta_estimate).
    pub fn feed(&mut self, chunk: &[u8]) -> ParsedChunk {
        self.buffer.extend_from_slice(chunk);

        let mut forward = Vec::with_capacity(chunk.len());
        let mut usage: Option<ParsedUsage> = None;
        let mut token_delta: u64 = 0;

        // Find complete events delimited by `\n\n` and process them. Anything
        // after the last `\n\n` is partial — keep it in buffer for next feed.
        loop {
            let Some(boundary) = find_event_boundary(&self.buffer) else {
                break;
            };
            // Split: event is buffer[..boundary]; everything after boundary+2 stays.
            let event_bytes: Vec<u8> = self.buffer.drain(..boundary + 2).collect();

            let event_outcome = match &mut self.state {
                ProtocolState::OpenAiChat(s) => {
                    openai_chat::process_event(s, &event_bytes, self.strip_usage_chunk)
                }
                ProtocolState::AnthropicMessages(s) => {
                    anthropic::process_event(s, &event_bytes)
                }
                ProtocolState::OpenAiResponses(s) => {
                    responses::process_event(s, &event_bytes)
                }
            };

            forward.extend_from_slice(&event_outcome.forward);
            token_delta = token_delta.saturating_add(event_outcome.token_delta);
            if let Some(u) = event_outcome.usage {
                if !self.usage_extracted {
                    self.usage_extracted = true;
                    usage = Some(u);
                }
            }
        }

        ParsedChunk { forward, usage, token_delta }
    }

    /// Flush any remaining partial buffer. Called once on stream end.
    /// For most well-formed streams this is a no-op (everything ended with `\n\n`).
    pub fn flush(&mut self) -> ParsedChunk {
        let mut forward = Vec::new();
        let mut usage: Option<ParsedUsage> = None;
        let mut token_delta: u64 = 0;

        if !self.buffer.is_empty() {
            // Treat residue as a final event (no trailing \n\n).
            let event_bytes = std::mem::take(&mut self.buffer);
            let event_outcome = match &mut self.state {
                ProtocolState::OpenAiChat(s) => {
                    openai_chat::process_event(s, &event_bytes, self.strip_usage_chunk)
                }
                ProtocolState::AnthropicMessages(s) => {
                    anthropic::process_event(s, &event_bytes)
                }
                ProtocolState::OpenAiResponses(s) => {
                    responses::process_event(s, &event_bytes)
                }
            };
            forward.extend_from_slice(&event_outcome.forward);
            token_delta = token_delta.saturating_add(event_outcome.token_delta);
            if let Some(u) = event_outcome.usage {
                if !self.usage_extracted {
                    self.usage_extracted = true;
                    usage = Some(u);
                }
            }
        }

        // Per-protocol final reconciliation (e.g., Anthropic: combine input+output_tokens
        // even if no single event carried the full picture).
        if usage.is_none() && !self.usage_extracted {
            let final_usage = match &self.state {
                ProtocolState::AnthropicMessages(s) => anthropic::finalize(s),
                ProtocolState::OpenAiResponses(s) => responses::finalize(s),
                ProtocolState::OpenAiChat(_) => None,
            };
            if let Some(u) = final_usage {
                self.usage_extracted = true;
                usage = Some(u);
            }
        }

        ParsedChunk { forward, usage, token_delta }
    }
}

/// Find the index of the byte that ENDS the next complete event (the position
/// of the first byte of `\n\n`). Returns `None` if no full event is in the buffer.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

#[derive(Debug, Default)]
pub(crate) struct EventOutcome {
    pub forward: Vec<u8>,
    pub usage: Option<ParsedUsage>,
    pub token_delta: u64,
}

/// Strip the leading `data:` prefix and any leading whitespace/colon. Returns
/// `None` if the line isn't a `data:` line (e.g., `event:`, `:` keepalive, empty).
pub(crate) fn extract_data_payload(line: &[u8]) -> Option<&[u8]> {
    if !line.starts_with(b"data:") {
        return None;
    }
    let mut payload = &line[5..];
    while let Some((b' ', rest)) = payload.split_first() {
        payload = rest;
    }
    Some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_event_boundary_basic() {
        assert_eq!(find_event_boundary(b"data: x\n\nrest"), Some(7));
        assert_eq!(find_event_boundary(b"data: x"), None);
        assert_eq!(find_event_boundary(b""), None);
    }

    #[test]
    fn extract_data_payload_basic() {
        assert_eq!(extract_data_payload(b"data: hello"), Some(&b"hello"[..]));
        assert_eq!(extract_data_payload(b"data:hello"), Some(&b"hello"[..]));
        assert_eq!(extract_data_payload(b"data:  hello"), Some(&b"hello"[..]));
        assert_eq!(extract_data_payload(b"event: foo"), None);
        assert_eq!(extract_data_payload(b": comment"), None);
        assert_eq!(extract_data_payload(b""), None);
    }

    #[test]
    fn buffers_partial_event_across_feeds() {
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        // First feed: incomplete event.
        let p1 = e.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"");
        assert_eq!(p1.usage, None);
        // Second feed: completes the event with \n\n.
        let p2 = e.feed(b"}}]}\n\n");
        // No usage chunk yet (just content delta), so usage is None.
        assert_eq!(p2.usage, None);
        // Combined output should be the concatenation of both inputs.
        let combined: Vec<u8> = p1.forward.iter().chain(p2.forward.iter()).copied().collect();
        assert!(combined.starts_with(b"data:"));
    }

    #[test]
    fn handles_keepalive_comment_lines() {
        let mut e = SseExtractor::new(SseProtocol::OpenAiChat, false);
        let p = e.feed(b": keepalive\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n");
        // No usage; both events forwarded; comment line forwarded unchanged.
        assert_eq!(p.usage, None);
        assert!(p.forward.contains(&b':'));
    }
}
