//! Program ID extraction helper for program-aware policies (e.g. ThunderPolicy).
//!
//! This module centralizes the call to `GenerationRequest::program_id_hint`
//! so future hooks (sanitization, tenant prefixing) have a single seam.

use openai_protocol::common::GenerationRequest;

/// Extract the program identifier hint from a typed generation request.
///
/// Today this is a thin pass-through to `GenerationRequest::program_id_hint`.
/// The indirection exists to give future per-tenant or per-deployment rewrites
/// a single place to land — e.g., if `metadata.program_id` ever needs
/// namespace prefixing for multi-tenant cross-isolation.
///
/// Returns `None` for requests whose protocol does not carry a program_id
/// concept (every type today except `CreateMessageRequest`).
pub fn extract<T: GenerationRequest + ?Sized>(req: &T) -> Option<&str> {
    req.program_id_hint()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, serde::Serialize)]
    struct Stub<'a> {
        pid: Option<&'a str>,
    }

    impl GenerationRequest for Stub<'_> {
        fn is_stream(&self) -> bool {
            false
        }
        fn get_model(&self) -> Option<&str> {
            None
        }
        fn extract_text_for_routing(&self) -> String {
            String::new()
        }
        fn program_id_hint(&self) -> Option<&str> {
            self.pid
        }
    }

    #[test]
    fn extract_returns_program_id_hint() {
        let req = Stub { pid: Some("agent-1") };
        assert_eq!(extract(&req), Some("agent-1"));
    }

    #[test]
    fn extract_returns_none_when_hint_is_none() {
        let req = Stub { pid: None };
        assert_eq!(extract(&req), None);
    }
}
