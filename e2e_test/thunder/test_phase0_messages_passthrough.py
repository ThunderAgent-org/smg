"""Phase 0 e2e: SMG forwards POST /v1/messages to a backend.

After Phase 0:
- crates/protocols implements GenerationRequest for CreateMessageRequest
- model_gateway/src/routers/http/router.rs::Router::route_messages exists
- /v1/messages route_to_endpoint label = "messages"

The test does NOT use ThunderPolicy (which doesn't exist until Phase 3);
it uses cache_aware and asserts the protocol seam works for arbitrary
GenerationRequest impls.
"""
from __future__ import annotations

import requests


def test_messages_non_streaming_passthrough(smg_router):
    """POST /v1/messages with a CreateMessageRequest body returns 200 +
    backend-shaped body + program_id reached the backend."""
    req_body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "hello, gateway!"},
        ],
        "metadata": {
            "program_id": "phase0-test-1",
            "user_id": "alice",
        },
    }
    r = requests.post(
        f"{smg_router}/v1/messages",
        json=req_body,
        headers={"content-type": "application/json"},
        timeout=10,
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    # Mock returns OpenAI-shape; gateway forwards bytes-as-is.
    assert body["object"] == "chat.completion"
    assert "choices" in body and body["choices"]
    # Backend received metadata.program_id and echoed it back.
    assert body.get("_mock_echo_program_id") == "phase0-test-1", \
        f"program_id was lost in transit; body={body}"


def test_messages_metadata_program_id_optional(smg_router):
    """Requests without metadata.program_id still succeed; backend gets None."""
    req_body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    r = requests.post(f"{smg_router}/v1/messages", json=req_body, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body.get("_mock_echo_program_id") is None


def test_messages_blocks_content_routes_correctly(smg_router):
    """Block-form content (Anthropic native) is forwarded; backend gets the body."""
    req_body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 16,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "block-form text"},
            ],
        }],
        "metadata": {"program_id": "phase0-blocks"},
    }
    r = requests.post(f"{smg_router}/v1/messages", json=req_body, timeout=10)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("_mock_echo_program_id") == "phase0-blocks"
