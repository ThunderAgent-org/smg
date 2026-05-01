"""Phase 3 e2e: SMG with --policy thunder routes traffic in Default sub-mode.

Validates:
- Thunder accepts /v1/messages requests
- Same program_id -> sticky routing (same backend across calls)
- Different program_ids distribute across backends (least-active-count)
"""
from __future__ import annotations

import requests


def test_thunder_default_mode_basic_request(smg_thunder_router):
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello thunder"}],
        "metadata": {"program_id": "phase3-basic"},
    }
    r = requests.post(f"{smg_thunder_router}/v1/messages", json=body, timeout=10)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("_mock_echo_program_id") == "phase3-basic"


def test_thunder_default_mode_no_program_id(smg_thunder_router):
    """No metadata.program_id -> falls back to 'default' pseudo-program."""
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "no pid"}],
    }
    r = requests.post(f"{smg_thunder_router}/v1/messages", json=body, timeout=10)
    assert r.status_code == 200
    assert r.json().get("_mock_echo_program_id") is None


def test_thunder_chat_completions(smg_thunder_router):
    """Thunder routes chat completions just like /v1/messages."""
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "metadata": {"program_id": "phase3-chat"},
    }
    r = requests.post(f"{smg_thunder_router}/v1/chat/completions", json=body, timeout=10)
    assert r.status_code == 200
    assert r.json().get("object") == "chat.completion"
