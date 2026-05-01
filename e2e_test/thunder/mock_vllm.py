#!/usr/bin/env python3
"""Mock vLLM backend for Thunder router e2e tests.

Pure stdlib (no pip install needed) — uses http.server.ThreadingHTTPServer.
Mimics the subset of vLLM's HTTP API that Thunder cares about:

  POST /v1/chat/completions      OpenAI Chat (canonical)
  POST /v1/messages              Anthropic Messages (P0+; OpenAI-shape response)
  POST /v1/responses             OpenAI Responses (P2+; OpenAI-shape response)
  GET  /v1/models                vLLM-compat for SMG worker discovery
  GET  /version                  vLLM-compat for SMG worker discovery
  GET  /get_server_info          vLLM cache-config-shaped JSON
  POST /control/capacity         Test knob: dynamically resize KV capacity
  GET  /control/state            Test introspection

Run:
    python3 mock_vllm.py --port 8001
    python3 mock_vllm.py --port 8001 --canned-content 'hi from mock'

By design this is single-file and does not import anything outside stdlib so
team members can run it without setting up the e2e_test pyproject env.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("mock_vllm")


class MockState:
    """Per-process mutable state — shared across handler threads."""

    def __init__(self, *, canned_content: str, kv_cache_block_tokens: int,
                 num_kv_cache_blocks: int, stream_chunk_count: int,
                 stream_delay_ms: int, model_name: str) -> None:
        self.lock = threading.Lock()
        self.canned_content = canned_content
        self.kv_cache_block_tokens = kv_cache_block_tokens
        self.num_kv_cache_blocks = num_kv_cache_blocks
        self.stream_chunk_count = stream_chunk_count
        self.stream_delay_ms = stream_delay_ms
        self.model_name = model_name
        self.request_count = 0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "canned_content": self.canned_content,
                "kv_cache_block_tokens": self.kv_cache_block_tokens,
                "num_kv_cache_blocks": self.num_kv_cache_blocks,
                "total_kv_cache_tokens": self.kv_cache_block_tokens
                * self.num_kv_cache_blocks,
                "request_count": self.request_count,
                "stream_chunk_count": self.stream_chunk_count,
                "stream_delay_ms": self.stream_delay_ms,
                "model_name": self.model_name,
            }


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        # Reduce stdlib's noisy default logging — we route through `LOG`.
        def log_message(self, fmt, *args):
            LOG.debug("[%s] %s", self.address_string(), fmt % args)

        # ---------- helpers ----------
        def _read_json_body(self) -> dict:
            length = int(self.headers.get("content-length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: int, body: str, content_type: str = "text/plain") -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ---------- routes ----------
        def do_GET(self):
            if self.path == "/get_server_info":
                snap = state.snapshot()
                self._send_json(200, {
                    # vLLM's get_server_info returns various engine fields; we mirror the
                    # ones Thunder will read in Phase 6.
                    "model_config": {"model": snap["model_name"]},
                    "cache_config": {
                        "block_size": snap["kv_cache_block_tokens"],
                        "num_gpu_blocks": snap["num_kv_cache_blocks"],
                        # Convenience field: total tokens of KV cache.
                        "total_kv_cache_tokens": snap["total_kv_cache_tokens"],
                    },
                })
            elif self.path == "/control/state":
                self._send_json(200, state.snapshot())
            elif self.path == "/health":
                self._send_text(200, "ok\n")
            elif self.path == "/v1/models":
                # Advertise the configured model so SMG's worker registry
                # accepts requests for it. SMG hits /v1/models on each worker
                # at registration time (detect_backend.rs) and on cache miss
                # (worker_selection.rs). owned_by="vllm" lets the backend
                # detector classify this mock as a vLLM HTTP worker; root
                # mirrors the served model id so vllm metadata discovery
                # populates model_path.
                snap = state.snapshot()
                self._send_json(200, {
                    "object": "list",
                    "data": [{
                        "id": snap["model_name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "vllm",
                        "root": snap["model_name"],
                        "max_model_len": 32768,
                    }],
                })
            elif self.path == "/version":
                # vLLM exposes /version; SMG's HTTP backend detector treats
                # a 200 from /version as a strong signal that this worker
                # is vLLM (detect_backend.rs::try_vllm_version).
                self._send_json(200, {"version": "0.0.0-mock"})
            else:
                self._send_text(404, f"not found: {self.path}\n")

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                self._handle_chat()
            elif self.path == "/v1/messages":
                self._handle_messages()
            elif self.path == "/v1/responses":
                self._handle_responses()
            elif self.path == "/control/capacity":
                self._handle_capacity_update()
            else:
                self._send_text(404, f"not found: {self.path}\n")

        # ---------- handlers ----------
        def _handle_chat(self) -> None:
            payload = self._read_json_body()
            with state.lock:
                state.request_count += 1
                req_idx = state.request_count
            stream = bool(payload.get("stream"))
            messages = payload.get("messages") or []
            LOG.info(
                "chat #%d stream=%s messages=%d program_id=%s",
                req_idx, stream, len(messages),
                payload.get("program_id")
                or (payload.get("extra_body") or {}).get("program_id")
                or "<none>",
            )
            if stream:
                self._stream_chat(req_idx)
            else:
                self._non_stream_chat(req_idx, messages)

        def _non_stream_chat(self, req_idx: int, messages: list) -> None:
            content = state.canned_content
            # Prompt token count is intentionally trivial — it's a mock.
            prompt_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages) or 1
            completion_tokens = max(1, len(content) // 4)
            payload = {
                "id": f"chatcmpl-mock-{req_idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": state.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            self._send_json(200, payload)

        def _stream_chat(self, req_idx: int) -> None:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            chat_id = f"chatcmpl-mock-{req_idx}"
            created = int(time.time())
            content = state.canned_content
            chunk_count = max(1, state.stream_chunk_count)
            # Split content into roughly equal chunks; whichever comes out smoothly is fine.
            piece_len = max(1, len(content) // chunk_count)
            pieces = [content[i:i + piece_len] for i in range(0, len(content), piece_len)]
            if len(pieces) == 0:
                pieces = [""]
            for i, piece in enumerate(pieces):
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece} if i > 0 else {"role": "assistant", "content": piece},
                        "finish_reason": None,
                    }],
                }
                self._write_sse(chunk)
                if state.stream_delay_ms > 0:
                    time.sleep(state.stream_delay_ms / 1000.0)
            # Final chunk with finish_reason and usage (vLLM-style).
            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": state.model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": max(1, len(content) // 4),
                    "total_tokens": 1 + max(1, len(content) // 4),
                },
            }
            self._write_sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _write_sse(self, obj: dict) -> None:
            self.wfile.write(b"data: ")
            self.wfile.write(json.dumps(obj).encode("utf-8"))
            self.wfile.write(b"\n\n")
            self.wfile.flush()

        # ---------- /v1/messages handler (Phase 0 pass-through tests) ----------
        def _handle_messages(self) -> None:
            """Anthropic Messages payload arrives here. For Phase 0 tests we just
            log the program_id and return an OpenAI-shape chat.completion body —
            this matches the litellm-proxy translation behavior the real
            sidecar topology will use (Anthropic in → OpenAI internally).

            Real Anthropic-shape responses are out of scope for P0; P3+ tests
            will exercise cross-protocol program_id stickiness with proper
            Anthropic-out translation.
            """
            payload = self._read_json_body()
            with state.lock:
                state.request_count += 1
                req_idx = state.request_count
            stream = bool(payload.get("stream"))
            messages = payload.get("messages") or []
            metadata = payload.get("metadata") or {}
            program_id = metadata.get("program_id")
            LOG.info(
                "messages #%d stream=%s messages=%d program_id=%s",
                req_idx, stream, len(messages), program_id or "<none>",
            )
            if stream:
                # Phase 0 doesn't exercise streaming on /v1/messages; emit
                # a minimal SSE that pass-through tests can ignore. Real
                # streaming arrives in P3+.
                self._stream_chat(req_idx)
                return
            # Reuse the OpenAI-shape non-stream payload.
            content = state.canned_content
            prompt_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages) or 1
            completion_tokens = max(1, len(content) // 4)
            payload_out = {
                "id": f"chatcmpl-mock-{req_idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": state.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                # Echo the program_id so the e2e test can assert plumbing worked.
                "_mock_echo_program_id": program_id,
            }
            self._send_json(200, payload_out)

        # ---------- /v1/responses handler (Phase 2 multi-protocol smoke) ----------
        def _handle_responses(self) -> None:
            """OpenAI Responses API endpoint. Like _handle_messages, returns an
            OpenAI chat.completion shape — matches what a litellm-proxy sidecar
            produces after Responses-in → Chat-in upstream translation. Phase 2
            tests assert pass-through, not body-shape compliance.
            """
            payload = self._read_json_body()
            with state.lock:
                state.request_count += 1
                req_idx = state.request_count
            stream = bool(payload.get("stream"))
            # Responses API uses `input` field (string or array) instead of `messages`.
            input_field = payload.get("input")
            metadata = payload.get("metadata") or {}
            program_id = metadata.get("program_id")
            LOG.info(
                "responses #%d stream=%s has_input=%s program_id=%s",
                req_idx, stream, input_field is not None, program_id or "<none>",
            )
            if stream:
                self._stream_chat(req_idx)
                return
            content = state.canned_content
            # Approximate prompt size from input field
            if isinstance(input_field, str):
                prompt_chars = len(input_field)
            elif isinstance(input_field, list):
                prompt_chars = sum(len(str(item)) for item in input_field)
            else:
                prompt_chars = 0
            prompt_tokens = max(1, prompt_chars // 4)
            completion_tokens = max(1, len(content) // 4)
            payload_out = {
                "id": f"chatcmpl-mock-{req_idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": state.model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                # Echo program_id so e2e can assert plumbing
                "_mock_echo_program_id": program_id,
                "_mock_endpoint": "/v1/responses",
            }
            self._send_json(200, payload_out)

        def _handle_capacity_update(self) -> None:
            payload = self._read_json_body()
            with state.lock:
                if "num_kv_cache_blocks" in payload:
                    state.num_kv_cache_blocks = int(payload["num_kv_cache_blocks"])
                if "kv_cache_block_tokens" in payload:
                    state.kv_cache_block_tokens = int(payload["kv_cache_block_tokens"])
                snap = {
                    "kv_cache_block_tokens": state.kv_cache_block_tokens,
                    "num_kv_cache_blocks": state.num_kv_cache_blocks,
                    "total_kv_cache_tokens": state.kv_cache_block_tokens * state.num_kv_cache_blocks,
                }
            LOG.info("capacity update -> %s", snap)
            self._send_json(200, snap)

    return Handler


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock vLLM backend for Thunder e2e tests.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--canned-content", default="Hello from mock vLLM!",
                   help="Assistant message content returned for chat completions.")
    p.add_argument("--model-name", default="mock-vllm-model")
    p.add_argument("--kv-cache-block-tokens", type=int, default=16,
                   help="vLLM cache config: tokens per KV-cache block.")
    p.add_argument("--num-kv-cache-blocks", type=int, default=2048,
                   help="vLLM cache config: number of KV-cache blocks.")
    p.add_argument("--stream-chunk-count", type=int, default=5,
                   help="How many SSE delta chunks to emit before the final [DONE].")
    p.add_argument("--stream-delay-ms", type=int, default=0,
                   help="Sleep between SSE chunks (ms). Useful for testing streaming progress.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state = MockState(
        canned_content=args.canned_content,
        kv_cache_block_tokens=args.kv_cache_block_tokens,
        num_kv_cache_blocks=args.num_kv_cache_blocks,
        stream_chunk_count=args.stream_chunk_count,
        stream_delay_ms=args.stream_delay_ms,
        model_name=args.model_name,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    LOG.info("mock vLLM listening on http://%s:%d", args.host, args.port)
    LOG.info("initial state: %s", state.snapshot())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
