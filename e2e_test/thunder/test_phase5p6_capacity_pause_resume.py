"""Phase 5+6 e2e: ThunderPolicy TR mode capacity gate + force-resume.

Two scenarios exercised end-to-end against a single mock backend:

1. **Capacity available → fast admit.** Set mock capacity high, send a
   /v1/messages request, assert it returns quickly (< 3s) — i.e. the TR
   gate did not pause.
2. **Capacity zero forever → force-resume.** Set mock capacity to 0,
   start SMG with `--thunder-resume-timeout-secs 5`, send a request,
   assert it blocks ~5s and then completes (200 OK via the
   force-admit-after-timeout fallback in `pick_tr`).

The "set 0 → unblock by raising capacity" variant is intentionally
skipped (D-22 sub-decision in the plan) because cross-thread timing for
the capacity-poll → usage-consumer broadcast chain adds e2e flakiness;
unit test `tr_mode_pauses_then_resumes_on_capacity_free` covers that
flow.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import closing

import pytest
import requests

THUNDER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THUNDER_DIR, "..", ".."))


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"timeout waiting for {url}: {last_err}")


@pytest.fixture
def thunder_tr_with_short_timeout(mock_backend):
    """SMG with --policy thunder --thunder-sub-mode tr and a short
    --thunder-resume-timeout-secs so the force-resume test doesn't
    take 30 minutes. Function-scoped so each test gets a fresh process
    (clean Thunder state)."""
    port = _free_port()
    pport = _free_port()
    binary = os.path.join(REPO_ROOT, "target", "debug", "smg")
    if not os.path.exists(binary):
        binary = os.path.join(REPO_ROOT, "target", "release", "smg")
    if not os.path.exists(binary):
        pytest.skip(
            f"smg binary not found at target/{{debug,release}}/smg; "
            f"run `cargo build -p model_gateway` from {REPO_ROOT} first"
        )
    cmd = [
        binary, "start",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--worker-urls", mock_backend,
        "--policy", "thunder",
        "--thunder-sub-mode", "tr",
        "--thunder-resume-timeout-secs", "5",
        "--thunder-capacity-poll-interval-secs", "1",
        "--prometheus-port", str(pport),
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        _wait_http(f"http://127.0.0.1:{port}/health", timeout=20)
        yield {"smg": f"http://127.0.0.1:{port}", "mock": mock_backend}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _set_capacity(mock_url: str, num_kv_cache_blocks: int) -> None:
    r = requests.post(
        f"{mock_url}/control/capacity",
        json={"num_kv_cache_blocks": num_kv_cache_blocks},
        timeout=2,
    )
    r.raise_for_status()


def test_tr_admits_when_capacity_available(thunder_tr_with_short_timeout):
    """Capacity > 0 → request admits without blocking."""
    fix = thunder_tr_with_short_timeout
    _set_capacity(fix["mock"], 1024)  # plenty (1024 blocks * default block_tokens)
    # Give the capacity-poll task (1s interval per the fixture flag) a
    # couple ticks to refresh the BackendState.
    time.sleep(2.5)
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "fast path"}],
        "metadata": {"program_id": "tr-fastpath"},
    }
    start = time.time()
    r = requests.post(f"{fix['smg']}/v1/messages", json=body, timeout=10)
    elapsed = time.time() - start
    assert r.status_code == 200, r.text
    assert elapsed < 3.0, f"should not have blocked (took {elapsed:.2f}s)"


def test_tr_force_resume_on_timeout(thunder_tr_with_short_timeout):
    """Tiny non-zero capacity + warmup primer → real request blocks because
    its estimate exceeds capacity → force-resumes after
    --thunder-resume-timeout-secs (5s) via the timeout-fallback path.

    Two subtleties bake into this test:

    1. `RouterState::has_capacity` treats `capacity_tokens == 0` as "backend
       not yet polled → optimistic admit" so we need a non-zero capacity
       (1 block = 16 tokens) for the gate to actually engage.

    2. The capacity-poll task only refreshes backends ALREADY in
       `RouterState.backends`. The map is populated lazily by the first
       `select_worker` call. So we send a tiny primer to seed the map,
       wait for one poll tick, then the real test request hits the gate
       and blocks.
    """
    fix = thunder_tr_with_short_timeout
    _set_capacity(fix["mock"], 1)  # 1 block * 16 tokens = 16 total, ~14 usable

    # Primer: seeds RouterState.backends so the capacity-poll task starts
    # refreshing this backend. Since cold-start `capacity_tokens` is 0,
    # has_capacity returns optimistic-true and the primer admits cleanly.
    primer = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "primer"}],
        "metadata": {"program_id": "tr-primer"},
    }
    r0 = requests.post(f"{fix['smg']}/v1/messages", json=primer, timeout=10)
    assert r0.status_code == 200, r0.text

    # Wait for the capacity-poll task (1s interval per fixture flag) to
    # refresh BackendState.capacity_tokens to 16.
    time.sleep(2.5)

    body = {
        "model": "Qwen/Qwen3-0.6B",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "force resume"}],
        "metadata": {"program_id": "tr-force"},
    }
    start = time.time()
    r = requests.post(f"{fix['smg']}/v1/messages", json=body, timeout=20)
    elapsed = time.time() - start
    assert r.status_code == 200, r.text
    # Should have blocked at least ~4s (5s timeout minus scheduling slop).
    assert elapsed >= 4.0, f"expected blocked ~5s, took {elapsed:.2f}s"
    # And resumed before 15s (no >>5s overhead).
    assert elapsed < 15.0, f"resume took too long: {elapsed:.2f}s"
