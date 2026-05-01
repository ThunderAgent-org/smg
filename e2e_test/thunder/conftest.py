"""Pytest fixtures for thunder Phase 0 e2e tests.

For Phase 0 we run the mock backend and the SMG binary on the same host
(the test runner — currently the SLURM compute node via srun). Phase 2
will introduce launcher scripts + 4 sglang backends; for P0 a single
mock + a single SMG is enough.
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
    """Bind a transient socket to find an unused TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 10.0) -> None:
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


@pytest.fixture(scope="session")
def mock_backend():
    """Start the mock_vllm.py backend; yield its base URL; stop it on teardown."""
    port = _free_port()
    # The Phase 0 e2e tests post requests with model="Qwen/Qwen3-0.6B"; have
    # the mock advertise that id via /v1/models so SMG's worker registry
    # accepts the routing.
    proc = subprocess.Popen(
        [
            "python3",
            os.path.join(THUNDER_DIR, "mock_vllm.py"),
            "--port", str(port),
            "--model-name", "Qwen/Qwen3-0.6B",
        ],
        cwd=THUNDER_DIR,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/health", timeout=10)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def smg_router(mock_backend):
    """Start SMG with cache_aware policy and one worker pointing at the mock.

    Phase 0 uses cache_aware (the default-friendly policy); ThunderPolicy
    arrives in Phase 3 and replaces this fixture's --policy arg.
    """
    port = _free_port()
    binary = os.path.join(REPO_ROOT, "target", "debug", "smg")
    if not os.path.exists(binary):
        # Fall back to release if a debug build hasn't been done yet.
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
        "--policy", "cache_aware",
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        _wait_http(f"http://127.0.0.1:{port}/health", timeout=20)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def smg_thunder_router(mock_backend):
    """SMG with --policy thunder, pointing at the same mock_backend.

    Used by Phase 3+ tests; coexists with smg_router (cache_aware) so
    Phase 0-2 tests keep passing under cache_aware.
    """
    port = _free_port()
    binary = os.path.join(REPO_ROOT, "target", "debug", "smg")
    if not os.path.exists(binary):
        binary = os.path.join(REPO_ROOT, "target", "release", "smg")
    if not os.path.exists(binary):
        pytest.skip(
            f"smg binary not found at target/{{debug,release}}/smg; "
            f"run `cargo build -p smg` from {REPO_ROOT} first"
        )
    metrics_port = _free_port()
    cmd = [
        binary, "start",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--worker-urls", mock_backend,
        "--policy", "thunder",
        "--thunder-sub-mode", "default",
        # Metrics server defaults to :29000; use a free port so we don't
        # collide with the other SMG fixture (smg_router) running in the
        # same pytest session.
        "--prometheus-port", str(metrics_port),
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        _wait_http(f"http://127.0.0.1:{port}/health", timeout=20)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
