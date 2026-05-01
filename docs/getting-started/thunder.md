---
title: ThunderAgent (Program-Aware Routing)
---

# ThunderAgent Algorithm as a Routing Policy

[ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) is a **program-level** scheduling algorithm designed for agent workloads, making automatic tradeoffs between load-balancing and kv cache affinity. Instead of scheduling each request independently, ThunderAgent groups requests by `program_id` and routes requests from the same program to the same backend whenever possible — maximizing KV cache reuse — while pausing agentic programs' incoming requests based on backend's kv cache capacity to avoid thrashing - and resuming programs on backends with the least workload.

ThunderAgent algorithm is implemented as a routing policy in SMG.

```bash
smg --policy thunder --thunder-sub-mode tr \
    --worker-urls http://w1:8000 http://w2:8000 \
    --thunder-resume-timeout-secs 1800
```

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Two or more sglang or vLLM workers running with KV-cache metrics exposed
- Your client adds a `program_id` to each request (see "Client contract" below)

</div>

---

## When to use ThunderAgent

ReAct style agents with reusable prefix cache across requests (e.g. requests from agentic clients such as Claude Code, Codex CLI, OpenClaw, Hermes Agents, etc.)

---

## Sub-modes

ThunderAgent ships two sub-modes, switchable via `--thunder-sub-mode`:

| Sub-mode | Admission | Pause/Resume | When to use |
|---|---|---|---|
| `default` | Every request admits immediately on the sticky-or-least-active backend | None | Most deployments — start here |
| `tr` (transactional) | Capacity-gated; new requests pause if backend is over its capacity threshold | Yes, with proactive preemption + BFD greedy resume | When backend OOM is a real concern OR strict per-program isolation is required |

`tr` adds end-to-end pause/resume bookkeeping, force-resume safety timeout, and
proactive eviction of low-progress programs when capacity gets tight. Both
modes do program-sticky routing.

> **BFD** = **Best-Fit Decreasing**, a bin-packing approximation algorithm.
> ThunderAgent uses it to repack paused programs onto backends with available
> KV-cache capacity. See [What ThunderAgent does on each request](#what-thunderagent-does-on-each-request)
> below for details.

---

## Configuration

```bash
smg --policy thunder \
    --thunder-sub-mode tr \
    --thunder-capacity-reserved-fraction 0.10 \
    --thunder-resume-timeout-secs 1800 \
    --thunder-scheduler-tick-ms 100 \
    --thunder-capacity-poll-interval-secs 5 \
    --worker-urls http://w1:8000 http://w2:8000
```

| Flag | Default | Description |
|---|---|---|
| `--thunder-sub-mode` | `default` | `default` (sticky least-active) or `tr` (capacity-gated). |
| `--thunder-capacity-reserved-fraction` | `0.10` | Fraction of each backend's KV-cache capacity kept free as headroom. `tr` mode pauses incoming requests when remaining capacity falls below this. |
| `--thunder-resume-timeout-secs` | `1800` | Maximum seconds a paused program will wait for capacity before being force-admitted to the least-active backend regardless of capacity. Prevents indefinite hangs. |
| `--thunder-scheduler-tick-ms` | `100` | Interval (ms) at which the scheduler runs proactive-pause + BFD-resume passes. Lower = faster reaction to capacity changes; higher = less lock contention. Default 100ms is appropriate for most deployments. |
| `--thunder-capacity-poll-interval-secs` | `5` | How often ThunderAgent polls `/get_server_info` on each backend to refresh KV-cache capacity numbers. |

**Tuning tips**

- If you see frequent `thunder TR pause (full)` log lines under steady load,
  raise `--thunder-capacity-reserved-fraction` (more headroom) or scale workers.
- If pauses last more than ~10 seconds in normal operation, your backend
  capacity numbers may be stale — lower `--thunder-capacity-poll-interval-secs`.
- If scheduler-tick CPU > 5% on idle traffic, raise `--thunder-scheduler-tick-ms`
  to 200ms.

---

## Client contract: program_id

ThunderAgent needs each request to declare which **program** it belongs to.
A program is just an opaque string identifier — typically your agent's session
ID, tool execution ID, or whatever logical unit groups consecutive requests.

### Anthropic Messages API (`/v1/messages`)

Send `program_id` as part of the request `metadata`:

```http
POST /v1/messages
Content-Type: application/json

{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1024,
  "stream": true,
  "metadata": {
    "program_id": "agent-session-42"
  },
  "messages": [
    {"role": "user", "content": "..."}
  ]
}
```

This is the **only protocol** with built-in `program_id` support today.

### OpenAI Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`)

These protocols do **not** have a native `program_id` field in their request
schema. Without one, ThunderAgent treats every request as belonging to a single
synthetic `default` program — meaning no per-program isolation.

Workarounds while a native field is added in a future SMG release:

- **Recommended**: route OpenAI-format streaming traffic through a small adapter
  that translates client requests into the Anthropic Messages format with
  `metadata.program_id`. This is what most agent frameworks already do.
- **Alternative**: use a different policy (`cache_aware` or `consistent_hashing`)
  for these endpoints, and reserve ThunderAgent for `/v1/messages` traffic.

### Streaming behavior

For streaming requests, ThunderAgent additionally:

- Forces `stream_options.include_usage=true` on outbound OpenAI Chat requests
  so it can extract usage at stream end (transparent — the usage chunk is
  stripped from the response if your client did not originally request it).
- Reads cumulative `output_tokens` from each `message_delta` event for
  Anthropic Messages — more accurate than counting events.
- Reads `usage` from the `response.completed` event for OpenAI Responses.

Three protocols (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) all
participate in ThunderAgent state when streaming.

---

## What ThunderAgent does on each request

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Extract program_id from request metadata                              │
│ 2. Lookup sticky backend for this program                                │
│    └── if assigned and healthy: prefer it                                │
│    └── else: pick least-active backend                                   │
│ 3. (TR only) Check backend has capacity for estimated tokens             │
│    └── if yes: reserve estimate, increment in_flight, admit              │
│    └── if no: pause; await scheduler wake or force-resume timeout        │
│ 4. Forward request to backend                                            │
│ 5. (Streaming) Extract usage at stream end, emit UsageEvent              │
│ 6. usage_consumer un-reserves estimate, adds actual tokens, decrements   │
│    in_flight, calibrates per-program chars/token ratio                   │
└──────────────────────────────────────────────────────────────────────────┘
```

In parallel, a 100ms scheduler tick runs:

- **Proactive pause**: any backend over `1 - capacity_reserved_fraction` of its
  capacity gets one of its lowest-progress programs paused and unbooked.
- **BFD greedy resume**: paused programs are placed back onto backends using a
  **Best-Fit Decreasing** bin-packing heuristic — the classic approximation
  algorithm for the bin-packing problem. Concretely: paused programs are
  sorted by their estimated KV-cache footprint (largest first; programs paused
  longer than 15 minutes get a priority boost so they are not starved), and
  each program is then placed on the backend with the **smallest remaining
  capacity that still fits it**. Putting big programs first onto tight-fit
  backends keeps free capacity contiguous on the rest of the fleet, which
  leaves room for the next batch of incoming traffic. A program that does
  not fit on any backend stays paused and is retried on the next tick.

---

## Limitations & known gaps

- **OpenAI Chat / Responses lack native `program_id`** — see "Client contract"
  above. Workaround: use Anthropic Messages format end-to-end if you need
  per-program isolation across all three protocols.
- **gRPC backends not yet validated**: ThunderAgent is HTTP-only. gRPC support
  arrives in a future Phase 7+.
- **Profiling endpoints (`/thunder/programs`, `/thunder/profiles`) deferred**:
  inspect state via SMG logs with `RUST_LOG=smg::policies::thunder=debug`.

---

## Observability

ThunderAgent emits `tracing` events under the `smg::policies::thunder` and
`smg::sse` targets. Enable them with:

```bash
RUST_LOG=smg::policies::thunder=debug,smg::sse=debug smg --policy thunder ...
```

Key log lines to watch in production:

- `thunder TR pause (full)` — backend over capacity threshold; a paused program
  was unbooked. Frequent occurrences = raise `--thunder-capacity-reserved-fraction`
  or scale workers.
- `thunder TR resume (BFD)` — paused program was resumed onto a backend with
  capacity. Long gaps between pause and resume = check backend KV-cache
  recovery or lower `--thunder-capacity-poll-interval-secs`.
- `thunder force resume (timeout)` — a paused program exceeded
  `--thunder-resume-timeout-secs` and was admitted regardless of capacity. If
  you see these in normal operation, your capacity numbers are too tight.
