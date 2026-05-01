---
title: ThunderAgent (Program-Aware Routing)
---

# ThunderAgent — Program-Aware Routing for Agentic Workloads

ThunderAgent is a **program-level** scheduling algorithm for agentic LLM
workloads. Where SMG's other policies make a routing decision per request,
ThunderAgent makes decisions per **program** — an opaque identifier that
groups consecutive requests belonging to the same agent task. Pinning every
request from a program to the same backend turns long ReAct-style trajectories
into a near-perfect KV-cache hit, while a capacity-aware scheduler pauses and
resumes programs across backends to keep the fleet from thrashing.

This page describes the algorithm and how it is implemented in SMG. For the
operator-facing setup guide, see
[Getting Started: ThunderAgent](../../getting-started/thunder.md).

> **Reference implementation**: ThunderAgent's algorithm originated in the
> Python project [ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent).
> SMG's `--policy thunder` is a Rust port that follows the Python reference
> closely while integrating with SMG's existing routing trait. Reported
> throughput gains on real agent traces (SWE-Agent, OpenHands, ToolOrchestra)
> are 1.5–3.6× over per-request load balancing.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-identifier: Program-Sticky Routing

A `program_id` on each request groups requests into one logical agent task.
All requests from the same program prefer the same backend, so the prompt
prefix the agent has built up stays warm in that worker's KV cache.

</div>

<div class="card" markdown>

### :material-scale-balance: Capacity-Aware Admission

In `tr` (transactional) sub-mode, a new request is admitted only when its
target backend has enough free KV-cache for the program's estimated
footprint. Otherwise the program **pauses** until capacity returns.

</div>

<div class="card" markdown>

### :material-package-variant: Best-Fit Decreasing Resume

A 100 ms scheduler tick repacks paused programs onto backends with free
capacity using a Best-Fit Decreasing bin-packing heuristic. Largest
programs go first onto the tightest fit, leaving big contiguous holes
for incoming traffic.

</div>

<div class="card" markdown>

### :material-tune: Self-Calibrating Token Estimates

Per-program EMA of `chars/token` ratio and completion fraction —
half-life decayed — lets ThunderAgent estimate the next request's
KV-cache reservation without trusting the user's `max_tokens`.

</div>

</div>

---

## Why ThunderAgent?

LLM agents do not look like chat: they emit many short, prefix-heavy requests
in rapid succession, separated by tool calls. Routing each request
independently fragments the prompt prefix across workers, multiplying prefill
cost and starving the cache. At the same time, agents have wildly variable
running lengths — a 30-step trajectory next to a 3-step one — so static
sharding by user or session causes pile-ups on whichever worker drew the long
program.

<div class="grid" markdown>

<div class="card" markdown>

### :material-robot: Long ReAct Trajectories

Tens to hundreds of requests per program, each appending one observation /
thought / action. The shared prefix grows monotonically — exactly the
shape KV-cache reuse rewards.

</div>

<div class="card" markdown>

### :material-tools: Tool-Heavy Agent Frameworks

Mini-SWE-Agent, OpenHands, Search-R1-style retrieval agents, Codex-CLI,
Claude Code: long-running clients that issue many requests under a
single session ID.

</div>

<div class="card" markdown>

### :material-server-network: Multi-Backend KV Pressure

When agents share a fleet, one runaway program can fill a worker's KV-cache
and trigger evictions that hurt every other program on it. ThunderAgent
proactively pauses low-progress programs before that happens.

</div>

<div class="card" markdown>

### :material-language-python: ML Training & RL Rollouts

Off-policy rollouts in agentic RL (slime, SkyRL) issue thousands of
parallel programs with diverse lengths. Program-aware scheduling keeps
the worker pool busy without one slow program blocking a whole rollout.

</div>

</div>

**The core insight:** an agent program's *sequence* of requests is the unit
that matters for cache locality, not the individual request. Cache-aware
prefix matching reacts to whatever already happened to land on a worker;
ThunderAgent decides up front where each program lives, then keeps it there.

---

## Algorithm

ThunderAgent's algorithm has four moving parts: a per-program lifecycle, a
per-request admission check, a periodic scheduler tick, and a calibration
loop fed by request-end usage events.

### Program Lifecycle

Each program tracked by ThunderAgent has a status:

| Status | Meaning |
|---|---|
| `Idle` | First request not yet seen, or program was just released. No backend reservation. |
| `Reasoning` | Currently has an in-flight request being served. KV-cache is being written. |
| `Acting` | Between requests — agent is running a tool or waiting on the user. KV-cache is held but quiescent. |
| `Paused` | Has been kicked off a backend due to capacity pressure. Waiting for a backend with space. |

Transitions:

```
                ┌─────── new program ─────────┐
                ▼                              │
              Idle ────first request───▶  Reasoning
                                              │
                request finishes ─────────────┤
                                              ▼
                                           Acting
                                              │
                next request arrives ─────────┤
                                              ▼
                                           Reasoning ◀──┐
                                              │         │
                  proactive_pause / oversub ───┤         │ wake & resume
                                              ▼         │
                                            Paused ─────┘
```

The Acting state matters: a program with no in-flight request still **owns**
its KV-cache footprint on the backend. ThunderAgent counts active reservations
per backend, not just in-flight requests, because the next request from a
program in Acting will land on that backend.

### Per-Request Admission (TR sub-mode)

When a request with `program_id = P` arrives at SMG:

```
1. Look up program P in the router state.
   - If P has a sticky backend assigned and the backend is healthy → use it.
   - Else → pick the backend with the lowest current load.

2. Estimate the request's KV-cache footprint:
       est_tokens = prefix_len + max_tokens * completion_fraction(P)
   where:
       prefix_len  = char_to_token_ratio(P) * len(prompt_chars)
       max_tokens  = the request's declared max_tokens
       completion_fraction(P)
                   = EMA estimate of "what fraction of max_tokens does P
                     actually generate" (defaults to 0.5)

3. Reserve est_tokens on the chosen backend.
   - If backend.active_tokens + est_tokens ≤ capacity * (1 - reserved_frac):
         book the reservation; transition P → Reasoning; admit request.
   - Else:
         pause P, wait on a tokio::sync::Notify keyed by program_id.
         A scheduler tick (or a usage event from another program freeing
         capacity) will set the Notify when this program can fit somewhere.
         When woken, retry from step 1 — possibly on a *different* backend.

4. Forward the request. The wait at step 3 is bounded by
   --thunder-resume-timeout-secs; on timeout, force-admit on the
   least-active backend regardless of capacity (safety valve).
```

In `default` sub-mode, step 3 is skipped: every request is admitted
immediately on its sticky-or-least-active backend. The state machine still
runs (so calibration and stickiness still work), but pause/resume is disabled.

### Scheduler Tick (TR sub-mode)

Every `--thunder-scheduler-tick-ms` (default 100 ms), an async task runs two
passes over the router state:

#### Proactive pause

For each backend whose `active_tokens` exceeds its high-water mark
(`capacity × (1 − reserved_frac)`):

1. Find the program on that backend with the **lowest progress**
   (smallest cumulative `output_tokens`).
2. If it is in `Acting` state, transition it to `Paused`, un-reserve its
   tokens, clear its sticky assignment, and emit a `WAITING` entry keyed by
   its `program_id`.
3. If it is in `Reasoning` state (request currently in flight), set a
   `marked_for_pause` flag and pause it after the request completes — never
   tear down a request mid-stream.

This keeps backends from running into hard KV-cache eviction.

#### BFD greedy resume

Among all paused programs (across the whole fleet), pack them onto backends
using **Best-Fit Decreasing**:

1. Sort paused programs by estimated KV-cache footprint, **largest first**.
   Programs that have been paused for more than 15 minutes get a priority
   boost so they are not starved by a steady stream of larger newer programs.
2. For each program, find the backend whose **remaining capacity is smallest
   but still ≥ the program's estimate**.
3. Resume the program on that backend: clear the paused flag, reserve its
   tokens, set the sticky pointer, and `notify_one()` the program's
   `Notify` so any blocked admission wakes up.
4. Programs that do not fit on any backend stay paused; the next tick
   retries.

Best-Fit Decreasing is a classical bin-packing approximation algorithm,
guaranteed within ~22% of the optimal packing for the offline problem. It
fits ThunderAgent's needs because (a) putting big programs first onto tight
fits leaves contiguous free space for incoming traffic, and (b) the offline
formulation matches the once-per-tick batch decision.

### Calibration

After every request — streaming or non-streaming — ThunderAgent receives a
`UsageEvent` with the actual `prompt_tokens`, `completion_tokens`, and the
declared `max_tokens`. Two scalars per program (with global fallbacks) are
updated using EMA with wall-time half-life decay:

| Statistic | Updated from | Used to estimate |
|---|---|---|
| `chars / prompt_token` | `len(prompt_chars) / prompt_tokens` | future prompt token counts before tokenizing |
| `completion_tokens / max_tokens` | `output_tokens / declared_max_tokens` | future output-side reservations |

Both decay toward a neutral prior with a 1-hour half-life so that a program's
old behavior fades as new requests arrive. For Anthropic Messages,
`cache_read_input_tokens` are excluded from `prompt_tokens` so prefix-cache
hits do not skew the chars/token ratio.

For streaming requests, ThunderAgent extracts usage incrementally (per
protocol — see below), and emits a `StreamingProgressEvent` every 20 output
tokens so the scheduler tick can preempt long-running programs in Acting
state if a backend gets tight.

---

## Implementation in SMG

ThunderAgent ships as a `LoadBalancingPolicy` implementation, alongside
`cache_aware`, `consistent_hashing`, etc. The policy is selected with
`--policy thunder`. All ThunderAgent-specific state is encapsulated behind
the existing trait; the rest of the SMG router pipeline does not need to
know about programs.

### Component Map

```
                             ┌────────────────────────────────────┐
                             │  HTTP / gRPC routers               │
                             │   - extract program_id from         │
                             │     metadata.program_id (Anthropic)│
                             │   - call select_worker_async        │
                             │   - relay request, optionally       │
                             │     wrap response in SseExtractor   │
                             └──────────────┬─────────────────────┘
                                            │
                                            ▼ SelectWorkerInfo
                             ┌────────────────────────────────────┐
                             │  ThunderPolicy                      │
                             │   ┌─────────────────────────────┐  │
                             │   │ RouterState (RwLock)         │  │
                             │   │  - programs: HashMap         │  │
                             │   │  - backends: HashMap         │  │
                             │   │  - waiting_events: Notify    │  │
                             │   │  - calibration: EMA + decay  │  │
                             │   └─────────────────────────────┘  │
                             │                                     │
                             │   spawned tasks:                    │
                             │   - capacity_poll  (every 5s)       │
                             │   - scheduler_tick (every 100ms)    │
                             │   - usage_consumer (mpsc)           │
                             └─────┬──────────────┬──────────────┘
                                   │              │
                                   │              │ on streaming response:
                                   │              ▼
                                   │   ┌────────────────────────┐
                                   │   │  sse module             │
                                   │   │  - openai_chat parser   │
                                   │   │  - anthropic parser     │
                                   │   │  - responses parser     │
                                   │   │  - emit Usage + Progress│
                                   │   └────────────────────────┘
                                   ▼
                             ┌────────────────────────────────────┐
                             │ HttpMetricsClient.get_server_info  │
                             │  → backend KV-cache capacity numbers│
                             └────────────────────────────────────┘
```

The single `RwLock<RouterState>` is intentional: ThunderAgent's decisions
need a coherent view of programs *and* backends in the same critical
section, so multi-mutex sharding would either reintroduce races or require
a global ordering anyway. Hot paths hold the lock for under a microsecond
because the data structures are small (one entry per active program /
backend, not per request).

### The `LoadBalancingPolicy` Trait Extensions

To make ThunderAgent work without breaking other policies, the trait gained
three small additions, each with a default that no-ops for non-ThunderAgent
policies:

| Method | Default behavior | ThunderAgent uses it for |
|---|---|---|
| `select_worker_async(&self, info: SelectWorkerInfo<'_>)` | Calls the existing sync `select_worker` | Async sticky lookup + capacity wait |
| `usage_sender(&self) -> Option<Sender<UsageEvent>>` | Returns `None` | Channel to feed end-of-request usage into the calibration loop |
| `streaming_progress_sender(&self) -> Option<Sender<StreamingProgressEvent>>` | Returns `None` | Channel for incremental token deltas during streams |

`SelectWorkerInfo` carries the request's `program_id` (extracted by the
router from `metadata.program_id` for Anthropic Messages, or
`"default"` for protocols that don't expose it), the prompt text length,
and the declared `max_tokens`.

### Per-Request Resource Lifetime

A request's reservation is owned by a `ProgramRequestGuard` RAII value.
Whichever way the request leaves the system — success, error, or client
disconnect mid-stream — the guard's `Drop` un-reserves the tokens and
decrements `in_flight`, so capacity bookkeeping cannot leak. The streaming
fast path explicitly calls `guard.complete()` on stream end to suppress the
fallback `Drop` un-reserve and instead apply the precise token counts from
the usage event.

### Streaming Usage Extraction

Three protocols emit usage information differently:

| Protocol | Where usage lives | Extra handling |
|---|---|---|
| OpenAI Chat (`/v1/chat/completions`) | A trailing chunk with `usage` after `stream_options.include_usage=true` | If the client did not opt in, ThunderAgent forces it on and strips the trailing chunk before forwarding |
| Anthropic Messages (`/v1/messages`) | Cumulative `output_tokens` on every `message_delta` event; `input_tokens` and `cache_read_input_tokens` on `message_start` | Cumulative reads avoid drift from per-event counting |
| OpenAI Responses (`/v1/responses`) | Final `response.completed` event carries `usage` | Single end-of-stream extraction |

The `model_gateway/src/sse/` module is a stand-alone state-machine library
(no `tokio`, no Thunder dependencies in its core) that splits each
protocol's parser into its own file and runs incrementally over byte
chunks. The router wires it in only when `--policy thunder` is active.

---

## Comparison with Other Policies

| Aspect | `thunder` | `cache_aware` | `consistent_hashing` | `random` |
|---|---|---|---|---|
| **Decision unit** | Program (sequence of requests) | Token prefix | Hash key | Per request |
| **Cache locality** | Excellent for agentic flows; perfect for repeated programs | Excellent for any prefix-shared traffic | Good when keys correlate with prefixes | None |
| **Cross-backend rebalancing** | Active (BFD resume) | Passive (load thresholds) | None | None |
| **Capacity awareness** | Yes (TR sub-mode) | No | No | No |
| **Memory cost** | O(programs + backends) | O(unique tokens) | O(workers) | O(1) |
| **Best for** | Long agent trajectories | Mixed workloads | Sticky users with no notion of program | Stateless / smoke tests |

The two are complementary, not competing: `cache_aware` is the right
default for general traffic where requests do not declare a program;
`thunder` is the right pick when the client *can* tag agent sessions with
a `program_id` and the workload is dominated by long, prefix-heavy
trajectories.

---

## Tuning Guidelines

| Setting | Default | When to change |
|---|---|---|
| `--thunder-sub-mode` | `default` | Switch to `tr` when worker OOM is observed under spike load, or when isolation between programs is a hard requirement. |
| `--thunder-capacity-reserved-fraction` | `0.10` | Raise (e.g. `0.20`) if you see frequent `thunder TR pause (full)` log lines. Lower if pauses are rare and you want to push utilization up. |
| `--thunder-resume-timeout-secs` | `1800` | Lower if your application has its own request timeout shorter than 30 min. Always keep it strictly less than your client's HTTP read timeout. |
| `--thunder-scheduler-tick-ms` | `100` | Raise to `200` if scheduler-tick CPU > 5% on idle traffic. Lower to `50` for very latency-sensitive deployments. |
| `--thunder-capacity-poll-interval-secs` | `5` | Lower if pause durations exceed 10 s under steady load — your capacity numbers are stale. |

---

## Monitoring

ThunderAgent emits structured `tracing` events under
`smg::policies::thunder` and `smg::sse`. Enable with
`RUST_LOG=smg::policies::thunder=debug,smg::sse=debug`.

Key signals to watch:

| Log line | Healthy frequency | What it means |
|---|---|---|
| `thunder TR pause (full)` | Rare under normal load | A program was paused because its backend was over the high-water mark. Frequent → raise reserved fraction or scale workers. |
| `thunder TR resume (BFD)` | Should pair 1:1 with pauses | A paused program found a backend with capacity. |
| `thunder force resume (timeout)` | Should be ~0 in production | A program waited more than `resume_timeout` and was force-admitted. Indicates capacity is genuinely undersized. |
| `thunder calibration update` (debug) | Once per request | Per-program EMA updated. Useful for diagnosing why a program's token estimate is off. |

For programmatic access to per-program state, the `ThunderPolicy::snapshot()`
method (used internally by tests) returns a debug-only view of the router
state. Profiling endpoints (`/thunder/programs`, `/thunder/profiles`) are
planned for a future release.

---

## Limitations

- **OpenAI Chat / Responses lack a native `program_id`**: those endpoints
  fall back to a single synthetic `default` program when used with
  `--policy thunder`, defeating per-program isolation. Anthropic Messages
  carries `metadata.program_id` natively. A native field on OpenAI-shape
  protocols is tracked for a future SMG release.
- **gRPC backends not yet validated end-to-end**: ThunderAgent's gRPC
  selection is wired but exercised only in tests. HTTP backends
  (sglang, vLLM) are the supported target today.
- **Single-instance state**: ThunderAgent state is per-SMG-process and not
  yet shared via mesh HA. Behind a mesh of SMG instances, use sticky
  routing at the load balancer (e.g. hash by `program_id` or by user) so
  a program lands consistently on one SMG.

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-rocket-launch: Set It Up

Operator-side configuration, CLI flags, and the `program_id` client
contract.

[Getting Started: ThunderAgent →](../../getting-started/thunder.md)

</div>

<div class="card" markdown>

### :material-cached: Compare With Cache-Aware

When prefix-tree routing fits your workload better than program-aware.

[Cache-Aware Routing →](cache-aware.md)

</div>

<div class="card" markdown>

### :material-scale-balance: Survey Other Policies

The full menu of routing policies and when each one wins.

[Load Balancing →](load-balancing.md)

</div>

</div>
