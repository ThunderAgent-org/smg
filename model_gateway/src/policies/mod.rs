//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc};

use async_trait::async_trait;
use openai_protocol::worker::WorkerLoadResponse;
use smg_mesh::OptionalMeshSyncManager;
use tokio::sync::mpsc::UnboundedSender;

use crate::worker::{HashRing, Worker};

mod bucket;
mod cache_aware;
mod consistent_hashing;
mod dp_min_token;
mod factory;
mod manual;
mod power_of_two;
mod prefix_hash;
mod random;
mod registry;
mod round_robin;
mod thunder;
mod thunder_metrics;
pub(crate) mod utils;

pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use consistent_hashing::ConsistentHashingPolicy;
pub use dp_min_token::MinimumTokensPolicy;
pub use factory::PolicyFactory;
// Re-export PrefixMatchResult from kv_index for production use
pub use kv_index::PrefixMatchResult;
pub use manual::{ManualConfig, ManualPolicy};
pub use power_of_two::PowerOfTwoPolicy;
pub use prefix_hash::{PrefixHashConfig, PrefixHashPolicy};
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;
pub use thunder::{ProgramRequestGuard, ThunderPolicy};
pub use thunder_metrics::{BackendCapacity, HttpMetricsClient, MetricsClient};

/// Per-request usage event emitted by routers after the upstream stream completes.
///
/// Stateless policies ignore this; ThunderPolicy (Phase 3+) consumes it via the
/// `usage_sender` channel to update `BackendState.active_program_tokens` and the
/// per-program `char_to_token_ratio` calibration.
///
/// `request_text_chars` is captured by the router at admission time (length of
/// the value returned by `GenerationRequest::extract_text_for_routing`) so the
/// consumer can compute `tokens_per_char = total_tokens / request_text_chars`.
#[derive(Debug, Clone)]
pub struct UsageEvent {
    /// Program identifier this usage belongs to (None for non-program requests
    /// or when the client did not send `metadata.program_id`).
    pub program_id: Option<String>,
    /// Backend URL the request was routed to (matches `worker.url()`).
    pub backend_url: String,
    /// Prompt tokens reported by upstream usage payload.
    pub prompt_tokens: u32,
    /// Completion tokens reported by upstream usage payload.
    pub completion_tokens: u32,
    /// Sum of prompt + completion (kept explicit so consumers don't repeat the math).
    pub total_tokens: u32,
    /// Char-length of the routing-extracted request text (for char→token ratio calibration).
    pub request_text_chars: usize,
    /// Anthropic-only: tokens served from prompt cache (excluded from prefill ratio in M3).
    /// `None` for OpenAI Chat / Responses where this concept doesn't exist.
    pub cache_read_input_tokens: Option<u32>,
    /// Client-declared `max_tokens` (or equivalent: `max_completion_tokens` for
    /// OpenAI Chat, `max_output_tokens` for Responses). Used by M3 completion
    /// fraction calibration. `None` if the request did not specify a limit.
    pub declared_max_tokens: Option<u32>,
}

/// Per-progress event emitted during streaming (every ~20 tokens or per Anthropic
/// `message_delta`). ThunderPolicy consumes this via `streaming_progress_sender`
/// to update `Program.total_tokens` incrementally — mirrors Python's
/// `update_program_tokens_streaming`.
#[derive(Debug, Clone)]
pub struct StreamingProgressEvent {
    pub program_id: String,
    pub delta_tokens: u64,
}

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
#[async_trait]
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    ///
    /// # Arguments
    /// * `workers` - Available workers to select from
    /// * `info` - Additional information for routing decisions
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize>;

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &std::collections::HashMap<String, WorkerLoadResponse>) {
        // Default: no-op for policies that don't use load information
    }

    /// Set mesh sync manager
    fn set_mesh_sync(&mut self, _mesh_sync: OptionalMeshSyncManager) {
        // Default: no-op for policies that don't use mesh sync
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Async variant of `select_worker`. The default implementation delegates
    /// to `select_worker` so existing policies keep working unchanged. Policies
    /// that need to do async work during selection (e.g., ThunderPolicy
    /// awaiting a per-program Notify after a pause) override this method.
    ///
    /// ## Why both sync and async?
    ///
    /// Most policies (cache_aware, round_robin, etc.) make selection decisions
    /// from in-memory state and don't need `.await`. Forcing them to be async
    /// would add overhead and complicate their tests. ThunderPolicy is the
    /// outlier — it may pause selection until capacity frees — and it's the
    /// only thing that overrides the default.
    async fn select_worker_async(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select_worker(workers, info)
    }

    /// Optional usage-event sender. Routers fire-and-forget a `UsageEvent`
    /// after the upstream stream completes. Stateless policies return `None`
    /// (the default) and routers short-circuit the send. ThunderPolicy returns
    /// `Some(&self.usage_tx)` so it can update `BackendState.active_program_tokens`
    /// and per-program `char_to_token_ratio`.
    fn usage_sender(&self) -> Option<&UnboundedSender<UsageEvent>> {
        None
    }

    /// Optional incremental streaming-progress sender. ThunderPolicy returns
    /// `Some(&self.progress_tx)` so streaming relays can emit per-chunk token
    /// deltas without grabbing the RouterState write lock on the hot path.
    /// Mirrors `usage_sender` precedent (P1).
    fn streaming_progress_sender(&self) -> Option<&UnboundedSender<StreamingProgressEvent>> {
        None
    }
}

pub trait DPRankLoadPolicy: Send + Sync + Debug {
    fn select_dp_rank(&self, worker: &dyn Worker, estimated_cost: isize) -> Option<isize>;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    /// Backend KV cache block size (tokens per block) for event-driven routing.
    /// Used by `compute_request_content_hashes` to chunk request tokens into blocks.
    /// Must match the backend's block size. Default: 16 (SGLang page size).
    pub block_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
            block_size: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker_can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

/// Helper function to normalize model_id to a key for policy lookups.
///
/// Returns UNKNOWN_MODEL_ID for empty model_ids to ensure consistent behavior
/// across single-model and multi-model deployments.
#[inline]
pub(crate) fn normalize_model_key(model_id: &str) -> &str {
    if model_id.is_empty() {
        crate::worker::UNKNOWN_MODEL_ID
    } else {
        model_id
    }
}

/// Information passed to policy for worker selection
#[derive(Debug, Clone, Default)]
pub struct SelectWorkerInfo<'a> {
    /// Request text for cache-aware routing
    pub request_text: Option<&'a str>,
    /// Tokenized request for prefix-hash routing
    /// Used by PrefixHashPolicy for token-based prefix hashing
    pub tokens: Option<&'a [u32]>,
    /// HTTP headers for header-based routing policies
    /// Policies can extract routing information from headers like:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by index
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    pub headers: Option<&'a http::HeaderMap>,
    /// Pre-computed hash ring for O(log n) consistent hashing
    /// Built and cached by WorkerRegistry, passed through to avoid per-request rebuilds
    pub hash_ring: Option<Arc<HashRing>>,
    /// Program identifier extracted from the request body (typically from
    /// `metadata.program_id` for Anthropic Messages requests). Read by
    /// program-aware policies (Thunder) for capacity tracking. Default None
    /// keeps existing policies' behavior unchanged.
    pub program_id: Option<&'a str>,
    /// Client-declared completion budget (`max_tokens` / `max_completion_tokens`
    /// / `max_output_tokens` depending on protocol). Used by Thunder M3
    /// completion-fraction calibration to estimate the completion side of
    /// reserve. `None` for legacy clients that omit the field.
    pub declared_max_tokens: Option<u32>,
    /// M7: backend URL to exclude from selection (set on retry attempts so
    /// the retry doesn't land on the same backend that just failed). `None`
    /// for first attempts; populated by `RetryExecutor` via the route layer.
    pub avoid_backend: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_status(WorkerStatus::NotReady);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn usage_event_struct_exists_and_is_constructible() {
        let ev = UsageEvent {
            program_id: Some("p1".to_string()),
            backend_url: "http://w1:8001".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            request_text_chars: 400,
            cache_read_input_tokens: None,
            declared_max_tokens: None,
        };
        assert_eq!(ev.total_tokens, 150);
        assert_eq!(ev.program_id.as_deref(), Some("p1"));
    }

    #[test]
    fn select_worker_info_carries_program_id() {
        let pid = "agent-step-7";
        let info = SelectWorkerInfo {
            request_text: Some("hello"),
            tokens: None,
            headers: None,
            hash_ring: None,
            program_id: Some(pid),
            declared_max_tokens: None,
            avoid_backend: None,
        };
        assert_eq!(info.program_id, Some("agent-step-7"));
    }

    #[test]
    fn select_worker_info_default_program_id_is_none() {
        let info = SelectWorkerInfo::default();
        assert_eq!(info.program_id, None);
    }

    #[tokio::test]
    async fn select_worker_async_default_delegates_to_sync() {
        struct Stub;
        impl Debug for Stub {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Stub")
            }
        }
        #[async_trait::async_trait]
        impl LoadBalancingPolicy for Stub {
            fn select_worker(
                &self,
                workers: &[Arc<dyn Worker>],
                _info: &SelectWorkerInfo,
            ) -> Option<usize> {
                if workers.is_empty() { None } else { Some(0) }
            }
            fn name(&self) -> &'static str {
                "stub"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let stub = Stub;
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let info = SelectWorkerInfo::default();
        // Default async impl must delegate to sync — same answer for empty + non-empty.
        let sync_result = stub.select_worker(&workers, &info);
        let async_result = stub.select_worker_async(&workers, &info).await;
        assert_eq!(sync_result, async_result);
    }

    #[test]
    fn usage_sender_default_returns_none() {
        struct Stub;
        impl Debug for Stub {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Stub")
            }
        }
        #[async_trait::async_trait]
        impl LoadBalancingPolicy for Stub {
            fn select_worker(
                &self,
                _workers: &[Arc<dyn Worker>],
                _info: &SelectWorkerInfo,
            ) -> Option<usize> {
                None
            }
            fn name(&self) -> &'static str {
                "stub"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let stub = Stub;
        assert!(stub.usage_sender().is_none());
    }

    /// Helper: build N healthy mock workers.
    fn mock_workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://w{i}:8000"))
                        .worker_type(WorkerType::Regular)
                        .api_key("test")
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Helper: assert sync and async selection give compatible results.
    /// "Compatible" means either both return None, or both return Some(idx)
    /// where idx is a valid index into workers. We don't require exact equality
    /// because stateful and RNG-based policies may advance between calls.
    async fn assert_parity(
        policy: &dyn LoadBalancingPolicy,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) {
        let sync = policy.select_worker(workers, info);
        let asy = policy.select_worker_async(workers, info).await;
        match (sync, asy) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert!(a < workers.len(), "sync idx out of range: {a}");
                assert!(b < workers.len(), "async idx out of range: {b}");
            }
            (s, a) => panic!(
                "policy {} parity violated: sync={:?} async={:?}",
                policy.name(),
                s,
                a
            ),
        }
    }

    #[tokio::test]
    async fn round_robin_parity() {
        let policy = RoundRobinPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn random_parity() {
        let policy = RandomPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn power_of_two_parity() {
        let policy = PowerOfTwoPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn consistent_hashing_parity() {
        let policy = ConsistentHashingPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn cache_aware_parity() {
        let policy = CacheAwarePolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo {
            request_text: Some("hello world"),
            ..Default::default()
        };
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn bucket_parity() {
        let policy = BucketPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn prefix_hash_parity() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig::default());
        let workers = mock_workers(3);
        let tokens = [1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_parity(&policy, &workers, &info).await;
    }

    #[tokio::test]
    async fn manual_parity() {
        let policy = ManualPolicy::new();
        let workers = mock_workers(3);
        let info = SelectWorkerInfo::default();
        assert_parity(&policy, &workers, &info).await;
    }

    #[test]
    fn minimum_tokens_policy_is_dp_rank_policy_only() {
        let policy = MinimumTokensPolicy::new(None);
        let workers = mock_workers(1);
        assert!(policy.select_dp_rank(workers[0].as_ref(), 1).is_none());
    }
}
