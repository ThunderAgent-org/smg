//! Backend metrics fetcher used by `ThunderPolicy`'s capacity poll task.
//!
//! Today: HTTP-only — polls `/get_server_info` on each backend and reads the
//! KV-cache fields out of the response. gRPC-backed capacity arrives in P7
//! (gRPC validation phase) via a different `MetricsClient` impl injected at
//! construction time.
//!
//! ## Why a small dedicated struct instead of reusing `ServerInfo`?
//!
//! The existing `workflow::steps::local::discover_metadata::ServerInfo` is the
//! flat sglang `/server_info` shape (used to populate worker labels). It does
//! not carry the nested `cache_config` object that vLLM's `/get_server_info`
//! returns. Extending it would ripple through the `flat_labels` flow that
//! drives backend metadata discovery. Parsing a tiny local struct here keeps
//! the blast radius to this file. The mock vLLM at
//! `e2e_test/thunder/mock_vllm.py` returns exactly the shape we parse here.

use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};

/// Capacity snapshot returned by a metrics fetch.
#[derive(Debug, Clone)]
pub struct BackendCapacity {
    /// Total KV cache capacity in tokens. 0 means "could not determine".
    pub capacity_tokens: u64,
    /// Backend-reported model name (informational).
    pub model_name: Option<String>,
}

/// Trait abstracting backend capacity fetches so unit tests can inject a mock
/// without spinning up an HTTP server.
#[async_trait]
pub trait MetricsClient: Send + Sync + std::fmt::Debug {
    async fn fetch_capacity(&self, worker_url: &str) -> Result<BackendCapacity, String>;
}

#[derive(Debug, Default)]
pub struct HttpMetricsClient;

/// Minimal subset of vLLM's `/get_server_info` response that ThunderPolicy
/// consumes. Extra fields are ignored (`deny_unknown_fields = false` by
/// default), so this stays robust against backend version drift.
#[derive(Debug, Default, Deserialize)]
struct GetServerInfoResponse {
    #[serde(default)]
    cache_config: Option<CacheConfig>,
    #[serde(default)]
    model_config: Option<ModelConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct CacheConfig {
    /// Tokens per KV-cache block (e.g. 16).
    #[serde(default)]
    block_size: Option<u64>,
    /// Number of KV-cache blocks allocated on the GPU.
    #[serde(default)]
    num_gpu_blocks: Option<u64>,
    /// Convenience field exposed by some vLLM versions and by our mock —
    /// preferred over `block_size * num_gpu_blocks` when present.
    #[serde(default)]
    total_kv_cache_tokens: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelConfig {
    #[serde(default)]
    model: Option<String>,
}

/// Shared HTTP client — reqwest pools connections internally.
#[expect(
    clippy::expect_used,
    reason = "Lazy static initialization — reqwest::Client::build() only fails on TLS backend misconfiguration which is unrecoverable at startup"
)]
static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("Failed to build reqwest::Client for thunder_metrics")
});

/// Normalize `worker.url()` to an HTTP base. Workers may be registered as
/// `host:port` without scheme; reqwest needs an absolute URL.
fn http_base(url: &str) -> String {
    if url.starts_with("http://") || url.starts_with("https://") {
        url.trim_end_matches('/').to_string()
    } else {
        let stripped = url
            .trim_start_matches("grpc://")
            .trim_end_matches('/')
            .to_string();
        format!("http://{stripped}")
    }
}

#[async_trait]
impl MetricsClient for HttpMetricsClient {
    async fn fetch_capacity(&self, worker_url: &str) -> Result<BackendCapacity, String> {
        let base = http_base(worker_url);
        let url = format!("{base}/get_server_info");
        let resp = HTTP_CLIENT
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("connect {url}: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("status {} from {}", resp.status(), url));
        }
        let info = resp
            .json::<GetServerInfoResponse>()
            .await
            .map_err(|e| format!("parse {url}: {e}"))?;

        let capacity_tokens = info
            .cache_config
            .as_ref()
            .map(|c| {
                // Prefer total_kv_cache_tokens when the backend exposes it,
                // else derive from block_size * num_gpu_blocks. Either source
                // is enough for ThunderPolicy's TR-mode capacity gate (P5+).
                c.total_kv_cache_tokens.unwrap_or_else(|| {
                    c.block_size
                        .zip(c.num_gpu_blocks)
                        .map(|(b, n)| b.saturating_mul(n))
                        .unwrap_or(0)
                })
            })
            .unwrap_or(0);

        if capacity_tokens == 0 {
            warn!(worker_url, "fetch_capacity: backend returned 0 capacity");
        } else {
            debug!(worker_url, capacity_tokens, "fetch_capacity ok");
        }
        Ok(BackendCapacity {
            capacity_tokens,
            model_name: info.model_config.and_then(|m| m.model),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_base_handles_scheme_variants() {
        assert_eq!(http_base("http://w0:8000"), "http://w0:8000");
        assert_eq!(http_base("https://w0:8000/"), "https://w0:8000");
        assert_eq!(http_base("w0:8000"), "http://w0:8000");
        assert_eq!(http_base("grpc://w0:8000"), "http://w0:8000");
    }

    #[test]
    fn parse_full_cache_config_prefers_total() {
        let json = serde_json::json!({
            "cache_config": {
                "block_size": 16,
                "num_gpu_blocks": 100,
                "total_kv_cache_tokens": 9999,
            },
            "model_config": {"model": "mock-model"},
        });
        let info: GetServerInfoResponse = serde_json::from_value(json).expect("parse");
        let cc = info.cache_config.expect("cache_config");
        assert_eq!(cc.total_kv_cache_tokens, Some(9999));
        assert_eq!(cc.block_size, Some(16));
        assert_eq!(cc.num_gpu_blocks, Some(100));
        assert_eq!(info.model_config.and_then(|m| m.model).as_deref(), Some("mock-model"));
    }

    #[test]
    fn parse_only_block_and_num_blocks_derives_total() {
        let json = serde_json::json!({
            "cache_config": {
                "block_size": 16,
                "num_gpu_blocks": 100,
            }
        });
        let info: GetServerInfoResponse = serde_json::from_value(json).expect("parse");
        let cc = info.cache_config.expect("cache_config");
        assert_eq!(cc.total_kv_cache_tokens, None);
        assert_eq!(cc.block_size.unwrap_or(0) * cc.num_gpu_blocks.unwrap_or(0), 1600);
    }

    #[test]
    fn parse_missing_cache_config_yields_zero() {
        let json = serde_json::json!({"model_config": {"model": "x"}});
        let info: GetServerInfoResponse = serde_json::from_value(json).expect("parse");
        assert!(info.cache_config.is_none());
    }
}
