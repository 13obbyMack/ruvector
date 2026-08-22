//! Error contract for scoped context retrieval.

/// Result returned by the scoped context index.
pub type Result<T> = core::result::Result<T, ContextIndexError>;

/// Fail-closed errors from scope validation, persistence, or retrieval.
#[derive(Debug, thiserror::Error)]
pub enum ContextIndexError {
    /// A namespace or path component violates the canonical scope contract.
    #[error("invalid context scope: {0}")]
    InvalidScope(&'static str),
    /// A point identifier is empty, too large, or contains unsafe bytes.
    #[error("invalid context point identifier")]
    InvalidPointId,
    /// An embedding contains NaN or infinity.
    #[error("context vectors must contain only finite values")]
    NonFiniteVector,
    /// A configured resource limit is zero or otherwise inconsistent.
    #[error("invalid context index configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// A query requested too many results.
    #[error("requested {requested} results, maximum is {maximum}")]
    ResultLimit {
        /// Requested top K value.
        requested: usize,
        /// Configured maximum top K value.
        maximum: usize,
    },
    /// A search would cross too many authorized descendant shards.
    #[error("scope fanout {actual} exceeds maximum {maximum}")]
    ScopeFanout {
        /// Number of matching shards.
        actual: usize,
        /// Configured maximum shard fanout.
        maximum: usize,
    },
    /// Creating another exact-scope shard would exceed the configured bound.
    #[error("context scope capacity {maximum} exhausted")]
    ScopeCapacity {
        /// Configured maximum shard count.
        maximum: usize,
    },
    /// An immutable point ID was reused with different vector bytes.
    #[error("context point ID already exists with different bytes")]
    ImmutableConflict,
    /// A discovered shard lacks a valid, hash-bound scope manifest.
    #[error("corrupt context shard: {0}")]
    CorruptShard(String),
    /// A shared index lock was poisoned by a panicking caller.
    #[error("context index lock poisoned")]
    LockPoisoned,
    /// Filesystem persistence failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Scope manifest encoding or decoding failed.
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    /// The underlying vector engine refused an operation.
    #[error(transparent)]
    Vector(#[from] ruvector_core::RuvectorError),
}
