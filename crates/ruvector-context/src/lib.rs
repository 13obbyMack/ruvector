//! Physically isolated vector retrieval for governed context namespaces.
//!
//! Metadata filtering after approximate-nearest-neighbor traversal is not a
//! tenant boundary. This crate places each exact context scope in a distinct
//! vector index and selects authorized descendant shards before any ANN call.
//!
//! # This crate is not the tenant boundary
//!
//! [`ScopedContextIndex`] enforces *physical* isolation between scopes; it does
//! not authenticate or authorize anyone. Every method — [`ScopedContextIndex::insert`],
//! [`ScopedContextIndex::search`], [`ScopedContextIndex::delete_point`],
//! [`ScopedContextIndex::erase_scope`], and [`ScopedContextIndex::scope_stats`]
//! — treats the [`ContextScope`] it is handed as already authorized for the
//! requesting principal. Deciding which principal may name which scope is the
//! caller's responsibility, and it must happen before the call. In particular
//! [`ScopedContextIndex::scope_stats`] is an existence oracle that reports
//! exact vector counts, so it needs the same authorization as a read.
//!
//! The scope manifest stored in each shard binds a scope to a *filename*, not
//! to shard contents; there is no message authentication code over the stored
//! vectors.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod index;
mod options;
mod quarantine;
mod scope;
mod shard;

pub use error::{ContextIndexError, Result};
pub use index::{ContextMatch, ContextPoint, ScopeStats, ScopedContextIndex};
pub use options::{ContextIndexOptions, MAX_RESULT_LIMIT};
pub use quarantine::QuarantinedShard;
pub use scope::{ContextNamespace, ContextScope};
