//! Physically isolated vector retrieval for governed context namespaces.
//!
//! Metadata filtering after approximate-nearest-neighbor traversal is not a
//! tenant boundary. This crate places each exact context scope in a distinct
//! vector index and selects authorized descendant shards before any ANN call.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod index;
mod scope;

pub use error::{ContextIndexError, Result};
pub use index::{ContextIndexOptions, ContextMatch, ContextPoint, ScopeStats, ScopedContextIndex};
pub use scope::{ContextNamespace, ContextScope};
