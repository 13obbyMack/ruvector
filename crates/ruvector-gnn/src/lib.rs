//! # RuVector GNN
//!
//! Graph Neural Network capabilities for RuVector, providing tensor operations,
//! GNN layers, compression, and differentiable search.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod compress;
pub mod error;
pub mod layer;
pub mod query;
pub mod search;
pub mod tensor;
pub mod training;

#[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
pub mod mmap;

// Re-export commonly used types
pub use compress::{CompressedTensor, CompressionLevel, TensorCompress};
pub use error::{GnnError, Result};
pub use layer::RuvectorLayer;
pub use query::{QueryMode, QueryResult, RuvectorQuery, SubGraph};
pub use search::{cosine_similarity, differentiable_search, hierarchical_forward};
pub use training::{info_nce_loss, local_contrastive_loss, sgd_step, OnlineConfig, TrainConfig};

#[cfg(all(not(target_arch = "wasm32"), feature = "mmap"))]
pub use mmap::{AtomicBitmap, MmapGradientAccumulator, MmapManager};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic smoke test to ensure the crate compiles
        assert!(true);
    }
}
