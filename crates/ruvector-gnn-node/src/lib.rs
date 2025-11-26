//! Node.js bindings for Ruvector GNN via NAPI-RS
//!
//! This module provides JavaScript bindings for the Ruvector GNN library,
//! enabling graph neural network operations, tensor compression, and
//! differentiable search in Node.js applications.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_gnn::{
    compress::{
        CompressedTensor as RustCompressedTensor, CompressionLevel as RustCompressionLevel,
        TensorCompress as RustTensorCompress,
    },
    layer::RuvectorLayer as RustRuvectorLayer,
    search::{
        differentiable_search as rust_differentiable_search,
        hierarchical_forward as rust_hierarchical_forward,
    },
};

// ==================== RuvectorLayer Bindings ====================

/// Graph Neural Network layer for HNSW topology
#[napi]
pub struct RuvectorLayer {
    inner: RustRuvectorLayer,
}

#[napi]
impl RuvectorLayer {
    /// Create a new Ruvector GNN layer
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input node embeddings
    /// * `hidden_dim` - Dimension of hidden representations
    /// * `heads` - Number of attention heads
    /// * `dropout` - Dropout rate (0.0 to 1.0)
    ///
    /// # Example
    /// ```javascript
    /// const layer = new RuvectorLayer(128, 256, 4, 0.1);
    /// ```
    #[napi(constructor)]
    pub fn new(input_dim: u32, hidden_dim: u32, heads: u32, dropout: f64) -> Result<Self> {
        if dropout < 0.0 || dropout > 1.0 {
            return Err(Error::new(
                Status::InvalidArg,
                "Dropout must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            inner: RustRuvectorLayer::new(
                input_dim as usize,
                hidden_dim as usize,
                heads as usize,
                dropout as f32,
            ),
        })
    }

    /// Forward pass through the GNN layer
    ///
    /// # Arguments
    /// * `node_embedding` - Current node's embedding
    /// * `neighbor_embeddings` - Embeddings of neighbor nodes
    /// * `edge_weights` - Weights of edges to neighbors
    ///
    /// # Returns
    /// Updated node embedding
    ///
    /// # Example
    /// ```javascript
    /// const node = [1.0, 2.0, 3.0, 4.0];
    /// const neighbors = [[0.5, 1.0, 1.5, 2.0], [2.0, 3.0, 4.0, 5.0]];
    /// const weights = [0.3, 0.7];
    /// const output = layer.forward(node, neighbors, weights);
    /// ```
    #[napi]
    pub fn forward(
        &self,
        node_embedding: Vec<f64>,
        neighbor_embeddings: Vec<Vec<f64>>,
        edge_weights: Vec<f64>,
    ) -> Result<Vec<f64>> {
        // Convert f64 to f32
        let node_f32: Vec<f32> = node_embedding.iter().map(|&x| x as f32).collect();
        let neighbors_f32: Vec<Vec<f32>> = neighbor_embeddings
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        let weights_f32: Vec<f32> = edge_weights.iter().map(|&x| x as f32).collect();

        let result = self.inner.forward(&node_f32, &neighbors_f32, &weights_f32);

        // Convert back to f64
        Ok(result.iter().map(|&x| x as f64).collect())
    }

    /// Serialize the layer to JSON
    #[napi]
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Serialization error: {}", e),
            )
        })
    }

    /// Deserialize the layer from JSON
    #[napi(factory)]
    pub fn from_json(json: String) -> Result<Self> {
        let inner: RustRuvectorLayer = serde_json::from_str(&json).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Deserialization error: {}", e),
            )
        })?;
        Ok(Self { inner })
    }
}

// ==================== TensorCompress Bindings ====================

/// Compression level for tensor compression
#[napi(object)]
pub struct CompressionLevelConfig {
    /// Type of compression: "none", "half", "pq8", "pq4", "binary"
    pub level_type: String,
    /// Scale factor (for "half" compression)
    pub scale: Option<f64>,
    /// Number of subvectors (for PQ compression)
    pub subvectors: Option<u32>,
    /// Number of centroids (for PQ8)
    pub centroids: Option<u32>,
    /// Outlier threshold (for PQ4)
    pub outlier_threshold: Option<f64>,
    /// Binary threshold (for binary compression)
    pub threshold: Option<f64>,
}

impl CompressionLevelConfig {
    fn to_rust(&self) -> Result<RustCompressionLevel> {
        match self.level_type.as_str() {
            "none" => Ok(RustCompressionLevel::None),
            "half" => Ok(RustCompressionLevel::Half {
                scale: self.scale.unwrap_or(1.0) as f32,
            }),
            "pq8" => Ok(RustCompressionLevel::PQ8 {
                subvectors: self.subvectors.unwrap_or(8) as u8,
                centroids: self.centroids.unwrap_or(16) as u8,
            }),
            "pq4" => Ok(RustCompressionLevel::PQ4 {
                subvectors: self.subvectors.unwrap_or(8) as u8,
                outlier_threshold: self.outlier_threshold.unwrap_or(3.0) as f32,
            }),
            "binary" => Ok(RustCompressionLevel::Binary {
                threshold: self.threshold.unwrap_or(0.0) as f32,
            }),
            _ => Err(Error::new(
                Status::InvalidArg,
                format!("Invalid compression level: {}", self.level_type),
            )),
        }
    }
}

/// Tensor compressor with adaptive level selection
#[napi]
pub struct TensorCompress {
    inner: RustTensorCompress,
}

#[napi]
impl TensorCompress {
    /// Create a new tensor compressor
    ///
    /// # Example
    /// ```javascript
    /// const compressor = new TensorCompress();
    /// ```
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RustTensorCompress::new(),
        }
    }

    /// Compress an embedding based on access frequency
    ///
    /// # Arguments
    /// * `embedding` - The input embedding vector
    /// * `access_freq` - Access frequency in range [0.0, 1.0]
    ///
    /// # Returns
    /// Compressed tensor as JSON string
    ///
    /// # Example
    /// ```javascript
    /// const embedding = [1.0, 2.0, 3.0, 4.0];
    /// const compressed = compressor.compress(embedding, 0.5);
    /// ```
    #[napi]
    pub fn compress(&self, embedding: Vec<f64>, access_freq: f64) -> Result<String> {
        let embedding_f32: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();

        let compressed = self
            .inner
            .compress(&embedding_f32, access_freq as f32)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Compression error: {}", e)))?;

        serde_json::to_string(&compressed).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Serialization error: {}", e),
            )
        })
    }

    /// Compress with explicit compression level
    ///
    /// # Arguments
    /// * `embedding` - The input embedding vector
    /// * `level` - Compression level configuration
    ///
    /// # Returns
    /// Compressed tensor as JSON string
    ///
    /// # Example
    /// ```javascript
    /// const embedding = [1.0, 2.0, 3.0, 4.0];
    /// const level = { level_type: "half", scale: 1.0 };
    /// const compressed = compressor.compressWithLevel(embedding, level);
    /// ```
    #[napi]
    pub fn compress_with_level(
        &self,
        embedding: Vec<f64>,
        level: CompressionLevelConfig,
    ) -> Result<String> {
        let embedding_f32: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();
        let rust_level = level.to_rust()?;

        let compressed = self
            .inner
            .compress_with_level(&embedding_f32, &rust_level)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Compression error: {}", e)))?;

        serde_json::to_string(&compressed).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Serialization error: {}", e),
            )
        })
    }

    /// Decompress a compressed tensor
    ///
    /// # Arguments
    /// * `compressed_json` - Compressed tensor as JSON string
    ///
    /// # Returns
    /// Decompressed embedding vector
    ///
    /// # Example
    /// ```javascript
    /// const decompressed = compressor.decompress(compressed);
    /// ```
    #[napi]
    pub fn decompress(&self, compressed_json: String) -> Result<Vec<f64>> {
        let compressed: RustCompressedTensor =
            serde_json::from_str(&compressed_json).map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Deserialization error: {}", e),
                )
            })?;

        let result = self.inner.decompress(&compressed).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Decompression error: {}", e),
            )
        })?;

        Ok(result.iter().map(|&x| x as f64).collect())
    }
}

// ==================== Search Functions ====================

/// Result from differentiable search
#[napi(object)]
pub struct SearchResult {
    /// Indices of top-k candidates
    pub indices: Vec<u32>,
    /// Soft weights for top-k candidates
    pub weights: Vec<f64>,
}

/// Differentiable search using soft attention mechanism
///
/// # Arguments
/// * `query` - The query vector
/// * `candidate_embeddings` - List of candidate embedding vectors
/// * `k` - Number of top results to return
/// * `temperature` - Temperature for softmax (lower = sharper, higher = smoother)
///
/// # Returns
/// Search result with indices and soft weights
///
/// # Example
/// ```javascript
/// const query = [1.0, 0.0, 0.0];
/// const candidates = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]];
/// const result = differentiableSearch(query, candidates, 2, 1.0);
/// console.log(result.indices); // [0, 1]
/// console.log(result.weights); // [0.x, 0.y]
/// ```
#[napi]
pub fn differentiable_search(
    query: Vec<f64>,
    candidate_embeddings: Vec<Vec<f64>>,
    k: u32,
    temperature: f64,
) -> Result<SearchResult> {
    let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
    let candidates_f32: Vec<Vec<f32>> = candidate_embeddings
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();

    let (indices, weights) =
        rust_differentiable_search(&query_f32, &candidates_f32, k as usize, temperature as f32);

    Ok(SearchResult {
        indices: indices.iter().map(|&i| i as u32).collect(),
        weights: weights.iter().map(|&w| w as f64).collect(),
    })
}

/// Hierarchical forward pass through GNN layers
///
/// # Arguments
/// * `query` - The query vector
/// * `layer_embeddings` - Embeddings organized by layer
/// * `gnn_layers_json` - JSON array of serialized GNN layers
///
/// # Returns
/// Final embedding after hierarchical processing
///
/// # Example
/// ```javascript
/// const query = [1.0, 0.0];
/// const layerEmbeddings = [[[1.0, 0.0], [0.0, 1.0]]];
/// const layer1 = new RuvectorLayer(2, 2, 1, 0.0);
/// const layers = [layer1.toJson()];
/// const result = hierarchicalForward(query, layerEmbeddings, layers);
/// ```
#[napi]
pub fn hierarchical_forward(
    query: Vec<f64>,
    layer_embeddings: Vec<Vec<Vec<f64>>>,
    gnn_layers_json: Vec<String>,
) -> Result<Vec<f64>> {
    let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();

    let embeddings_f32: Vec<Vec<Vec<f32>>> = layer_embeddings
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|v| v.iter().map(|&x| x as f32).collect())
                .collect()
        })
        .collect();

    let gnn_layers: Vec<RustRuvectorLayer> = gnn_layers_json
        .iter()
        .map(|json| {
            serde_json::from_str(json).map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Layer deserialization error: {}", e),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let result = rust_hierarchical_forward(&query_f32, &embeddings_f32, &gnn_layers);

    Ok(result.iter().map(|&x| x as f64).collect())
}

// ==================== Helper Functions ====================

/// Get the compression level that would be selected for a given access frequency
///
/// # Arguments
/// * `access_freq` - Access frequency in range [0.0, 1.0]
///
/// # Returns
/// String describing the compression level: "none", "half", "pq8", "pq4", or "binary"
///
/// # Example
/// ```javascript
/// const level = getCompressionLevel(0.9); // "none" (hot data)
/// const level2 = getCompressionLevel(0.5); // "half" (warm data)
/// ```
#[napi]
pub fn get_compression_level(access_freq: f64) -> String {
    if access_freq > 0.8 {
        "none".to_string()
    } else if access_freq > 0.4 {
        "half".to_string()
    } else if access_freq > 0.1 {
        "pq8".to_string()
    } else if access_freq > 0.01 {
        "pq4".to_string()
    } else {
        "binary".to_string()
    }
}

/// Module initialization
#[napi]
pub fn init() -> String {
    "Ruvector GNN Node.js bindings initialized".to_string()
}
