# ruvector-attention

Advanced attention mechanisms for vector search and geometric AI, implementing 7 mathematical theories for transformer attention.

[![Crates.io](https://img.shields.io/crates/v/ruvector-attention.svg)](https://crates.io/crates/ruvector-attention)
[![Documentation](https://docs.rs/ruvector-attention/badge.svg)](https://docs.rs/ruvector-attention)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)]()

## Overview

`ruvector-attention` provides production-ready implementations of advanced attention mechanisms based on mathematical foundations from differential geometry, information theory, and optimal transport. The library combines theoretical rigor with practical optimizations including SIMD acceleration, caching, and quantization.

## Features

- 🚀 **High-Performance**: SIMD-accelerated with 4-way unrolled accumulators
- 🎯 **Ergonomic API**: Fluent builder pattern and preset configurations
- 📦 **Modular Design**: Mix and match attention mechanisms
- 🔧 **Flexible**: Support for standard, sparse, graph, and geometric attention
- 🧠 **7 Mathematical Theories**: Optimal Transport, Mixed Curvature, Topology, Information Geometry, Information Bottleneck, PDE/Diffusion, and Unified Diagnostics
- 📊 **Unified Reporting**: Health monitoring and automatic attention mode selection
- 🔢 **Quantization-Friendly**: Component-wise precision control (8-bit Euclidean, 5-bit Hyperbolic/Spherical)

## Supported Attention Mechanisms

### Standard Attention
- **Scaled Dot-Product**: `softmax(QK^T / √d)V`
- **Multi-Head**: Parallel attention heads with diverse representations

### Sparse Attention (Memory Efficient)
- **Flash Attention**: O(n) memory complexity with tiled computation
- **Linear Attention**: O(n) complexity using kernel approximation
- **Local-Global**: Sliding window + global tokens (Longformer-style)

### Geometric Attention
- **Hyperbolic Attention**: Attention in hyperbolic space for hierarchical data
- **Mixed Curvature**: Dynamic curvature for complex geometries

### Graph Attention
- **Edge-Featured GAT**: Graph attention with edge features
- **RoPE**: Rotary Position Embeddings for graphs

### Mixture-of-Experts
- **MoE Attention**: Learned routing to specialized expert modules
- **Top-k Routing**: Efficient expert selection

## 7 Mathematical Theories

This crate implements attention mechanisms grounded in 7 distinct mathematical theories:

| # | Theory | Module | Key Types | Use Case |
|---|--------|--------|-----------|----------|
| 1 | **Optimal Transport** | `transport` | `SlicedWassersteinAttention`, `CentroidOTAttention` | Distribution matching, Earth mover distance |
| 2 | **Mixed Curvature** | `curvature` | `MixedCurvatureFusedAttention`, `TangentSpaceMapper` | Product spaces E^e × H^h × S^s |
| 3 | **Topology** | `topology` | `TopologyGatedAttention`, `WindowCoherence` | Coherence-based mode switching |
| 4 | **Information Geometry** | `info_geometry` | `FisherMetric`, `NaturalGradient` | Natural gradient descent |
| 5 | **Information Bottleneck** | `info_bottleneck` | `InformationBottleneck`, `KLDivergence` | Compression via KL minimization |
| 6 | **PDE/Diffusion** | `pde_attention` | `DiffusionAttention`, `GraphLaplacian` | Heat equation on similarity graph |
| 7 | **Unified Diagnostics** | `unified_report` | `GeometryReport`, `ReportBuilder` | Health monitoring & mode selection |

### Theory 1: Optimal Transport Attention

Attention as mass transport between query and key distributions using Wasserstein distance.

```rust
use ruvector_attention::{SlicedWassersteinAttention, SlicedWassersteinConfig};

// Configure Sliced Wasserstein with 16 random projections
let config = SlicedWassersteinConfig {
    num_projections: 16,
    num_candidates: 64,
    dim: 512,
    ..Default::default()
};

let ot_attention = SlicedWassersteinAttention::new(config);

// Compute OT-based attention scores
let query = vec![0.5; 512];
let keys: Vec<&[f32]> = key_data.iter().map(|k| k.as_slice()).collect();
let values: Vec<&[f32]> = value_data.iter().map(|v| v.as_slice()).collect();

let output = ot_attention.compute_sliced(&query, &keys, &values)?;
```

**Key Features:**
- Sliced Wasserstein with cached sorted projections
- Two-stage filtering: cheap dot-product → expensive OT kernel
- Centroid OT: cluster keys into M centroids for O(M) transport

### Theory 2: Mixed Curvature Attention

Attention in product manifolds combining Euclidean (E), Hyperbolic (H), and Spherical (S) spaces.

```rust
use ruvector_attention::{
    MixedCurvatureFusedAttention, FusedCurvatureConfig,
    TangentSpaceMapper, TangentSpaceConfig
};

// Configure mixed curvature with component dimensions
let config = FusedCurvatureConfig {
    euclidean_dim: 256,
    hyperbolic_dim: 128,
    spherical_dim: 128,
    curvature_h: -1.0,  // Negative for hyperbolic
    curvature_s: 1.0,   // Positive for spherical
    ..Default::default()
};

let mixed_attention = MixedCurvatureFusedAttention::new(config);

// Map hyperbolic vectors to tangent space for efficient computation
let mapper = TangentSpaceMapper::new(TangentSpaceConfig::default());
let tangent_keys = mapper.map_to_tangent(&hyperbolic_keys);
```

**Key Features:**
- Tangent space mapping (avoids expensive geodesic computations)
- Fused dot kernel: single vectorized loop for E+H+S similarities
- Per-head learned mixing weights
- Component quantization: 8-bit E, 5-bit H/S

### Theory 3: Topology-Gated Attention

Adaptive attention that switches modes based on local coherence metrics.

```rust
use ruvector_attention::{
    TopologyGatedAttention, TopologyGatedConfig,
    AttentionMode, PolicyConfig, CoherenceMetric
};

let config = TopologyGatedConfig {
    dim: 512,
    policy: PolicyConfig {
        stable_threshold: 0.8,    // High coherence → Stable mode
        cautious_threshold: 0.5,  // Medium → Cautious mode
        freeze_threshold: 0.3,    // Low → Freeze mode
        hysteresis: 0.05,         // Prevents mode oscillation
        ..Default::default()
    },
    ..Default::default()
};

let gated = TopologyGatedAttention::new(config);

// Attention automatically adjusts based on window coherence
let output = gated.compute_gated(&query, &keys, &values)?;
let mode = gated.current_mode(); // Stable, Cautious, or Freeze
```

**Coherence Metrics:**
| Metric | Description |
|--------|-------------|
| `BoundaryMass` | Mass near window boundaries |
| `CutProxy` | Proxy for graph cut quality |
| `Disagreement` | Variance in attention weights |
| `SimilarityVariance` | Local similarity variance |

### Theory 4: Information Geometry

Natural gradient optimization using the Fisher Information Matrix.

```rust
use ruvector_attention::{FisherMetric, FisherConfig, NaturalGradient, NaturalGradientConfig};

// Fisher metric for probability distributions
let fisher = FisherMetric::new(FisherConfig {
    eps: 1e-8,
    max_cg_iters: 50,
    cg_tol: 1e-6,
});

// Compute F * v (Fisher-vector product)
let probs = vec![0.25, 0.25, 0.25, 0.25];
let direction = vec![0.1, -0.1, 0.05, -0.05];
let fv = fisher.apply(&probs, &direction);

// Natural gradient optimizer
let ng = NaturalGradient::new(NaturalGradientConfig {
    lr: 0.1,
    use_diagonal: false,  // Full CG solve (more accurate)
    fisher: FisherConfig::default(),
});

// Update logits using natural gradient: θ ← θ - lr * F^{-1} * ∇L
let new_logits = ng.step_logits(&logits, &grad_logits);
```

**Key Features:**
- Conjugate gradient solver for F^{-1} * v
- Diagonal approximation for speed
- SIMD-accelerated matrix-vector operations

### Theory 5: Information Bottleneck

Attention compression via the Information Bottleneck principle.

```rust
use ruvector_attention::{InformationBottleneck, IBConfig, KLDivergence, DiagonalGaussian};

// Information bottleneck layer
let ib = InformationBottleneck::new(IBConfig {
    beta: 0.1,        // Compression strength
    z_dim: 64,        // Bottleneck dimension
    anneal_steps: 1000,
    ..Default::default()
});

// Compute KL divergence between Gaussian and unit normal
let gaussian = DiagonalGaussian {
    mean: vec![0.1; 64],
    log_var: vec![-1.0; 64],
};
let kl = KLDivergence::gaussian_to_unit(&gaussian);

// Compress attention weights
let (compressed, kl_loss) = ib.compress_attention_weights(&weights, temperature);

// Reparameterized sampling
let z = ib.sample(&mean, &log_var, &epsilon);
```

**Key Features:**
- KL divergence: Gaussian→Unit, Categorical, Jensen-Shannon
- Variational Information Bottleneck (VIB)
- Temperature annealing for curriculum learning

### Theory 6: PDE/Diffusion Attention

Attention as heat diffusion on the key similarity graph.

```rust
use ruvector_attention::{
    DiffusionAttention, DiffusionConfig,
    GraphLaplacian, LaplacianType
};

// Build graph Laplacian from keys
let laplacian = GraphLaplacian::from_keys(
    &keys,
    sigma,  // Gaussian kernel bandwidth
    LaplacianType::SymmetricNormalized
);

// Diffusion attention with heat equation
let config = DiffusionConfig {
    t: 1.0,           // Diffusion time
    num_steps: 10,    // Discretization steps
    sigma: 1.0,       // Kernel bandwidth
    use_knn: true,    // Sparse Laplacian
    k: 16,            // k-NN neighbors
    laplacian_type: LaplacianType::SymmetricNormalized,
    ..Default::default()
};

let diffusion = DiffusionAttention::new(config);

// Compute diffused attention
let output = diffusion.compute_diffusion(&query, &keys, &values)?;

// Multi-scale diffusion (captures different granularities)
let scales = diffusion.compute_multiscale(&query, &keys, 4);
```

**Laplacian Types:**
| Type | Formula | Properties |
|------|---------|------------|
| `Unnormalized` | D - W | Graph spectrum analysis |
| `SymmetricNormalized` | I - D^{-1/2}WD^{-1/2} | Symmetric, eigenvalues in [0,2] |
| `RandomWalk` | I - D^{-1}W | Probability transitions |

### Theory 7: Unified Geometry Report

Diagnostic dashboard combining all metrics for intelligent attention mode selection.

```rust
use ruvector_attention::{
    ReportBuilder, ReportConfig, GeometryReport,
    MetricType, AttentionRecommendation
};

// Build comprehensive geometry report
let report = ReportBuilder::new(ReportConfig::default())
    .with_ot_distance(0.15)
    .with_topology_coherence(0.82)
    .with_ib_kl(0.05)
    .with_diffusion_energy(0.3)
    .with_attention_entropy(2.1)
    .build();

// Get health score (0-1)
println!("Health: {:.2}", report.health_score);

// Get automatic attention mode recommendation
match report.recommendation {
    AttentionRecommendation::Standard => { /* Use standard attention */ }
    AttentionRecommendation::Sparse => { /* Switch to sparse */ }
    AttentionRecommendation::Geometric => { /* Use hyperbolic/mixed */ }
    AttentionRecommendation::Diffusion => { /* Use diffusion attention */ }
}

// Check individual metrics
for metric in &report.metrics {
    println!("{:?}: {} ({})",
        metric.metric_type,
        metric.value,
        metric.status()
    );
}
```

**Metrics Tracked:**
| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| OT Distance | 0.0 - 0.5 | > 0.3 | > 0.7 |
| Topology Coherence | 0.5 - 1.0 | < 0.3 | < 0.1 |
| IB KL | 0.0 - 0.2 | > 0.5 | > 1.0 |
| Diffusion Energy | 0.0 - 1.0 | > 2.0 | > 5.0 |
| Attention Entropy | 1.0 - 4.0 | < 0.5 | < 0.1 |

## Quick Start

```rust
use ruvector_attention::sdk::*;

// Simple multi-head attention
let attention = multi_head(768, 12)
    .dropout(0.1)
    .causal(true)
    .build()?;

// Use preset configurations
let bert = AttentionPreset::Bert.builder(768).build()?;
let gpt = AttentionPreset::Gpt.builder(768).build()?;

// Build pipelines with normalization
let pipeline = AttentionPipeline::new()
    .add_attention(attention)
    .add_norm(NormType::LayerNorm)
    .add_residual();

// Compute attention
let query = vec![0.5; 768];
let keys = vec![&query[..]; 10];
let values = vec![&query[..]; 10];

let output = pipeline.run(&query, &keys, &values)?;
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvector-attention = "0.1"
```

Or with specific features:

```toml
[dependencies]
ruvector-attention = { version = "0.1", features = ["simd", "wasm"] }
```

## SDK Overview

### Builder API

The builder provides a fluent interface for configuring attention:

```rust
use ruvector_attention::sdk::*;

// Flash attention for long sequences
let flash = flash(1024, 128)  // dim, block_size
    .causal(true)
    .dropout(0.1)
    .build()?;

// Linear attention for O(n) complexity
let linear = linear(512, 256)  // dim, num_features
    .build()?;

// MoE attention with 8 experts
let moe = moe(512, 8, 2)  // dim, num_experts, top_k
    .expert_capacity(1.25)
    .jitter_noise(0.01)
    .build()?;

// Hyperbolic attention for hierarchies
let hyperbolic = hyperbolic(512, -1.0)  // dim, curvature
    .build()?;
```

### Pipeline API

Compose attention with pre/post processing:

```rust
use ruvector_attention::sdk::*;

let attention = multi_head(768, 12).build()?;

let pipeline = AttentionPipeline::new()
    .add_norm(NormType::LayerNorm)     // Pre-normalization
    .add_attention(attention)           // Attention layer
    .add_dropout(0.1)                   // Dropout
    .add_residual()                     // Residual connection
    .add_norm(NormType::RMSNorm);      // Post-normalization

let output = pipeline.run(&query, &keys, &values)?;
```

### Preset Configurations

Pre-configured attention for popular models:

```rust
use ruvector_attention::sdk::presets::*;

// Model-specific presets
let bert = AttentionPreset::Bert.builder(768).build()?;
let gpt = AttentionPreset::Gpt.builder(768).build()?;
let longformer = AttentionPreset::Longformer.builder(512).build()?;
let flash = AttentionPreset::FlashOptimized.builder(1024).build()?;
let t5 = AttentionPreset::T5.builder(768).build()?;
let vit = AttentionPreset::ViT.builder(768).build()?;

// Smart selection based on use case
let attention = for_sequences(512, max_len).build()?;  // Auto-select by length
let graph_attn = for_graphs(256, hierarchical).build()?;  // Graph attention
let fast_attn = for_large_scale(1024).build()?;  // Flash attention

// By model name
let bert = from_model_name("bert", 768)?;
let gpt2 = from_model_name("gpt2", 768)?;
```

## Architecture

```
ruvector-attention/
├── src/
│   ├── lib.rs                 # Main crate entry
│   ├── error.rs               # Error types
│   ├── traits.rs              # Core attention traits
│   │
│   ├── attention/             # Standard attention
│   │   ├── scaled_dot_product.rs
│   │   └── multi_head.rs
│   │
│   ├── sparse/                # Sparse attention (O(n) memory)
│   │   ├── flash.rs           # Flash attention (tiled)
│   │   ├── linear.rs          # Kernel approximation
│   │   └── local_global.rs    # Longformer-style
│   │
│   ├── graph/                 # Graph attention
│   │   ├── edge_featured.rs   # GAT with edge features
│   │   ├── dual_space.rs      # Dual-space attention
│   │   └── rope.rs            # Rotary embeddings
│   │
│   ├── hyperbolic/            # Hyperbolic geometry
│   │   ├── hyperbolic_attention.rs
│   │   ├── mixed_curvature.rs
│   │   └── poincare.rs
│   │
│   ├── moe/                   # Mixture-of-Experts
│   │   ├── expert.rs          # Expert modules
│   │   ├── router.rs          # Top-k routing
│   │   └── moe_attention.rs
│   │
│   ├── transport/             # [Theory 1] Optimal Transport
│   │   ├── sliced_wasserstein.rs   # Sliced OT attention
│   │   ├── centroid_ot.rs          # Centroid-based OT
│   │   └── cached_projections.rs   # Projection caching
│   │
│   ├── curvature/             # [Theory 2] Mixed Curvature
│   │   ├── tangent_space.rs        # Tangent space mapping
│   │   ├── fused_attention.rs      # Fused E+H+S kernel
│   │   └── component_quantizer.rs  # 8-bit/5-bit quantization
│   │
│   ├── topology/              # [Theory 3] Topology Gating
│   │   ├── coherence.rs            # Window coherence metrics
│   │   ├── policy.rs               # 3-mode policy (Stable/Cautious/Freeze)
│   │   └── gated_attention.rs      # Adaptive gated attention
│   │
│   ├── info_geometry/         # [Theory 4] Information Geometry
│   │   ├── fisher.rs               # Fisher information matrix
│   │   └── natural_gradient.rs     # Natural gradient descent
│   │
│   ├── info_bottleneck/       # [Theory 5] Information Bottleneck
│   │   ├── kl_divergence.rs        # KL, JS divergences
│   │   └── bottleneck.rs           # VIB layer
│   │
│   ├── pde_attention/         # [Theory 6] PDE/Diffusion
│   │   ├── laplacian.rs            # Graph Laplacian construction
│   │   └── diffusion.rs            # Heat equation attention
│   │
│   ├── unified_report/        # [Theory 7] Unified Diagnostics
│   │   ├── metrics.rs              # Metric types and values
│   │   ├── report.rs               # Geometry report builder
│   │   └── recommendation.rs       # Attention mode recommendations
│   │
│   ├── training/              # Training utilities
│   │   ├── loss.rs            # InfoNCE, contrastive losses
│   │   ├── optimizer.rs       # SGD, Adam, AdamW
│   │   └── curriculum.rs      # Curriculum scheduling
│   │
│   └── sdk/                   # High-level SDK
│       ├── builder.rs         # Fluent builder API
│       ├── pipeline.rs        # Composable pipelines
│       └── presets.rs         # Model presets (BERT, GPT, etc.)
```

## Examples

### Transformer Block

```rust
use ruvector_attention::sdk::*;

fn create_transformer_block(dim: usize) -> AttentionResult<AttentionPipeline> {
    let attention = multi_head(dim, 12)
        .dropout(0.1)
        .build()?;

    Ok(AttentionPipeline::new()
        .add_norm(NormType::LayerNorm)
        .add_attention(attention)
        .add_dropout(0.1)
        .add_residual())
}
```

### Long Context Processing

```rust
use ruvector_attention::sdk::*;

fn create_long_context_attention(dim: usize, max_len: usize)
    -> AttentionResult<Box<dyn Attention>> {
    if max_len <= 2048 {
        multi_head(dim, 12).build()
    } else if max_len <= 16384 {
        local_global(dim, 512).build()
    } else {
        linear(dim, dim / 4).build()
    }
}
```

### Graph Neural Network

```rust
use ruvector_attention::sdk::*;

fn create_graph_attention(dim: usize, is_tree: bool)
    -> AttentionResult<Box<dyn Attention>> {
    if is_tree {
        hyperbolic(dim, -1.0).build()  // Hyperbolic for tree-like
    } else {
        multi_head(dim, 8).build()     // Standard for general graphs
    }
}
```

## Performance

### Complexity Comparison

| Mechanism | Time | Memory | Use Case |
|-----------|------|--------|----------|
| Scaled Dot-Product | O(n²) | O(n²) | Short sequences |
| Multi-Head | O(n²) | O(n²) | Standard transformers |
| Flash Attention | O(n²) | O(n) | Long sequences |
| Linear Attention | O(n) | O(n) | Very long sequences |
| Local-Global | O(n·w) | O(n·w) | Document processing |
| Hyperbolic | O(n²) | O(n²) | Hierarchical data |
| MoE | O(n²/E) | O(n²) | Specialized tasks |

### Advanced Mechanisms Complexity

| Theory | Mechanism | Time | Memory | Notes |
|--------|-----------|------|--------|-------|
| OT | Sliced Wasserstein | O(n·P·log n) | O(n·P) | P = num projections |
| OT | Centroid OT | O(n + M²) | O(M·d) | M = num centroids |
| Curvature | Mixed Curvature | O(n²) | O(n²) | Fused E+H+S kernel |
| Topology | Gated Attention | O(n²) | O(n²) | + O(n) coherence |
| Info Geo | Natural Gradient | O(n²) | O(n) | CG solver |
| Info Bottle | VIB | O(n·z) | O(z) | z = bottleneck dim |
| PDE | Diffusion | O(n²·T) | O(n²) | T = diffusion steps |

Where:
- `n` = sequence length
- `w` = local window size
- `E` = number of experts
- `P` = number of random projections (typically 8-16)
- `M` = number of centroids (typically 16-32)
- `z` = bottleneck dimension
- `T` = number of diffusion time steps

### Benchmarks

On a typical workload (batch_size=32, seq_len=512, dim=768):

- **Flash Attention**: 2.3x faster, 5x less memory than standard
- **Linear Attention**: O(n) scaling for sequences >4096
- **Local-Global**: 60% of standard attention cost for w=256
- **Sliced Wasserstein**: 1.8x slower than standard, but better distribution matching
- **Mixed Curvature**: ~1.3x standard with tangent space optimization
- **Diffusion Attention**: 2-10x slower depending on T, but captures multi-scale structure

## Tutorials

### Tutorial 1: Building a Geometry-Aware Transformer

Combine multiple geometric attention mechanisms for hierarchical data.

```rust
use ruvector_attention::*;
use ruvector_attention::sdk::*;

fn create_geometry_aware_block(dim: usize) -> AttentionResult<AttentionPipeline> {
    // Use hyperbolic attention for hierarchy + standard for local patterns
    let hyperbolic_attn = hyperbolic(dim, -1.0).build()?;

    // Create a pipeline with pre-norm
    Ok(AttentionPipeline::new()
        .add_norm(NormType::RMSNorm)
        .add_attention(hyperbolic_attn)
        .add_dropout(0.1)
        .add_residual())
}
```

### Tutorial 2: Adaptive Attention with Unified Report

Use the unified report to automatically select the best attention mode.

```rust
use ruvector_attention::*;

fn adaptive_attention(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
) -> AttentionResult<Vec<f32>> {
    // Build a diagnostic report
    let report = ReportBuilder::new(ReportConfig::default())
        .analyze_keys(keys)  // Automatically compute metrics
        .build();

    // Select attention based on recommendation
    match report.recommendation {
        AttentionRecommendation::Standard => {
            let attn = ScaledDotProductAttention::new(query.len());
            attn.compute(query, keys, values)
        }
        AttentionRecommendation::Sparse => {
            let attn = FlashAttention::new(query.len(), 64);
            attn.compute(query, keys, values)
        }
        AttentionRecommendation::Geometric => {
            let config = HyperbolicAttentionConfig {
                dim: query.len(),
                curvature: -1.0,
                ..Default::default()
            };
            let attn = HyperbolicAttention::new(config);
            attn.compute(query, keys, values)
        }
        AttentionRecommendation::Diffusion => {
            let config = DiffusionConfig::default();
            let attn = DiffusionAttention::new(config);
            attn.compute_diffusion(query, keys, values)
        }
    }
}
```

### Tutorial 3: Information Bottleneck for Attention Compression

Use VIB to learn compressed attention representations.

```rust
use ruvector_attention::*;

struct CompressedAttention {
    ib: InformationBottleneck,
    encoder_mean: Vec<f32>,     // Learned weights
    encoder_log_var: Vec<f32>,  // Learned weights
}

impl CompressedAttention {
    fn new(input_dim: usize, bottleneck_dim: usize) -> Self {
        let ib = InformationBottleneck::new(IBConfig {
            beta: 0.1,
            z_dim: bottleneck_dim,
            ..Default::default()
        });

        Self {
            ib,
            encoder_mean: vec![0.0; input_dim * bottleneck_dim],
            encoder_log_var: vec![0.0; input_dim * bottleneck_dim],
        }
    }

    fn forward(&self, x: &[f32], epsilon: &[f32]) -> (Vec<f32>, f32) {
        // Encode to mean and log_var (simplified)
        let mean = self.encode_mean(x);
        let log_var = self.encode_log_var(x);

        // Sample from posterior
        let z = self.ib.sample(&mean, &log_var, epsilon);

        // Compute KL loss
        let kl_loss = self.ib.compute_kl_loss(&mean, &log_var);

        (z, kl_loss)
    }

    fn encode_mean(&self, _x: &[f32]) -> Vec<f32> {
        // Linear transform (simplified)
        vec![0.0; self.ib.config().z_dim]
    }

    fn encode_log_var(&self, _x: &[f32]) -> Vec<f32> {
        vec![-1.0; self.ib.config().z_dim]  // Initialize to low variance
    }
}
```

### Tutorial 4: Multi-Scale Diffusion for Document Understanding

Use diffusion attention at multiple scales for long documents.

```rust
use ruvector_attention::*;

fn document_understanding(
    query: &[f32],
    document_keys: &[&[f32]],  // Keys from document chunks
) -> Vec<Vec<f32>> {
    // Configure diffusion with k-NN sparsity for large documents
    let config = DiffusionConfig {
        t: 2.0,           // Larger t for more diffusion
        num_steps: 20,
        sigma: 1.0,
        use_knn: true,
        k: 32,            // Sparse Laplacian
        laplacian_type: LaplacianType::SymmetricNormalized,
    };

    let diffusion = DiffusionAttention::new(config);

    // Get attention at 4 different scales
    // Scale 0: Local (small t) - captures nearby relationships
    // Scale 3: Global (large t) - captures document-level structure
    let scales = diffusion.compute_multiscale(query, document_keys, 4);

    scales
}
```

### Tutorial 5: Natural Gradient Training Loop

Train attention parameters with geometry-aware optimization.

```rust
use ruvector_attention::*;

fn natural_gradient_step(
    logits: &[f32],
    target_probs: &[f32],
    config: &NaturalGradientConfig,
) -> Vec<f32> {
    let ng = NaturalGradient::new(config.clone());

    // Compute cross-entropy gradient w.r.t. logits
    let probs = softmax(logits);
    let grad: Vec<f32> = probs.iter()
        .zip(target_probs.iter())
        .map(|(p, t)| p - t)
        .collect();

    // Apply natural gradient update
    // This uses F^{-1} to rescale gradients, accounting for
    // the geometry of the probability simplex
    ng.step_logits(logits, &grad)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}
```

## Features

- `simd` - SIMD acceleration (default, enabled)
- `wasm` - WebAssembly support
- `napi` - Node.js bindings

## Documentation

- [SDK Guide](docs/SDK_GUIDE.md) - Comprehensive SDK usage guide
- [API Documentation](https://docs.rs/ruvector-attention) - Full API reference
- [Examples](examples/) - Working code examples

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Citation

If you use this crate in your research, please cite:

```bibtex
@software{ruvector_attention,
  title = {ruvector-attention: Advanced Attention Mechanisms for Vector Search},
  author = {ruvector contributors},
  year = {2025},
  url = {https://github.com/ruvnet/ruvector}
}
```

## Related Projects

- [ruvector](../ruvector) - Core vector search engine
- [ruvector-graph](../ruvector-graph) - Graph neural networks
- [ruvector-gnn](../ruvector-gnn) - Geometric neural networks
