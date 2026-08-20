//! Unit tests for KV-cache cross-model migration with synthetic geometry.

use ndarray::Array2;

use super::mapper::{CalibrationLayerPair, KvMapper, LinearKvMapper, DEFAULT_RIDGE_LAMBDA};
use super::quality::{GateDecision, QualityGateConfig, TransferQualityGate};
use super::{migrate_kv_cache, KvCacheSnapshot, LayerKvTensor, MigrationResult, MlpKvMapper};

/// Deterministic pseudo-random matrix (xorshift), avoiding rand API churn.
fn synthetic_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    Array2::from_shape_fn((rows, cols), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to roughly [-1, 1).
        ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    })
}

fn tensor(rows: usize, cols: usize, seed: u64) -> LayerKvTensor {
    LayerKvTensor::new(
        synthetic_matrix(rows, cols, seed),
        synthetic_matrix(rows, cols, seed.wrapping_add(1)),
    )
    .unwrap()
}

/// Identity-family pair: source and target activations are identical.
fn identity_pairs(
    layers: usize,
    tokens: usize,
    dim: usize,
    seed: u64,
) -> Vec<CalibrationLayerPair> {
    (0..layers)
        .map(|l| {
            let t = tensor(tokens, dim, seed + l as u64 * 100);
            CalibrationLayerPair::new(t.clone(), t).unwrap()
        })
        .collect()
}

/// Pair generated from a known ground-truth linear transform per layer.
fn linear_transform_pairs(
    layers: usize,
    tokens: usize,
    src_dim: usize,
    tgt_dim: usize,
    seed: u64,
) -> (Vec<CalibrationLayerPair>, Vec<(Array2<f32>, Array2<f32>)>) {
    let mut pairs = Vec::new();
    let mut truths = Vec::new();
    for l in 0..layers {
        let s = l as u64 * 1000 + seed;
        let w_k = synthetic_matrix(src_dim, tgt_dim, s + 7);
        let w_v = synthetic_matrix(src_dim, tgt_dim, s + 8);
        let source = tensor(tokens, src_dim, s);
        let target = LayerKvTensor::new(source.keys.dot(&w_k), source.values.dot(&w_v)).unwrap();
        pairs.push(CalibrationLayerPair::new(source, target).unwrap());
        truths.push((w_k, w_v));
    }
    (pairs, truths)
}

/// (a) Identity-family mapping recovers the cache exactly (within fp tolerance).
#[test]
fn identity_family_mapping_recovers_cache_exactly() {
    let (layers, tokens, dim) = (3, 64, 16);
    let calibration = identity_pairs(layers, tokens, dim, 42);
    let validation = identity_pairs(layers, tokens, dim, 999);

    let mapper = LinearKvMapper::fit(&calibration, DEFAULT_RIDGE_LAMBDA).unwrap();
    let gate = TransferQualityGate::with_defaults(validation).unwrap();

    let source = KvCacheSnapshot::new(
        (0..layers)
            .map(|l| tensor(48, dim, 5000 + l as u64))
            .collect(),
    );
    let result = migrate_kv_cache(&source, &mapper, &gate).unwrap();

    match result {
        MigrationResult::Migrated { cache, report } => {
            assert_eq!(report.decision, GateDecision::Migrate);
            assert!(
                report.worst_cosine > 0.999,
                "cosine {}",
                report.worst_cosine
            );
            for (mapped, orig) in cache.layers.iter().zip(source.layers.iter()) {
                for (m, o) in mapped.keys.iter().zip(orig.keys.iter()) {
                    assert!((m - o).abs() < 1e-2, "key mismatch: {m} vs {o}");
                }
                for (m, o) in mapped.values.iter().zip(orig.values.iter()) {
                    assert!((m - o).abs() < 1e-2, "value mismatch: {m} vs {o}");
                }
            }
        }
        MigrationResult::Reprefill { report } => {
            panic!("identity mapping was vetoed: {report:?}")
        }
    }
}

/// (b) A known linear transform is recovered by fitting within tolerance.
#[test]
fn known_linear_transform_is_recovered_by_fit() {
    let (layers, tokens, src_dim, tgt_dim) = (2, 128, 12, 12);
    let (calibration, truths) = linear_transform_pairs(layers, tokens, src_dim, tgt_dim, 7);

    let mapper = LinearKvMapper::fit(&calibration, 1e-6).unwrap();

    // Held-out probe data mapped by the fitted W must match ground truth W.
    for (l, (w_k, w_v)) in truths.iter().enumerate() {
        let probe = tensor(32, src_dim, 8_000 + l as u64);
        let (pred_k, pred_v) = mapper.map_layer(l, &probe.keys, &probe.values).unwrap();
        let true_k = probe.keys.dot(w_k);
        let true_v = probe.values.dot(w_v);
        for (p, t) in pred_k.iter().zip(true_k.iter()) {
            assert!((p - t).abs() < 1e-2, "layer {l} key: {p} vs {t}");
        }
        for (p, t) in pred_v.iter().zip(true_v.iter()) {
            assert!((p - t).abs() < 1e-2, "layer {l} value: {p} vs {t}");
        }
    }
}

/// (c) The quality gate vetoes a deliberately-mismatched mapping.
#[test]
fn quality_gate_vetoes_mismatched_mapping() {
    let (layers, tokens, dim) = (2, 64, 8);

    // Mapper fitted on identity pairs...
    let calibration = identity_pairs(layers, tokens, dim, 1);
    let mapper = LinearKvMapper::fit(&calibration, DEFAULT_RIDGE_LAMBDA).unwrap();

    // ...but validated against targets that are unrelated random tensors
    // (a "degrading pair" in the paper's terms).
    let mismatched: Vec<CalibrationLayerPair> = (0..layers)
        .map(|l| {
            CalibrationLayerPair::new(
                tensor(tokens, dim, 300 + l as u64),
                tensor(tokens, dim, 77_000 + l as u64),
            )
            .unwrap()
        })
        .collect();
    let gate = TransferQualityGate::with_defaults(mismatched).unwrap();

    let source = KvCacheSnapshot::new(
        (0..layers)
            .map(|l| tensor(16, dim, 40 + l as u64))
            .collect(),
    );
    let result = migrate_kv_cache(&source, &mapper, &gate).unwrap();

    match result {
        MigrationResult::Reprefill { report } => {
            assert_eq!(report.decision, GateDecision::Reprefill);
            assert!(
                report.worst_cosine < QualityGateConfig::default().min_cosine,
                "worst cosine {} should be below threshold",
                report.worst_cosine
            );
        }
        MigrationResult::Migrated { report, .. } => {
            panic!("mismatched mapping must be vetoed, got Migrate: {report:?}")
        }
    }
}

/// (d) Dimension-mismatch mapping produces correctly-shaped output.
#[test]
fn dimension_mismatch_mapping_produces_correct_shapes() {
    // Source: 4 heads x 8 dim = 32; target: 2 heads x 12 dim = 24.
    let (layers, tokens, src_dim, tgt_dim) = (2, 96, 32, 24);
    let (calibration, _) = linear_transform_pairs(layers, tokens, src_dim, tgt_dim, 21);
    let (validation, _) = linear_transform_pairs(layers, 48, src_dim, tgt_dim, 21);

    let mapper = LinearKvMapper::fit(&calibration, 1e-6).unwrap();
    assert_eq!(mapper.source_dim(), src_dim);
    assert_eq!(mapper.target_dim(), tgt_dim);

    let gate = TransferQualityGate::with_defaults(validation).unwrap();
    let source = KvCacheSnapshot::new(
        (0..layers)
            .map(|l| tensor(10, src_dim, 60 + l as u64))
            .collect(),
    );
    let result = migrate_kv_cache(&source, &mapper, &gate).unwrap();

    match result {
        MigrationResult::Migrated { cache, .. } => {
            assert_eq!(cache.num_layers(), layers);
            for layer in &cache.layers {
                assert_eq!(layer.tokens(), 10);
                assert_eq!(layer.dim(), tgt_dim);
            }
        }
        MigrationResult::Reprefill { report } => {
            panic!("consistent dim-mismatch transform should migrate: {report:?}")
        }
    }
}

/// from_flat adapts the TwoTierKvCache::get_all_kv flat layout.
#[test]
fn from_flat_roundtrip_matches_stride_layout() {
    let (num_kv_heads, head_dim, tokens) = (2, 4, 3);
    let stride = num_kv_heads * head_dim;
    let keys: Vec<f32> = (0..tokens * stride).map(|i| i as f32).collect();
    let values: Vec<f32> = (0..tokens * stride).map(|i| (i as f32) * 0.5).collect();

    let t = LayerKvTensor::from_flat(&keys, &values, num_kv_heads, head_dim).unwrap();
    assert_eq!(t.tokens(), tokens);
    assert_eq!(t.dim(), stride);
    assert_eq!(t.keys[[1, 0]], stride as f32);
    assert_eq!(t.values[[2, 7]], (2 * stride + 7) as f32 * 0.5);

    // Length not divisible by stride must be rejected.
    assert!(LayerKvTensor::from_flat(
        &keys[..stride + 1],
        &values[..stride + 1],
        num_kv_heads,
        head_dim
    )
    .is_err());
}

/// The MLP fallback is a documented stub in this slice.
#[test]
fn mlp_mapper_stub_returns_not_implemented() {
    let stub = MlpKvMapper;
    let m = synthetic_matrix(2, 4, 3);
    let err = stub.map_layer(0, &m, &m).unwrap_err();
    assert!(err.to_string().contains("Not implemented"), "{err}");
}

/// Layer-count mismatch between cache and mapper is rejected up front.
#[test]
fn layer_count_mismatch_is_rejected() {
    let calibration = identity_pairs(2, 32, 8, 5);
    let validation = identity_pairs(2, 32, 8, 6);
    let mapper = LinearKvMapper::fit(&calibration, DEFAULT_RIDGE_LAMBDA).unwrap();
    let gate = TransferQualityGate::with_defaults(validation).unwrap();

    let source = KvCacheSnapshot::new(vec![tensor(4, 8, 9)]); // 1 layer vs 2
    assert!(migrate_kv_cache(&source, &mapper, &gate).is_err());
}
