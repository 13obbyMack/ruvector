//! Turbo4 quantized HNSW index (ADR-296).
//!
//! The HNSW element type is `u8`: each point's data **is** the packed Turbo4
//! code blob (`D/2 + 8` bytes — nibbles + α + Σlevel²). No `Vec<f32>` is
//! retained anywhere in this index; at 1536-D the per-vector payload drops
//! from 6144 B to 776 B (7.9×).
//!
//! Blob roles are structurally disjoint by length (code `D/2+8`, query
//! `D+8`), so one [`Distance<u8>`] functor serves both phases:
//! * insert / graph maintenance → symmetric code×code kernel;
//! * traversal → asymmetric int8-query×code kernel (higher fidelity);
//! * final ranking → the top `k · rescore_multiplier` candidates are
//!   re-scored with the exact f32 rotated query and truncated to `k`.

use crate::error::{Result, RuvectorError};
use crate::index::VectorIndex;
use crate::types::{DistanceMetric, HnswConfig, SearchPolicy, SearchResult, VectorId};
use dashmap::DashMap;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use ruvector_turboquant::{score, Metric, Turbo4Codec, Turbo4Query};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Escalation parameters derived from a [`SearchPolicy`] (ADR-297 §3/§9):
/// relative score-margin below which a query is "uncertain", the ef widening
/// factor per escalation round, and how many rounds are allowed.
struct EscalationParams {
    margin_threshold: f32,
    ef_mult: usize,
    max_rounds: usize,
}

impl EscalationParams {
    fn for_policy(policy: SearchPolicy) -> Self {
        match policy {
            SearchPolicy::Quality => Self {
                margin_threshold: 0.05,
                ef_mult: 3,
                max_rounds: 2,
            },
            SearchPolicy::Balanced => Self {
                margin_threshold: 0.02,
                ef_mult: 2,
                max_rounds: 1,
            },
            SearchPolicy::MaxCompression => Self {
                margin_threshold: 0.0,
                ef_mult: 1,
                max_rounds: 0,
            },
        }
    }
}

/// Distance functor over Turbo4 blobs. Chooses the kernel from blob lengths:
/// hnsw_rs passes the search query straight through to `eval`, which is what
/// makes true asymmetric traversal possible without forking hnsw_rs.
struct Turbo4DistanceFn {
    metric: Metric,
    dim: usize,
    code_len: usize,
    query_len: usize,
}

impl Distance<u8> for Turbo4DistanceFn {
    #[inline(always)]
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        if a.len() == self.code_len && b.len() == self.code_len {
            score::symmetric_distance(self.metric, a, b, self.dim)
        } else if a.len() == self.query_len && b.len() == self.code_len {
            score::asymmetric_distance(self.metric, a, b, self.dim)
        } else if b.len() == self.query_len && a.len() == self.code_len {
            score::asymmetric_distance(self.metric, b, a, self.dim)
        } else {
            debug_assert!(
                false,
                "unrecognized Turbo4 blob lengths {}/{}",
                a.len(),
                b.len()
            );
            f32::MAX
        }
    }
}

fn to_turbo_metric(metric: DistanceMetric) -> Result<Metric> {
    match metric {
        DistanceMetric::Euclidean => Ok(Metric::Euclidean),
        DistanceMetric::Cosine => Ok(Metric::Cosine),
        DistanceMetric::DotProduct => Ok(Metric::DotProduct),
        DistanceMetric::Manhattan => Err(RuvectorError::InvalidParameter(
            "Turbo4 quantization does not support the Manhattan metric \
             (it does not decompose over the level dot product); use \
             Euclidean, Cosine, or DotProduct, or disable quantization"
                .into(),
        )),
    }
}

struct Turbo4Inner {
    hnsw: Hnsw<'static, u8, Turbo4DistanceFn>,
    /// id → packed code blob; the rescoring source and (with the graph's own
    /// copy) the only representation of the vector.
    codes: DashMap<VectorId, Vec<u8>>,
    id_to_idx: DashMap<VectorId, usize>,
    idx_to_id: DashMap<usize, VectorId>,
    next_idx: usize,
}

/// HNSW index storing only Turbo4 packed codes.
pub struct Turbo4HnswIndex {
    inner: Arc<RwLock<Turbo4Inner>>,
    codec: Arc<Turbo4Codec>,
    config: HnswConfig,
    metric: Metric,
    dimensions: usize,
    rescore_multiplier: usize,
    escalation: EscalationParams,
    /// Adaptive-plane telemetry: total queries / queries that escalated.
    queries: AtomicU64,
    escalated: AtomicU64,
}

impl Turbo4HnswIndex {
    /// Create a Turbo4-quantized HNSW index.
    pub fn new(
        dimensions: usize,
        metric: DistanceMetric,
        config: HnswConfig,
        rotation_seed: u64,
        rescore_multiplier: usize,
        policy: SearchPolicy,
    ) -> Result<Self> {
        let metric = to_turbo_metric(metric)?;
        let codec = Turbo4Codec::new(dimensions, rotation_seed).map_err(|e| {
            RuvectorError::InvalidParameter(format!("Turbo4 codec init failed: {e}"))
        })?;
        let distance_fn = Turbo4DistanceFn {
            metric,
            dim: dimensions,
            code_len: codec.code_len(),
            query_len: codec.query_len(),
        };
        let hnsw = Hnsw::<u8, Turbo4DistanceFn>::new(
            config.m,
            config.max_elements,
            dimensions,
            config.ef_construction,
            distance_fn,
        );
        Ok(Self {
            inner: Arc::new(RwLock::new(Turbo4Inner {
                hnsw,
                codes: DashMap::new(),
                id_to_idx: DashMap::new(),
                idx_to_id: DashMap::new(),
                next_idx: 0,
            })),
            codec: Arc::new(codec),
            config,
            metric,
            dimensions,
            rescore_multiplier: rescore_multiplier.max(1),
            escalation: EscalationParams::for_policy(policy),
            queries: AtomicU64::new(0),
            escalated: AtomicU64::new(0),
        })
    }

    /// Stored bytes per vector (`D/2 + 8`).
    pub fn code_len(&self) -> usize {
        self.codec.code_len()
    }

    /// Adaptive-plane telemetry: `(total_queries, escalated_queries)`.
    /// A healthy workload escalates only a small fraction (~5–15 %).
    pub fn adaptive_stats(&self) -> (u64, u64) {
        (
            self.queries.load(Ordering::Relaxed),
            self.escalated.load(Ordering::Relaxed),
        )
    }

    /// One traversal + exact-rescore pass. Returns the full rescored
    /// candidate list, ascending — the margin between `list[k-1]` and
    /// `list[k]` is the stability signal for adaptive escalation.
    fn traverse_and_rescore(
        &self,
        inner: &Turbo4Inner,
        prepared: &Turbo4Query,
        fetch: usize,
        ef: usize,
    ) -> Vec<SearchResult> {
        let neighbors = inner.hnsw.search(&prepared.blob, fetch, ef);
        let mut results: Vec<SearchResult> = neighbors
            .into_iter()
            .filter_map(|n| {
                let id = inner.idx_to_id.get(&n.d_id)?.clone();
                let code = inner.codes.get(&id)?;
                let dist = score::rescore(self.metric, prepared, code.value(), self.dimensions);
                Some(SearchResult {
                    id,
                    score: dist,
                    vector: None,
                    metadata: None,
                })
            })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Relative margin between the last kept and first dropped candidate.
    /// `f32::INFINITY` when nothing is dropped (result already exhaustive).
    fn boundary_margin(results: &[SearchResult], k: usize) -> f32 {
        if results.len() <= k || k == 0 {
            return f32::INFINITY;
        }
        let kept = results[k - 1].score;
        let dropped = results[k].score;
        (dropped - kept) / kept.abs().max(1e-6)
    }

    /// Search with an explicit `ef_search`.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }
        if k == 0 {
            return Ok(vec![]);
        }
        let prepared = self
            .codec
            .encode_query(query)
            .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;

        let inner = self.inner.read();
        // hnsw_rs panics on empty indexes (unguarded heap peek) — return early.
        if inner.codes.is_empty() {
            return Ok(vec![]);
        }
        self.queries.fetch_add(1, Ordering::Relaxed);

        // Base pass: over-fetch for the exact rescoring step.
        let mut fetch = k.saturating_mul(self.rescore_multiplier);
        let mut ef = ef_search.max(fetch);
        let mut results = self.traverse_and_rescore(&inner, &prepared, fetch, ef);

        // Adaptive escalation (ADR-297 §3): if the kept/dropped boundary sits
        // inside the quantization noise band, widen the search. Stop as soon
        // as the top-k membership is stable across rounds — that's the
        // candidate-stability signal, independent of absolute scores.
        let mut round = 0;
        let mut did_escalate = false;
        while round < self.escalation.max_rounds
            && Self::boundary_margin(&results, k) < self.escalation.margin_threshold
        {
            did_escalate = true;
            fetch = fetch.saturating_mul(2);
            ef = ef.saturating_mul(self.escalation.ef_mult);
            let wider = self.traverse_and_rescore(&inner, &prepared, fetch, ef);

            let stable = wider.len() >= k
                && results.len() >= k
                && wider[..k]
                    .iter()
                    .zip(&results[..k])
                    .all(|(a, b)| a.id == b.id);
            results = wider;
            round += 1;
            if stable {
                break;
            }
        }
        if did_escalate {
            self.escalated.fetch_add(1, Ordering::Relaxed);
        }

        results.truncate(k);
        Ok(results)
    }
}

impl VectorIndex for Turbo4HnswIndex {
    fn add(&mut self, id: VectorId, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        let blob = self
            .codec
            .encode(&vector)
            .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;
        drop(vector); // floats are not retained

        let mut inner = self.inner.write();
        let idx = inner.next_idx;
        inner.next_idx += 1;
        inner.hnsw.insert_data(&blob, idx);
        inner.codes.insert(id.clone(), blob);
        inner.id_to_idx.insert(id.clone(), idx);
        inner.idx_to_id.insert(idx, id);
        Ok(())
    }

    fn add_batch(&mut self, entries: Vec<(VectorId, Vec<f32>)>) -> Result<()> {
        for (_, vector) in &entries {
            if vector.len() != self.dimensions {
                return Err(RuvectorError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: vector.len(),
                });
            }
        }
        // Encode outside the lock — the expensive part (O(D log D) each).
        let mut encoded = Vec::with_capacity(entries.len());
        for (id, vector) in entries {
            let blob = self
                .codec
                .encode(&vector)
                .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;
            encoded.push((id, blob));
        }

        let mut inner = self.inner.write();
        let base = inner.next_idx;
        inner.next_idx += encoded.len();

        // Same parallel-insert gate as the f32 index (hnsw_rs rule of thumb:
        // parallel pays off only for large batches).
        const PARALLEL_THRESHOLD: usize = 10_000;
        if encoded.len() >= PARALLEL_THRESHOLD {
            let datas: Vec<(&[u8], usize)> = encoded
                .iter()
                .enumerate()
                .map(|(i, (_, blob))| (blob.as_slice(), base + i))
                .collect();
            inner.hnsw.parallel_insert_slice(&datas);
        } else {
            for (i, (_, blob)) in encoded.iter().enumerate() {
                inner.hnsw.insert_data(blob, base + i);
            }
        }

        for (i, (id, blob)) in encoded.into_iter().enumerate() {
            inner.codes.insert(id.clone(), blob);
            inner.id_to_idx.insert(id.clone(), base + i);
            inner.idx_to_id.insert(base + i, id);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    fn remove(&mut self, id: &VectorId) -> Result<bool> {
        let inner = self.inner.write();
        // hnsw_rs has no deletion; drop the mappings (graph node remains,
        // same limitation as the f32 index).
        let removed = inner.codes.remove(id).is_some();
        if removed {
            if let Some((_, idx)) = inner.id_to_idx.remove(id) {
                inner.idx_to_id.remove(&idx);
            }
        }
        Ok(removed)
    }

    fn len(&self) -> usize {
        self.inner.read().codes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Gaussian test vectors (SplitMix64 + Box–Muller).
    fn gauss_vecs(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        (0..count)
            .map(|_| {
                let mut v = Vec::with_capacity(dim);
                while v.len() < dim {
                    let u1 = (next() >> 11) as f64 / (1u64 << 53) as f64;
                    let u2 = (next() >> 11) as f64 / (1u64 << 53) as f64;
                    let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
                    let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
                    v.push((r * c) as f32);
                    if v.len() < dim {
                        v.push((r * s) as f32);
                    }
                }
                v
            })
            .collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn manhattan_is_rejected() {
        let err = Turbo4HnswIndex::new(
            64,
            DistanceMetric::Manhattan,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
        );
        assert!(err.is_err());
    }

    /// Clustered vectors: `n` points around `n / cluster_size` Gaussian
    /// centers — the neighbor structure real embedding corpora have.
    fn clustered_vecs(
        n: usize,
        dim: usize,
        cluster_size: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let n_centers = n.div_ceil(cluster_size);
        let centers = gauss_vecs(n_centers, dim, seed);
        let noise = gauss_vecs(n, dim, seed ^ 0xABCD_EF01);
        (0..n)
            .map(|i| {
                centers[i / cluster_size]
                    .iter()
                    .zip(&noise[i])
                    .map(|(c, e)| c + spread * e)
                    .collect()
            })
            .collect()
    }

    /// Recall@10 vs brute force, measured for BOTH the f32 HNSW and the
    /// Turbo4 HNSW on identical data/params — the ADR-296 acceptance metric
    /// is the *delta* to the f32 baseline. Data is clustered, like real
    /// embedding corpora (and SIFT/GIST). i.i.d. Gaussian data is deliberately
    /// NOT used here: with 500 points at 128-D all pairwise distances
    /// concentrate within a few percent, so the ~1% distance error of ANY
    /// 4-bit quantizer scrambles top-10 membership — that regime is covered
    /// by the absolute-floor smoke test below instead.
    #[test]
    fn recall_close_to_f32_hnsw_baseline() -> Result<()> {
        use crate::index::hnsw::HnswIndex;

        let dim = 128;
        let n = 500;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 1000,
        };
        let vectors = clustered_vecs(n, dim, 10, 0.35, 7);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();

        let mut t4 = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config.clone(),
            42,
            8,
            SearchPolicy::Balanced,
        )?;
        t4.add_batch(entries.clone())?;
        assert_eq!(t4.len(), n);
        let mut f32_ix = HnswIndex::new(dim, DistanceMetric::Euclidean, config)?;
        f32_ix.add_batch(entries)?;

        // Queries: data points perturbed with fresh noise — same clusters,
        // so each query has a meaningful true neighborhood.
        let qnoise = gauss_vecs(20, dim, 998877);
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|j| {
                vectors[(j * 25) % n]
                    .iter()
                    .zip(&qnoise[j])
                    .map(|(v, e)| v + 0.2 * e)
                    .collect()
            })
            .collect();
        let (mut t4_hits, mut f32_hits, mut total) = (0usize, 0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();

            let t4_res = t4.search(q, 10)?;
            assert_eq!(t4_res.len(), 10);
            t4_hits += t4_res.iter().filter(|r| top10.contains(&r.id)).count();
            f32_hits += f32_ix
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }
        let t4_recall = t4_hits as f32 / total as f32;
        let f32_recall = f32_hits as f32 / total as f32;
        assert!(
            t4_recall >= f32_recall - 0.02,
            "Turbo4 recall@10 {t4_recall} vs f32 baseline {f32_recall}: loss above 2pp \
             on clustered data ({t4_hits}/{f32_hits}/{total})"
        );
        assert!(
            t4_recall >= 0.90,
            "Turbo4 absolute recall@10 {t4_recall} below floor"
        );
        Ok(())
    }

    /// Worst-case smoke test: i.i.d. Gaussian data has near-total distance
    /// concentration (500 pts @ 128-D ⇒ all pairwise distances within a few
    /// percent), so any 4-bit quantizer loses top-10 members to ~1% distance
    /// noise. Assert a floor, not parity with f32.
    #[test]
    fn gaussian_worst_case_recall_floor() -> Result<()> {
        let dim = 128;
        let n = 500;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_elements: 1000,
        };
        let vectors = gauss_vecs(n, dim, 7);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        let mut t4 = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            8,
            SearchPolicy::Balanced,
        )?;
        t4.add_batch(entries)?;

        let queries = gauss_vecs(20, dim, 12345);
        let (mut hits, mut total) = (0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();
            hits += t4
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }
        let recall = hits as f32 / total as f32;
        assert!(
            recall >= 0.75,
            "Turbo4 worst-case Gaussian recall@10 {recall} below floor ({hits}/{total})"
        );
        Ok(())
    }

    #[test]
    fn self_query_returns_self_first() -> Result<()> {
        let dim = 64;
        let config = HnswConfig::default();
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Cosine,
            config,
            42,
            4,
            SearchPolicy::Balanced,
        )?;
        let vectors = gauss_vecs(50, dim, 3);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.clone())?;
        }
        let results = index.search(&vectors[7], 5)?;
        assert_eq!(results[0].id, "v7");
        Ok(())
    }

    /// Adaptive escalation (ADR-297 §3): MaxCompression must never escalate;
    /// Quality must escalate on distance-concentrated (uncertain) queries and
    /// must not lose recall relative to the non-escalating policy.
    #[test]
    fn adaptive_escalation_follows_policy() -> Result<()> {
        let dim = 128;
        let n = 400;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 60,
            max_elements: 1000,
        };
        // i.i.d. Gaussian ⇒ tight boundary margins ⇒ escalation should fire.
        let vectors = gauss_vecs(n, dim, 21);
        let entries: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();

        let mut fixed = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config.clone(),
            42,
            2,
            SearchPolicy::MaxCompression,
        )?;
        fixed.add_batch(entries.clone())?;
        let mut adaptive = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            2,
            SearchPolicy::Quality,
        )?;
        adaptive.add_batch(entries)?;

        let queries = gauss_vecs(25, dim, 4242);
        let (mut fixed_hits, mut adaptive_hits, mut total) = (0usize, 0usize, 0usize);
        for q in &queries {
            let mut truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, l2(q, v)))
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            let top10: std::collections::HashSet<String> =
                truth[..10].iter().map(|(i, _)| format!("v{i}")).collect();
            fixed_hits += fixed
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            adaptive_hits += adaptive
                .search(q, 10)?
                .iter()
                .filter(|r| top10.contains(&r.id))
                .count();
            total += 10;
        }

        let (fq, fe) = fixed.adaptive_stats();
        assert_eq!(fq, 25);
        assert_eq!(fe, 0, "MaxCompression must never escalate");

        let (aq, ae) = adaptive.adaptive_stats();
        assert_eq!(aq, 25);
        assert!(
            ae > 0,
            "Quality policy should escalate on concentrated Gaussian queries"
        );
        assert!(
            adaptive_hits >= fixed_hits,
            "escalation must not lose recall: adaptive {adaptive_hits} vs fixed {fixed_hits} of {total}"
        );
        Ok(())
    }

    /// Easy well-separated queries must not trigger escalation under
    /// Balanced — the ~5–15 % escalation budget depends on margins staying
    /// wide when results are unambiguous.
    #[test]
    fn easy_queries_do_not_escalate() -> Result<()> {
        let dim = 64;
        let config = HnswConfig::default();
        let mut index = Turbo4HnswIndex::new(
            dim,
            DistanceMetric::Euclidean,
            config,
            42,
            4,
            SearchPolicy::Balanced,
        )?;
        // Well-separated clusters of exactly k members, so the kept/dropped
        // boundary falls BETWEEN clusters (wide margin). A boundary inside a
        // cluster is genuinely ambiguous and escalating on it is correct.
        let vectors = clustered_vecs(200, dim, 5, 0.05, 33);
        for (i, v) in vectors.iter().enumerate() {
            index.add(format!("v{i}"), v.iter().map(|x| x * 10.0).collect())?;
        }
        for j in 0..10 {
            let q: Vec<f32> = vectors[j * 20].iter().map(|x| x * 10.0).collect();
            index.search(&q, 5)?;
        }
        let (queries, escalated) = index.adaptive_stats();
        assert_eq!(queries, 10);
        assert!(
            escalated <= 2,
            "well-separated queries escalated {escalated}/10 times"
        );
        Ok(())
    }

    #[test]
    fn empty_index_is_safe() -> Result<()> {
        let index = Turbo4HnswIndex::new(
            64,
            DistanceMetric::Euclidean,
            HnswConfig::default(),
            42,
            4,
            SearchPolicy::Balanced,
        )?;
        assert!(index.search(&vec![0.5; 64], 5)?.is_empty());
        Ok(())
    }
}
