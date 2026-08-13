//! Retrieval-receipt benchmark: measures the real cost and tamper-detection
//! reliability of attaching witness-chained provenance to ANN query results.
//!
//! Usage:
//!   cargo run --release -p ruvector-retrieval-receipt --bin benchmark
//!   cargo run --release -p ruvector-retrieval-receipt --bin benchmark -- 5000 128 10 200

use std::time::{Duration, Instant};

use ruvector_retrieval_receipt::{
    query_hash, synthetic_queries, ReceiptVariant, ResultItem, RetrievalIndex, RetrievalReceipt,
};

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct VariantStats {
    variant: ReceiptVariant,
    gen_mean_ns: f64,
    gen_p95_ns: u64,
    verify_worst_mean_ns: f64,
    proof_bytes_worst: usize,
    total_receipt_bytes_mean: f64,
    tamper_trials: usize,
    tamper_detected: usize,
}

/// Deterministic xorshift for tamper-trial selection (which query, which
/// tamper kind), independent of the dataset/query streams so tamper trials
/// are reproducible without adding an external RNG dependency.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[derive(Clone, Copy)]
enum TamperKind {
    ScoreMutation,
    VectorIdSubstitution,
    RankSwap,
    WriteReceiptHashFlip,
}

fn apply_tamper(results: &mut [ResultItem], kind: TamperKind, rng: &mut Xorshift64) -> usize {
    let idx = rng.next_range(results.len());
    match kind {
        TamperKind::ScoreMutation => {
            results[idx].score += 0.5;
        }
        TamperKind::VectorIdSubstitution => {
            results[idx].vector_id = results[idx].vector_id.wrapping_add(999_999);
        }
        TamperKind::RankSwap => {
            let other = (idx + 1) % results.len();
            results.swap(idx, other);
        }
        TamperKind::WriteReceiptHashFlip => {
            results[idx].write_receipt.payload_hash[0] ^= 0xFF;
        }
    }
    idx
}

fn run_variant(
    variant: ReceiptVariant,
    index: &RetrievalIndex,
    queries: &[Vec<f32>],
    k: usize,
    index_root: [u8; 32],
    tamper_trials_per_kind: usize,
) -> VariantStats {
    let mut gen_latencies = Vec::with_capacity(queries.len());
    let mut verify_worst_latencies = Vec::with_capacity(queries.len());
    let mut total_bytes = Vec::with_capacity(queries.len());
    let mut proof_bytes_worst = 0usize;

    for query in queries {
        let results = index.search(query, k);
        let qh = query_hash(query);

        let t0 = Instant::now();
        let receipt = RetrievalReceipt::build(variant, qh, index_root, &results);
        gen_latencies.push(t0.elapsed());

        let worst_idx = results.len().saturating_sub(1);
        proof_bytes_worst = receipt.proof_bytes_for(worst_idx);
        total_bytes.push(receipt.total_bytes() as f64);

        if variant != ReceiptVariant::None {
            let t1 = Instant::now();
            let ok = receipt.verify_item(worst_idx, qh, index_root, &results[worst_idx]);
            verify_worst_latencies.push(t1.elapsed());
            assert!(ok, "honest receipt must verify");
        }
    }

    gen_latencies.sort_unstable();
    let gen_mean_ns = gen_latencies.iter().map(|d| d.as_nanos()).sum::<u128>() as f64
        / gen_latencies.len() as f64;
    let gen_p95_ns = percentile(&gen_latencies, 95.0).as_nanos() as u64;

    let verify_worst_mean_ns = if verify_worst_latencies.is_empty() {
        0.0
    } else {
        verify_worst_latencies
            .iter()
            .map(|d| d.as_nanos())
            .sum::<u128>() as f64
            / verify_worst_latencies.len() as f64
    };

    let total_receipt_bytes_mean = total_bytes.iter().sum::<f64>() / total_bytes.len() as f64;

    // Tamper-detection trials: for each tamper kind, corrupt a fresh honest
    // result set and confirm verify_full() rejects it. None/no-receipt is
    // skipped: there is nothing to verify, by design.
    let mut tamper_trials = 0usize;
    let mut tamper_detected = 0usize;
    if variant != ReceiptVariant::None {
        let kinds = [
            TamperKind::ScoreMutation,
            TamperKind::VectorIdSubstitution,
            TamperKind::RankSwap,
            TamperKind::WriteReceiptHashFlip,
        ];
        let mut rng = Xorshift64(0xF00D_CAFE_1234_5678);
        for kind in kinds {
            for trial in 0..tamper_trials_per_kind {
                let query = &queries[trial % queries.len()];
                let results = index.search(query, k);
                let qh = query_hash(query);
                let receipt = RetrievalReceipt::build(variant, qh, index_root, &results);
                let mut tampered = results.clone();
                apply_tamper(&mut tampered, kind, &mut rng);
                tamper_trials += 1;
                if !receipt.verify_full(qh, index_root, &tampered) {
                    tamper_detected += 1;
                }
            }
        }
    }

    VariantStats {
        variant,
        gen_mean_ns,
        gen_p95_ns,
        verify_worst_mean_ns,
        proof_bytes_worst,
        total_receipt_bytes_mean,
        tamper_trials,
        tamper_detected,
    }
}

fn variant_name(v: ReceiptVariant) -> &'static str {
    match v {
        ReceiptVariant::None => "NoReceipt",
        ReceiptVariant::PerResult => "PerResultReceipt",
        ReceiptVariant::Merkle => "MerkleReceipt",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let num_queries: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let tamper_trials_per_kind = 50usize; // 4 kinds x 50 = 200 trials per variant

    println!("=== ruvector-retrieval-receipt benchmark ===");
    println!(
        "n={n} dims={dims} k={k} queries={num_queries} tamper_trials_per_kind={tamper_trials_per_kind}"
    );
    println!(
        "hardware: {} logical CPUs (see `nproc`), rustc build profile: release-required for meaningful numbers",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );

    let t_ingest = Instant::now();
    let index = RetrievalIndex::ingest(n, dims, 0xC0FF_EE01_D00D);
    let ingest_elapsed = t_ingest.elapsed();
    assert!(index.verify_write_history(), "write history must verify");
    println!(
        "ingest: {n} vectors in {:.3} ms ({:.1} writes/ms), index_state_root non-zero: {}",
        ingest_elapsed.as_secs_f64() * 1000.0,
        n as f64 / (ingest_elapsed.as_secs_f64() * 1000.0),
        index.index_state_root() != [0u8; 32]
    );

    let queries = synthetic_queries(num_queries, dims, 0xA5A5_5A5A_1111);
    let index_root = index.index_state_root();

    // Baseline search-only latency (shared across all variants; receipts
    // are built *after* the same brute-force scan, so this isolates the
    // provenance layer's added cost from retrieval cost).
    let mut search_latencies = Vec::with_capacity(queries.len());
    for query in &queries {
        let t0 = Instant::now();
        let _ = index.search(query, k);
        search_latencies.push(t0.elapsed());
    }
    search_latencies.sort_unstable();
    let search_mean_ns = search_latencies.iter().map(|d| d.as_nanos()).sum::<u128>() as f64
        / search_latencies.len() as f64;
    println!(
        "\nbaseline brute-force search: mean={:.0}ns p95={:.0}ns over {} queries",
        search_mean_ns,
        percentile(&search_latencies, 95.0).as_nanos(),
        queries.len()
    );

    let variants = [
        ReceiptVariant::None,
        ReceiptVariant::PerResult,
        ReceiptVariant::Merkle,
    ];
    let stats: Vec<VariantStats> = variants
        .iter()
        .map(|&v| run_variant(v, &index, &queries, k, index_root, tamper_trials_per_kind))
        .collect();

    println!(
        "\n{:<18} {:>14} {:>14} {:>16} {:>14} {:>18} {:>16}",
        "variant",
        "gen_mean_ns",
        "gen_p95_ns",
        "verify_worst_ns",
        "proof_bytes",
        "total_bytes_mean",
        "tamper_detect"
    );
    for s in &stats {
        let tamper_str = if s.tamper_trials == 0 {
            "n/a".to_string()
        } else {
            format!("{}/{}", s.tamper_detected, s.tamper_trials)
        };
        println!(
            "{:<18} {:>14.0} {:>14} {:>16.0} {:>14} {:>18.1} {:>16}",
            variant_name(s.variant),
            s.gen_mean_ns,
            s.gen_p95_ns,
            s.verify_worst_mean_ns,
            s.proof_bytes_worst,
            s.total_receipt_bytes_mean,
            tamper_str
        );
    }

    // ── Acceptance evaluation (thresholds fixed before this run; see the
    // nightly research README for the formalized hypothesis) ──────────────
    let per_result = &stats[1];
    let merkle = &stats[2];

    let all_tamper_detected = stats
        .iter()
        .filter(|s| s.tamper_trials > 0)
        .all(|s| s.tamper_detected == s.tamper_trials);

    let merkle_proof_smaller = merkle.proof_bytes_worst < per_result.proof_bytes_worst;

    let overhead_threshold = 0.15; // receipt generation must add < 15% of baseline search latency
    let merkle_overhead = merkle.gen_mean_ns / search_mean_ns;
    let per_result_overhead = per_result.gen_mean_ns / search_mean_ns;
    let overhead_ok =
        merkle_overhead < overhead_threshold && per_result_overhead < overhead_threshold;

    println!("\n=== acceptance ===");
    println!("tamper detection 100% across all kinds: {all_tamper_detected}");
    println!(
        "merkle worst-case proof bytes ({}) < per-result worst-case proof bytes ({}): {merkle_proof_smaller}",
        merkle.proof_bytes_worst, per_result.proof_bytes_worst
    );
    println!(
        "generation overhead < {:.0}% of baseline search: merkle={:.1}% per_result={:.1}% -> {overhead_ok}",
        overhead_threshold * 100.0,
        merkle_overhead * 100.0,
        per_result_overhead * 100.0
    );

    let verdict = if all_tamper_detected && merkle_proof_smaller && overhead_ok {
        "ACCEPT"
    } else if all_tamper_detected && merkle_proof_smaller {
        "INCONCLUSIVE" // core provenance claims hold, overhead threshold missed
    } else {
        "REJECT"
    };
    println!("\nACCEPTANCE RESULT: {verdict}");
}
