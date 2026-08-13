//! Witness-chained provenance receipts for ANN retrieval results.
//!
//! `ruvector-proof-gate` answers "what was written, and can I prove it
//! hasn't been tampered with?" for the *write* path. This crate answers
//! the symmetric question for the *read* path: "what did this query
//! actually return, against which index state, and can an auditor
//! independently confirm that — without re-running the query or trusting
//! whoever ran it?"
//!
//! Every result item is bound into its receipt together with the
//! `WriteReceipt` produced when that vector was ingested, so a retrieval
//! receipt is not just "these IDs and scores were returned" but "these IDs,
//! scores, and their entire write-provenance chain were returned" — closing
//! the loop between ingestion integrity and retrieval integrity that a RAG
//! auditor needs to replay an agent's evidence trail.
//!
//! # Variants
//!
//! | Variant         | Per-query build cost | Evidence to verify 1 of k results | Guarantee |
//! |-----------------|----------------------|-------------------------------------|-----------|
//! | `NoReceipt`      | 0                    | none (unverifiable)                 | none |
//! | `PerResultReceipt` | O(k) hashes        | O(idx) bytes, O(idx) work            | sequential tamper-evidence |
//! | `MerkleReceipt`  | O(k) hashes          | O(log k) bytes, O(log k) work        | membership-proof tamper-evidence |

mod index;
mod receipt;

pub use index::{synthetic_queries, ResultItem, RetrievalIndex};
pub use receipt::{query_hash, MerkleReceipt, PerResultReceipt, ReceiptVariant};

/// A built receipt for one query's result set, in whichever variant was
/// requested. Carries enough state to answer `proof_bytes_for` /
/// `verify_item` / `verify_full` uniformly across variants for benchmarking.
pub enum RetrievalReceipt {
    None,
    PerResult(PerResultReceipt),
    Merkle(MerkleReceipt),
}

impl RetrievalReceipt {
    pub fn build(
        variant: ReceiptVariant,
        qh: [u8; 32],
        index_root: [u8; 32],
        results: &[ResultItem],
    ) -> Self {
        match variant {
            ReceiptVariant::None => RetrievalReceipt::None,
            ReceiptVariant::PerResult => {
                RetrievalReceipt::PerResult(PerResultReceipt::build(qh, index_root, results))
            }
            ReceiptVariant::Merkle => {
                RetrievalReceipt::Merkle(MerkleReceipt::build(qh, index_root, results))
            }
        }
    }

    pub fn variant(&self) -> ReceiptVariant {
        match self {
            RetrievalReceipt::None => ReceiptVariant::None,
            RetrievalReceipt::PerResult(_) => ReceiptVariant::PerResult,
            RetrievalReceipt::Merkle(_) => ReceiptVariant::Merkle,
        }
    }

    /// Bytes of evidence required to verify a single result at `idx`
    /// without trusting a live index/gate instance. `None` returns `0` —
    /// by construction there is nothing to verify, which is the point.
    pub fn proof_bytes_for(&self, idx: usize) -> usize {
        match self {
            RetrievalReceipt::None => 0,
            RetrievalReceipt::PerResult(r) => r.proof_bytes_for(idx),
            RetrievalReceipt::Merkle(r) => r.proof_bytes_for(idx),
        }
    }

    /// Total receipt payload size (all results), independent of which item
    /// is later spot-checked.
    pub fn total_bytes(&self) -> usize {
        match self {
            RetrievalReceipt::None => 0,
            RetrievalReceipt::PerResult(r) => r.total_bytes(),
            RetrievalReceipt::Merkle(r) => r.total_bytes(),
        }
    }

    /// Verify a single result item. `NoReceipt` always returns `false`:
    /// there is no evidence to check, which is intentional and distinct
    /// from "verification passed."
    pub fn verify_item(
        &self,
        idx: usize,
        qh: [u8; 32],
        index_root: [u8; 32],
        item: &ResultItem,
    ) -> bool {
        match self {
            RetrievalReceipt::None => false,
            RetrievalReceipt::PerResult(r) => r.verify_item(idx, qh, index_root, item),
            RetrievalReceipt::Merkle(r) => {
                let proof = r.proof_for(idx);
                r.verify_item(idx, qh, index_root, item, r.root, &proof)
            }
        }
    }

    pub fn verify_full(&self, qh: [u8; 32], index_root: [u8; 32], results: &[ResultItem]) -> bool {
        match self {
            RetrievalReceipt::None => false,
            RetrievalReceipt::PerResult(r) => r.verify_full(qh, index_root, results),
            RetrievalReceipt::Merkle(r) => r.verify_full(qh, index_root, results),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(
        n: usize,
        dims: usize,
        k: usize,
    ) -> (
        RetrievalIndex,
        Vec<f32>,
        Vec<ResultItem>,
        [u8; 32],
        [u8; 32],
    ) {
        let index = RetrievalIndex::ingest(n, dims, 0xC0FF_EE01);
        let query = synthetic_queries(1, dims, 0xA5A5_5A5A)[0].clone();
        let results = index.search(&query, k);
        let qh = query_hash(&query);
        let root = index.index_state_root();
        (index, query, results, qh, root)
    }

    #[test]
    fn ingestion_produces_verifiable_write_history() {
        let (index, _, _, _, _) = setup(200, 16, 5);
        assert!(index.verify_write_history());
        assert_ne!(index.index_state_root(), [0u8; 32]);
    }

    #[test]
    fn search_returns_k_results_in_descending_score_order() {
        let (_, _, results, _, _) = setup(200, 16, 5);
        assert_eq!(results.len(), 5);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn no_receipt_never_verifies() {
        let (_, _, results, qh, root) = setup(100, 16, 4);
        let receipt = RetrievalReceipt::build(ReceiptVariant::None, qh, root, &results);
        assert_eq!(receipt.proof_bytes_for(0), 0);
        assert!(!receipt.verify_item(0, qh, root, &results[0]));
        assert!(!receipt.verify_full(qh, root, &results));
    }

    #[test]
    fn per_result_receipt_verifies_honest_results() {
        let (_, _, results, qh, root) = setup(500, 32, 10);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        assert!(receipt.verify_full(qh, root, &results));
        for (i, item) in results.iter().enumerate() {
            assert!(receipt.verify_item(i, qh, root, item));
        }
    }

    #[test]
    fn merkle_receipt_verifies_honest_results() {
        let (_, _, results, qh, root) = setup(500, 32, 10);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        assert!(receipt.verify_full(qh, root, &results));
        for (i, item) in results.iter().enumerate() {
            assert!(receipt.verify_item(i, qh, root, item));
        }
    }

    #[test]
    fn per_result_receipt_detects_score_tamper() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let mut tampered = results.clone();
        tampered[3].score += 1.0;
        assert!(!receipt.verify_item(3, qh, root, &tampered[3]));
    }

    #[test]
    fn merkle_receipt_detects_score_tamper() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let mut tampered = results.clone();
        tampered[3].score += 1.0;
        assert!(!receipt.verify_item(3, qh, root, &tampered[3]));
    }

    #[test]
    fn per_result_receipt_detects_reorder() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let mut tampered = results.clone();
        tampered.swap(2, 5);
        assert!(!receipt.verify_full(qh, root, &tampered));
    }

    #[test]
    fn merkle_receipt_detects_reorder() {
        let (_, _, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let mut tampered = results.clone();
        tampered.swap(2, 5);
        assert!(!receipt.verify_full(qh, root, &tampered));
    }

    #[test]
    fn merkle_receipt_detects_vector_id_substitution() {
        let (index, query, results, qh, root) = setup(300, 24, 8);
        let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        // Substitute a truthful-looking but different vector's write receipt
        // in place of result 0 — simulates an adversary swapping in evidence
        // for a different, unretrieved memory.
        let other = index.search(&query, 9)[8].clone();
        let mut tampered = results.clone();
        tampered[0] = other;
        assert!(!receipt.verify_item(0, qh, root, &tampered[0]));
    }

    #[test]
    fn merkle_proof_bytes_are_sublinear_vs_per_result_at_k10() {
        let (_, _, results, qh, root) = setup(2000, 64, 10);
        let per_result = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &results);
        let merkle = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);
        let worst_idx = results.len() - 1;
        assert!(
            merkle.proof_bytes_for(worst_idx) < per_result.proof_bytes_for(worst_idx),
            "merkle worst-case proof ({} bytes) must be smaller than per-result worst-case ({} bytes)",
            merkle.proof_bytes_for(worst_idx),
            per_result.proof_bytes_for(worst_idx)
        );
    }

    #[test]
    fn index_state_root_changes_receipts_across_reingestion() {
        let index_a = RetrievalIndex::ingest(50, 8, 1);
        let index_b = RetrievalIndex::ingest(50, 8, 2);
        assert_ne!(index_a.index_state_root(), index_b.index_state_root());
    }
}
