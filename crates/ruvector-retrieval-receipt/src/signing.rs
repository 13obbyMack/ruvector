//! Ed25519 root-signing for retrieval receipts — closes the non-repudiation
//! gap the unsigned [`crate::receipt`] module leaves open (see the crate
//! docs and ADR-304's Threat Model): an unsigned receipt lets a holder
//! detect post-issuance tamper, but nothing binds a root to *which issuer*
//! vouched for it. Two anchoring strategies, both layered on top of an
//! existing [`crate::RetrievalReceipt`] rather than replacing it:
//!
//! - [`Issuer::sign_root`] / [`verify_root`]: sign a single receipt's root
//!   (`PerResultReceipt::chain_head` or `MerkleReceipt::root`, exposed via
//!   [`crate::RetrievalReceipt::root`]) directly. One Ed25519 signature per
//!   query — a batch of size 1 in [`BatchAnchor`] terms.
//! - [`BatchAnchor`]: build a second Merkle tree over *B* receipt roots and
//!   sign only the batch root. A verifier who has already checked the
//!   batch signature needs only an O(log B) inclusion proof (no further
//!   signature operations) to authenticate any one query's receipt root
//!   against it — a real amortization, but only if the verifier actually
//!   caches the one signature check per batch; see the benchmark's
//!   `verify_naive` vs `verify_cached` split for what happens if it
//!   doesn't.
//!
//! Neither strategy changes what an unsigned receipt already proves; they
//! add proof of *origin*, non-repudiably, at a measured cost. What signing
//! does **not** do: it does not make the issuer honest (a malicious issuer
//! can sign a false root just as validly as a true one), and a batch
//! signature does not exist — and so cannot be checked — until the batch
//! closes, a real end-to-end latency cost this in-process benchmark does
//! not model (see the nightly research README's Limitations section).

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

/// An Ed25519 keypair standing in for a query engine's receipt-issuance
/// identity. In production this would be a long-lived, HSM- or
/// KMS-backed key; here it is generated fresh per benchmark/test run.
pub struct Issuer {
    signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl Issuer {
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Sign a 32-byte receipt root or batch root. Returns the raw 64-byte
    /// Ed25519 signature.
    pub fn sign_root(&self, root: [u8; 32]) -> [u8; 64] {
        self.signing_key.sign(&root).to_bytes()
    }
}

/// Verify a root's signature against `vk`. Returns `false` (never panics)
/// on a malformed signature, matching the fail-closed convention the rest
/// of the crate uses for verification.
pub fn verify_root(vk: &VerifyingKey, root: [u8; 32], sig_bytes: [u8; 64]) -> bool {
    let sig = Signature::from_bytes(&sig_bytes);
    vk.verify(&root, &sig).is_ok()
}

const BATCH_LEAF_DOMAIN: &[u8] = b"ruvector:retrieval:batch:leaf:";
const BATCH_NODE_DOMAIN: &[u8] = b"ruvector:retrieval:batch:node:";

fn batch_leaf(receipt_root: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(BATCH_LEAF_DOMAIN);
    h.update(receipt_root);
    h.finalize().into()
}

fn batch_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(BATCH_NODE_DOMAIN); // domain-separated from `receipt::node_hash`
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// A Merkle tree over a *batch* of receipt roots, anchored by a single
/// Ed25519 signature over `root`. Trades per-query signing latency for
/// throughput: a batch of size B pays one signature for B queries, at the
/// cost of an O(log B) inclusion proof per query and — not modeled here —
/// the wall-clock delay of waiting for the batch to fill before a signed
/// anchor exists at all. `B = 1` degenerates to per-query signing (one
/// extra domain-separated hash over the raw root, negligible next to an
/// Ed25519 signature).
pub struct BatchAnchor {
    levels: Vec<Vec<[u8; 32]>>,
    pub root: [u8; 32],
}

impl BatchAnchor {
    pub fn build(receipt_roots: &[[u8; 32]]) -> Self {
        assert!(
            !receipt_roots.is_empty(),
            "batch must contain at least one receipt root"
        );
        let leaves: Vec<[u8; 32]> = receipt_roots.iter().map(batch_leaf).collect();
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len().div_ceil(2));
            let mut i = 0;
            while i < cur.len() {
                if i + 1 < cur.len() {
                    next.push(batch_node(&cur[i], &cur[i + 1]));
                } else {
                    // odd tail: duplicate the last node, same documented
                    // scheme (and same non-issue in this deployment shape)
                    // as `receipt::MerkleReceipt`.
                    next.push(batch_node(&cur[i], &cur[i]));
                }
                i += 2;
            }
            levels.push(next);
        }
        let root = levels.last().unwrap()[0];
        Self { levels, root }
    }

    pub fn proof_for(&self, idx: usize) -> Vec<([u8; 32], bool)> {
        let mut proof = Vec::new();
        let mut i = idx;
        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_idx = if i % 2 == 0 { i + 1 } else { i - 1 };
            let sibling = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                level[i]
            };
            proof.push((sibling, i % 2 == 0));
            i /= 2;
        }
        proof
    }

    pub fn proof_bytes_for(&self, idx: usize) -> usize {
        32 + self.proof_for(idx).len() * 32
    }

    /// Verify that `receipt_root` is a leaf committed under `root`, given
    /// an inclusion `proof`. Does **not** itself check any signature —
    /// callers verify `root`'s signature once (via [`verify_root`]) and
    /// may reuse that result across every query in the batch.
    pub fn verify_inclusion(
        receipt_root: [u8; 32],
        proof: &[([u8; 32], bool)],
        root: [u8; 32],
    ) -> bool {
        let mut node = batch_leaf(&receipt_root);
        for (sibling, sibling_is_right) in proof {
            node = if *sibling_is_right {
                batch_node(&node, sibling)
            } else {
                batch_node(sibling, &node)
            };
        }
        node == root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_roundtrip_succeeds() {
        let issuer = Issuer::generate();
        let root = [7u8; 32];
        let sig = issuer.sign_root(root);
        assert!(verify_root(&issuer.verifying_key, root, sig));
    }

    #[test]
    fn tampered_root_fails_verification() {
        let issuer = Issuer::generate();
        let root = [7u8; 32];
        let sig = issuer.sign_root(root);
        let mut wrong_root = root;
        wrong_root[0] ^= 0xFF;
        assert!(!verify_root(&issuer.verifying_key, wrong_root, sig));
    }

    #[test]
    fn tampered_signature_byte_fails_verification() {
        let issuer = Issuer::generate();
        let root = [7u8; 32];
        let mut sig = issuer.sign_root(root);
        sig[0] ^= 0xFF;
        assert!(!verify_root(&issuer.verifying_key, root, sig));
    }

    #[test]
    fn wrong_issuer_key_fails_verification() {
        let issuer = Issuer::generate();
        let impostor = Issuer::generate();
        let root = [7u8; 32];
        let sig = issuer.sign_root(root);
        assert!(!verify_root(&impostor.verifying_key, root, sig));
    }

    fn sample_roots(n: usize) -> Vec<[u8; 32]> {
        (0..n)
            .map(|i| {
                let mut r = [0u8; 32];
                r[0] = i as u8;
                r[1] = (i >> 8) as u8;
                r
            })
            .collect()
    }

    #[test]
    fn batch_anchor_verifies_all_members() {
        for n in [1usize, 2, 3, 8, 17, 128] {
            let roots = sample_roots(n);
            let anchor = BatchAnchor::build(&roots);
            for (i, root) in roots.iter().enumerate() {
                let proof = anchor.proof_for(i);
                assert!(
                    BatchAnchor::verify_inclusion(*root, &proof, anchor.root),
                    "batch size {n}, index {i} must verify"
                );
            }
        }
    }

    #[test]
    fn batch_of_one_root_equals_its_own_leaf_hash() {
        let roots = sample_roots(1);
        let anchor = BatchAnchor::build(&roots);
        assert_eq!(anchor.root, batch_leaf(&roots[0]));
    }

    #[test]
    fn batch_anchor_rejects_wrong_leaf() {
        let roots = sample_roots(8);
        let anchor = BatchAnchor::build(&roots);
        let proof = anchor.proof_for(3);
        let mut wrong = roots[3];
        wrong[0] ^= 0xFF;
        assert!(!BatchAnchor::verify_inclusion(wrong, &proof, anchor.root));
    }

    #[test]
    fn batch_anchor_rejects_tampered_proof_sibling() {
        let roots = sample_roots(8);
        let anchor = BatchAnchor::build(&roots);
        let mut proof = anchor.proof_for(3);
        proof[0].0[0] ^= 0xFF;
        assert!(!BatchAnchor::verify_inclusion(
            roots[3],
            &proof,
            anchor.root
        ));
    }

    #[test]
    fn batch_anchor_proof_bytes_grow_logarithmically() {
        let small = BatchAnchor::build(&sample_roots(2));
        let large = BatchAnchor::build(&sample_roots(128));
        assert!(small.proof_bytes_for(0) < large.proof_bytes_for(0));
        // log2(128) = 7 levels -> 32 (root) + 7*32 = 256 bytes
        assert_eq!(large.proof_bytes_for(0), 32 + 7 * 32);
    }

    #[test]
    fn end_to_end_batch_signing_detects_root_and_signature_tamper() {
        let issuer = Issuer::generate();
        let roots = sample_roots(16);
        let anchor = BatchAnchor::build(&roots);
        let sig = issuer.sign_root(anchor.root);
        assert!(verify_root(&issuer.verifying_key, anchor.root, sig));

        let idx = 5;
        let proof = anchor.proof_for(idx);
        assert!(BatchAnchor::verify_inclusion(
            roots[idx],
            &proof,
            anchor.root
        ));

        // Signature tamper: batch root itself is untouched but the
        // signature is corrupted -> signature check must fail even though
        // inclusion still holds.
        let mut bad_sig = sig;
        bad_sig[10] ^= 0xFF;
        assert!(!verify_root(&issuer.verifying_key, anchor.root, bad_sig));

        // Root citation tamper: a different receipt root is claimed to be
        // member `idx` -> inclusion check must fail even though the
        // signature (over the true, untouched batch root) still verifies.
        let mut forged_root = roots[idx];
        forged_root[0] ^= 0xFF;
        assert!(!BatchAnchor::verify_inclusion(
            forged_root,
            &proof,
            anchor.root
        ));
    }
}
