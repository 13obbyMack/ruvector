# ruvector-retrieval-receipt

**Witness-chained provenance for ANN retrieval results** — cryptographic receipts that commit a
query's top-k results (together with copies of each vector's ingestion `WriteReceipt`) so that a
receipt/result pair, once issued, cannot be silently mutated in transit or in storage. Part of
the [ruvector](https://github.com/ruvnet/ruvector) ecosystem.

> `ruvector-proof-gate` proves what was *written*. This crate makes the record of what a query
> *returned* tamper-evident after issuance — a read-side provenance primitive no major vector
> database (Qdrant, Milvus, Weaviate, LanceDB, FAISS, pgvector, Chroma, Vespa, Pinecone)
> documents today.

## What it gives you

Search a `RetrievalIndex` (a brute-force cosine index whose ingestion path is a real
`ruvector_proof_gate::HashChainGate`), wrap the result set in a `RetrievalReceipt`, and a later
holder of the receipt can check — offline, without talking to the query engine — that the
results they hold are the ones the engine committed to at query time.

**Threat model, stated plainly:** receipts are unsigned commitments produced by the query
engine itself. They detect *post-issuance mutation* of a receipt/result pair. They do **not**
protect against a dishonest query engine (leaves are engine-chosen; nothing binds a score to an
actual cosine computation or the committed set to the true top-k), and they do **not** prove
write-chain membership — leaves commit to *copies* of `WriteReceipt` fields, verification never
consults the write gate, so mutating the ingestion history after issuance leaves existing
receipts verifying. Anchoring leaves to `MerkleGate`'s MMR membership proofs is the named
future-work item. See ADR-304's Threat Model section.

## Variants

| Variant | Generation | Verify 1-of-k (worst case, k=10) | Proof size (worst case, k=10) | Guarantee |
|---|---|---|---|---|
| `NoReceipt` | ~0 | N/A | 0 bytes | none (baseline) |
| `PerResultReceipt` | O(k) hashes | O(idx) work | O(idx) bytes | sequential tamper-evidence |
| `MerkleReceipt` | O(k) hashes | O(log k) work | O(log k) bytes | membership-proof tamper-evidence |

## Usage

```rust
use ruvector_retrieval_receipt::{
    query_hash, ReceiptVariant, RetrievalIndex, RetrievalReceipt,
};

let index = RetrievalIndex::ingest(5_000, 128, 0xC0FF_EE01);
let query = vec![0.1; 128];
let results = index.search(&query, 10);

let qh = query_hash(&query);
let root = index.index_state_root();
let receipt = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &results);

assert!(receipt.verify_full(qh, root, &results));
```

## Performance

Measured (n=5,000, dims=128, k=10, release build): `MerkleReceipt` generation ≈ 19.6 µs
(1.8% of a 1.1 ms brute-force search), single-result verification ≈ 3.8 µs, worst-case proof
size 160 bytes, vs 320 bytes / 8.2 µs for `PerResultReceipt` — where the per-result figure is
defined as the genesis-anchored chain replay (O(idx)); the durable comparison is the
asymptotic O(log k) vs O(k) proof size, not the specific constant at k=10. Both variants
rejected all 200/200 injected tamper trials — expected from SHA-256 by construction, a
regression check rather than an empirical detection rate. Full methodology and raw output:
[`docs/research/nightly/2026-08-13-retrieval-receipts/README.md`](../../docs/research/nightly/2026-08-13-retrieval-receipts/README.md).

See [`ADR-304`](../../docs/adr/ADR-304-retrieval-receipts.md) for the design rationale,
documented limitations (Merkle padding malleability), and rejection criteria for production
promotion.
