# ruvector-retrieval-receipt

**Witness-chained provenance for ANN retrieval results** — cryptographic receipts that bind a
query's top-k results to the write-provenance of every returned vector, so a retrieval event can
be audited independently of the system that ran the query. Part of the
[ruvector](https://github.com/ruvnet/ruvector) ecosystem.

> `ruvector-proof-gate` proves what was *written*. This crate proves what a query actually
> *returned* — the read-side half of agent-memory provenance that no major vector database
> (Qdrant, Milvus, Weaviate, LanceDB, FAISS, pgvector, Chroma, Vespa, Pinecone) documents today.

## What it gives you

Search a `RetrievalIndex` (a brute-force cosine index whose ingestion path is a real
`ruvector_proof_gate::HashChainGate`), wrap the result set in a `RetrievalReceipt`, and hand the
receipt to a verifier who never has to trust — or even talk to — the system that ran the query.

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
size 160 bytes — 2x smaller and 2.1x faster to verify than `PerResultReceipt`'s equivalent
(320 bytes, 8.2 µs). Both variants detect 100% (200/200) of injected tampering across four
tamper kinds. Full methodology and raw output:
[`docs/research/nightly/2026-08-13-retrieval-receipts/README.md`](../../docs/research/nightly/2026-08-13-retrieval-receipts/README.md).

See [`ADR-304`](../../docs/adr/ADR-304-retrieval-receipts.md) for the design rationale,
documented limitations (Merkle padding malleability), and rejection criteria for production
promotion.
