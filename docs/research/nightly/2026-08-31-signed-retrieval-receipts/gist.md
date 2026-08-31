# Closing a Non-Repudiation Gap in Retrieval Receipts With Ed25519, Honestly Benchmarked

## Problem

Two weeks ago, a nightly research run in the RuVector project shipped
`ruvector-retrieval-receipt`: cryptographic commitments over ANN query
result sets, so a holder of a receipt could detect if the result set they
were handed was silently tampered with after the query engine issued it.
The design was explicit about what it did *not* prove: the receipts were
unsigned. Anyone holding the same result leaves could reproduce the same
root hash. A receipt was internally consistent, but not attributable — it
couldn't be shown to a third party as evidence that a *specific* engine
instance vouched for it. The prior run's own README named the fix
directly: sign the root.

## Hypothesis

Given a Merkle receipt root per query, does signing it with Ed25519 close
that gap at an acceptable, measurable cost — and does batching many
receipt roots under one signature actually amortize that cost, or is that
a plausible-sounding claim that falls apart under an uncaching verifier?

Formally:

```
Given a MerkleReceipt root per query,
when signed either per-query (batch size 1) or batched (B roots folded
  into a second Merkle tree, signed once),
then batched signing's amortized per-query cost should drop by roughly
  the batch factor relative to per-query signing,
subject to: every tamper (root, signature, or inclusion-proof sibling)
  stays detected, and an uncaching verifier's per-query cost must NOT
  drop with batch size — if it did, the benchmark would be quietly
  rewarding an unrealistic verifier.
```

## Technical Design

Two additions, layered on the existing crate without touching it:

- **Per-query signing:** `Issuer::sign_root(root) -> [u8; 64]`, an
  Ed25519 signature over the receipt's existing Merkle root.
- **Batched signing:** `BatchAnchor` — a second Merkle tree built over B
  receipt roots (domain-separated hashing keeps it from colliding with
  the existing per-result tree), signed once. Each query gets an O(log B)
  inclusion proof against the signed batch root.

`ed25519-dalek 2.1` was already used at that exact version in five other
places in this workspace (`cognitum-gate-tilezero`, `rvm-checkpoint`,
`rvf-crypto`, `rvforge-registry`, `mcp-brain-server`) — reused verbatim
rather than re-decided.

## Implementation

~250 new lines in `signing.rs`, one accessor added to the existing
`RetrievalReceipt` enum, and a new benchmark section reusing the existing
benchmark file's helpers. No changes to the unsigned-receipt code path.
11 new unit tests plus the 13 pre-existing ones, all passing; `cargo
clippy --all-targets --release` and `cargo fmt --check` both clean.

## Benchmark Evidence

`cargo run --release -p ruvector-retrieval-receipt --bin benchmark --
5000 128 10 200`, 4 logical CPUs, rustc 1.94.1, 3 repeated runs:

| batch_size | sign amortized (ns/query) | verify naive (ns/query) | verify cached (ns/query) |
|---:|---:|---:|---:|
| 1   | 29,042 | 76,936 | 785   |
| 8   | 5,107  | 55,943 | 2,932 |
| 32  | 2,521  | 42,322 | 4,081 |
| 128 | 1,688  | 46,708 | 5,893 |

Amortized signing drops ~17x from batch=1 to batch=128, landing at
5.5–6.1% of the per-query cost — under the 10% threshold fixed before the
run. Naive (uncaching) verify cost stays flat, 42,000–77,000ns, across
every batch size — confirming the hypothesis's guard condition held: an
Ed25519 verify (~40–65µs here) dominates regardless of batching, so a
verifier that skips caching gets no benefit from it. 100% tamper
detection (root-byte, signature-byte, inclusion-proof-sibling flips)
across every batch size in all 3 runs.

## The Bug the Process Caught

The very first run reported 50/150 undetected tampers at batch size 1 —
a REJECT. Rather than re-run until it passed, the raw output was traced:
the root-tamper trial swapped a root with `(idx + 1) % batch_len`, which
for a one-element batch is `idx` itself — a swap with nothing, a no-op.
Fifty trials "tampered" a batch by doing nothing to it, and the verifier
correctly verified the untouched data, which the trial-counting logic
misread as "not detected." Fixed by replacing the swap with a direct byte
flip (no degenerate case at batch size 1), confirmed clean across 3
subsequent runs. Both the bug and the fix are recorded in the raw
evidence file, not silently absorbed into a clean final number.

## Limitations

This benchmark measures only in-process CPU cost. A batch signature does
not exist until a batch closes — the wall-clock delay of waiting for B
queries to arrive is a real, deployment-specific cost this run does not
model. Batching is therefore a measured CPU-throughput win, not a
demonstrated end-to-end latency win. The dataset is synthetic and
brute-force (inherited scope from the crate's original design, which
isolates the provenance layer's cost from ANN recall). WASM/edge cost is
analyzed as plausible (the same dependency is already used in a WASM
target elsewhere in the workspace) but not measured here.

## Production Relevance

Signed receipts are directly useful anywhere a retrieval result needs to
be shown to a party that doesn't trust the query engine by default: a
compliance audit of an agent's cited evidence, a regulated-industry RAG
deployment, or a swarm of agents that need to accept each other's
retrieved context without re-querying. Batching is the right choice when
throughput matters more than per-query signature immediacy (e.g., a
multi-tenant retrieval service); per-query signing is the right choice
when a caller needs a signed receipt the instant a query returns.

## RuVector Ecosystem Implications

This connects five points of ecosystem leverage from one crate extension:
`ruvector-proof-gate`'s existing write-side hash chains, the read-side
receipts this extends, the workspace's established Ed25519 pattern, a
plausible MCP verification tool (read-only, no signing exposed), and a
natural fit for RVF's signed-lineage goals (a `BatchAnchor` is exactly
the shape of a portable, independently-verifiable provenance unit).

## Future Direction

Model real wall-clock batch-fill latency under a realistic query
arrival-rate distribution. Evaluate BLS aggregate signatures, which could
avoid batch-fill latency entirely by combining independently-issued
per-query signatures after the fact — deferred here because it requires
vetting a new pairing-curve dependency. Consider an independent,
periodically-signed `index_state_root` anchor, decoupled from any single
query, as a complementary mechanism for auditors who want to verify index
state without holding a specific receipt.

## References

- ADR-304: Retrieval Receipts (prior nightly run, the origin of this
  run's hypothesis).
- ADR-340: Signed Retrieval-Receipt Anchoring (this run's design record).
- `ed25519-dalek` 2.1 — the signature library, already in use across this
  workspace.
