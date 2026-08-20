# ADR-312: Shared Witness Record Schema and Cross-Layer Anchoring Contract (rvm-witness ↔ autogenous witness)

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-305 (PIR, depends on); ADR-306, ADR-307, ADR-309, ADR-315 (PIR, consumers); ruflo ADR-322C (flywheel receipt/ledger/verification protocol); ruflo ADR-381 (sequential promotion evidence); ruvector ADR-134 (witness schema/log format); see `docs/research/perpetual-intelligence-runtime/04-verification-addendum.md` §3, §5
- **Tags**: pir, witness, provenance, security, cross-repo

## Context

The program's original framing (from `03-program-plan.md`'s initial draft)
proposed resolving a perceived duplication: `ruvnet/rvm`'s `rvm-witness`
crate and `ruvnet/autogenous`'s `witness` crate both implement an
append-only, hash-chained, tamper-evident log, described as having
"near-identical framing." The proposed decision was to make `rvm-witness`
canonical and have `autogenous`'s `witness` crate depend on or converge with
it.

**Direct comparison of both crates does not support that framing** (full
detail in `04-verification-addendum.md` §3):

| | `autogenous/crates/witness` | `rvm`'s `crates/rvm-witness` (vendored in `ruvector`) |
|---|---|---|
| Size | 302 LOC, one file | 4,405 LOC, multi-file |
| Hash | SHA-256 over canonical JSON | u64 chain/record hashes |
| Auth | Ed25519, per-role `SigningAuthority` | Keyed chain MAC with derivation + ratcheting |
| Record shape | JSON artifacts (genome, mutation, antibody, incident) | 64-byte cache-line-aligned records in a fixed-capacity ring buffer |
| Runtime | std, service-side, offline+deterministic | `no_std`, hypervisor-side |

The two crates are correct for their respective layers — SHA-256-over-JSON
does not fit a `no_std` 64-byte-aligned ring buffer, and `rvm-witness`'s
keyed-MAC chain is deliberately not a signature scheme suitable for
cross-service, offline verification. **Merging them, or making one depend on
the other, would be a design error**, not a simplification.

A better anchor already exists. `ruflo ADR-322C` (part of the Accepted,
phases-0–2-implemented ADR-322 flywheel-integration series) defines a
receipt/ledger/verification protocol built specifically to be a portable,
offline-verifiable, cross-layer wire format: **RFC 8785 JCS canonical JSON,
SHA-256 digests, Ed25519 signatures with domain separation**
(`Ed25519(domainPrefix || 0x00 || canonicalBytes)`), UUIDv7 run IDs,
deterministic paired-bootstrap statistical recomputation, and fail-closed
verification (unknown fields, non-finite numbers, and negative zero all
rejected). Separately, `ruflo ADR-381` defines an anytime-valid sequential
statistical evidence scheme (`α_k = α_total · 6/(π²k²)`, measured 0.6%
family-wise false-promotion rate over 1,000 simulated nulls) for the
statistical claims a witness chain needs to carry when it records a
promotion decision.

## Decision

Resolve the witness-layer question as a **shared record schema plus a
cross-layer anchoring contract**, not a crate merge or dependency edge:

1. `rvm-witness` remains the canonical implementation for its layer —
   hypervisor-side, `no_std`, privileged-action recording. `autogenous`'s
   `witness` crate remains the canonical implementation for its layer —
   service-side, std, JSON-artifact governance records. Neither is refactored
   to depend on the other.
2. Both layers adopt **ruflo ADR-322C's canonical encoding and
   domain-separated Ed25519 signature scheme** as the shared, portable
   record format for any witness record that needs to be verified *across*
   layers — specifically, an autogenous promotion decision that needs to be
   anchored into an RVM witness chain (or vice versa) is represented using
   ADR-322C's canonical-JSON + domain-separated-Ed25519 encoding at the
   anchoring boundary, regardless of each crate's own internal
   representation.
3. Where a witness record makes a statistical claim (e.g. "this mutation's
   improvement was significant"), that claim's evidence follows ruflo
   ADR-381's sequential-evidence scheme rather than a single uncorrected
   significance test — this is the concrete mechanism that makes "outperform
   its parent" (invariant 5) auditable across an arbitrarily long sequence of
   promotion attempts, not just a single one.
4. ruvector ADR-134's witness schema (the 64-byte cache-line-aligned,
   hash-chained record format used by `rvm-witness`) is unchanged by this
   ADR; this decision governs the boundary where a non-RVM witness record
   needs to be anchored into or verified against an RVM chain, not RVM's own
   internal format.

## Consequences

### Positive

- Avoids a design error (merging two crates built for genuinely different
  runtimes and threat models) that the original framing would have produced.
- Reuses an already-Accepted, already-implemented cross-service verification
  protocol (ruflo ADR-322C) instead of this program inventing a fifth
  witness-record format.
- Gives every PIR ADR that emits witness records (ADR-306, ADR-307, ADR-309,
  ADR-315) one anchoring contract to target, regardless of which underlying
  crate produced the record.

### Negative

- Requires cross-repo sign-off from `ruvnet/rvm`, `ruvnet/autogenous`, and
  `ruvnet/ruflo` maintainers on the shared boundary format — more
  coordination surface than a single-repo crate-dependency change would have
  been, even though it is architecturally the correct choice.
- `autogenous` self-labels "research prototype" status; adopting ruflo
  ADR-322C's format at the anchoring boundary does not itself stabilize
  autogenous's own witness crate's internal API, which may still churn.
- This ADR does not eliminate the two independent witness implementations —
  it deliberately keeps both. Anyone expecting "one witness crate for the
  whole program" from the original framing needs to be told explicitly that
  this ADR chose interoperability over consolidation.

## Security / Validation Gates

- **Witness-chain requirement**: every cross-layer anchoring event (an
  autogenous promotion record anchored into an RVM chain, or the reverse)
  must use ruflo ADR-322C's canonical encoding and produce a verifiable
  domain-separated Ed25519 signature before either side treats the record as
  committed.
- **Fail-closed verification**: unknown fields, non-finite numbers, and
  negative zero are rejected at the anchoring boundary, per ADR-322C's own
  verification rules — this program does not weaken that contract at the
  boundary.
- **Sequential-evidence discipline**: any witness record carrying a
  statistical promotion claim states its evidence per ruflo ADR-381's
  anytime-valid scheme, not a single-test p-value, so the acceptance
  harness's (ADR-306, WP12) day-30 comparison remains statistically sound
  across many promotion attempts.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to signing, canonicalization, or verification code in either
  witness crate or the anchoring boundary.

## Affected Repos

- `ruvnet/rvm` (`rvm-witness`, unchanged internally)
- `ruvnet/autogenous` (`witness` crate, unchanged internally)
- `ruvnet/ruflo` (ADR-322C canonical encoding/signature scheme, ADR-381 statistics — consumed, not modified, by this ADR)
- `ruvnet/ruvector` (anchoring-boundary implementation, RVM witness consumer)

## Dependencies

Depends on ADR-305 (adopts the cross-repo coordination posture this ADR
requires). ADR-306 (Dream Machine), ADR-307 (three-level memory), ADR-309
(LatentMesh transport), and ADR-315 (constitutional gate) all depend on this
ADR for how their respective witness records get anchored across layers.

## Alternatives Considered

- **Make `autogenous`'s witness crate depend on `rvm-witness`** (the
  program's original framing): rejected — see Context; the two crates target
  incompatible runtimes (`no_std` hypervisor-side ring buffer vs. std
  service-side JSON), and forcing a dependency edge between them would
  either break `rvm-witness`'s `no_std` constraint or strip `autogenous`'s
  witness crate of the JSON/Ed25519 shape its own consumers expect.
- **Invent a new, program-specific witness schema instead of adopting ruflo
  ADR-322C's**: rejected — ADR-322C is already Accepted and implemented with
  a fail-closed verification suite; inventing a fifth schema would add
  coordination surface without adding capability.
- **Leave the two witness crates fully independent, with no anchoring
  contract at all**: rejected — this would leave invariant 3 ("every agent
  communication is attributable") and invariant 5 ("every promoted mutation
  must outperform its parent") unauditable across the autogenous/RVM
  boundary specifically, which is exactly the boundary this program's
  Governance & Constitution context needs to cross.
