# Verification Addendum — Direct Source Checks Against LatentMesh and Autogenous

Status: addendum to `02-asset-map.md` and `03-program-plan.md`, compiled
2026-08-19 by a third research pass (asset-scout) that cloned both external
repos directly rather than relying on `gh` CLI metadata or org-search
results. **This document does not edit `02-asset-map.md` or
`03-program-plan.md` in place** — it records what those two documents still
get wrong or omit as of this program's ADR-authoring pass, so the correction
is traceable rather than silently folded in. Where a claim below contradicts
`02-asset-map.md`, the section header says so explicitly.

Repos cloned: `ruvnet/LatentMesh` (HEAD `922ea196`, dated 2026-08-18) and
`ruvnet/autogenous` (HEAD `b5c6e838`, dated 2026-08-19).

## 1. LatentMesh — accurate but understated in scale

`02-asset-map.md` §8 correctly describes LatentMesh's crates, ADR sequence,
and "not implemented" markers, and its headline finding's loop diagram and
"statistical primitive and admission gate implemented; closed loop not
wired" quote both match source verbatim, including the per-stage markers on
ADR-004 through ADR-008 (004 not wired to live MidStream; 005 not wired to a
live RuVector instance; 006 no live Darwin loop; 007 no live Radio/RuView;
008's admission gate implemented in `crates/latentmesh-gate`, RVF packaging
and RVM enforcement not wired).

**What §8 does not convey: LatentMesh is a small research prototype.** The
entire workspace is 1,407 LOC across four crates:

| crate | LOC | tests |
|---|---|---|
| `latentmesh-core` | 272 | 6 |
| `latentmesh-align` | 454 | 6 |
| `latentmesh-gate` (incl. `causal.rs`) | 534 | 11 |
| `latentmesh-bench` | 147 | 0 |

That is real, tested code, but roughly 1/23rd the size of `ruvector-core`
alone (32,380 LOC / 503 tests). **There is no network transport crate
anywhere in the workspace** — no QUIC/TCP/transport module exists. This is
consistent with, and reinforces, `02-asset-map.md`'s own §8 correction that
`ruvector`-side LatentMesh work is greenfield, not integration — but it also
means "adopt LatentMesh ADR-009 as the spine" (this program's ADR-305) is
adopting a *design contract and ~1.4K LOC of primitives*, not inheriting a
mature system. Scope every ADR and work package that references LatentMesh
accordingly.

**Minor date correction**: `02-asset-map.md` §8 states LatentMesh was "last
updated 2026-08-19." The correct date, per the cloned HEAD commit, is
**2026-08-18**.

## 2. Autogenous — `02-asset-map.md` is materially incomplete

### 2a. The ADR catalog is undercounted, and the missing ADRs are the program itself

`02-asset-map.md` §9 cites autogenous's ADRs as **"391/392/393/397."** The
actual catalog, confirmed by directory listing, runs **391 through 403 — 13
ADRs, not 4**:

```
ADR-391-autogenous-governed-self-evolving-architecture.md
ADR-392-autogenous-genome-language-antibody-protocol.md
ADR-393-autogenous-product-thesis-adaptive-agent-firewall.md
ADR-394-cryptographic-closure-of-the-promotion-path.md
ADR-395-radio-realtime-streaming-peer-expert-mesh.md
ADR-396-peer-expert-protocol-security-and-governed-evolution.md
ADR-397-autogenous-streaming-mixture-of-agents.md
ADR-398-applications-and-development-loop-integration.md
ADR-399-provider-backed-mesh-run-rvm-rvf.md
ADR-400-self-evolving-mesh-perpetual-loop.md
ADR-401-perpetual-intelligence-machine.md
ADR-402-ruview-cognitum-spaces-spatial-intelligence.md
ADR-403-verifiable-execution-loop.md
```

The eight missing from the asset map's list matter directly to this program:

- **ADR-401 — "The Perpetual Intelligence Machine."** Status: **Accepted**
  (framing + capability map) · **Partial** (implementation). Dated
  2026-08-16. Defines a 10-capability map with an honest Built/Partial/
  Gap/Narrative status legend, a stage-by-stage operating loop, and a V1 +
  30-day acceptance test. Carries an explicit honesty anchor: *"'Perpetual'
  does not mean conscious, immortal, or continuously improving… The system
  targets operational continuity, not infallibility."* This is, on its face,
  substantially the same product this program's brief describes. See
  ADR-305 for the required reconciliation.
- **ADR-400 — "Self-Evolving Mesh Perpetual Loop."** Status: **Accepted —
  Implemented (first flywheel turn measured).**
- ADR-394 (cryptographic closure of the promotion path), ADR-395/396 (radio
  realtime streaming peer-expert mesh + its security/governed-evolution
  counterpart), ADR-398 (applications/dev-loop integration), ADR-399
  (provider-backed mesh runs on RVM/RVF).
- **ADR-402 — "RuView/Cognitum Spaces Spatial Intelligence."** Status:
  Accepted; read-side adapters, fail-closed seam, production HTTPS API
  complete; write-side sync and a 30-day acceptance test outstanding.
  Directly relevant to this program's RF Sensing Substrate context (WP14).
- **ADR-403 — "Verifiable Execution Loop."** Status: Accepted;
  `VerifiedPromotion` implemented, concurrent-promotion fencing and durable
  replay partially implemented, enforced isolation accepted-design-pending.

ADR-401 also cross-references **"metaharness ADR-322 (flywheel receipts/
promotion)"** — a third ADR series appearing in none of this program's
research documents. Its content is unlocated as of this addendum; see
ADR-305's Decision §3 for the resulting action item.

### 2b. The governance implementation is TypeScript, not the Rust crates `02-asset-map.md` implies

`02-asset-map.md` §9 lists autogenous's crates as a Rust workspace (`witness,
antibody, agl-types, generator, evaluator, verifier, lineage, ledger,
promotion, deployment, constitution, runtime, midstream-adapter, service`).
That crate list is correct as far as it goes, but ADR-401's capability map
points almost entirely at **`packages/radio-moe/src/*.ts`**, not the Rust
workspace:

- `packages/radio-moe/src` — 6,171 LOC; `packages/radio-moe/test` — 3,161 LOC
- All 11 modules ADR-401 names exist: `mesh.ts` (252), `mesh-evolve.ts`
  (277), `mixture.ts` (685), `action-gate.ts` (440), `failover.ts` (455),
  `disclosure.ts` (125), `reputation.ts` (154), `rvf-trajectory.ts` (67),
  `lineage-independence.ts` (243), `relevance.ts` (99), `capability.ts` (31)
- The Rust crates total 8,096 LOC / 91 test functions — real, but a
  different layer (crypto/protocol primitives) than the governance loop
  ADR-401 describes as Built/Partial.

Any PIR ADR describing autogenous's governance loop (ADR-315, this program)
should point readers at `radio-moe`, not only at the Rust crate names.

### 2c. Crate list omission

`02-asset-map.md` §9 lists 14 autogenous crates and gets 14 right, but
**misses `envelope`** (1,009 LOC, 7 tests) — the second-largest crate in the
workspace after `runtime` (1,471 LOC). Everything else in the asset map's
list is confirmed accurate.

### 2d. Cloud Run deployment — confirmed, not just inferred

`02-asset-map.md` documents Cloud Run deployment concretely for
`mcp-brain-server`/`ruvbrain`, but does not confirm it for autogenous
specifically. The root `Dockerfile` in `ruvnet/autogenous` is headed
*"autogenous-service — Cloud Run image (the Autogenous AGL control-plane as
a service)"*, multi-stage `rust:1-bookworm` → `debian:bookworm-slim`, `ENV
PORT=8080`, with a comment noting Cloud Run sets `$PORT`. Backed by
`crates/service` (980 LOC).

### 2e. The prototype-badge / Accepted-ADR tension

`02-asset-map.md` §9 correctly cites autogenous's README "research prototype"
status badge and its "Honest status" note that it is "not wired to live
MidStream/MetaHarness/RVF/RVM yet" and that "performance and economics claims
in the ADRs are hypotheses until benchmarked." What is worth flagging
explicitly: several of autogenous's own ADRs (400, 401, 402, 403) carry
**Accepted** status despite the repo-level prototype badge. When citing
autogenous maturity in any PIR ADR, cite the README badge for overall repo
maturity and the individual ADR's status line for that specific capability —
the two are not always in agreement, and neither alone tells the full story.

## 3. Witness-crate comparison (grounds ADR-312's reframe)

`02-asset-map.md` §9's "duplication note" (repeated in `03-program-plan.md`'s
original ADR #8 draft) described `rvm-witness` and autogenous's `witness`
crate as having "near-identical framing," recommending autogenous depend on
or converge with `rvm-witness`. A direct comparison of both crates does not
support a merge/dependency framing:

| | `autogenous/crates/witness` | `ruvector`'s `crates/rvm/crates/rvm-witness` |
|---|---|---|
| Size | 302 LOC, 4 tests, one file | 4,405 LOC, multi-file (`v2.rs` 880, `replay.rs` 253, `hash.rs` 204, `record.rs`) |
| Hash | SHA-256 over canonical JSON | u64 chain/record hashes (`compute_chain_hash`, `compute_record_hash`) |
| Auth | Ed25519 (`ed25519_dalek`), per-role `SigningAuthority` | Keyed chain MAC with derivation + ratcheting (`derive_chain_key`, `ratchet_chain_key`, `erase_key`, `CHAIN_KEY_CONTEXT`) |
| Record shape | `WitnessRecord` over JSON artifacts (genome, mutation, antibody, incident) | 64-byte cache-line-aligned privileged-action records in a fixed-capacity ring (`WitnessLog<const N>`, `WitnessLogV2<const N, const SEG>`, `DEFAULT_RING_CAPACITY = 262_144`) |
| Extras | `WitnessSeal`, `verify_seal`, `verify_chain` | `CoveragePolicy`/`CoverageError`, `verify_chain_v2_ratcheted`, `v1_head_to_genesis`, `verify_log_bytes`, queries by partition/action-kind/time-range |
| Runtime | std, service-side, offline+deterministic | `no_std`, hypervisor-side |

The shared element is the abstract idea of an append-only, hash-chained,
tamper-evident log — cryptographic primitives, data model, storage strategy,
and target runtime all differ, and each choice is correct for its own layer:
SHA-256-over-JSON does not fit in a `no_std` 64-byte-aligned ring buffer, and
`rvm-witness`'s u64 keyed-MAC chain is deliberately not a signature scheme.
**ADR-312 in this program set is written against a shared witness record
schema plus a cross-layer verification/anchoring contract, not a crate
merge or a dependency edge — merging the two crates as originally proposed
would be a design error.**

## 4. Summary of corrections to carry into this program's ADRs

| Claim in `02-asset-map.md` | Correction here | PIR ADR affected |
|---|---|---|
| LatentMesh "last updated 2026-08-19" | 2026-08-18 | ADR-305 |
| LatentMesh described without scale context | 1,407 LOC / 4 crates / no transport crate | ADR-305, ADR-309 |
| Autogenous ADRs "391/392/393/397" | 391–403 (13 ADRs); 400/401/402/403 previously uncited | ADR-305, ADR-315 |
| Autogenous governance described via Rust crate list only | Governance loop lives in `packages/radio-moe/src/*.ts` (6,171 LOC) | ADR-315 |
| Autogenous crate list (14 named) | Missing `envelope` (1,009 LOC) | ADR-312 |
| Autogenous Cloud Run presence unconfirmed for this repo specifically | Confirmed via root `Dockerfile` (`autogenous-service`, `crates/service`) | ADR-315 |
| `rvm-witness`/autogenous `witness` "near-identical framing" → merge/depend | Different primitives, data models, std/no_std, layers; reframe as shared schema + anchoring contract | ADR-312 |

## 5. `metaharness ADR-322` located — it is `ruflo` ADR-322, misattributed by autogenous

A fourth research pass located the ADR autogenous ADR-400/401 cite as
"metaharness ADR-322": it lives in **`ruvnet/ruflo`**, at
`v3/docs/adr/ADR-322-metaharness-flywheel-integration.md` — **"Adopt
`@metaharness/{flywheel,darwin}` as pluggable engines behind ruflo's ADR-176
self-improvement flywheel,"** status Accepted (phases 0–2 implemented),
dated 2026-07-28. It is ruflo's own ADR *about consuming* MetaHarness, not an
ADR published by the MetaHarness/`agent-harness-generator` project (which was
itself renamed to `ruvnet/metaharness`, 230 ADRs topping out at ADR-250 —
no ADR-322 exists there). **Cite this dependency as "ruflo ADR-322," never
"metaharness ADR-322."**

ADR-322 has three Accepted, implemented children directly relevant to this
program's promotion-gating ADRs (ADR-306, ADR-312):

- **322A** — evaluation↔promotion transaction model (`RUFLO_FLYWHEEL_TRANSACTION_V1`).
- **322B** — Darwin proposer adapter, enforcing a separation-of-powers
  invariant this program's own ADR-401 reconciliation (ADR-305) is reaching
  for independently: *"A proposer produces untrusted candidates only. It
  cannot issue promotion decisions or mutate active policy."*
  `promoteFlywheelCandidate` is the sole promotion authority.
- **322C** — receipt/ledger/verification protocol: RFC 8785 JCS canonical
  JSON, SHA-256 digests, **Ed25519 with domain separation**
  (`Ed25519(domainPrefix || 0x00 || canonicalBytes)`), UUIDv7 run IDs,
  independent statistical recomputation via deterministic paired bootstrap.

Separately, **ruflo ADR-381** defines a sequential-promotion-evidence scheme
(anytime-valid e-process, `α_k = α_total · 6/(π²k²)` so `Σα_k` is bounded
across arbitrarily many adaptively-chosen candidates; measured 0.6%
family-wise false-promotion rate over 1,000 simulated nulls) — the concrete
prior art for any PIR ADR that claims statistical promotion gating.

ruflo has already suffered the exact ADR-number collision this program is
trying to avoid: a dream-cycle research PR independently proposed
`ADR-322-dream-cycle-memory-typed-provenance.md` and was closed as superseded
when it collided with the already-merged flywheel-integration ADR-322 — a
live precedent for verifying numbers against the actual filename list
immediately before committing, not against a max+1 assumption.

## 6. `ruvnet/dream-machine` is a real, Accepted, shipped repo — corrects §"Dream Machine" reasoning above and in `02-asset-map.md`/`03-program-plan.md`

An earlier pass of this research reported "Dream Machine" as a name with no
matching asset, later corrected in `02-asset-map.md`'s revision to point at
`ruvector`'s own `scripts/research-gate/` + `ruvector-sota-bench/harness`
(ADR-282) as the functional equivalent. **Both were incomplete**:
`ruvnet/dream-machine` is itself a real, public, published-to-npm repo
(pushed 2026-08-19), and its own **ADR-0001** — *"The Dream Machine engine —
a config-driven, evidence-gated nightly evolution loop composed from the
ruvnet stack"* — is **Accepted, engine v0.1.0 shipped** (compile / ledger /
witness / schedule / memory modules + CLI/TUI, 85 tests, ~2,558 LOC across
six packages), dated 2026-08-13.

ADR-0001's own thesis is the one this program's brief and ADR-306 are
circling: *"Freeze the model. Evolve the harness. Evaluation is not
promotion — the machine never merges; a human does."* It **composes**
`@metaharness/flywheel` 0.1.10, `@metaharness/darwin` 0.9.1,
`@metaharness/redblue` 0.1.4, `metaharness` CLI 0.4.5, `ruvector` 0.2.41, and
`agentdb` 3.0.0-alpha (all optional/peer dependencies), and its own text
names `ruvnet/ruflo`'s and `ruvnet/metaharness`'s nightly dream-cycle
routines as prior instances it **subsumes** — specifically metaharness
ADR-251 ("MetaHarness Nightly Dream Cycle") and the Ruflo Nightly Dream Cycle
v3.

**Net effect**: the "Dream Machine" mechanism now exists in **four** places
— `ruvnet/dream-machine` (the generalized, already-shipped engine),
`ruvnet/ruflo`'s nightly dream cycle, `ruvnet/metaharness` ADR-251, and
`ruvector`'s own `scripts/research-gate/` + ADR-282. ADR-306 in this program
is revised to cite `ruvnet/dream-machine` ADR-0001 as the consolidating
design ruvector's `research-gate` should align with, not as a name this
program invents or as work fully covered by `research-gate` alone.

**Version-drift note**: `ruvnet/dream-machine` composes `@metaharness/darwin`
0.9.1 / `@metaharness/flywheel` 0.1.10; `ruvector` currently pins `darwin`
0.8.0 / `flywheel` 0.1.7 — three of the nine `@metaharness/*` packages
`ruvector` depends on (per `02-asset-map.md` §1) are behind the versions
`dream-machine` itself composes against. Worth a line in WP0b's dependency
remediation scope, not a blocker for this ADR set.

## 7. Open items not resolved by this addendum

None remaining from the original metaharness-ADR-322 question (resolved in
§5). Still open: no repo checked so far publishes a formal reconciliation
between `ruvnet/dream-machine` ADR-0001, `ruvector`'s ADR-282
(`research-gate`), and autogenous ADR-400/401's flywheel — WP1/WP2 own
producing that reconciliation, per ADR-305 and ADR-306.
