# ADR-315: Governance Constitution for Capability Expansion

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-312 (PIR, depends on); ADR-306 (PIR, distinct from ordinary promotion gating); autogenous ADR-393 "Product Thesis — Adaptive Agent Firewall" (autogenous repo); autogenous ADR-401 capability 5 (governed self-improvement)
- **Tags**: pir, governance, constitution, capability-expansion, security

## Context

The acceptance test requires "zero unapproved capability expansion" — no
mutation may silently grant an agent a new tool, a new physical action
class, or a new communication peer without an explicit, higher-bar approval
distinct from ordinary behavioral-mutation promotion (ADR-306).
`ruvnet/autogenous` already implements a constitution/admission-gate pattern
for exactly this purpose (`constitution` crate, part of its
observe → explain → mutate → prove → canary → promote/rollback loop), and its
own ADR-401 (see ADR-305's reconciliation) frames governed self-improvement
as capability 5 of its 10-capability map: `Promote = Better ∧ Safe ∧
Authorized ∧ Reversible`. ADR-401 admits this predicate is **not yet fully
closed as one checked gate** — the "Authorized" conjunct in particular is
enforced by a human-anchor flywheel step (autogenous ADR-399/ruflo ADR-322),
not yet a single automated predicate.

**Maturity caveat, corrected in this program's verification pass** (see
`04-verification-addendum.md` §2): `ruvnet/autogenous`'s README self-labels
the whole repo **"research prototype"** status, and states plainly that
"performance and economics claims in the ADRs are hypotheses until
benchmarked" — even though several of its individual ADRs (400, 401, 402,
403) carry **Accepted** status. This program treats the repo-level badge as
the governing signal for API stability: autogenous's constitution/
admission-gate API should be treated as unstable, and this program budgets
time for API churn rather than assuming production-grade stability. The
governance loop's actual implementation lives in `packages/radio-moe/src/*.ts`
(action-gate.ts, mesh-evolve.ts, disclosure.ts, reputation.ts — 6,171 LOC
total), not the Rust `constitution` crate alone.

## Decision

Adopt `autogenous`'s constitution/admission-gate pattern as the enforcement
point for zero-unapproved-capability-expansion, while closing the specific
gap ADR-401 itself admits is open:

1. Every mutation that would expand an agent's capability set — new tool
   access, new physical action class, new communication peer — requires
   explicit constitutional approval logged to the witness chain, **distinct
   from** ordinary behavioral-mutation promotion (ADR-306's evaluation
   pipeline). A mutation passing ADR-306's promotion gate does not
   automatically pass this gate; capability expansion needs its own,
   separately-authorized approval record.
2. This program contributes toward closing autogenous ADR-401's own
   admitted gap — converging `Promote = Better ∧ Safe ∧ Authorized ∧
   Reversible` into one checked predicate — specifically for the
   "Authorized" conjunct as it applies to capability-expanding mutations,
   rather than leaving it a human-anchor-only step indefinitely.
3. Approval records for capability expansion are anchored via this
   program's shared witness contract (ADR-312), so the 30-day acceptance
   harness (per `03-program-plan.md`) can query "zero unapproved capability
   expansion" directly by diffing the capability/tool/action tables at day 0
   vs. day 30 against this gate's approval log.
4. Given autogenous's self-labeled research-prototype status, this program's
   integration against its constitution/admission-gate API is built with an
   explicit API-stability buffer (versioned interface, integration tests
   that fail loudly on upstream API changes) rather than a tight, assumed-
   stable coupling.

## Consequences

### Positive

- Reuses `autogenous`'s already-designed governance concept (constitution,
  admission gate) instead of inventing a parallel capability-expansion
  approval mechanism.
- Directly operationalizes the acceptance test's "zero unapproved capability
  expansion" requirement as a queryable log, not just a stated goal.
- Explicitly separating capability-expansion approval from ordinary mutation
  promotion (ADR-306) prevents the specific failure mode where a
  statistically-significant, well-tested mutation quietly also expands an
  agent's capability set without anyone noticing.

### Negative

- Coupling to a self-labeled research-prototype API means this integration
  carries real churn risk; the API-stability buffer in Decision §4 adds
  engineering overhead specifically to manage that risk.
- Converging autogenous's own `Better ∧ Safe ∧ Authorized ∧ Reversible`
  predicate is, by ADR-401's own admission, not yet done even inside
  autogenous itself — this program's contribution to closing that gap is
  real, unstarted work, not an integration against a finished mechanism.
- Two separate approval paths (behavioral promotion via ADR-306, capability
  expansion via this ADR) add process overhead versus a single unified
  promotion gate — deliberate, per the acceptance test's own requirement
  that these be distinct, but a real cost.

## Security / Validation Gates

- **Constitutional admission gate** (this ADR's core mechanism): every
  capability-expanding mutation requires explicit, separately-logged
  approval before it can take effect — blocking, not advisory.
- **Witness-chain requirement**: every constitutional approval or denial is
  anchored via ADR-312's shared witness contract, queryable end-to-end for
  the 30-day acceptance harness's capability-table diff.
- **Distinctness from behavioral promotion**: this gate is never satisfied
  merely by a mutation passing ADR-306's evaluation pipeline — the two are
  independently blocking.
- **Hosted-RVM honesty discipline** (ruvector ADR-285): if any part of this
  gate's enforcement runs in a hosted (non-bare-metal) mode, isolation
  claims must match what has actually been tested.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change touching capability tables or the admission-gate integration.

## Affected Repos

- `ruvnet/autogenous` (`constitution` crate, `packages/radio-moe/src/action-gate.ts` and `mesh-evolve.ts`)
- `ruvnet/ruvector` (integration point, capability/tool tables)
- `ruvnet/rvm` (witness anchoring via ADR-312)

## Dependencies

Depends on ADR-312 (the shared witness/anchoring contract this gate's
approval records use). Distinct from, but coordinates with, ADR-306's
promotion pipeline — a mutation may need to pass both gates independently.

## Alternatives Considered

- **Fold capability-expansion approval into ADR-306's ordinary promotion
  gate as one more check**: rejected — the acceptance test explicitly
  requires querying capability expansion independently of general
  performance-based promotion; conflating the two would make that query
  impossible to answer cleanly.
- **Build a `ruvector`-native constitutional gate instead of integrating
  with autogenous's**: rejected — autogenous's constitution/admission-gate
  concept is already designed for exactly this purpose, and building a
  parallel mechanism would repeat the same anti-pattern ADR-305 already
  rejected at the program level (parallel architecture instead of adopting
  existing design work).
- **Treat autogenous's Accepted-status ADRs (400/401) as sufficient
  evidence of API stability, ignoring the repo-level prototype badge**:
  rejected — per this program's verification pass, the badge and individual
  ADR statuses are not always in agreement, and the badge is the more
  conservative signal for integration planning.
