# ADR-313: SHAPER-Pattern Skill/Harness Evolution Loop (Frozen Weights)

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-306 (PIR, depends on); ADR-308 (PIR, downstream consumer); ADR-305 (PIR, separation-of-powers invariant); ruvector ADR-150 (optionalDependencies policy); ruvector ADR-259 (ruvllm mutator backend); ruvector ADR-271 (darwin_guard); ruflo ADR-322B (proposer/promotion separation)
- **Tags**: pir, shaper, darwin, frozen-weights, evolution

## Context

The acceptance test's central constraint is frozen foundation-model weights:
the model never changes, only the skills, context, and execution harness
around it. Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md`, this
is grounded in a grade-A source:

- **SHAPER** — [arXiv:2608.11350](https://arxiv.org/abs/2608.11350),
  "Self-Evolving Embodied Agents via Skill-Harness Evolution," submitted
  2026-08-11 (matches the program brief's date exactly). Confirms a
  train-free framework that keeps model parameters frozen while evolving
  reusable skills and a context/code harness through target-environment
  rollouts; the same frozen model serves as both planner and optimizer.
  Evaluated on VLABench and ESI-Bench. Grade A; no name collision found.

`ruvector` already has a real evolutionary loop that maps onto this pattern:
Darwin (`@metaharness/darwin`), called from `harness/src/darwin.ts` (ANN
config evolution), `examples/mragent`'s `scorePolicy` function, and
`crates/sona/src/darwin_guard.rs` (ruvector ADR-271) — profile → baseline →
mutate (seven approved surfaces via `CodeGenerator`, e.g. `OpenRouterMutator`,
`RuvllmMutator`) → sandbox → 6-term score → archive-as-tree → repeat.
`ruvllm` is a real, wired local mutator backend (ruvector ADR-259).

**Two concrete, already-identified bugs block this work package** and must
be fixed first (program plan WP0b), not discovered mid-implementation:

1. `METAHARNESS-README.md` claims ruvector ADR-150 `optionalDependencies`
   compliance, but the nine `@metaharness/*` packages in
   `crates/ruvector-sota-bench/harness` are plain (hard) dependencies — the
   harness currently hard-fails to install without them, contradicting its
   own documented policy.
2. A known HTTP-307 redirect bug in `ruvllm`'s model-download path blocks
   end-to-end live-serve testing of the mutator backend.

## Decision

Implement the physical-intelligence evolution loop with foundation-model
weights frozen throughout, following SHAPER's pattern (arXiv:2608.11350):

1. The same frozen model serves as both planner and optimizer; only skills,
   context, and the execution harness evolve — mapped onto Darwin's existing
   mutation surfaces (`harness/src/darwin.ts`, `examples/mragent`
   `scorePolicy`, `crates/sona/src/darwin_guard.rs`).
2. This is enforced **structurally**, not by policy: a CI check fails the
   build if any mutation surface reachable from the promotion pipeline
   imports a training/fine-tuning API. Policy alone ("please don't
   fine-tune") is not sufficient for the acceptance test's frozen-weights
   verification requirement (day-30 re-hash must be bit-identical to day 0).
3. Darwin's mutation proposals are exactly that — proposals. Per ruflo
   ADR-322B's separation-of-powers invariant, adopted as a governing
   invariant in ADR-305: *"a proposer produces untrusted candidates only; it
   cannot issue promotion decisions or mutate active policy."* Darwin's
   mutation surfaces never gain promotion authority; every proposal routes
   through ADR-306's adopted evaluation pipeline before any change to active
   policy.
4. WP0b's two blocking bugs are fixed before this work package's live-serve
   testing begins: the ADR-150 `optionalDependencies` non-compliance (make
   the nine `@metaharness/*` packages genuinely optional, or correct the
   documentation to state the real hard-dependency requirement) and the
   `ruvllm` HTTP-307 redirect bug in the model-download path.

## Consequences

### Positive

- Grounds the frozen-weights constraint in a concrete, exact-date-matching
  grade-A source rather than only the program brief's prose.
- Reuses Darwin's already-real evolutionary loop instead of building a new
  mutation-proposal mechanism.
- The structural (CI-enforced) frozen-weights check, combined with
  ADR-322B's separation-of-powers invariant, gives this program two
  independent enforcement layers for "the model never changes and a
  proposer never self-promotes" — a single-point-of-failure policy
  violation in either layer alone does not compromise the other.

### Negative

- This work package cannot start live-serve testing until WP0b's two bugs
  are fixed — an explicit, tracked blocking dependency, not a soft
  preference.
- Darwin currently exists as an external npm dependency
  (`@metaharness/darwin`) called from three sites, not a first-class
  in-repo asset; this ADR does not itself resolve that architectural
  question (vendor vs. formalize-as-dependency), leaving it to the owning
  work package.
- A CI check for "no training/fine-tuning API import" is a static,
  import-based check; it does not catch a sufficiently obfuscated or
  dynamically-loaded fine-tuning path. This is a known limitation, not
  claimed to be closed by this ADR.

## Security / Validation Gates

- **Structural frozen-weights enforcement**: CI fails the build if any
  mutation surface reachable from the promotion pipeline imports a
  training/fine-tuning API.
- **Separation-of-powers invariant** (ruflo ADR-322B, adopted via ADR-305):
  Darwin's mutation surfaces never gain promotion authority; enforced by
  ADR-306's adopted evaluation/promotion pipeline, not by this ADR directly.
- **Proof-gated promotion**: every mutation still passes through
  `ruvector-proof-gate`/`rvm-proof` regardless of SHAPER-pattern compliance.
- **WP0b blocking gate**: this work package's live-serve acceptance criteria
  cannot be claimed complete while either the ADR-150 compliance bug or the
  ruvllm HTTP-307 bug remains open.

## Affected Repos

- `ruvnet/ruvector` (`crates/sona`, `crates/ruvllm` mutator backend, `crates/ruvector-sota-bench/harness`, `agent-harness-generator`/Darwin integration sites)

## Dependencies

Depends on ADR-306 (Dream Machine evaluation pipeline — every mutation
proposal needs a verdict before promotion) and on ADR-305's adopted
separation-of-powers invariant. ADR-308 (WorldCycle verification) depends on
this ADR — it verifies the physical-action outputs this evolution loop
produces.

## Alternatives Considered

- **Allow a fine-tuning fallback path for cases the frozen-weights harness
  can't handle**: rejected — this directly contradicts the acceptance test's
  central constraint and SHAPER's own pattern; any capability gap should be
  addressed via richer harness/skill mutation surfaces, not weight updates.
- **Defer fixing WP0b's two bugs until they actually block a specific
  task**: rejected — both are already identified, small, and cheap to fix;
  deferring them guarantees they surface mid-implementation of this ADR's
  higher-priority work instead of being resolved ahead of time.
