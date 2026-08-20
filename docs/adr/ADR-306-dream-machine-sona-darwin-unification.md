# ADR-306: Dream Machine — Adopt the Consolidating Evaluation Engine, Wired to research-gate and Darwin

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-305 (PIR, depends on); ADR-312 (PIR, shares verification stack); ADR-313 (PIR, downstream consumer); ruvector ADR-282 (research-gate); ruflo ADR-322/322A/322B/322C (flywheel integration); ruflo ADR-381 (sequential promotion evidence); dream-machine ADR-0001 (Accepted, engine v0.1.0 shipped); metaharness ADR-251 (Nightly Dream Cycle); see `docs/research/perpetual-intelligence-runtime/04-verification-addendum.md` §6
- **Tags**: pir, dream-machine, promotion, evaluation, cross-repo

## Context

Invariant 5 of the program is: *"every promoted mutation must outperform its
parent."* This ADR's premise changed twice during research for this program
(see `04-verification-addendum.md` §6 for the full trail):

1. **First pass**: no asset named "Dream Machine" exists; the closest
   analogue is SONA's dream-replay engine (`05-MEMORY-DREAMS.md`), and this
   would be a from-scratch build.
2. **Second pass**: real, CI-wired, cryptographically-signed statistical
   promotion machinery already exists inside `ruvector` under a different
   name — `scripts/research-gate/` + `.github/workflows/research-*.yml`,
   backed by `crates/ruvector-sota-bench/harness` (`statistics.ts` paired
   bootstrap, `vetoes.ts` hard vetoes, `flywheel.ts` Ed25519-signed replay
   bundles), documented in ruvector ADR-282. This significantly de-risked
   what had been flagged as the program's top risk.
3. **Third pass (this ADR)**: **a literal `ruvnet/dream-machine` repo
   exists**, is real, public, published to npm, and its own **ADR-0001** is
   Accepted with engine v0.1.0 shipped (compile / ledger / witness /
   schedule / memory modules, CLI/TUI, 85 tests, ~2,558 LOC). Its thesis:
   *"Freeze the model. Evolve the harness. **Evaluation is not promotion —
   the machine never merges; a human does.**"* It composes
   `@metaharness/flywheel`, `@metaharness/darwin`, `@metaharness/redblue`,
   the `metaharness` CLI, `ruvector`, and `agentdb` as optional/peer
   dependencies, and its own text names two prior instances it
   **subsumes**: `ruvnet/ruflo`'s nightly dream cycle and metaharness
   ADR-251 ("MetaHarness Nightly Dream Cycle").

**"Dream Machine" therefore now exists in four places**, and this program
must name which is canonical for which layer rather than building a fifth:

| Layer | Canonical implementation |
|---|---|
| Generalized, product-level evaluation engine (compile/ledger/witness/schedule/memory, CLI/TUI) | `ruvnet/dream-machine` ADR-0001 (Accepted, v0.1.0 shipped) |
| Statistical significance + hard vetoes + signed replay bundles, already CI-wired inside `ruvector` | `ruvector` ADR-282 (`scripts/research-gate/`, `crates/ruvector-sota-bench/harness`) |
| Evaluation↔promotion transaction model, separation of proposer vs. promotion authority | ruflo ADR-322A/322B (see ADR-305, ADR-313) |
| Sequential/anytime-valid statistical evidence across adaptively-chosen candidates | ruflo ADR-381 (0.6% measured family-wise false-promotion rate over 1,000 simulated nulls) |
| Earlier reference instances, now subsumed | `ruvnet/ruflo` nightly dream cycle; `ruvnet/metaharness` ADR-251 |

**Version drift**: `ruvnet/dream-machine` composes `@metaharness/darwin`
0.9.1 / `@metaharness/flywheel` 0.1.10; `ruvector` currently pins `darwin`
0.8.0 / `flywheel` 0.1.7. This program's WP0b (MetaHarness
dependency-compliance remediation) should account for this drift when it
fixes the ADR-150 `optionalDependencies` non-compliance bug.

## Decision

The Perpetual Intelligence Runtime does not build a new evaluation service.
It **adopts `ruvnet/dream-machine`'s engine (ADR-0001) as the evaluation
core**, wired to `ruvector`'s already-CI-integrated statistical layer:

1. `ruvnet/dream-machine`'s compile/ledger/witness/schedule/memory pipeline
   becomes the orchestration layer for PIR's mutation-evaluation cycle,
   consuming Darwin mutation proposals (`@metaharness/darwin`, per ADR-313)
   and producing evidence-gated verdicts.
2. `ruvector` ADR-282's `research-gate` (`statistics.ts` paired bootstrap,
   `vetoes.ts` hard vetoes, `flywheel.ts` Ed25519-signed replay bundles)
   supplies the statistical significance layer dream-machine's own engine
   calls into for `ruvector`-hosted mutation evaluation — this program does
   not reimplement paired-bootstrap significance testing a second time.
3. SONA's dream-replay engine (`05-MEMORY-DREAMS.md`) remains an optional
   pre-filter feeding candidates into this pipeline before the more
   expensive statistical evaluation runs, not a replacement for it.
4. The evaluation↔promotion transaction boundary follows ruflo ADR-322A/
   322B: an evaluation verdict is advisory input, never a promotion
   decision in itself.
5. **ADR-0001's constitutional principle is adopted verbatim and is not
   negotiable within this program: "Evaluation is not promotion — the
   machine never merges; a human does."** Automated gates (research-gate's
   statistics, dream-machine's ledger/witness pipeline, ruflo's promotion
   transaction) may recommend promote/reject; only a human-authorized action
   performs the final merge into an active policy or codebase. This
   constrains the frozen-weights-and-governed-mutation loop this program
   builds (ADR-313, ADR-315) — no PIR work package may wire an unattended
   `/loop`-style auto-merge path, mirroring ADR-0001's own explicit
   phase-3/4 gating of unattended promotion pending separate privilege,
   spend, and rollout controls.

## Consequences

### Positive

- Converts what was the program's single largest identified risk (building
  a promotion-evaluation system from nothing) into a consolidation task
  across four already-Accepted-or-implemented pieces of prior art.
- Inherits ruflo ADR-381's measured statistical guarantee (0.6% family-wise
  false-promotion rate) instead of needing to re-derive or re-validate a
  sequential-testing scheme from scratch.
- The "evaluation is not promotion, a human merges" principle gives
  invariant 5 a hard human-in-the-loop backstop, directly addressing the
  acceptance test's "zero unapproved capability expansion" requirement
  (ADR-315) at the evaluation layer as well as the constitutional layer.

### Negative

- Four separate prior instances (dream-machine, ruflo dream cycle,
  metaharness ADR-251, ruvector research-gate) must be reconciled into one
  coherent pipeline for this program; no repo checked by this program's
  research passes has published that reconciliation yet — it is WP1/WP2's
  deliverable, not a pre-existing fact this ADR can cite.
- Adds a real external dependency (`ruvnet/dream-machine`, v0.1.0, a young
  shipped engine) to `ruvector`'s promotion pipeline, with its own version
  churn risk on top of the already-identified `@metaharness/*` version
  drift.
- The "a human merges" principle is a deliberate throughput ceiling: it
  rules out a fully unattended 30-day acceptance run unless the acceptance
  harness (WP12) is itself designed around scheduled human checkpoints
  rather than continuous autonomous promotion — a scope clarification this
  ADR surfaces but does not resolve.

## Security / Validation Gates

- **Proof-gated promotion**: `ruvector-proof-gate`/`rvm-proof` (ruvector
  ADR-227 and sibling RVM ADRs) still gates every promotion; dream-machine's
  verdict and research-gate's statistical result are inputs to that gate,
  never a bypass of it.
- **Witness-chain requirement**: every verdict (promote or reject) emits an
  RVM witness record (ruvector ADR-134 schema); this program's shared
  witness/anchoring contract (ADR-312) is the mechanism that makes
  dream-machine's own ledger/witness modules and RVM's witness chain
  cross-verifiable.
- **Human-merge constitutional gate**: no CI or scheduled-worker path may
  execute a final merge/promotion without an explicit human-authorized
  action recorded in the witness chain, per ADR-0001's principle adopted
  above.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the promotion pipeline's signing or verification code.

## Affected Repos

- `ruvnet/dream-machine` (adopted engine, external dependency)
- `ruvnet/ruvector` (`scripts/research-gate/`, `crates/ruvector-sota-bench/harness`, ADR-282; SONA dream-replay)
- `ruvnet/ruflo` (ADR-322A/322B transaction model, ADR-381 statistics)
- `ruvnet/metaharness` (ADR-251, prior reference instance)

## Dependencies

Depends on ADR-305 (adopts the control-loop and cross-repo coordination
posture this evaluation core plugs into). Shares its verification stack with
ADR-312. ADR-313 (SHAPER-pattern evolution loop) depends on this ADR —
Darwin's mutation proposals need a verdict from this pipeline before
promotion.

## Alternatives Considered

- **Build a new evaluation service from scratch** (the program's original
  framing): rejected — three separate research passes converged on the fact
  that this would duplicate at least four existing pieces of prior art, the
  newest of which (`ruvnet/dream-machine`) is already shipped and Accepted.
- **Adopt only `ruvector` ADR-282's research-gate and ignore
  `ruvnet/dream-machine`**: rejected — research-gate is a strong statistical
  layer but does not provide dream-machine's orchestration (compile/ledger/
  schedule/memory/CLI-TUI) layer, and ADR-0001 explicitly positions itself
  as the consolidating design other implementations should align with.
- **Allow unattended promotion once statistical significance is met**:
  rejected — contradicts ADR-0001's explicit constitutional principle and
  autogenous ADR-401's own admission that its `Better ∧ Safe ∧ Authorized ∧
  Reversible` promotion predicate is not yet fully closed; keeping a human
  merge step is the safer default until that predicate is fully wired and
  independently audited.
