# ADR-337: Adaptive Runtime Monitoring with Value-of-Information Escalation

- **Status**: Proposed
- **Date**: 2026-08-23
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-331 (PIR — VoI cost-aware routing; **consumed as-is, not reimplemented**); ADR-317 (PIR — HarnessRisk lifecycle-security gate); `crates/ruvector-tiny-dancer-core/src/voi.rs`; `crates/mcp-gate/`; see `docs/research/perpetual-intelligence-runtime/12-wave5-evidence-review.md` and `13-wave5-program-plan.md`
- **Tags**: pir, wave-5, monitoring, voi, escalation, mcp-gate, security

## Context

OpenAI's "Pacing model development in an era of cyber-critical
capabilities" (2026-08-18) describes a production runtime-monitoring
architecture, verbatim:

> "**activation classifiers** — detectors that inspect a model's internal
> activity — that **run at every sampled token** and **escalate potential
> concerns to increasingly sophisticated and high-compute automated
> investigators**"

examining "**tool actions, available reasoning, and the full sequence of
activity** for unauthorized access, data theft, destructive behavior, and
attempts to defeat safeguards."

**The cost figure must always carry its denominator.** OpenAI states
"monitoring overhead at roughly 20% of **the inference compute being
monitored**" — not total inference, and not training. And monitoring is
deliberately narrow: required for RL training and evaluations with tools
for Sol-capability-or-higher models, extended on August 7 to all Astra
inference with tools. **An ADR that reads "monitoring costs 20%" without
that scoping overstates the budget by a large and unknown factor.** The
fleet-wide cost is a design choice, not a constant.

**Evidence grade A−**: `openai.com` returns HTTP 403 to automated fetches,
so this was verified via three independent outlets quoting the post
verbatim and agreeing word-for-word. A human should open the primary URL
before this citation ships externally.

**The more actionable detail, which the briefing omitted**: a **30-minute
alerting SLO**. On a likely critical-boundary violation the system raises
a highest-priority alert, pages the safety/security/research teams, "and
if they cannot conclusively determine within 30 minutes that the flag is a
false positive, those teams are expected to **pause the activity**." A
fail-closed human-in-the-loop gate on a wall clock.

**What already exists here — and it is the hard part.**
`crates/ruvector-tiny-dancer-core/src/voi.rs` shipped in Wave 4
(ADR-331), and its own module docstring names this exact use case:

> "`decide` is a standalone pure function so the same primitive can gate
> model selection (this crate's Router integration...), retrieval depth,
> **verifier invocations**, agent spawning, and **escalation**. Only the
> router integration is implemented here; the others are expected to
> construct their own `EstimatorSpec` ladders over the same API."

The mapping is one-to-one: `decide(belief, alternative, estimators,
config) -> VoiDecision::Buy(usize) | VoiDecision::Route`, where each
`EstimatorSpec { cost, latency_us, noise_std }` **is one investigator
rung** — cheap and noisy first, expensive and sharp last. The purchase
inequality `value_of_success × VoI − monetized_cost > 0` **is** "investigate
when P(violation) × damage > verification cost", with `value_of_success`
carrying the damage term. `observe()` performs the Bayesian update between
rungs (posterior variance never increases, proptested).

**Greenfield here**: the cheap detector, the risk-score plumbing, the
ladder wiring, and any token-level hook. `crates/mcp-gate/` is the natural
in-repo mount point (it already serves and gates schemas).

**A name-confusion warning worth recording.** Token-level stream
inspection belongs in **`ruvnet/midstream`** (external — "analyzes
responses as they stream in real-time"), with `ruvnet/llm-stream-reformat`
built on it. `ruvector`'s own `crates/mcp-brain-server/src/midstream.rs`
is the **pi.ruv.io brain AGI-diagnostics subsystem** (ADR-077/078) — a
completely different thing that happens to share the name. Do not conflate
them.

## Decision

Build an escalation ladder in `crates/mcp-gate/` that **consumes**
`voi::decide` unchanged.

1. **Ladder shape.** stream → cheap detector → risk score. Low risk
   continues; medium buys a local verifier; high buys a strong-model
   investigator; critical stops execution via RVM. Each purchasable rung
   is an `EstimatorSpec`; `decide` chooses whether the next rung is worth
   its price.
2. **`voi.rs` is consumed, not forked.** No reimplementation of the
   closed-form index, no copy of the Gaussian machinery. If the ladder
   needs something the primitive lacks, it is added *there*, under
   ADR-331's gates.
3. **The unconditional-inspection floor is not subject to the
   economics.** Privilege escalation, network access, credential use,
   runtime mutation, and destructive operations are inspected **always**,
   whatever the VoI computation returns. The economics govern
   **discretionary** investigation only. This is the monitoring analog of
   ADR-331's escalate-only invariant: the failure direction is
   conservative by construction.
4. **Cost target, stated as a target and not as a claim**: under 5%
   average monitoring overhead, measured against a real unmonitored
   baseline on the identical path, with downgrade-only accounting — a
   skipped inspection may never be scored as a cheap inspection. Until
   measured, the figure is a design goal.
5. **Four binding caveats inherited from the primitive**, each recorded in
   `voi.rs` and each a real failure mode here:
   - **Every rung must have `cost > 0`, or the caller must cap rounds.**
     `EstimatorSpec::validate` permits `cost == 0`, and a free rung is
     bought forever — the module's own docstring records a probe running
     100,000 rounds still returning `Buy`. A monitoring ladder that never
     terminates is a denial of service on itself.
   - **A `noise_std == 0` rung is an oracle: take its verdict and exit the
     protocol.** `decide()` may return `Buy` for such a rung, and
     `observe()` deliberately refuses it. Piping an oracle purchase into
     `observe` is a contract violation.
   - **`value_of_success` must be the currency value of catching a real
     violation.** VoI is bounded by ≈`0.4σ`; left at a nominal 1.0, the
     gate degenerates into a **never-investigate switch that still looks
     configured** — the identical failure mode ADR-331 records for
     routing, and the reason it is repeated here rather than cross-
     referenced.
   - **Rescale utilities to unit range.** The Abramowitz–Stegun 7.1.26
     `norm_cdf` approximation (abs. error ≤ 1.5e-7) biases
     one-directionally in the deep tail, toward worthless purchases.
6. **Adopt the fail-closed alerting SLO.** A flagged likely violation of a
   critical boundary that cannot be cleared as a false positive within a
   bounded wall-clock window **pauses the activity**. The window is
   configurable; the fail-closed direction is not.
7. **Token-level inspection is out of scope for this repo.** It belongs in
   `ruvnet/midstream`. This ADR covers request- and tool-call-level
   monitoring at the `mcp-gate` boundary.

## Consequences

### Positive

- The hard part — a calibrated, closed-form, auditable purchase rule — is
  already built, tested, and proptested. This ADR is wiring, not
  mathematics.
- One shared primitive now governs model selection *and* monitoring
  escalation, so the two cannot drift into divergent cost heuristics.
- The unconditional floor means the economically-cheapest configuration is
  still safe on the operations that matter, which is what makes an
  adaptive budget defensible at all.

### Negative

- **A monitoring gate that decides not to investigate is a gate that can
  fail open.** Every one of the four caveats above is a way to
  accidentally build a never-investigate switch. The unconditional floor
  is the backstop, and it is why the floor is not negotiable.
- The "under 5%" target is unvalidated. It is materially below OpenAI's
  ~20%, but their denominator is a deliberately narrow high-risk slice
  while ours would be broader — the two numbers are **not comparable**,
  and this ADR does not claim to beat theirs.
- Risk-score calibration is the whole ballgame and is unsolved here. A
  miscalibrated detector makes every downstream purchase decision wrong in
  the same direction.

## Security / Validation Gates

- **Unconditional-inspection floor** (blocking, Decision §3) — privilege
  escalation, network access, credential use, runtime mutation, and
  destructive operations are inspected regardless of VoI output.
  Regression-tested by configuring an economics that would suppress all
  discretionary investigation and asserting the five classes are still
  inspected.
- **Termination** (blocking, Decision §5) — every rung `cost > 0` or an
  explicit round cap, validated at ladder construction, not at first use.
- **Oracle rungs exit the protocol** (blocking) — a `noise_std == 0`
  purchase must never reach `observe()`.
- **Non-finite rejection at the choke point** — inherited from ADR-331;
  any non-finite surviving a clamp resolves toward **investigating**,
  never toward skipping.
- **Downgrade-only accounting** — a skipped or partial inspection may
  never be recorded as a completed one. The 5% figure cannot be achieved
  by accounting.
- **Fail-closed alerting** (Decision §6) — an uncleared critical flag
  pauses the activity.
- **Standard repo gate**: `cargo nextest run -p mcp-gate` plus
  `npx @claude-flow/cli@latest security scan` (this touches a trust
  surface).

## Affected Repos

- `ruvnet/ruvector` — `crates/mcp-gate/` (detector, ladder, accounting).
  Consumes `crates/ruvector-tiny-dancer-core/src/voi.rs` unchanged.
- `ruvnet/midstream` — token-level stream inspection, **out of scope this
  wave**, recorded so the boundary is explicit.

## Dependencies

Hard: ADR-331's `voi.rs` (merged, WP28). Soft: ADR-317 (HarnessRisk) for
the lifecycle-security posture; ADR-336 for the "critical → stop
execution" arm, which is an RVM capability and therefore cross-repo.

## Alternatives Considered

- **Copy OpenAI's architecture including the 20% budget**: rejected — the
  denominator is a narrow high-risk slice, not fleet-wide inference.
  Copying the ratio without the scoping would misbudget by an unknown
  factor.
- **Reimplement the VoI computation inside `mcp-gate`**: rejected — that
  is exactly the "more independent primitives instead of strengthening the
  substrate" failure mode the Wave-5 briefing names. `voi.rs` was written
  for this.
- **Inspect everything, always**: rejected as the default — it is the safe
  direction and is retained *for the five high-risk classes*, but applying
  it universally is the cost profile the adaptive ladder exists to avoid.
- **Activation-level classifiers at every sampled token**: rejected for
  this repo — that requires model-internal access this program does not
  have at the `mcp-gate` boundary. The stream-level analog belongs in
  `ruvnet/midstream`.
- **Name a module "midstream"**: rejected — `mcp-brain-server/src/
  midstream.rs` already means something else here (pi.ruv.io brain
  diagnostics, ADR-077/078).
