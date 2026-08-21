import assert from "node:assert/strict";
import test from "node:test";
import * as designer from "../src/environmentDesigner.js";
import {
  SPADE_CITATION,
  assertExternalEvidence,
  atFrontier,
  buildEnvironmentProposal,
  evidenceReadingAgent,
  externalGroundingVetoProvider,
  generateEnvironment,
  groundingGatedVetoProvider,
  measureRegret,
  staticEvidenceResolver,
  toContextCandidateMutation,
  toDreamCandidate,
  verificationKeyFor,
  type EnvironmentProvenance,
  type EnvironmentSpec,
  type ExternalEvidence,
  type RegretMeasurement,
} from "../src/environmentDesigner.js";
import { composeVetoProviders, type VetoContext } from "../src/vetoes.js";
import { expandsCapabilities, type ContextGenome } from "../src/genome.js";

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

// A REAL external-evidence trace: a test in this very harness (repo-test kind).
const REAL_EVIDENCE: ExternalEvidence = {
  kind: "repo-test",
  locator: "crates/ruvector-sota-bench/harness/test/skillForge.test.ts",
  contentSha256: "a".repeat(64),
  description: "SkillForge covered-target corroboration test — a real, checked-in test.",
};

const GROUNDED_PROVENANCE: EnvironmentProvenance = {
  evidence: [REAL_EVIDENCE],
  designerNote: "Derived a curriculum from the corroboration test's covered behavior.",
  citation: SPADE_CITATION,
};

/** A self-invented, closed-loop provenance — NO external evidence. */
const UNGROUNDED_PROVENANCE: EnvironmentProvenance = {
  evidence: [],
  designerNote: "Invented from my own prior generations.",
  citation: SPADE_CITATION,
};

const groundedSpec: EnvironmentSpec = {
  id: "env-grounded-1",
  task: "Recover the behavior pinned by the covered test.",
  provenance: GROUNDED_PROVENANCE,
};

const ungroundedSpec: EnvironmentSpec = {
  id: "env-selfinvented-1",
  task: "A closed-loop challenge with no external anchor.",
  provenance: UNGROUNDED_PROVENANCE,
};

const PARENT_CONTEXT: ContextGenome = {
  surface: "context",
  version: 3,
  entries: { "env:seed": "existing curriculum entry" },
};

// A resolver that knows the REAL evidence (and nothing forged/invented).
const resolver = staticEvidenceResolver([REAL_EVIDENCE]);

test("GENERATE: an executable environment is built from an external-evidence fixture", () => {
  const generated = generateEnvironment(groundedSpec);
  assert.equal(generated.spec.id, "env-grounded-1");
  assert.match(generated.verificationKey, /^[0-9a-f]{16}$/);
  // The key is derived FROM the external evidence — uncomputable without it.
  assert.equal(generated.verificationKey, verificationKeyFor(GROUNDED_PROVENANCE));

  const env = generated.environment;
  // Gym-style reset()/step()/reward()/verify() contract (SPADE arXiv:2608.19197).
  const start = env.reset();
  assert.equal(start.solved, false);
  assert.equal(env.reward(start), 0);
  // A wrong action does not solve; the right (evidence-derived) key does.
  const wrong = env.step(start, "nope");
  assert.equal(wrong.done, true);
  assert.equal(wrong.reward, 0);
  assert.equal(env.verify(wrong.state), false);
  const right = env.step(start, generated.verificationKey);
  assert.equal(right.reward, 1);
  assert.equal(env.verify(right.state), true);
  // Privileged reset reveals the key in the observation; normal does not.
  assert.match(env.reset({ privileged: true }).observation.prompt, /key=[0-9a-f]{16}/);
  assert.doesNotMatch(env.reset().observation.prompt, /key=/);
  // No performance magnitude leaks into the environment text (preprint rule).
  assert.doesNotMatch(start.observation.prompt, /\d+(\.\d+)?\s*(%|percent|points?|pp)\b/i);

  // Shape validation: malformed evidence is rejected at generation.
  assert.throws(() => assertExternalEvidence({ ...REAL_EVIDENCE, contentSha256: "z" }), /64 lowercase hex/);
  // A self-invented environment is REPRESENTABLE (so the veto can reject it).
  assert.doesNotThrow(() => generateEnvironment(ungroundedSpec));
});

test("REGRET: the difficulty signal is the reward gap between normal and privileged play", async () => {
  const generated = generateEnvironment(groundedSpec);
  const measurement = await measureRegret(generated.environment, evidenceReadingAgent, 5);
  // The reference agent solves WITH privileged evidence and cannot WITHOUT it,
  // so a fresh grounded environment yields the maximal regret signal.
  assert.equal(measurement.normalReward, 0);
  assert.equal(measurement.privilegedReward, 1);
  assert.equal(measurement.regret, 1);
  assert.equal(measurement.normalSamples.length, 5);
  assert.equal(measurement.privilegedSamples.length, 5);

  // A MASTERED environment (an agent that solves even without privilege) yields
  // no regret — difficulty follows the learner, so this drops out of scope.
  const masteredAgent = () => 1;
  const mastered = await measureRegret(generated.environment, masteredAgent, 4);
  assert.equal(mastered.regret, 0);

  assert.rejects(() => measureRegret(generated.environment, evidenceReadingAgent, 1), /at least 2 episodes/);
});

test("EXTERNAL-GROUNDING VETO (headline security gate): grounded passes, ungrounded/forged REJECTED", async () => {
  const context: VetoContext = { policy: { ef_search: "100" }, suite: { id: "fixture", items: [] }, observations: [] };

  // 1. A grounded environment whose evidence RESOLVES passes the gate (no veto).
  const grounded = generateEnvironment(groundedSpec);
  const passReasons = await externalGroundingVetoProvider(() => grounded, resolver)(context);
  assert.deepEqual(passReasons, []);

  // 2. A self-invented environment (no external trace) is VETOED.
  const ungrounded = generateEnvironment(ungroundedSpec);
  const absentReasons = await externalGroundingVetoProvider(() => ungrounded, resolver)(context);
  assert.ok(absentReasons.includes("external_grounding_absent"));

  // 3. A FORGED trace (a locator that resolves to nothing) is VETOED.
  const forged = generateEnvironment({
    id: "env-forged-1",
    task: "Cites a source that does not exist.",
    provenance: {
      evidence: [{ ...REAL_EVIDENCE, locator: "test/does-not-exist.test.ts" }],
      designerNote: "citation of convenience",
      citation: SPADE_CITATION,
    },
  });
  const forgedReasons = await externalGroundingVetoProvider(() => forged, resolver)(context);
  assert.ok(forgedReasons.includes("external_grounding_unresolved"));

  // 4. A DRIFTED trace (resolves, but the content hash changed) is VETOED — a
  //    real cite cannot be silently reused after its source moved.
  const drifted = generateEnvironment({
    id: "env-drift-1",
    task: "Cites a real source at a stale hash.",
    provenance: {
      evidence: [{ ...REAL_EVIDENCE, contentSha256: "b".repeat(64) }],
      designerNote: "stale",
      citation: SPADE_CITATION,
    },
  });
  const driftReasons = await externalGroundingVetoProvider(() => drifted, resolver)(context);
  assert.ok(driftReasons.includes("external_grounding_drift"));

  // 5. CANNOT BE PROMOTED: composed conjunctively with a "no objection"
  //    downstream, an ungrounded environment still yields a non-empty veto set,
  //    which the flywheel promotion rule treats as a hard block.
  const composed = composeVetoProviders(
    externalGroundingVetoProvider(() => ungrounded, resolver),
    () => [],
  );
  assert.ok((await composed(context)).length > 0);
});

test("VETO FIRES BEFORE EVALUATION: the grounding gate short-circuits the dream-machine stage", async () => {
  const context: VetoContext = { policy: { ef_search: "100" }, suite: { id: "fixture", items: [] }, observations: [] };

  // The downstream provider models the dream-machine evaluation stage.
  let evaluationRan = false;
  const evaluationStage = () => {
    evaluationRan = true;
    return [];
  };

  // Ungrounded ⇒ the gate returns its reason and the evaluation stage NEVER runs.
  const ungrounded = generateEnvironment(ungroundedSpec);
  const blockedGate = groundingGatedVetoProvider(() => ungrounded, resolver, evaluationStage);
  const blocked = await blockedGate(context);
  assert.ok(blocked.includes("external_grounding_absent"));
  assert.equal(evaluationRan, false, "an ungrounded environment must never reach evaluation");

  // Grounded ⇒ the gate passes control to the evaluation stage.
  evaluationRan = false;
  const grounded = generateEnvironment(groundedSpec);
  const openGate = groundingGatedVetoProvider(() => grounded, resolver, evaluationStage);
  assert.deepEqual(await openGate(context), []);
  assert.equal(evaluationRan, true, "a grounded environment proceeds to evaluation");
});

test("FRONTIER: promotable only if solvable WITH evidence AND gap-exposing WITHOUT it", () => {
  const measurement = (normal: number, privileged: number): RegretMeasurement => ({
    normalReward: normal,
    privilegedReward: privileged,
    regret: privileged - normal,
    normalSamples: [normal, normal],
    privilegedSamples: [privileged, privileged],
  });

  // Solvable (privileged=1 ≥ 0.5) AND gap-exposing (regret=1 ≥ 0.1) ⇒ frontier.
  const frontier = atFrontier(measurement(0, 1));
  assert.equal(frontier.atFrontier, true);
  assert.equal(frontier.solvable, true);
  assert.equal(frontier.exposesGap, true);

  // Mastered: no regret ⇒ NOT at the frontier (retirement candidate).
  const mastered = atFrontier(measurement(1, 1));
  assert.equal(mastered.atFrontier, false);
  assert.ok(mastered.reasons.includes("mastered_no_regret_signal"));

  // Unsolvable even with privileged evidence ⇒ NOT at the frontier.
  const unsolvable = atFrontier(measurement(0, 0.2));
  assert.equal(unsolvable.atFrontier, false);
  assert.ok(unsolvable.reasons.includes("not_solvable_with_privileged_evidence"));
});

test("CONSTITUTIONAL BOUNDARY: an environment is a PROPOSAL, no promotion reachable", async () => {
  // 1. No export looks like a promotion capability (same contract as WP2/WP9/WP19).
  const forbidden = /promote|merge|approve|apply|commit|push/i;
  for (const [name] of Object.entries(designer)) {
    assert.ok(!forbidden.test(name), `export "${name}" looks like a promotion capability`);
  }

  const generated = generateEnvironment(groundedSpec);
  const frontier = atFrontier(await measureRegret(generated.environment));
  const proposal = buildEnvironmentProposal(PARENT_CONTEXT, generated, frontier);

  // 2. The proposal is frozen data with no callable members.
  assert.equal(proposal.kind, "environment-proposal");
  assert.ok(Object.isFrozen(proposal));
  for (const [key, value] of Object.entries(proposal)) {
    assert.notEqual(typeof value, "function", `proposal.${key} must be data, not a capability`);
  }

  // 3. It feeds WP9's genome as a CANDIDATE context mutation (+1 entry, version
  //    bumped) — an untrusted proposal, not a merge.
  const mutation = proposal.candidateMutation;
  assert.equal(mutation.candidate.surface, "context");
  assert.equal(mutation.candidate.version, PARENT_CONTEXT.version + 1);
  if (mutation.candidate.surface !== "context") assert.fail("expected context surface");
  assert.equal(Object.keys(mutation.candidate.entries).length, Object.keys(PARENT_CONTEXT.entries).length + 1);
  // A non-capability-expanding proposal is not routed to the gate at all.
  assert.equal(proposal.blockedByGate, false);
  assert.equal(proposal.capabilityGate, undefined);

  // 4. The parent genome is not mutated in place.
  assert.equal(Object.keys(PARENT_CONTEXT.entries).length, 1);
  const dream = toDreamCandidate(generated, await measureRegret(generated.environment), COMMIT);
  assert.equal(dream.commit, COMMIT);
});

test("CAPABILITY EXPANSION: an environment certifying a new capability routes to the ADR-315 stub and is BLOCKED", () => {
  const generated = generateEnvironment({
    id: "env-expanding-1",
    task: "Certifies a capability that would grant a new tool.",
    provenance: GROUNDED_PROVENANCE,
    capabilityDelta: { addsTools: ["shell_exec"], addsActionClasses: [], addsCommunicationPeers: [] },
  });
  assert.ok(expandsCapabilities(generated.spec.capabilityDelta));

  const frontier = atFrontier({
    normalReward: 0, privilegedReward: 1, regret: 1, normalSamples: [0, 0], privilegedSamples: [1, 1],
  });
  const proposal = buildEnvironmentProposal(PARENT_CONTEXT, generated, frontier);

  // Routed to the constitutional gate stub — blocked by default (no WP11 wiring).
  assert.equal(proposal.blockedByGate, true);
  assert.ok(proposal.capabilityGate);
  assert.equal(proposal.capabilityGate.allowed, false);
  assert.ok(proposal.capabilityGate.reasons.some((r) => r.includes("constitutional_gate_stub_unwired")));

  // The capabilityDelta rides onto the WP9 candidate mutation so shaperLoop's
  // own gate re-adjudicates it — this source bypasses no ADR-313/ADR-315 gate.
  assert.ok(expandsCapabilities(proposal.candidateMutation.capabilityDelta));

  // Even at the frontier, a capability-expanding environment stays blocked.
  assert.equal(proposal.frontier.atFrontier, true);
  assert.equal(toContextCandidateMutation(PARENT_CONTEXT, generated).candidate.surface, "context");
});
