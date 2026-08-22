import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import {
  AI4AI_BENCH_CITATION,
  ai4aiEvidenceClass,
  ai4aiLineageVetoProvider,
  ai4aiPairedDecision,
  ai4aiRunRoot,
  ai4aiTaskIdentity,
  assertAi4aiTaskManifest,
  commandAi4aiExecutor,
  newAi4aiRunNonce,
  parseAi4aiEvaluatorOutput,
  runAi4aiLineage,
  runAi4aiMutation,
  verifyAi4aiLineage,
  type Ai4aiExecutor,
  type Ai4aiMutation,
  type Ai4aiTaskManifest,
} from "../src/ai4aiBench.js";

const hex = (seed: string) => createHash("sha256").update(seed).digest("hex");
const fixture = resolve(import.meta.dirname, "../../test/fixtures/fake-ai4ai-evaluator.mjs");

// The manifest pins the evaluator by its REAL content hash: the adapter binds
// what actually ran to this value, so a stand-in constant would not run.
const evaluatorSha256 = createHash("sha256").update(await readFile(fixture)).digest("hex");

const manifest: Ai4aiTaskManifest = {
  benchmark: "einsia-ai4ai-bench",
  taskId: "task-03-policy-gradient",
  algorithmFamily: "policy-gradient",
  repoSnapshotSha256: hex("repo"),
  evaluatorSha256,
  citation: AI4AI_BENCH_CITATION,
};

const mutation = (id: string, seed: number, description = "reshape the update rule"): Ai4aiMutation => ({
  id, seed, description, diffSha256: hex(`diff-${id}`),
});

const fixtureExecutor = commandAi4aiExecutor({
  executorPath: fixture,
  commandPrefixArgs: [process.execPath],
});

/** An in-process executor that reports the pinned evaluator identity. */
const okExecutor: Ai4aiExecutor = async () => ({
  raw: { score: 0.18, algorithm_changed: true },
  executorIdentity: evaluatorSha256,
});

test("manifest without hashes is refused", () => {
  assert.throws(
    () => assertAi4aiTaskManifest({ ...manifest, repoSnapshotSha256: "" }),
    /SHA-256 repo snapshot identity/,
  );
  assert.throws(
    () => assertAi4aiTaskManifest({ ...manifest, evaluatorSha256: "DEADBEEF" }),
    /SHA-256 evaluator identity/,
  );
  assert.throws(
    () => assertAi4aiTaskManifest({ ...manifest, citation: "AI4AI" }),
    /exact citation/,
  );
});

test("fixture end-to-end lineage chains parent digests from the run root", async () => {
  const lineage = await runAi4aiLineage(
    manifest,
    [mutation("mut-001", 3), mutation("mut-002", 7), mutation("mut-003", 5, "tuning sweep only")],
    fixtureExecutor,
  );
  const { records } = lineage;
  assert.equal(records.length, 3);
  assert.equal(records[0]!.parentDigest, ai4aiRunRoot(manifest, lineage.runNonce));
  assert.notEqual(records[0]!.parentDigest, ai4aiTaskIdentity(manifest));
  assert.equal(records[1]!.parentDigest, records[0]!.digest);
  assert.equal(records[2]!.parentDigest, records[1]!.digest);
  assert.equal(records[0]!.evaluation.submissionKind, "algorithm-change");
  assert.equal(records[2]!.evaluation.submissionKind, "tuning-only");
  assert.ok(records.every((record) => record.executorIdentity === evaluatorSha256));
  assert.ok(records.every((record) => record.wrapperIndirection === false));
  assert.deepEqual(verifyAi4aiLineage(manifest, lineage), []);
});

test("byte-derived identity cannot be self-reported: the echo-executor probe fails", async () => {
  // The probe: an executor that runs NOTHING, echoes the manifest's public
  // pin, and returns a perfect score. It must not be able to mint a lineage
  // that passes as evaluator-bound evidence.
  const echo: Ai4aiExecutor = async () => ({
    raw: { score: 0.99, algorithm_changed: true },
    executorIdentity: manifest.evaluatorSha256,
  });
  const minted = await runAi4aiLineage(manifest, [mutation("mut-echo", 1)], echo);

  // It runs and records honestly — but as `injected`, never byte-derived.
  assert.equal(minted.records[0]!.executorClass, "injected");
  assert.equal(ai4aiEvidenceClass(minted), "injected");
  // A gate demanding evaluator-bound evidence gets a distinct, actionable reason
  // instead of the indistinguishable clean verdict this used to return.
  assert.deepEqual(
    verifyAi4aiLineage(manifest, minted, { requireEvaluatorBound: true }),
    ["ai4ai_executor_not_byte_bound"],
  );
  const strict = ai4aiLineageVetoProvider(
    () => ({ manifest, lineage: minted }),
    { requireEvaluatorBound: true },
  );
  assert.deepEqual(
    await strict({ policy: {}, suite: { id: "s", items: [] }, observations: [] } as never),
    ["ai4ai_executor_not_byte_bound"],
  );
  // The class is inside the digest body, so it cannot be flipped after the fact.
  const forged = {
    runNonce: minted.runNonce,
    records: [{ ...minted.records[0]!, executorClass: "byte-derived-command" as const }],
  };
  assert.ok(verifyAi4aiLineage(manifest, forged).includes("ai4ai_lineage_digest_mismatch"));
});

test("a real command run is byte-derived and bound to the pinned evaluator", async () => {
  const lineage = await runAi4aiLineage(manifest, [mutation("mut-001", 3)], fixtureExecutor);
  assert.equal(lineage.records[0]!.executorClass, "byte-derived-command");
  assert.equal(lineage.records[0]!.executorIdentity, evaluatorSha256);
  assert.equal(ai4aiEvidenceClass(lineage), "byte-derived-command");
  // Passes even under the strictest policy — this is what real evidence looks like.
  assert.deepEqual(verifyAi4aiLineage(manifest, lineage, { requireEvaluatorBound: true }), []);

  // A manifest pinning a different evaluator is refused at run time, because
  // here the module itself knows what ran.
  const wrongManifest: Ai4aiTaskManifest = { ...manifest, evaluatorSha256: hex("declared-but-not-run") };
  await assert.rejects(
    runAi4aiMutation(
      wrongManifest,
      mutation("mut-x", 1),
      fixtureExecutor,
      ai4aiRunRoot(wrongManifest, newAi4aiRunNonce()),
    ),
    /does not match the manifest's pinned evaluator/,
  );
});

test("a declared wrapper is honestly classed and distinguishable under policy", async () => {
  const wrapped: Ai4aiExecutor = async () => ({
    raw: { score: 0.2, algorithm_changed: true },
    executorIdentity: hex("a-declared-wrapper"),
    wrapperIndirection: true,
  });
  const wrappedLineage = await runAi4aiLineage(manifest, [mutation("mut-w", 2)], wrapped);
  assert.equal(wrappedLineage.records[0]!.wrapperIndirection, true);
  assert.equal(wrappedLineage.records[0]!.executorClass, "wrapper-waived");
  // Not blocked by default — real Docker/GPU runs legitimately wrap.
  assert.deepEqual(verifyAi4aiLineage(manifest, wrappedLineage), []);
  // But a gate that needs evaluator-bound evidence can tell it apart.
  assert.deepEqual(
    verifyAi4aiLineage(manifest, wrappedLineage, { requireEvaluatorBound: true }),
    ["ai4ai_evaluator_identity_waived"],
  );
  assert.equal(ai4aiEvidenceClass(wrappedLineage), "wrapper-waived");
});

test("run nonce distinguishes runs: identical inputs cannot be byte-identical", async () => {
  const nonce = newAi4aiRunNonce();
  const first = await runAi4aiLineage(manifest, [mutation("mut-001", 3)], okExecutor, nonce);
  const replay = await runAi4aiLineage(manifest, [mutation("mut-001", 3)], okExecutor);
  // Identical inputs, different runs: digests must differ. This is a run
  // DISTINGUISHER only — re-presenting a whole {runNonce, records} pair still
  // verifies; anti-replay is out of scope for this slice.
  assert.notEqual(first.records[0]!.digest, replay.records[0]!.digest);
  // A chain presented under someone else's nonce does not verify.
  assert.ok(
    verifyAi4aiLineage(manifest, { runNonce: replay.runNonce, records: first.records })
      .includes("ai4ai_lineage_broken_chain"),
  );
  assert.deepEqual(
    verifyAi4aiLineage(manifest, { runNonce: "not-a-nonce", records: first.records }),
    ["ai4ai_lineage_run_nonce_invalid"],
  );
});

test("executor crash throws and is never scored", async () => {
  await assert.rejects(
    runAi4aiMutation(
      manifest,
      mutation("mut-bad", 1, "crash on purpose"),
      fixtureExecutor,
      ai4aiRunRoot(manifest, newAi4aiRunNonce()),
    ),
    /crash is not a score/,
  );
});

test("non-finite and out-of-range scores are refused at ingest", async () => {
  assert.throws(() => parseAi4aiEvaluatorOutput({ score: Number.NaN }), /finite/);
  assert.throws(() => parseAi4aiEvaluatorOutput({ score: Number.POSITIVE_INFINITY }), /finite/);
  assert.throws(() => parseAi4aiEvaluatorOutput({ score: 1.5 }), /out of range/);
  assert.throws(() => parseAi4aiEvaluatorOutput({ score: "0.2" }), /finite/);
  const nanExecutor: Ai4aiExecutor = async () => ({
    raw: { score: Number.NaN },
    executorIdentity: evaluatorSha256,
  });
  await assert.rejects(
    runAi4aiMutation(manifest, mutation("mut-nan", 2), nanExecutor, ai4aiRunRoot(manifest, newAi4aiRunNonce())),
    /finite/,
  );
});

test("classification is ingested only when reported, never fabricated", () => {
  assert.equal(parseAi4aiEvaluatorOutput({ score: 0.2 }).submissionKind, "unreported");
  assert.equal(
    parseAi4aiEvaluatorOutput({ score: 0.2, algorithm_changed: false }).submissionKind,
    "tuning-only",
  );
  assert.throws(() => parseAi4aiEvaluatorOutput({ score: 0.2, algorithm_changed: "yes" }), /boolean/);
});

test("lineage verification detects tampering and the veto provider objects", async () => {
  const lineage = await runAi4aiLineage(
    manifest,
    [mutation("mut-001", 3), mutation("mut-002", 7)],
    okExecutor,
  );
  const tampered = {
    runNonce: lineage.runNonce,
    records: [
      lineage.records[0]!,
      {
        ...lineage.records[1]!,
        evaluation: { ...lineage.records[1]!.evaluation, score: 0.99 },
      },
    ],
  };
  assert.deepEqual(verifyAi4aiLineage(manifest, tampered), ["ai4ai_lineage_digest_mismatch"]);
  assert.ok(
    verifyAi4aiLineage(manifest, { runNonce: lineage.runNonce, records: [lineage.records[1]!] })
      .includes("ai4ai_lineage_broken_chain"),
  );

  const context = { policy: {}, suite: { id: "s", items: [] }, observations: [] } as never;
  const provider = ai4aiLineageVetoProvider(() => ({ manifest, lineage: tampered }));
  assert.deepEqual(await provider(context), ["ai4ai_lineage_digest_mismatch"]);
  const clean = ai4aiLineageVetoProvider(() => ({ manifest, lineage }));
  assert.deepEqual(await clean(context), []);
});

test("paired decision refuses non-finite samples and otherwise defers to the bootstrap", () => {
  assert.throws(() => ai4aiPairedDecision([0.1, Number.NaN], [0.2, 0.2]), /non-finite/);
  const decision = ai4aiPairedDecision(
    [0.10, 0.11, 0.10, 0.12, 0.11],
    [0.18, 0.19, 0.17, 0.20, 0.18],
  );
  assert.equal(decision.outcome, "pass");
  const flat = ai4aiPairedDecision([0.1, 0.1, 0.1], [0.1, 0.1, 0.1]);
  assert.notEqual(flat.outcome, "pass");
});
