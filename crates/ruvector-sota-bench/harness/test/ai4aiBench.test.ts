import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import test from "node:test";
import {
  AI4AI_BENCH_CITATION,
  ai4aiLineageVetoProvider,
  ai4aiPairedDecision,
  ai4aiTaskIdentity,
  assertAi4aiTaskManifest,
  commandAi4aiExecutor,
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

const manifest: Ai4aiTaskManifest = {
  benchmark: "einsia-ai4ai-bench",
  taskId: "task-03-policy-gradient",
  algorithmFamily: "policy-gradient",
  repoSnapshotSha256: hex("repo"),
  evaluatorSha256: hex("evaluator"),
  citation: AI4AI_BENCH_CITATION,
};

const mutation = (id: string, seed: number, description = "reshape the update rule"): Ai4aiMutation => ({
  id, seed, description, diffSha256: hex(`diff-${id}`),
});

const okExecutor: Ai4aiExecutor = async () => ({
  raw: { score: 0.18, algorithm_changed: true },
  executorIdentity: hex("executor-ok"),
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

test("fixture end-to-end lineage chains parent digests from the task identity", async () => {
  const executor = commandAi4aiExecutor({
    executorPath: fixture,
    commandPrefixArgs: [process.execPath],
  });
  const records = await runAi4aiLineage(
    manifest,
    [mutation("mut-001", 3), mutation("mut-002", 7), mutation("mut-003", 5, "tuning sweep only")],
    executor,
  );
  assert.equal(records.length, 3);
  assert.equal(records[0]!.parentDigest, ai4aiTaskIdentity(manifest));
  assert.equal(records[1]!.parentDigest, records[0]!.digest);
  assert.equal(records[2]!.parentDigest, records[1]!.digest);
  assert.equal(records[0]!.evaluation.submissionKind, "algorithm-change");
  assert.equal(records[2]!.evaluation.submissionKind, "tuning-only");
  assert.ok(records.every((record) => record.executorIdentity === records[0]!.executorIdentity));
  assert.deepEqual(verifyAi4aiLineage(manifest, records), []);
});

test("executor crash throws and is never scored", async () => {
  const executor = commandAi4aiExecutor({
    executorPath: fixture,
    commandPrefixArgs: [process.execPath],
  });
  await assert.rejects(
    runAi4aiMutation(manifest, mutation("mut-bad", 1, "crash on purpose"), executor, ai4aiTaskIdentity(manifest)),
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
    executorIdentity: hex("executor-nan"),
  });
  await assert.rejects(
    runAi4aiMutation(manifest, mutation("mut-nan", 2), nanExecutor, ai4aiTaskIdentity(manifest)),
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
  const records = await runAi4aiLineage(manifest, [mutation("mut-001", 3), mutation("mut-002", 7)], okExecutor);
  const tampered = [
    records[0]!,
    { ...records[1]!, evaluation: { ...records[1]!.evaluation, score: 0.99 } },
  ];
  assert.deepEqual(verifyAi4aiLineage(manifest, tampered), ["ai4ai_lineage_digest_mismatch"]);
  const broken = [records[1]!];
  assert.ok(verifyAi4aiLineage(manifest, broken).includes("ai4ai_lineage_broken_chain"));

  const provider = ai4aiLineageVetoProvider(() => ({ manifest, records: tampered }));
  assert.deepEqual(
    await provider({ policy: {}, suite: { id: "s", items: [] }, observations: [] } as never),
    ["ai4ai_lineage_digest_mismatch"],
  );
  const clean = ai4aiLineageVetoProvider(() => ({ manifest, records }));
  assert.deepEqual(
    await clean({ policy: {}, suite: { id: "s", items: [] }, observations: [] } as never),
    [],
  );
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
